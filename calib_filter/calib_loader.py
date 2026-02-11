#!/usr/bin/env python3
"""DROID calibration data loader.

Loads intrinsics, cam2base extrinsics, cam2cam extrinsics, and camera serials.
Supports chaining cam2cam relative poses to recover missing cam2base transforms.
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = _PROJECT_ROOT / "calibration"


class CalibrationDB:

    def __init__(self, calib_dir=None):
        calib_dir = Path(calib_dir) if calib_dir else CALIB_DIR

        print("Loading calibration data...")
        with open(calib_dir / "intrinsics.json") as f:
            self._intrinsics = json.load(f)
        with open(calib_dir / "cam2base_extrinsics.json") as f:
            self._extrinsics = json.load(f)
        with open(calib_dir / "camera_serials.json") as f:
            self._serials = json.load(f)

        c2c_path = calib_dir / "cam2cam_extrinsics.json"
        self._cam2cam = json.load(open(c2c_path)) if c2c_path.exists() else {}

        # Reverse index: relative_path -> calibration key
        self._path_to_key = {}
        for key, entry in self._extrinsics.items():
            rpath = entry.get('relative_path', '')
            if rpath:
                self._path_to_key[rpath] = key

        print(f"  intrinsics={len(self._intrinsics)}, extrinsics={len(self._extrinsics)}, "
              f"serials={len(self._serials)}, cam2cam={len(self._cam2cam)}, "
              f"path_index={len(self._path_to_key)}")

    def file_path_to_relative(self, file_path: str) -> Optional[str]:
        """Convert DROID episode file_path to relative path for calibration lookup."""
        idx = file_path.find('r2d2-data-full/')
        if idx >= 0:
            rel = file_path[idx + len('r2d2-data-full/'):]
            return rel.replace('/trajectory.h5', '').rstrip('/')

        known_labs = ['RAIL', 'RPL', 'AUTOLab', 'ILIAD', 'TRI', 'RAD',
                      'CLVR', 'IRIS', 'REAL', 'WEIRD', 'MILA', 'IPRL']
        parts = file_path.split('/')
        for i, p in enumerate(parts):
            if p in known_labs:
                rel = '/'.join(parts[i:])
                return rel.replace('/trajectory.h5', '').rstrip('/')
        return None

    def get_calib_key(self, file_path: str) -> Optional[str]:
        rel = self.file_path_to_relative(file_path)
        if rel and rel in self._path_to_key:
            return self._path_to_key[rel]
        return None

    def get_camera_serials(self, file_path: str) -> Optional[dict]:
        calib_key = self.get_calib_key(file_path)
        if calib_key and calib_key in self._serials:
            return self._serials[calib_key]
        return None

    def get_extrinsics(self, file_path: str) -> Optional[dict]:
        calib_key = self.get_calib_key(file_path)
        if calib_key and calib_key in self._extrinsics:
            return self._extrinsics[calib_key]
        return None

    def get_intrinsics(self, file_path: str, camera_serial: str = None) -> Optional[dict]:
        calib_key = self.get_calib_key(file_path)
        if calib_key is None or calib_key not in self._intrinsics:
            return None
        cam_intrinsics = self._intrinsics[calib_key]
        return cam_intrinsics.get(camera_serial) if camera_serial else cam_intrinsics

    def get_render_params(self, file_path: str, camera_type: str = 'ext1') -> Optional[dict]:
        """Get complete render parameters for a specific camera."""
        serials = self.get_camera_serials(file_path)
        if serials is None:
            return None

        serial_key = {'ext1': 'ext1_cam_serial', 'ext2': 'ext2_cam_serial',
                      'wrist': 'wrist_cam_serial'}.get(camera_type)
        if serial_key is None or serial_key not in serials:
            return None

        camera_serial = serials[serial_key]

        ext_data = self.get_extrinsics(file_path)
        if ext_data is None or camera_serial not in ext_data:
            return None

        intr_data = self.get_intrinsics(file_path, camera_serial)
        if intr_data is None:
            return None

        cam_matrix = intr_data['cameraMatrix']
        if cam_matrix[0] <= 0 or cam_matrix[2] <= 0:
            return None
        if intr_data['width'] <= 0 or intr_data['height'] <= 0:
            return None

        return {
            'intrinsics': cam_matrix,
            'extrinsics': ext_data[camera_serial],
            'calib_width': intr_data['width'],
            'calib_height': intr_data['height'],
            'camera_serial': camera_serial,
            'quality_metric': ext_data.get('quality_metric'),
            'source': ext_data.get('source'),
        }

    def _chain_cam2base(self, calib_key, known_serial, known_ext_6d, target_serial):
        """Compute target camera's cam2base via cam2cam relative pose chain.

        T_target_to_base = T_known_to_base @ inv(T_known_in_ref) @ T_target_in_ref
        """
        if calib_key not in self._cam2cam:
            return None

        c2c = self._cam2cam[calib_key]
        left, right = c2c.get('left_cam'), c2c.get('right_cam')
        if not left or not right:
            return None

        # Match serials to left/right by focal length
        ep_intr = self._intrinsics.get(calib_key, {})
        known_fx = ep_intr.get(known_serial, {}).get('cameraMatrix', [0])[0]
        target_fx = ep_intr.get(target_serial, {}).get('cameraMatrix', [0])[0]
        if known_fx <= 0 or target_fx <= 0:
            return None

        dl = abs(known_fx - left.get('focal', 0))
        dr = abs(known_fx - right.get('focal', 0))
        if dl < dr:
            known_cam, target_cam = left, right
            if abs(target_fx - right.get('focal', 0)) > 5:
                return None
        else:
            known_cam, target_cam = right, left
            if abs(target_fx - left.get('focal', 0)) > 5:
                return None

        def to44(pose):
            T = np.array(pose)
            return np.vstack([T, [0, 0, 0, 1]]) if T.shape == (3, 4) else T

        T_known_in_ref = to44(known_cam['pose'])
        T_target_in_ref = to44(target_cam['pose'])

        rot = Rotation.from_euler('xyz', known_ext_6d[3:6]).as_matrix()
        T_known_to_base = np.eye(4)
        T_known_to_base[:3, :3] = rot
        T_known_to_base[:3, 3] = known_ext_6d[:3]

        T_target_to_base = T_known_to_base @ np.linalg.inv(T_known_in_ref) @ T_target_in_ref

        t = T_target_to_base[:3, 3].tolist()
        r = Rotation.from_matrix(T_target_to_base[:3, :3]).as_euler('xyz').tolist()
        return t + r

    def get_all_render_params(self, file_path: str) -> dict:
        """Get render params for all available cameras, using cam2cam chaining as fallback."""
        result = {}
        for cam_type in ['ext1', 'ext2', 'wrist']:
            params = self.get_render_params(file_path, cam_type)
            if params is not None:
                result[cam_type] = params
        if not result:
            return result

        # Try cam2cam chaining for missing ext cameras
        calib_key = self.get_calib_key(file_path)
        serials = self.get_camera_serials(file_path)
        if not (calib_key and serials and self._cam2cam):
            return result

        known_type = next(iter(result))
        known = result[known_type]
        serial_keys = {'ext1': 'ext1_cam_serial', 'ext2': 'ext2_cam_serial'}

        for cam_type in ['ext1', 'ext2']:
            if cam_type in result:
                continue

            target_serial = serials.get(serial_keys[cam_type])
            if not target_serial:
                continue

            chained = self._chain_cam2base(
                calib_key, known['camera_serial'], known['extrinsics'], target_serial
            )
            if chained is None:
                continue

            intr_data = self.get_intrinsics(file_path, target_serial)
            if intr_data is None:
                continue
            if intr_data['cameraMatrix'][0] <= 0 or intr_data['cameraMatrix'][2] <= 0:
                continue

            result[cam_type] = {
                'intrinsics': intr_data['cameraMatrix'],
                'extrinsics': chained,
                'calib_width': intr_data['width'],
                'calib_height': intr_data['height'],
                'camera_serial': target_serial,
                'quality_metric': None,
                'source': 'cam2cam_chained',
            }

        return result
