#!/usr/bin/env python3
"""MuJoCo offscreen renderer for Franka Panda.

Coordinate convention:
    OpenCV:  X-right, Y-down,  Z-forward  (looks at +Z)
    MuJoCo:  X-right, Y-up,    Z-backward (looks at -Z)
    Convert: R_mj = R_cv @ diag(1, -1, -1)
"""

import mujoco
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE_XML = _PROJECT_ROOT / "franka_model" / "scene_with_cam.xml"

_CV2MJ = np.diag([1.0, -1.0, -1.0])


def cam2base_to_mujoco(extrinsics_6d):
    """Convert DROID cam2base [tx,ty,tz,rx,ry,rz] to MuJoCo camera pose.

    DROID convention: p_base = R @ p_cam + t, R = euler('xyz', [rx,ry,rz])
    """
    pos = np.array(extrinsics_6d[:3])
    rot_cv = Rotation.from_euler('xyz', extrinsics_6d[3:6]).as_matrix()
    return pos, rot_cv @ _CV2MJ


def scale_intrinsics(intrinsics, calib_w, calib_h, target_w, target_h):
    fx, cx, fy, cy = intrinsics
    sx, sy = target_w / calib_w, target_h / calib_h
    return fx * sx, cx * sx, fy * sy, cy * sy


class FrankaRenderer:

    def __init__(self, model_path=None):
        model_path = model_path or str(DEFAULT_SCENE_XML)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'ext_cam')
        if self._cam_id < 0:
            raise RuntimeError("Model must have a camera named 'ext_cam'")

        self._renderers = {}
        self._floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')

    def _get_renderer(self, width, height):
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def set_joint_angles(self, joint_angles):
        n = min(len(joint_angles), self.model.nq)
        self.data.qpos[:n] = joint_angles[:n]
        if len(joint_angles) == 7 and self.model.nq >= 9:
            self.data.qpos[7] = self.data.qpos[8] = 0.04
        mujoco.mj_forward(self.model, self.data)

    def set_camera_from_droid(self, intrinsics, extrinsics,
                               calib_width, calib_height,
                               render_width, render_height):
        """Set camera from DROID calibration parameters."""
        pos, rot_mj = cam2base_to_mujoco(extrinsics)
        self.model.cam_pos[self._cam_id] = pos

        # cam_quat overrides cam_mat0 when non-zero
        q = Rotation.from_matrix(rot_mj).as_quat()  # scipy: [x,y,z,w]
        self.model.cam_quat[self._cam_id] = [q[3], q[0], q[1], q[2]]  # mujoco: [w,x,y,z]

        fx_s, cx_s, fy_s, cy_s = scale_intrinsics(
            intrinsics, calib_width, calib_height, render_width, render_height
        )

        if hasattr(self.model, 'cam_sensorsize'):
            W, H = render_width, render_height
            self.model.cam_sensorsize[self._cam_id] = [1.0, 1.0]
            self.model.cam_resolution[self._cam_id] = [W, H]
            self.model.cam_intrinsic[self._cam_id] = [
                fx_s / W, fy_s / H,
                (cx_s - W / 2.0) / W, (cy_s - H / 2.0) / H,
            ]
            self.model.cam_fovy[self._cam_id] = 0
        else:
            # Fallback: fovy only (no principal point offset)
            self.model.cam_fovy[self._cam_id] = np.degrees(
                2.0 * np.arctan(render_height / (2.0 * fy_s))
            )

    def _render_scene(self, width, height):
        renderer = self._get_renderer(width, height)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = self._cam_id
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def render(self, width, height):
        return self._render_scene(width, height)

    def render_robot_mask(self, width, height):
        """Render segmentation mask: 255 = robot, 0 = background."""
        renderer = self._get_renderer(width, height)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = self._cam_id
        renderer.update_scene(self.data, camera=camera)

        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        renderer.disable_segmentation_rendering()

        geom_ids = seg[:, :, 0]
        mask = np.logical_and(geom_ids >= 0, geom_ids != self._floor_id)
        return (mask * 255).astype(np.uint8)

    def project_fk_joints(self, intrinsics, extrinsics,
                           calib_width, calib_height,
                           output_width, output_height):
        """Project FK body positions to image pixels. Call after set_joint_angles."""
        rot_cv = Rotation.from_euler('xyz', extrinsics[3:6]).as_matrix()
        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = rot_cv
        T_cam2base[:3, 3] = extrinsics[:3]
        T_base2cam = np.linalg.inv(T_cam2base)

        fx, cx, fy, cy = intrinsics
        sx, sy = output_width / calib_width, output_height / calib_height

        points = []
        for bid in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not name or name == 'world':
                continue
            p_cam = T_base2cam[:3, :3] @ self.data.xpos[bid] + T_base2cam[:3, 3]
            if p_cam[2] <= 0:
                continue
            px = int((fx * p_cam[0] / p_cam[2] + cx) * sx)
            py = int((fy * p_cam[1] / p_cam[2] + cy) * sy)
            if 0 <= px < output_width and 0 <= py < output_height:
                points.append((px, py, name))
        return points

    def render_with_droid_params(self, joint_angles, intrinsics, extrinsics,
                                  calib_width, calib_height,
                                  output_width=None, output_height=None):
        out_w = output_width or 320
        out_h = output_height or 180

        self.set_joint_angles(joint_angles)
        self.set_camera_from_droid(intrinsics, extrinsics,
                                    calib_width, calib_height, out_w, out_h)
        # Must re-run forward so data.cam_xmat picks up the new cam_quat
        mujoco.mj_forward(self.model, self.data)

        return self.render(out_w, out_h), self.render_robot_mask(out_w, out_h)
