#!/usr/bin/env python3
"""Batch calibration quality filtering for DROID dataset.

Usage: python batch_filter.py [--save-vis] [--max-episodes N]
"""

import sys
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import time
import argparse
import numpy as np
import cv2

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_datasets as tfds

from mujoco_renderer import FrankaRenderer
from calib_loader import CalibrationDB
from quality_metrics import evaluate_calibration_quality


DATA_DIR = str(_PROJECT_ROOT / "droid_100" / "1.0.0")
RESULTS_DIR = str(_SCRIPT_DIR / "results")
VIS_DIR = os.path.join(RESULTS_DIR, "visualizations")

ALL_CAMS = ['ext1', 'ext2']
OBS_KEY_MAP = {
    'ext1': 'exterior_image_1_left',
    'ext2': 'exterior_image_2_left',
}


def _draw_label(img, text, pos=(8, 20), color=(0, 255, 255)):
    """Draw text with black background for readability."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x, y = pos
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def _make_cam_row(real_img, renderer, params, target_w, target_h, cam_label, score=None):
    """Generate a visualization row for one camera.

    With calibration: [FK projection | overlay | MuJoCo render]
    Without:          [raw image stretched to 3x width]
    """
    h, w = real_img.shape[:2]

    if params is None:
        row = cv2.resize(real_img, (target_w * 3, target_h))
        _draw_label(row, f"{cam_label}  [No Calibration]", color=(180, 180, 180))
        return row

    rgb, mask = renderer.render_with_droid_params(
        joint_angles=params['joints'],
        intrinsics=params['intrinsics'],
        extrinsics=params['extrinsics'],
        calib_width=params['calib_width'],
        calib_height=params['calib_height'],
        output_width=w,
        output_height=h,
    )

    # Left panel: FK joint projections as red dots
    fk_img = real_img.copy()
    fk_points = renderer.project_fk_joints(
        params['intrinsics'], params['extrinsics'],
        params['calib_width'], params['calib_height'],
        w, h,
    )
    for px, py, _ in fk_points:
        cv2.circle(fk_img, (px, py), 4, (255, 0, 0), -1)

    # Middle panel: green contour overlay + red dots
    overlay = real_img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    green_fill = real_img.copy().astype(np.float32)
    mask_bool = mask > 128
    green_fill[mask_bool] = green_fill[mask_bool] * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
    overlay = cv2.addWeighted(overlay.astype(np.float32), 0.7, green_fill, 0.3, 0).astype(np.uint8)
    for px, py, _ in fk_points:
        cv2.circle(overlay, (px, py), 4, (255, 0, 0), -1)

    row = np.hstack([fk_img, overlay, rgb])
    if row.shape[1] != target_w * 3 or row.shape[0] != target_h:
        row = cv2.resize(row, (target_w * 3, target_h))

    label = cam_label
    if score is not None:
        label += f"  Score: {score:.3f}"
    _draw_label(row, label)
    return row


def _extract_joints(step):
    """Extract 9-DOF joint angles (7 arm + 2 gripper) from a DROID step."""
    joint_pos = step['observation']['joint_position'].numpy()
    gripper_pos = step['observation']['gripper_position'].numpy()
    full = np.zeros(9)
    full[:7] = joint_pos
    full[7] = full[8] = gripper_pos[0] * 0.04
    return full


TOP_K_FRAMES = 5  # average the top-K frames by robot visibility


def process_episode(episode_idx, episode, calib, renderer, save_vis=False, sample_steps=15):
    file_path = episode['episode_metadata']['file_path'].numpy().decode('utf-8')
    rel_path = file_path.split('r2d2-data-full/')[-1] if 'r2d2-data-full/' in file_path else file_path
    lab = rel_path.split('/')[0]

    result = {
        'episode_idx': episode_idx,
        'file_path': rel_path,
        'lab': lab,
        'has_calibration': False,
        'calibration_complete': False,
        'cameras': {},
    }

    all_params = calib.get_all_render_params(file_path)
    if not all_params:
        return result

    result['has_calibration'] = True
    ext_cams = {k for k in all_params if k in ('ext1', 'ext2')}
    result['calibration_complete'] = ext_cams == {'ext1', 'ext2'}
    steps = list(episode['steps'])
    n_steps = len(steps)
    result['n_steps'] = n_steps

    # Evenly spaced sample across the episode (skip first/last 10%)
    margin = max(1, n_steps // 10)
    usable = n_steps - 2 * margin
    if usable <= 0 or sample_steps >= usable:
        step_indices = list(range(margin, n_steps - margin))
    else:
        step_indices = [margin + int(i * usable / sample_steps) for i in range(sample_steps)]

    mid_step = steps[n_steps // 2]

    # Score each camera across sampled steps
    for cam_type, params in all_params.items():
        if cam_type == 'wrist':
            continue
        obs_key = OBS_KEY_MAP.get(cam_type)
        if obs_key is None:
            continue

        cam_scores = []
        for step_idx in step_indices:
            step = steps[step_idx]
            if obs_key not in step['observation']:
                continue

            real_img = step['observation'][obs_key].numpy()
            real_h, real_w = real_img.shape[:2]
            full_joints = _extract_joints(step)

            try:
                rgb, mask = renderer.render_with_droid_params(
                    joint_angles=full_joints,
                    intrinsics=params['intrinsics'],
                    extrinsics=params['extrinsics'],
                    calib_width=params['calib_width'],
                    calib_height=params['calib_height'],
                    output_width=real_w,
                    output_height=real_h,
                )
                fk_pts = renderer.project_fk_joints(
                    params['intrinsics'], params['extrinsics'],
                    params['calib_width'], params['calib_height'],
                    real_w, real_h,
                )
                cam_scores.append(evaluate_calibration_quality(real_img, rgb, mask, fk_pts))
            except Exception as e:
                cam_scores.append({
                    'keypoint_appearance': 0, 'edge_alignment': 0,
                    'gradient_consistency': 0, 'brightness_contrast': 0,
                    'texture_presence': 0, 'mask_coverage': 0,
                    'overall_score': 0, 'error': str(e),
                })

        # Pick the top-K frames by robot visibility (mask_coverage), then average
        valid_scores = [s for s in cam_scores if 'error' not in s]
        valid_scores.sort(key=lambda s: s.get('mask_coverage', 0), reverse=True)
        valid_scores = valid_scores[:TOP_K_FRAMES]

        if valid_scores:
            avg_metrics = {}
            for key in valid_scores[0]:
                values = [s[key] for s in valid_scores]
                if values:
                    avg_metrics[key] = float(np.mean(values))

            result['cameras'][cam_type] = {
                'camera_serial': params['camera_serial'],
                'calib_quality': params['quality_metric'],
                'calib_source': params['source'],
                'n_samples': len(valid_scores),
                'metrics': avg_metrics,
            }

    # Visualization: stack ext1 + ext2 rows into one image
    if save_vis and result['cameras']:
        full_joints = _extract_joints(mid_step)
        ROW_W, ROW_H = 320, 180
        rows = []

        for cam_type in ALL_CAMS:
            obs_key = OBS_KEY_MAP[cam_type]
            if obs_key not in mid_step['observation']:
                continue

            real_img = mid_step['observation'][obs_key].numpy()
            calib_params = all_params.get(cam_type)
            if calib_params:
                render_params = {**calib_params, 'joints': full_joints}
                score = result['cameras'].get(cam_type, {}).get('metrics', {}).get('overall_score')
            else:
                render_params = None
                score = None

            rows.append(_make_cam_row(real_img, renderer, render_params, ROW_W, ROW_H, cam_type, score))

        if rows:
            vis = np.vstack(rows)
            vis_path = os.path.join(VIS_DIR, f"ep{episode_idx}.png")
            cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return result


def run_batch(max_episodes=None, save_vis=False, sample_steps=3):
    print("=" * 80)
    print("DroidCalib")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    print("\nLoading...")
    calib = CalibrationDB()
    renderer = FrankaRenderer()
    builder = tfds.builder_from_directory(DATA_DIR)
    ds = builder.as_dataset(split='train')

    results = []
    total = 0
    with_calib = 0
    complete = 0
    partial = 0
    t0 = time.time()

    for i, episode in enumerate(ds):
        if max_episodes and i >= max_episodes:
            break

        total += 1
        result = process_episode(i, episode, calib, renderer, save_vis, sample_steps)
        results.append(result)

        if result['has_calibration']:
            with_calib += 1
            if result['calibration_complete']:
                complete += 1
            else:
                partial += 1

            cam_info = []
            for cam_type, cam_data in result['cameras'].items():
                score = cam_data['metrics'].get('overall_score', 0)
                cam_info.append(f"{cam_type}={score:.3f}")
            tag = "OK" if result['calibration_complete'] else "PARTIAL"

            if cam_info:
                elapsed = time.time() - t0
                eps = total / elapsed if elapsed > 0 else 0
                print(f"  Ep {i:4d} | {result['lab']:10s} | {', '.join(cam_info):30s} | "
                      f"{tag:7s} | ({total} done, {eps:.1f} ep/s)")

    elapsed = time.time() - t0

    print(f"\n{'=' * 80}")
    print(f"Done: {total} episodes in {elapsed:.1f}s ({total/elapsed:.1f} ep/s)")
    print(f"  calibration: {with_calib}/{total} ({with_calib/total*100:.0f}%)")
    print(f"  complete (ext1+ext2): {complete}")
    print(f"  partial (one cam only): {partial}")
    print(f"  none: {total - with_calib}")

    all_scores = []
    for r in results:
        for cam_type, cam_data in r['cameras'].items():
            score = cam_data['metrics'].get('overall_score', 0)
            all_scores.append({
                'episode': r['episode_idx'],
                'lab': r['lab'],
                'camera': cam_type,
                'score': score,
                'droid_quality': cam_data['calib_quality'],
            })

    if all_scores:
        scores_array = [s['score'] for s in all_scores]
        print(f"\nScore distribution ({len(all_scores)} camera views):")
        print(f"  min={min(scores_array):.3f}  p25={np.percentile(scores_array, 25):.3f}  "
              f"median={np.percentile(scores_array, 50):.3f}  p75={np.percentile(scores_array, 75):.3f}  "
              f"max={max(scores_array):.3f}  mean={np.mean(scores_array):.3f}")

        labs = {}
        for s in all_scores:
            labs.setdefault(s['lab'], []).append(s['score'])

        print(f"\nBy lab:")
        for lab, lab_scores in sorted(labs.items()):
            print(f"  {lab:15s}: n={len(lab_scores):3d}, "
                  f"mean={np.mean(lab_scores):.3f}, "
                  f"min={min(lab_scores):.3f}, max={max(lab_scores):.3f}")

        threshold = 0.3
        low_quality = [s for s in all_scores if s['score'] < threshold]
        if low_quality:
            print(f"\nLow quality (< {threshold}): {len(low_quality)}/{len(all_scores)}")
            for s in low_quality[:10]:
                dq = f"{s['droid_quality']:.3f}" if s['droid_quality'] is not None else "N/A"
                print(f"  Ep {s['episode']:4d} | {s['lab']:10s} | {s['camera']} | "
                      f"score={s['score']:.3f} | droid_q={dq}")

    scores_array = [s['score'] for s in all_scores] if all_scores else []
    output = {
        'summary': {
            'total_episodes': total,
            'episodes_with_calibration': with_calib,
            'calibration_complete': complete,
            'calibration_partial': partial,
            'no_calibration': total - with_calib,
            'total_camera_views': len(all_scores),
            'processing_time_seconds': elapsed,
        },
        'score_distribution': {
            'min': float(min(scores_array)) if scores_array else 0,
            'p25': float(np.percentile(scores_array, 25)) if scores_array else 0,
            'median': float(np.percentile(scores_array, 50)) if scores_array else 0,
            'p75': float(np.percentile(scores_array, 75)) if scores_array else 0,
            'max': float(max(scores_array)) if scores_array else 0,
            'mean': float(np.mean(scores_array)) if scores_array else 0,
        },
        'episodes': results,
    }

    output_path = os.path.join(RESULTS_DIR, 'batch_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DroidCalib')
    parser.add_argument('--save-vis', action='store_true', help='save visualization images')
    parser.add_argument('--max-episodes', type=int, default=None, help='max episodes to process')
    parser.add_argument('--sample-steps', type=int, default=15, help='candidate frames to sample (top 5 by visibility are scored)')
    args = parser.parse_args()

    run_batch(
        max_episodes=args.max_episodes,
        save_vis=args.save_vis,
        sample_steps=args.sample_steps,
    )
