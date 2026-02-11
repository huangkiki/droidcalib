#!/usr/bin/env python3
"""Calibration quality scoring: rendered robot vs real image.

How it works
------------
We can't do pixel-wise comparison because the rendered image only has a white
robot on a checkerboard floor, while the real image has a complex scene. Instead
we use 5 proxy signals that together indicate whether the rendered robot mask
lands on the actual robot in the real image.

Scoring weights (tweak these to change behavior):

    WEIGHTS = {
        'edge':       0.30,  # rendered contour aligns with real edges
        'gradient':   0.25,  # edge direction matches in mask region
        'brightness': 0.20,  # mask region is brighter (Franka is white)
        'texture':    0.15,  # mask region has texture (not empty background)
        'coverage':   0.10,  # robot is visible in frame (not off-screen)
    }

Typical scores:
    0.80+ = good alignment, calibration is reliable
    0.50~0.70 = rough alignment, usable with caution
    < 0.40 = clearly off, calibration is bad or cam2cam chain drifted
"""

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────
# Weights for the overall score. Change these to emphasize different
# aspects. They should sum to 1.0.
# ──────────────────────────────────────────────────────────────────────
WEIGHTS = {
    'edge':       0.30,
    'gradient':   0.25,
    'brightness': 0.20,
    'texture':    0.15,
    'coverage':   0.10,
}

# If rendered robot covers less than this fraction of the image,
# it's probably off-screen and we give a near-zero score.
MIN_COVERAGE = 0.005

# edge_alignment: max pixel distance to count as "aligned"
EDGE_MAX_DIST = 15.0

# gradient_consistency: min gradient magnitude to consider
GRAD_THRESHOLD = 10.0

# brightness_contrast: scaling factors
BRIGHT_MEAN_SCALE = 3.0   # how much weight for mean brightness diff
BRIGHT_STD_SCALE = 0.5    # how much weight for inner region variance

# texture_presence: local variance normalization
TEXTURE_NORM = 500.0       # variance value that maps to score=1.0


# ──────────────────────────────────────────────────────────────────────
# Individual metrics
# ──────────────────────────────────────────────────────────────────────

def edge_alignment_score(real_image, rendered_mask):
    """How well the rendered contour sits on real image edges.

    For each point on the rendered mask contour, compute the distance to the
    nearest Canny edge in the real image (chamfer-like). Points within
    EDGE_MAX_DIST pixels score linearly from 1.0 (on edge) to 0.0.
    """
    mask_bool = rendered_mask > 128
    if mask_bool.sum() < 10:
        return 0.0

    contours, _ = cv2.findContours(rendered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0

    contour_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    if len(contour_pts) == 0:
        return 0.0

    gray = cv2.cvtColor(real_image, cv2.COLOR_RGB2GRAY) if real_image.ndim == 3 else real_image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
    edges = cv2.Canny(gray, 25, 80)

    # Only look for edges near the mask (no point searching the whole image)
    kernel_size = int(EDGE_MAX_DIST * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    search_region = cv2.dilate(rendered_mask, kernel)
    edges_nearby = edges & search_region

    if (edges_nearby > 0).sum() == 0:
        return 0.0

    # Distance transform from nearby edge pixels
    inv_edges = (255 - edges_nearby).astype(np.uint8)
    if inv_edges.min() > 0:
        return 0.0
    dist_map = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 3)

    h, w = dist_map.shape
    distances = []
    for x, y in contour_pts:
        if 0 <= y < h and 0 <= x < w:
            d = dist_map[y, x]
            if d < 1e10:
                distances.append(d)

    if not distances:
        return 0.0

    scores = np.clip(1.0 - np.array(distances) / EDGE_MAX_DIST, 0.0, 1.0)
    return float(scores.mean())


def gradient_consistency_score(real_image, rendered_image, rendered_mask):
    """Whether edge directions match in the mask region.

    Computes Sobel gradients on both images, then measures cos(angle) between
    gradient vectors at pixels where both have strong gradients. Uses abs(cos)
    because brightness can invert but edge orientation should still match.
    """
    mask_bool = rendered_mask > 128
    if mask_bool.sum() < 20:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    expanded = cv2.dilate(rendered_mask, kernel) > 128

    def to_gray_f32(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        return g.astype(np.float32)

    real_g = to_gray_f32(real_image)
    rend_g = to_gray_f32(rendered_image)

    real_gx = cv2.Sobel(real_g, cv2.CV_32F, 1, 0, ksize=3)
    real_gy = cv2.Sobel(real_g, cv2.CV_32F, 0, 1, ksize=3)
    rend_gx = cv2.Sobel(rend_g, cv2.CV_32F, 1, 0, ksize=3)
    rend_gy = cv2.Sobel(rend_g, cv2.CV_32F, 0, 1, ksize=3)

    real_mag = np.sqrt(real_gx**2 + real_gy**2)
    rend_mag = np.sqrt(rend_gx**2 + rend_gy**2)

    valid = expanded & (real_mag > GRAD_THRESHOLD) & (rend_mag > GRAD_THRESHOLD)
    if valid.sum() < 5:
        return 0.0

    dot = real_gx[valid] * rend_gx[valid] + real_gy[valid] * rend_gy[valid]
    cos_diff = dot / (real_mag[valid] * rend_mag[valid] + 1e-8)
    return float(np.abs(cos_diff).mean())


def brightness_contrast_score(real_image, rendered_mask):
    """Whether the mask region is brighter than background.

    Franka Panda is white/light gray, so the mask region should have higher
    brightness than the surroundings. Also checks that the region has some
    intensity variation (joints, shadows, etc).
    """
    mask_bool = rendered_mask > 128
    if mask_bool.sum() < 10 or (~mask_bool).sum() < 10:
        return 0.0

    gray = cv2.cvtColor(real_image, cv2.COLOR_RGB2GRAY) if real_image.ndim == 3 else real_image
    inner = gray[mask_bool].astype(np.float64)
    outer = gray[~mask_bool].astype(np.float64)

    mean_diff = abs(inner.mean() - outer.mean()) / 255.0
    inner_std = inner.std() / 128.0

    return float(min(mean_diff * BRIGHT_MEAN_SCALE + inner_std * BRIGHT_STD_SCALE, 1.0))


def texture_presence_score(real_image, rendered_mask):
    """Whether the mask region contains an object (not empty background).

    Measures local variance inside the mask. A flat wall/table has low variance;
    a robot arm with joints and shadows has high variance.
    """
    mask_bool = rendered_mask > 128
    if mask_bool.sum() < 10:
        return 0.0

    gray = cv2.cvtColor(real_image, cv2.COLOR_RGB2GRAY) if real_image.ndim == 3 else real_image
    gray_f = gray.astype(np.float32)
    mu = cv2.blur(gray_f, (5, 5))
    local_var = cv2.blur(gray_f**2, (5, 5)) - mu**2

    return float(min(local_var[mask_bool].mean() / TEXTURE_NORM, 1.0))


# ──────────────────────────────────────────────────────────────────────
# Overall score
# ──────────────────────────────────────────────────────────────────────

def evaluate_calibration_quality(real_image, rendered_image, rendered_mask):
    """Compute all metrics and weighted overall score.

    Args:
        real_image:    (H, W, 3) uint8 RGB from DROID
        rendered_image: (H, W, 3) uint8 RGB from MuJoCo
        rendered_mask:  (H, W) uint8, 255=robot

    Returns:
        dict with individual scores and 'overall_score'
    """
    coverage = float((rendered_mask > 128).sum() / rendered_mask.size) if rendered_mask.size > 0 else 0.0
    edge = edge_alignment_score(real_image, rendered_mask)
    gradient = gradient_consistency_score(real_image, rendered_image, rendered_mask)
    brightness = brightness_contrast_score(real_image, rendered_mask)
    texture = texture_presence_score(real_image, rendered_mask)

    if coverage < MIN_COVERAGE:
        # Robot barely visible => calibration probably way off
        overall = coverage * 20  # caps at ~0.1
    else:
        overall = (
            WEIGHTS['edge'] * edge +
            WEIGHTS['gradient'] * gradient +
            WEIGHTS['brightness'] * brightness +
            WEIGHTS['texture'] * texture +
            WEIGHTS['coverage'] * min(coverage * 5, 1.0)  # saturates at 20% coverage
        )

    return {
        'edge_alignment': edge,
        'gradient_consistency': gradient,
        'brightness_contrast': brightness,
        'texture_presence': texture,
        'mask_coverage': coverage,
        'overall_score': overall,
    }
