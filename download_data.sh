#!/usr/bin/env bash
# 一键下载 DROID 数据集和标定文件
# 用法: bash download_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "  DroidCalib — 数据下载"
echo "================================================"

# --------------------------------------------------
# 1. DROID 100-episode 数据集 (~2.1GB)
# --------------------------------------------------
if [ -d "droid_100/1.0.0" ] && ls droid_100/1.0.0/*.tfrecord-* &>/dev/null; then
    echo "[1/2] droid_100/ 已存在，跳过"
else
    echo "[1/2] 下载 DROID 100-episode 数据集 (~2.1GB)..."
    mkdir -p droid_100/1.0.0
    gsutil -m cp "gs://gresearch/robotics/droid_100/1.0.0/*" droid_100/1.0.0/
    echo "  完成"
fi

# --------------------------------------------------
# 2. DROID 标定文件 (~340MB)
# --------------------------------------------------
CALIB_BASE="https://github.com/droid-dataset/droid/raw/main"
CALIB_FILES=(
    "cam2base_extrinsics.json"
    "cam2cam_extrinsics.json"
    "camera_serials.json"
    "intrinsics.json"
    "episode_id_to_path.json"
)

mkdir -p calibration
NEED_CALIB=0
for f in "${CALIB_FILES[@]}"; do
    [ -f "calibration/$f" ] || NEED_CALIB=1
done

if [ "$NEED_CALIB" -eq 0 ]; then
    echo "[2/2] calibration/ 已存在，跳过"
else
    echo "[2/2] 下载 DROID 标定文件 (~340MB)..."
    for f in "${CALIB_FILES[@]}"; do
        if [ ! -f "calibration/$f" ]; then
            echo "  下载 $f ..."
            curl -fSL -o "calibration/$f" "${CALIB_BASE}/${f}" 2>/dev/null \
                || wget -q -O "calibration/$f" "${CALIB_BASE}/${f}" \
                || echo "  警告: $f 下载失败，请手动从 DROID 仓库获取"
        fi
    done
    echo "  完成"
fi

# --------------------------------------------------
# 验证
# --------------------------------------------------
echo ""
echo "================================================"
echo "  验证"
echo "================================================"

check() {
    if [ -e "$1" ]; then
        echo "  OK  $1 ($(du -sh "$1" | cut -f1))"
    else
        echo "  !!  $1 缺失"
    fi
}

check "droid_100/1.0.0"
check "calibration/intrinsics.json"
check "calibration/cam2base_extrinsics.json"
check "calibration/camera_serials.json"
check "calibration/episode_id_to_path.json"

echo ""
echo "接下来运行:"
echo "  MUJOCO_GL=egl uv run python calib_filter/batch_filter.py --save-vis"
