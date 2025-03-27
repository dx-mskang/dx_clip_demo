#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "$0")")
pushd $SCRIPT_DIR

# Function to display help message
show_help() {
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo "Options:"
  echo "  --app_type=<str>             Set Application type (pyqt | opencv)"
  echo "  --dxrt_src_path=<path>       Set DXRT source path (default: /deepx/dx_rt/)"
  echo "  --docker_volume_path=<path>  Set Docker volume path (required in container mode)"
  echo "  --help                       Show this help message"

  if [ "$1" == "error" ]; then
    echo "Error: Invalid or missing arguments."
    exit 1
  fi
  exit 0
}

# Default values
DXRT_SRC_PATH="/deepx/dx_rt/"
DOCKER_VOLUME_PATH=${DOCKER_VOLUME_PATH}

# Parse arguments
for i in "$@"; do
  case $i in
    --app_type=*)
      APP_TYPE="${i#*=}"
      ;;
    --dxrt_src_path=*)
      DXRT_SRC_PATH="${i#*=}"
      ;;
    --docker_volume_path=*)
      DOCKER_VOLUME_PATH="${i#*=}"
      ;;
    --help)
      show_help
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
shift
done

# Check if APP_TYPE is valid
if [ "$APP_TYPE" != "pyqt" ] && [ "$APP_TYPE" != "opencv" ]; then
  echo "Error: APP_TYPE ($APP_TYPE) is invalid. It must be set to either 'pyqt' or 'opencv'."
  show_help "error"
fi

# Check if DXRT_SRC_PATH exists
if [ ! -d "$DXRT_SRC_PATH" ]; then
  echo "Error: DXRT_SRC_PATH ($DXRT_SRC_PATH) does not exist."
  show_help "error"
fi

# Detect system architecture (amd64 or aarch64)
ARCH_TYPE=$(uname -m)
if [[ "$ARCH_TYPE" == "x86_64" ]]; then
  ARCH_TYPE="amd64"
elif [[ "$ARCH_TYPE" == "aarch64" ]]; then
  ARCH_TYPE="aarch64"
else
  echo "Unsupported architecture: $ARCH_TYPE"
  exit 1
fi

ASSET_PATH=./assets
VIDEO_PATH=./assets/demo_videos
VENV_PATH="./venv-${APP_TYPE}"
CONTAINER_MODE=false

# Check if running in a container
if grep -qE "/docker|/lxc|/containerd" /proc/1/cgroup || [ -f /.dockerenv ]; then
    CONTAINER_MODE=true
    echo "(container mode detected)"
    
    if [ -z "$DOCKER_VOLUME_PATH" ]; then
        echo "Error: --docker_volume_path must be provided in container mode."
        show_help "error"
        exit 1
    fi

    SETUP_CLIP_ASSET_ARGS="--output=${ASSET_PATH} --symlink_target_path=${DOCKER_VOLUME_PATH}/res/clip"
    SETUP_CLIP_VIDEO_ARGS="--output=${VIDEO_PATH} --symlink_target_path=${DOCKER_VOLUME_PATH}/res/videos"
    VENV_SYMLINK_TARGET_PATH="${DOCKER_VOLUME_PATH}/venv/clip/${APP_TYPE}"
    VENV_SYMLINK_TARGET_PATH_ARGS="--venv_symlink_target_path=${VENV_SYMLINK_TARGET_PATH}"
else
    echo "(host mode detected)"
    SETUP_CLIP_ASSET_ARGS="--output=${ASSET_PATH} --symlink_target_path=../workspace/res/clip"
    SETUP_CLIP_VIDEO_ARGS="--output=${VIDEO_PATH} --symlink_target_path=../workspace/res/videos"
    VENV_SYMLINK_TARGET_PATH="../workspace/venv/clip/${APP_TYPE}"
    VENV_SYMLINK_TARGET_PATH_ARGS="--venv_symlink_target_path=${VENV_SYMLINK_TARGET_PATH}"
fi

echo "ASSET_PATH: ${ASSET_PATH}"
ASSET_REAL_PATH=$(readlink -f "$ASSET_PATH")
# Check and set up assets
if [ ! -d "$ASSET_REAL_PATH" ]; then
  echo "Assets directory not found. Running setup assets script... ($ASSET_REAL_PATH)"
  ./setup_clip_assets.sh $SETUP_CLIP_ASSET_ARGS || { echo "Setup assets script failed."; rm -rf $ASSET_PATH; exit 1; }
else
  echo "Assets directory found. ($ASSET_REAL_PATH)"
fi

echo "VIDEO_PATH: ${VIDEO_PATH}"
VIDEO_REAL_PATH=$(readlink -f "$VIDEO_PATH")
# Check and set up assets
if [ ! -d "$VIDEO_REAL_PATH" ]; then
  echo "Video directory not found. Running setup assets script... ($VIDEO_REAL_PATH)"
  ./setup_clip_videos.sh $SETUP_CLIP_VIDEO_ARGS || { echo "Setup assets script failed."; rm -rf $VIDEO_PATH; exit 1; }
else
  echo "Video directory found. ($VIDEO_REAL_PATH)"
fi

echo "VENV_PATH: ${VENV_PATH}"
VENV_REAL_PATH=$(readlink -f "$VENV_PATH")


# Check and set up virtual environment
if [ ! -f "$VENV_REAL_PATH/bin/activate" ]; then
  echo "Virtual environment not found. Running setup script... ($VENV_REAL_PATH)"
  RUN_SETUP_CMD="./scripts/setup_clip_demo_app.sh --app_type=${APP_TYPE} --arch_type=${ARCH_TYPE} --venv_path=${VENV_PATH} --dxrt_src_path=${DXRT_SRC_PATH} ${VENV_SYMLINK_TARGET_PATH_ARGS}"
  echo "CMD : $RUN_SETUP_CMD"
  
  $RUN_SETUP_CMD || { echo "Setup script failed."; rm -rf $VENV_PATH; exit 1; }
else
  echo "Virtual environment found. ($VENV_REAL_PATH)"
fi

popd
