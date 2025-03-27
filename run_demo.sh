#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "$0")")
pushd $SCRIPT_DIR

# Function to display help message
show_help() {
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo "Options:"
  echo "  --app_type=<str>             Set Application type (pyqt | opencv)"
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
      shift
      ;;
    --help)
      show_help
      ;;
    *)
      # Unknown option
      ;;
  esac
done

# Check if APP_TYPE is valid
if [ "$APP_TYPE" != "pyqt" ] && [ "$APP_TYPE" != "opencv" ]; then
  echo "Error: APP_TYPE ($APP_TYPE) is invalid. It must be set to either 'pyqt' or 'opencv'."
  show_help "error"
fi

# Run the demo application
pushd clip_demo_app_${APP_TYPE}
./run_demo.sh
popd

popd