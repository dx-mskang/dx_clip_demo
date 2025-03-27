#!/bin/bash

APP_TYPE=""
ARCH_TYPE=""
SCRIPT_DIR=$(realpath "$(dirname "$0")")
DXRT_SRC_PATH="/deepx/dx_rt"
VENV_PATH="$SCRIPT_DIR/../../venv-${APP_TYPE}"
VENV_SYMLINK_TARGET_PATH=""

# Function to display help message
show_help() {
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo "Example: $0 --app_type=pyqt --dxrt_src_path=/deepx/dx_rt --venv_path=./venv-${APP_TYPE} --symlink_target_path=../workspace/venv/clip/pyqt"
  echo "Options:"
  echo "  --app_type=<str>                     Set Application type (pyqt | opencv)"
  echo "  --arch_type=<str>                    Set Archtecture type (aarch64 | amd64)"
  echo "  --dxrt_src_path=<dir>                Set DXRT source path (default: /deepx/dx_rt/)"
  echo "  --venv_path=<dir>                    Set virtual environment path (default: PROJECT_ROOT/venv-${APP_TYPE})"
  echo "  [--venv_symlink_target_path=<dir>]   Set symlink target path for venv (default: PROJECT_ROOT/../workspace/venv/clip/${APP_TYPE})"
  echo "  [--help]                             Show this help message"

  if [ "$1" == "error" ]; then
    echo "Error: Invalid or missing arguments."
    exit 1
  fi
  exit 0
}

make_venv(){
  echo "=== make_venv() ==="
  #### 1. Set up Virtual Environment

  # Print VENV_PATH for verification
  echo "Using virtual environment at: $VENV_PATH"

  # setting venv location 
  VENV_ORIGIN_DIR="$VENV_PATH"

  if [ -n "$VENV_SYMLINK_TARGET_PATH" ]; then
      # if '--venv_symlink_target_path' option is exist.
      VENV_ORIGIN_DIR="$VENV_SYMLINK_TARGET_PATH"
      echo "creating python venv to --venv_symlink_target_path: $VENV_ORIGIN_DIR"
  else
      echo "creating python venv to this path: $VENV_ORIGIN_DIR"
  fi

  # create venv
  python3 -m venv ${VENV_ORIGIN_DIR} --system-site-packages
      source ${VENV_ORIGIN_DIR}/bin/activate

  # create venv failed check
  if [ $? -ne 0 ]; then
      echo "Creation venv failed!"
      rm -rf "$VENV_ORIGIN_DIR"
      exit 1
  fi

  echo "Creation venv complete."
}

make_symlink() {
  echo "=== make_symlink() ==="
  # if '--symlink_target_path' option is exist, make symbolic link
  if [ -n "$VENV_SYMLINK_TARGET_PATH" ]; then
      mkdir -p "$(dirname "$VENV_PATH")"

      ln -s "$VENV_SYMLINK_TARGET_REAL_PATH" "$VENV_PATH"
      echo "Created symbolic link: $VENV_PATH -> $VENV_SYMLINK_TARGET_REAL_PATH"
  fi
}

setup_dx_engine(){
  echo "=== setup_dx_engine() ==="
  ### Setup DX-RunTime python package
  #### 2. Install dx_engine (DX-Runtime Python package)
  pushd ${DXRT_SRC_PATH}
  ./build.sh
  pushd ${DXRT_SRC_PATH}/python_package
  pip install .
  popd
  popd
}

setup_demo_app(){
  echo "=== setup_demo_app() ==="
  ### Setup Demo APP
  if [[ "$APP_TYPE" == "opencv" ]]; then
    echo "Running in OpenCV mode"
    #### 3. Install packages (gstreamer, qt5 multimedia plugins for play mp3, mp4, gif files)
    pip install -r requirements.${APP_TYPE}.txt
    pip install ./assets/CLIP
  elif [[ "$APP_TYPE" == "pyqt" ]]; then
    echo "Running in PyQt mode"
    #### 3. Install packages (gstreamer, qt5 multimedia plugins for play mp3, mp4, gif files)
    sudo apt-get install -y libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xfixes0 libxcb-shape0 libxcb-sync1 libxkbcommon-x11-0 libxcb-xkb1
    sudo apt-get install -y libqt5multimedia5-plugins libpulse-mainloop-glib0
    sudo apt-get install -y python3-pyqt5 python3-pyqt5.sip python3-pyqt5.qtmultimedia

    #### 4. Install pip packages
    pip install -r requirements.${APP_TYPE}.txt
    pip install ./assets/CLIP
  else
    echo "Error: APP_TYPE must be either 'opencv' or 'pyqt'." >&2
    exit 1
  fi
}

# Parse arguments
for i in "$@"; do
  case $i in
    --app_type=*)
      APP_TYPE="${i#*=}"
      ;;
    --arch_type=*)
      ARCH_TYPE="${i#*=}"
      ;;  
    --dxrt_src_path=*)
      DXRT_SRC_PATH="${i#*=}"
      ;;
    --venv_path=*)
      VENV_PATH="${i#*=}"

      # Symbolic link cannot be created when output_dir is the current directory.
      VENV_REAL_DIR=$(readlink -f "$VENV_PATH")
      CURRENT_REAL_DIR=$(readlink -f "./")
      if [ "$VENV_REAL_DIR" == "$CURRENT_REAL_DIR" ]; then
          echo "'--venv_path' is the same as the current directory. Please specify a different directory."
          exit 1
      fi
      ;;
    --venv_symlink_target_path=*)
      VENV_SYMLINK_TARGET_PATH="${1#*=}"
      VENV_SYMLINK_TARGET_REAL_PATH=$(readlink -f "$VENV_SYMLINK_TARGET_PATH")
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

# Check if DXRT_SRC_PATH exists
if [ ! -d "$DXRT_SRC_PATH" ]; then
  echo "Error: DXRT_SRC_PATH ($DXRT_SRC_PATH) does not exist."
  show_help "error"
fi

### Pre-Requisite
if [[ "$APP_TYPE" == "opencv" ]]; then
  echo "Running in OpenCV mode"
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update && sudo apt-get install -y python3 python3-dev python3-venv python3-tk
elif [[ "$APP_TYPE" == "pyqt" ]]; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update && sudo apt-get install -y python3 python3-dev python3-venv libxcb-xinerama0
else
  echo "Error: APP_TYPE must be either 'opencv' or 'pyqt'." >&2
  exit 1
fi

# If the VENV_SYMLINK_TARGET_PATH option is used and VENV_SYMLINK_TARGET_REAL_PATH exists
if [ -n "$VENV_SYMLINK_TARGET_PATH" ] && [ -d "$VENV_SYMLINK_TARGET_REAL_PATH" ]; then
    # Skip file download and create a symlink
    echo "VENV_SYMLINK_TARGET_PATH($VENV_SYMLINK_TARGET_PATH) is already exist. Skip to setup venv, then create a symlink"
    make_symlink
else
    make_venv
    make_symlink
    setup_dx_engine
    setup_demo_app
fi

exit 0

