#!/bin/bash

APP_TYPE=""
SCRIPT_DIR=$(realpath "$(dirname "$0")")
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_PATH="${PROJECT_ROOT}/venv-${APP_TYPE}"

source ${SCRIPT_DIR}/scripts/color_env.sh

check_virtualenv() {
    if [ -n "$VIRTUAL_ENV" ]; then
        venv_name=$(basename "$VIRTUAL_ENV")
        if [ "$venv_name" = "$APP_TYPE" ]; then
            echo "✅ Virtual environment '$venv_name' is currently active."
            return 0
        else
            echo "⚠️ A different virtual environment '$venv_name' is currently active."
            return 1
        fi
    else
        echo "❌ No virtual environment is currently active."
        return 1
    fi
}

install_deps(){
  echo -e "=== install_deps() ${TAG_START:-[START]} ==="

  ### Pre-Requisite
  if [[ "$APP_TYPE" == "opencv" ]]; then
    echo "Running in OpenCV mode"
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update && sudo apt-get install -y python3 python3-dev python3-venv python3-tk
  elif [[ "$APP_TYPE" == "pyqt" ]]; then
    echo "Running in PyQT mode"
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update && sudo apt-get install -y python3 python3-dev python3-venv libxcb-xinerama0
  else
    echo -e "${TAG_ERROR:-[ERROR]} APP_TYPE must be either 'opencv' or 'pyqt'." >&2
    exit 1
  fi

  ### Install dependencies for Demo APP
  if [[ "$APP_TYPE" == "opencv" ]]; then
    #### 3. Install packages (gstreamer, qt5 multimedia plugins for play mp3, mp4, gif files)
    pip install -r requirements.${APP_TYPE}.txt
    pip install ./assets/CLIP
  elif [[ "$APP_TYPE" == "pyqt" ]]; then
    #### 3. Install packages (gstreamer, qt5 multimedia plugins for play mp3, mp4, gif files)
    sudo apt-get install -y libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xfixes0 libxcb-shape0 libxcb-sync1 libxkbcommon-x11-0 libxcb-xkb1
    sudo apt-get install -y libqt5multimedia5-plugins libpulse-mainloop-glib0
    sudo apt-get install -y python3-pyqt5 python3-pyqt5.sip python3-pyqt5.qtmultimedia

    #### 4. Install pip packages
    pip install -r requirements.${APP_TYPE}.txt
    pip install ./assets/CLIP
  else
    echo -e "${TAG_ERROR:-[ERROR]} APP_TYPE must be either 'opencv' or 'pyqt'." >&2
    exit 1
  fi
  echo -e "=== install_deps() ${TAG_DONE:-[DONE]} ==="
}

activate_venv() {
    echo -e "=== activate_venv() ${TAG_START} ==="

    # activate venv
    source ${VENV_PATH}/bin/activate
    if [ $? -ne 0 ]; then
        echo -e "${TAG_ERROR} Activate venv failed! Please try installing again with the '--force' option."
        rm -rf "$VENV_PATH"
        echo -e "${TAG_ERROR} === ACTIVATE VENV FAIL ==="
        exit 1
    fi

    echo -e "=== activate_venv() ${TAG_DONE} ==="
}

main() {
    if check_virtualenv; then
        install_deps
    else
        if [ -d "$VENV_PATH" ]; then
            activate_venv
            install_deps
        else
            echo -e "${TAG_ERROR:-[ERROR]}${COLOR_BRIGHT_RED_ON_BLACK} Virtual environment '${VENV_PATH}' is not exist.\nPlease run 'setup.sh' to set up and activate the environment first.${COLOR_RESET}"
        fi
    fi
}

# Function to display help message
show_help() {
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo "Example: $0 --app_type=pyqt --venv_path=./venv-${APP_TYPE}"
  echo "Options:"
  echo "  --app_type=<str>                     Set Application type (pyqt | opencv)"
  echo "  [--help]                             Show this help message"

  if [ "$1" == "error" ]; then
    echo "Error: Invalid or missing arguments."
    exit 1
  fi
  exit 0
}

# Parse arguments
for i in "$@"; do
  case $i in
    --app_type=*)
      APP_TYPE="${i#*=}"
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

main

exit 0

