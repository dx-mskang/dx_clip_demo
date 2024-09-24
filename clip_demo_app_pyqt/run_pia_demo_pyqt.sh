#!/bin/bash
cd ..
source ./pia-package-executor-pyqt/bin/activate
python -m clip_demo_app_pyqt.dx_realtime_demo_pyqt "$@"
