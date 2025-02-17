@echo off
setlocal

call venv-pyqt\Scripts\activate

start cmd /K "python" -m debugpy --listen 15346 clip_demo_app_pyqt\dx_realtime_demo_pyqt.py

endlocal