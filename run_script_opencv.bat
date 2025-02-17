@echo off
setlocal

call clip-demo-package-executor-opencv\Scripts\activate

start cmd /K "python" clip_demo_app_opencv\dx_realtime_multi_demo.py

endlocal