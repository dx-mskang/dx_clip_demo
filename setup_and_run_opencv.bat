@echo off

set "ENV_FOLDER=venv-opencv"
set "SETUP_VENV=setup_clip_demo_app_opencv.bat"
set "RUN_DEMO=run_script_opencv.bat"

if exist "%~dp0%ENV_FOLDER%\" (
	echo call %RUN_DEMO% file ... 
) else (
	echo call %SETUP_VENV% file ....  
	call "%~dp0\scripts\x86_64_win\%SETUP_VENV%"
)

start "" dxrtd

call "%~dp0%RUN_DEMO%"

exit /b 0
