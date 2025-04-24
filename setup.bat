@echo off
setlocal

set "MODELS_URL=https://sdk.deepx.ai/res/assets/clip_assets.tar.gz"
set "VIDEOS_URL=https://sdk.deepx.ai/res/video/videos.tar.gz"
set "MODELS_PATH=assets"
set "VIDEOS_PATH=assets\demo_videos"

if not exist "%MODELS_PATH%" (
    echo "not found models, make dir %MODELS_PATH%"
    mkdir "%MODELS_PATH%"
    echo "Download clip_assets.tar.gz file URL : %MODELS_URL%"
    curl -o clip_assets.tar.gz %MODELS_URL%
    if %errorlevel%==0 (
        echo "extract clip_assets.tar.gz to %MODELS_PATH%"
        tar -xzvf clip_assets.tar.gz -C %MODELS_PATH%
        if %errorlevel%==0 (
            del /f /q clip_assets.tar.gz
            echo "remove clip_assets.tar.gz"
        )
    )
    if %errorlevel%==0 (
        echo "Download clip_assets.tar.gz and extract models : Complete"
    )
) else (
    echo "models directory found (%MODELS_PATH%)"
    echo "stop downloading models tar gz"
)

if not exist "%VIDEOS_PATH%" (
    echo "not found videos, make dir %VIDEOS_PATH%"
    mkdir "%VIDEOS_PATH%"
    echo "Download videos tar.gz file URL : %VIDEOS_URL%"
    curl -o videos.tar.gz %VIDEOS_URL%
    if %errorlevel%==0 (
        echo "extract videos.tar.gz to %VIDEOS_PATH%"
        tar -xzvf videos.tar.gz -C %VIDEOS_PATH%
        if %errorlevel%==0 (
            del /f /q videos.tar.gz
            echo "remove videos.tar.gz"
        )
    )
    if %errorlevel%==0 (
        echo "Download videos.tar.gz and extract videos : Complete"
    )
) else (
    echo "videos directory found (%VIDEOS_PATH%)"
    echo "stop downloading videos tar gz"
)

pause
endlocal