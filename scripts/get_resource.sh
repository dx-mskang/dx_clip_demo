#!/bin/bash

BASE_URL="https://sdk.deepx.ai/"

# default value
SOURCE_PATH=""
DOWNLOAD_DIR="./download"
OUTPUT_DIR=""
SYMLINK_TARGET_PATH=""

# Function to display help message
show_help() {
  
  echo "Usage: $(basename "$0") --src_path=<source_path> --output=<dir> [--symlink_target_path=<dir>]"
  echo "Example: $0 --src_path=res/assets/clip_assets.tar.gz --output=./assets --symlink_target_path=../workspace/res/clip_assets"
  echo "Options:"
  echo "  --src_path=<path>                Set source path for file server endpoint (example: res/assets/clip_assets.tar.gz)"
  echo "  --output=<path>                  Set output path (example: ./assets)"
  echo "  [--symlink_target_path=<path>]   Set symlink target path for output path (example: ../workspace/res/clip_assets)"
  echo "  [--help]                         Show this help message"

  if [ "$1" == "error" ]; then
    echo "Error: Invalid or missing arguments."
    exit 1
  fi
  exit 0
}

download_and_extract() {
    echo "=== download_and_extract() ==="
    URL="${BASE_URL}${SOURCE_PATH}"
    FILENAME=$(basename "$URL")

    # check curl and install curl
    if ! command -v curl &> /dev/null; then
        echo "curl is not installed. Installing..."
        sudo apt update && sudo apt install -y curl

        # curl install failed
        if ! command -v curl &> /dev/null; then
            echo "Failed to install curl. Exiting."
            exit 1
        fi
    fi

    mkdir -p "$DOWNLOAD_DIR"

    # download file
    echo "Downloading $FILENAME from $URL..."
    curl -o "$DOWNLOAD_DIR/$FILENAME" "$URL"

    # download failed check
    if [ $? -ne 0 ]; then
        echo "Download failed!"
        rm -rf "$DOWNLOAD_DIR"
        exit 1
    fi

    echo "Download complete."

    # setting extract location 
    EXTRACT_DIR="$OUTPUT_DIR"

    if [ -n "$SYMLINK_TARGET_PATH" ]; then
        # if '--symlink_target_path' option is exist.
        EXTRACT_DIR="$SYMLINK_TARGET_PATH"
        echo "Extracting to --symlink_target_path: $EXTRACT_DIR"
    else
        echo "Extracting to output path: $OUTPUT_DIR"
    fi

    # extract tar.gz
    mkdir -p "$EXTRACT_DIR"
    tar -xzf "$DOWNLOAD_DIR/$FILENAME" -C "$EXTRACT_DIR"

    # extract failed check
    if [ $? -ne 0 ]; then
        echo "Extraction failed!"
        rm -rf "$DOWNLOAD_DIR"
        rm -rf "$EXTRACT_DIR"
        exit 1
    fi

    echo "Extraction complete."
}

make_symlink() {
    echo "=== make_symlink() ==="
    # if '--symlink_target_path' option is exist, make symbolic link
    if [ -n "$SYMLINK_TARGET_PATH" ]; then
        mkdir -p "$(dirname "$OUTPUT_DIR")"
        SYMLINK_TARGET_REAL_PATH=$(readlink -f "$SYMLINK_TARGET_PATH")
        ln -s "$SYMLINK_TARGET_REAL_PATH" "$OUTPUT_DIR"
        echo "Created symbolic link: $OUTPUT_DIR -> $SYMLINK_TARGET_REAL_PATH"
    fi
}

# parse args
for i in "$@"; do
    case "$1" in
        --src_path=*)
            SOURCE_PATH="${1#*=}"
            ;;
        --output=*)
            OUTPUT_DIR="${1#*=}"

            # Symbolic link cannot be created when output_dir is the current directory.
            OUTPUT_REAL_DIR=$(readlink -f "$OUTPUT_DIR")
            CURRENT_REAL_DIR=$(readlink -f "./")
            if [ "$OUTPUT_REAL_DIR" == "$CURRENT_REAL_DIR" ]; then
                echo "'--output' is the same as the current directory. Please specify a different directory."
                exit 1
            fi
            ;;
        --symlink_target_path=*)
            SYMLINK_TARGET_PATH="${1#*=}"
            SYMLINK_TARGET_REAL_PATH=$(readlink -f "$SYMLINK_TARGET_PATH")
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

# usage
if [ -z "$SOURCE_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: SOURCE_PATH ($SOURCE_PATH) or OUTPUT_DIR ($OUTPUT_DIR) does not exist."
    show_help "error"
fi

# If the SYMLINK_TARGET_PATH option is used and SYMLINK_TARGET_REAL_PATH exists
if [ -n "$SYMLINK_TARGET_PATH" ] && [ -d "$SYMLINK_TARGET_REAL_PATH" ]; then
    # Skip file download and create a symlink
    make_symlink
else
    # Download and extract the files, then create a symlink
    download_and_extract
    make_symlink
fi

# cleanup
rm -rf "$DOWNLOAD_DIR"

exit 0

