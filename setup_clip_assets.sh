#!/bin/bash
sudo mkdir -p /mnt/regression_storage
sudo mount -o nolock 192.168.30.201:/do/regression /mnt/regression_storage

ls -al /mnt/regression_storage/atd/clip_assets.tar.gz

cp /mnt/regression_storage/atd/clip_assets.tar.gz .

tar xvfz ./clip_assets.tar.gz

rm -rf ./xvfz clip_assets.tar.gz

pushd ./assets
pip install dx_engine-1.0.0-py3-none-any.whl
popd