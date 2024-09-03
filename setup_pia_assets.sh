#!/bin/bash
sudo mkdir -p /mnt/regression_storage
sudo mount -o nolock 192.168.30.201:/do/regression /mnt/regression_storage

ls -al /mnt/regression_storage/atd/pia_assets.tar.gz

cp /mnt/regression_storage/atd/pia_assets.tar.gz .

tar xvfz ./pia_assets.tar.gz

rm -rf ./xvfz pia_assets.tar.gz

