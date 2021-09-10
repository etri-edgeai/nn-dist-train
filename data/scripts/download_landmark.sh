#!/usr/bin/env bash

cd data/benchmarks/landmark
wget --no-check-certificate --no-proxy https://fedcv.s3-us-west-1.amazonaws.com/landmark/data_user_dict.zip
wget --no-check-certificate --no-proxy https://fedcv.s3-us-west-1.amazonaws.com/landmark/images.zip
unzip data_user_dict.zip
unzip images.zip
cd ../../../