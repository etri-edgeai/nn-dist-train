#!/usr/bin/env bash

cd data
mkdir ./landmark
cd landmark
wget --no-check-certificate --no-proxy https://fedcv.s3-us-west-1.amazonaws.com/landmark/data_user_dict.zip
wget --no-check-certificate --no-proxy https://fedcv.s3-us-west-1.amazonaws.com/landmark/images.zip
tar xvf data_user_dict.zip
tar xvf images.zip
cd ../../