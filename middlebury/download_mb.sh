#! /bin/sh

mkdir -p data.mb/unzip
cd data.mb/unzip

# 2014 dataset
wget --no-check-certificate -r -np -A png,pfm,txt -X "/stereo/data/scenes2014/datasets/*-perfect/" http://vision.middlebury.edu/stereo/data/scenes2014/datasets/

# eval3 train/test set
wget --no-check-certificate http://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip
unzip MiddEval3-data-H.zip
