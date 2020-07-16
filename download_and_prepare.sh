#!/bin/bash
wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
wget http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat
tar -xvf cars_train.tgz
tar -xvf cars_test.tgz
tar -xvf car_devkit.tgz
mv cars_test_annos_withlabels.mat devkit
python3 prepare_directories.py
