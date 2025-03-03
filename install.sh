#! /bin/bash

cd submodules/diff-surfel-rasterization
python -m pip install -e .
cd ../simple-knn
python -m pip install -e .
cd ../../