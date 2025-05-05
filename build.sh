#!/bin/bash
rm -rf build && mkdir -p build && cd build
cmake -DBUILD_DOCS=ON ..
make -j
make docs

# Note: currently torch is not needed. Just for reference.
# wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcu121.zip
# unzip libtorch-cxx11-abi-shared-with-deps-2.5.0+cu121.zip
# cmake .. -DCMAKE_PREFIX_PATH="$(pwd)/../libtorch"
