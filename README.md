# QoZ: Dynamic Quality Metric Oriented Error Bounded Lossy Compression for Scientific Datasets

## Introduction

This is the source code of the QoZ data compression introduced in the paper: QoZ: Dynamic Quality Metric Oriented Error Bounded Lossy Compression for Scientific Datasets ([paper link](https://www.computer.org/csdl/proceedings-article/sc/2022/544400a892/1I0bT6kfcas)). 

## Dependencies

Please Install the following dependencies before running the artiact evaluation experiments:


* numpy >= 1.21.2
* pandas >= 1.4.1
* cmake>=3.13
* gcc>=6.0

## 3rd party libraries/tools

* Zstd >= 1.3.5 (https://facebook.github.io/zstd/). Not mandatory to be mannually installed as Zstandard v1.4.5 is included and will be used if libzstd can not be found by pkg-config.

## Installation

* mkdir build && cd build
* cmake -DCMAKE_INSTALL_PREFIX:PATH=[INSTALL_DIR] ..
* make
* make install

Then, you'll find all the executables in [INSTALL_DIR]/bin and header files in [INSTALL_DIR]/include. A Cmake version >= 3.13.0 is needed and we recommend to use gcc version 9.x to compile the code. 
Before you proceed to the following evaluations, please add the installation path of QoZ to your system path so that you can directly run qoz command in your machine for further evaluations.

## Single compression/decompression testing Examples

You can use the executable 'qoz' command to do the compression/decompression. Just run "qoz" command without any argument to check the instructions for its arguments.
The qoz executable includes the SZ3.1 compression, but the qoz features are automatically involved. You can add argument "-q 0" to disable them for comparing the compression results with SZ3-based compression (adding -q 0 will make qoz command perform the same as SZ3.1).
Currently you need to add a configuration file to the argument line (-c) for modifying the qoz-related parameters in the compression. 
By running 
* python generate_config.py

you can create some examples of the configuration file. The qoz-related parameters are on the line 14-20 of generate_config.py.



