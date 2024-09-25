# QoZ 2.0

## Introduction

The second major version of QoZ: QoZ 2.0, is based on both QoZ 1.1 and the paper: "High-performance Effective Scientific Error-bounded Lossy Compression with Auto-tuned Multi-component Interpolation". [ACM Paper Link](https://dl.acm.org/doi/abs/10.1145/3639259) [arXiv Paper Link](https://arxiv.org/abs/2311.12133) 

## Dependencies

Please Install the following dependencies before compiling QoZ 2.0:

* cmake>=3.13
* gcc>=6.0

## 3rd party libraries/tools

* Zstd >= 1.3.5 (https://facebook.github.io/zstd/). Not mandatory to be manually installed as Zstandard v1.4.5 is included and will be used if libzstd can not be found by pkg-config.

## Installation

* mkdir build && cd build
* cmake -DCMAKE_INSTALL_PREFIX:PATH=[INSTALL_DIR] ..
* make
* make install

Then, you'll find all the executables in [INSTALL_DIR]/bin and header files in [INSTALL_DIR]/include. A Cmake version >= 3.13.0 is needed. 
Before you proceed to the following evaluations, please add the installation path of HPEZ to your system path so that you can directly run qoz command in your machine for further evaluations.

## Single compression/decompression testing Examples

You can use the executable 'qoz' command to do the compression/decompression (the input data should be float or double binary files). Just run "qoz" command without any argument to check the instructions for its arguments.
For the convenience of tests, the qoz executable includes the SZ3.1 compression, QoZ 1.1 compression, and 3 optimization levels of QoZ 2.0 compression. In the command:
* -q 0: SZ3.1 compression.
* Containing -q 1: QoZ 1.1 compression.
* Containing -q 2 or -q 3: 2 intermediate optimization levels of QoZ 2.0 compression (having faster speeds but slightly worse rate-distortion).
* Not containing -q argument or containing -q 4: Full QoZ 2.0 compression (for the results reported in the paper).

Notice: the integrated SZ3.1 and QoZ 1.1 in QoZ 2.0 have already leveraged the Fast-varying-first interpolation (proposed in our paper), therefore their compression ratios are sometimes higher than the original public released versions of SZ3.1 and QoZ 1.1.

## Test Dataset

Please download test datasets from: https://sdrbench.github.io/. 

