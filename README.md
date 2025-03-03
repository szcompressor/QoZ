# QoZ 2.0 (HPEZ)

## Introduction

The second major version of QoZ: QoZ 2.0 (HPEZ), is based on both QoZ 1.1 and the paper: "High-performance Effective Scientific Error-bounded Lossy Compression with Auto-tuned Multi-component Interpolation". [ACM Paper Link](https://dl.acm.org/doi/abs/10.1145/3639259) [arXiv Paper Link](https://arxiv.org/abs/2311.12133) 

## Dependencies

Please install the following dependencies before compiling QoZ 2.0:

* cmake>=3.13
* gcc>=6.0

## 3rd party libraries/tools

* Zstd >= 1.3.5 (https://facebook.github.io/zstd/). It is not mandatory to manually install as Zstandard v1.4.5 is included and will be used if libzstd can not be found by pkg-config.

## Installation

* mkdir build && cd build
* cmake -DCMAKE_INSTALL_PREFIX:PATH=[INSTALL_DIR] ..
* make
* make install

Then, you'll find all the executables in [INSTALL_DIR]/bin and the header files in [INSTALL_DIR]/include. A Cmake version >= 3.13.0 is needed. 
Before you proceed to the following evaluations, please add the installation path of HPEZ to your system path so that you can directly run the qoz command in your machine for further evaluations.

## Command line compression/decompression 

You can use the executable **qoz** command to compress/decompress (the input data should be float or double binary files). Just run the **qoz** command without any argument to check the instructions for its arguments.
For the convenience of tests, the qoz executable includes the SZ3.1 compression, QoZ 1.1 compression, and 3 optimization levels of QoZ 2.0 compression. In the command:
* **-q 0**: SZ3.1 compression.
*  **-q 1**: QoZ 1.1 compression.
*  **-q 2** or **-q 3**: 2 intermediate optimization levels of QoZ 2.0 compression (having faster speeds but slightly worse rate-distortion).
*  **-q 4**: Full QoZ 2.0 compression (for the results reported in the paper).
Currently, **-q 3** is the default optimization level.

Notice: the integrated SZ3.1 and QoZ 1.1 in QoZ 2.0 have already leveraged the Fast-varying-first interpolation (proposed in our paper). Therefore, their compression ratios are sometimes higher than the original public released versions of SZ3.1 and QoZ 1.1.

## Bug report

If you have encountered any errors or abnormal results when using QoZ, please contact Jinyang Liu via jliu447@ucr.edu. 

## Citations

**Kindly note**: If you mention QoZ 2.0 (HPEZ) in your paper, the most appropriate citations are the following references:

* QoZ 2.0 (HPEZ): **[SIGMOD 24]** Jinyang Liu, Sheng Di, Kai Zhao, Xin Liang, Sian Jin, Zizhe Jian, Jiajun Huang, Shixun Wu, Zizhong Chen, and Franck Cappello. 2023. "[High-performance Effective Scientific Error-bounded Lossy Compression with Auto-tuned Multi-component Interpolation.](https://dl.acm.org/doi/abs/10.1145/3639259)" in Proceedings of the ACM on Management of Data 2, no. 1 (2024): 1-27.
* QoZ 1.0: **[SC 22]** Jinyang Liu, Sheng Di, Sian Jin, Kai Zhao, Xin Liang, Zizhong Chen, and Franck Cappello. "[Dynamic quality metric oriented error bounded lossy compression for scientific datasets.](https://ieeexplore.ieee.org/abstract/document/10046076)" In SC22: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1-15. IEEE, 2022.
