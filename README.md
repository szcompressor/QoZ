# QoZ: Dynamic Quality Metric Oriented Error Bounded Lossy Compression for Scientific Datasets

## Introduction

This is the source code of the QoZ data compression introduced in the paper: QoZ: Dynamic Quality Metric Oriented Error Bounded Lossy Compression for Scientific Datasets ([paper link](https://www.computer.org/csdl/proceedings-article/sc/2022/544400a892/1I0bT6kfcas)). 

## Dependencies

Please Install the following dependencies before running the artiact evaluation experiments:


* numpy >= 1.21.2
* pandas >= 1.4.1
* cmake>=3.13
* gcc>=6.0

The following dependencies are only for SC 22' artifact evaluation:
* Python >= 3.6 
* qcat (from https://github.com/Meso272/qcat, check its readme for installation guides. Make sure the following executables are successfully installed: calculateSSIM and computeErrAutoCorrelation).

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

## SC 22' Evaluation guides (only for SC 22' AD evaluation)

### Steps
Step 1: Download the dataset from the following links,then unzip them:

* CESM-ATM: https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/CESM-ATM/SDRBENCH-CESM-ATM-cleared-1800x3600.tar.gz 
* Miranda: https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Miranda/SDRBENCH-Miranda-256x384x384.tar.gz
* Hurricane-ISABEL: https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500_log.tar.gz
* NYX: https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512_log.tar.gz
* SCALE-LETKF: https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/SCALE_LETKF/SDRBENCH-SCALE_98x1200x1200_log.tar.gz
* Data download command: wget {DataLink} --no-check-certificate (DataLink: The link of data)
* Data unzip command: tar -zxvf {DataFile} (DataFile: the .tar.gz file of downloaded data)

Step 2: Preprocess the downloaded Miranda data with the preprocess_data.py:

* python preprocess_data -m {MirandaPath} (MirandaPath is the folder of the Miranda Dataset)

Step 3: Run the generate_config.py to generate the configuration files for QoZ:

* python generate_config.py

After that, the configuration files for QoZ test will be generated in configs/ folder in the name format of {DatasetName}\_{Target}.config. 

* DatasetName: The name of dataset (cesm, miranda, hurricane, nyx, scale)
* Target: The optimization target, including cr (compression ratio), psnr (PSNR), ssim (SSIM) and ac (Autocorrelation).

Step 4: Run test_qoz.py to generate the test results.

* Command: python test_qoz.py -i {Datapath} -o {OutputPath} -d {DatasetName} -t {Target}
* Datapath: the folder path of the dataset.
* OutputPath: the output data file prefix. The output files will be in format of Outputpath_{Metric}.tsv
* DatasetName: See step 3
* Target: See step 3
* Metric: Includes compression ratio (overall_cr), psnr (overall_psnr), ssim (overall_ssim), autocorrelation (overall_ac), compression speed (cspeed) and decompression speed (dspeed).

### Output examples

The output files contain:
* overall_cr: the overall compression ratio under different vr-rel error bounds.
* overall_psnr: the overall compression PSNR under different vr-rel error bounds.
* cspeed: the compression speed under different vr-rel error bounds (the speed results in the paper are generated with psnr mode).
* dspeed: the decompression speed under different vr-rel error bounds (the speed results in the paper are generated with psnr mode).
* overall_ssim: the overall compression SSIM under different vr-rel error bounds (ssim target only).
* overall_ac: the overall compression Autocorrelation under different vr-rel error bounds (ac target only).

* For fast execution only part of the data points (error bounds) are covered in the output.
* The example outputs are in the results folder. 

The relationship of generated results and the provided results in the paper:

QoZ is a user-specific metric driven lossy compression, which means that it can provide different compression results with the same input and error bound if different compression mode (optimization target) is specified. Particularly speaking, currently there are 4 compression modes in QoZ, and the compression results of all the 4 modes are presented in table 3 and figure 8/9/10 of the paper. 
* The compression ratios generated using mode cr correspond to the table 3 in the paper.
* The cr and PSNR generated using mode psnr correspond to the figure 8 in the paper.
* The cr and SSIM generated using mode ssim correspond to the figure 9 in the paper.
* The cr and AC generated using mode ac correspond to the figure 10 in the paper.
* The compression/decompression speeds generated using mode psnr correspond to the table 4 in the paper (different results may achieved if not run on the same nodes listed in the paper).
* Bit rate = 32 / compression ratio.

### A full evaluation exmaple for MIRANDA dataset

* git clone https://github.com/Meso272/QoZ.git -b sc
* mkdir build
* cd build
* cmake -DCMAKE_INSTALL_PREFIX:PATH=qoz-install
* make
* make install
* add qoz-install to the system path (make sure that the qoz command is runable)
* cd .. (now in the root of QoZ)
* wget https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Miranda/SDRBENCH-Miranda-256x384x384.tar.gz
* tar -xzvf SDRBENCH-Miranda-256x384x384.tar.gz
* python3 preprocess_data.py -m build/SDRBENCH-Miranda-256x384x384
* python3 generate_config.py 
* python3 test_qoz.py -i SDRBENCH-Miranda-256x384x384 -o results/SDRBENCH-Miranda-256x384x384 -d miranda -t psnr
* cat results/SDRBENCH-Miranda-256x384x384_overall_cr.tsv

The output of this example is part of the results in figure 8 (b) of the paper.

### Figure Plotting:

In the plotting/ folder, there is an example of generating the plots in the paper, which corresponds to the Figure 8(b) in the paper. When running the test_qoz.py, if  mode psnr and dataset miranda are used, the output will be a part of the plotting/miranda.txt. To plot the rate-distortion, the gnuplot is need to be installed and please run the following command for plotting:

* cd plotting
* gnuplot psnr-MIRANDA.p

After that, a eps figure psnr-MIRANDA.eps will be generated, which is the Figure 8(b).

