# CLVisc3.1


       ######  ##       ##     ## ####  ######   ######  
      ##    ## ##       ##     ##  ##  ##    ## ##    ## 
      ##       ##       ##     ##  ##  ##       ##       
      ##       ##       ##     ##  ##   ######  ##       
      ##       ##        ##   ##   ##        ## ##       
      ##    ## ##         ## ##    ##  ##    ## ##    ## 
       ######  ########    ###    ####  ######   ###### 

### CCNU-LBNL-Viscous hydrodynamic model

> Original Version: https://github.com/lgpang/clvisc <br>
> CLVisc3.0 Version: https://github.com/wangyunxiang1986/clvisc

### When using this model, please cite:
    
> (a) L.-G. Pang, H. Petersen, and X.-N. Wang, [PhysRevC 97, 064918](https://link.aps.org/doi/10.1103/PhysRevC.97.064918) <br>
> (b) X.-Y. Wu, G.-Y. Qin, L.-G. Pang, and X.-N. Wang, [Phys.Rev.C 105, 034909](https://link.aps.org/doi/10.1103/PhysRevC.105.034909) <br>
> (c) J.-Q. Tao, X. Fan, B.-W. Zhang, (In writing)

# What's New

1. It can run with python3.11 and other new version dependence (except GCC, it still needs GCC7.5) now.

2. It can use T<sub>R</sub>ENTo2.0 to produce the nucleus's initial state and can tune the T<sub>R</sub>ENTo's parameters in the `hydro.info` now.

3. Fixed some minor issues.

# Installtion

### 1. Install CUDA (example for Ubuntu22.04.4 LTS and CUDA12.4):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```   

   __Note__: CUDA already includes OpenCL, you do NOT need to install OpenCL additionally! You only need to write these commands,
         
```bash
export OpenCL_INCPATH=/usr/local/cuda-12.4/include
export OpenCL_LIBPATH=/usr/local/cuda-12.4/lib64
```
           
   in your `~/.bashrc` file.
         
   > CUDA official download website: https://developer.nvidia.com/cuda-downloads

### 2. Install Anaconda (example for Ubuntu22.04.4 LTS and Anaconda3):

   Download Anaconda from the website, https://www.anaconda.com/download/, and run

```bash   
sh Anaconda3-2024.02-1-Linux-x86_64.sh
```

   __Note__: Anaconda3 already includes BOOST, you do NOT need to install BOOST additionally!

### 3. Install PyOpenCL:

```bash
pip install pyopencl
```

### 4. Install GCC7.5 (example for Ubuntu22.04.4 LTS):

```bash
sudo add-apt-repository "deb [arch=amd64] http://archive.ubuntu.com/ubuntu focal main universe"
sudo apt-get install gcc-7 g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --config gcc
```

### 5. Install CMAKE (example for Ubuntu22.04.4 LTS):

```bash
sudo apt install cmake
```

### 6. Install GSL (example for Ubuntu22.04.4 LTS):

```bash
sudo apt install libgsl-dev
```

# Using

1. Please enter the `CLVisc3.1/3rdparty/trento_1_3_with_participant_plane`, `CLVisc3.1/3rdparty/trento_2_0_with_participant_plane`, `CLVisc3.1/CLSmoothSpec`,
   `CLVisc3.1/sampler/sampler_cpu`, `CLVisc3.1/sampler/sampler_gpu` and run the following commands separately,

```bash
mkdir build
cd build
cmake ..
make -j4
```

2. Then you can enter the `CLVisc3.1\pyvisc` and modify the parameters in the `hydro.info`.

3. Run the following command in the `CLVisc3.1\pyvisc`,
   
```bash
python CLVisc.py hydro.info 0 0 1
```

   where the first `0` is the GPU device code, the second `0` and `1` are the first event number and the last event number that you want to produce.

# Issue

* If you find the error that `can not find "GL.h" file`, please run command,

```bash
sudo apt install mesa-common-dev libglu1-mesa-dev freeglut3-dev
```

* If you find the error that ``anaconda3/lib/libboost_program_options.so.1.82.0: undefined reference to `std::__throw_bad_array_new_length()@GLIBCXX_3.4.29'``, 
please use the new version of GCC.

* If you find the error that `raise ImportError("Failed to import any qt binding")`,
please comment out `import matplotlib.pyplot as plt` in the PYTHON files.

# Other

If you want to run CLVisc3.1 on the computer cluster and you can't use the `sudo` command, 
please download the corresponding installation package for local installation.