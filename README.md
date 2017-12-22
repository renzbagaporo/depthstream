# depthstream

A GPU-accelerated stereo vision algorithm used in http://ieeexplore.ieee.org/document/7848074/


## Setup
To fully use the library, you will need the following software:

* Visual Studio 2013
* CUDA SDK 7.5 
* OpenCV 3.0
* Qt 5.0
* MATLAB 2011

The implementation was targeted for a Kepler-based NVIDIA (GT735M) card with features used introduced in compute capability 3.0. It should be easy to re-target it for newer cards and compute capabilities. 

## Directory

The root directory is /dstream. Inside the code files are logically grouped into subdirectories.

* dscalib - contains the code used for calibrating the stereo cameras before use
* dscore - contains the CUDA kernels used for computing disparities between stereo images
* dsdemo - contains three demo applications (1) depthmap display (2) point cloud construction (3) tracked objected distance display
* dseval - contains utilty code for evaluating depthstream against the KITTI and Middlebury datasets
* dsmain - contains the main classes used for using the algorihtm; wraps the kernels in dscore for use in an object-oriented fashion
* dsmeasure - contains code used to measure object distances to the stereo camera
* kitteval, middeval - KITTI and Middlebury dataset kits
