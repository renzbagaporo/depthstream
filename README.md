# depthstream

A GPU-accelerated stereo vision algorithm used in http://ieeexplore.ieee.org/document/7848074/


## Setup
The development setup is as follows:

* Visual Studio 2013
* CUDA SDK 7.5 
* OpenCV 3.0

The implementation was targeted for a Kepler-based NVIDIA (GT735M) card with features used introduced in compute capability 3.0. It should be easy to re-target it for newer cards and compute capabilities. 
