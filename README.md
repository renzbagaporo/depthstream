# depthstream

A GPU-accelerated stereo vision algorithm used in http://ieeexplore.ieee.org/document/7848074/


## Setup
To work with the code, you will need the following software:

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

## Usage

```c++
int width = 320, height = 240, disparities = 128;
int gamma = 8 , arm_length = 17, max_arm_length = 34, arm_threshold = 15, strict_arm_threshold = 6, region_voting_iterations = 4, disparity_tolerance = 1;

// Optional. Create a rectifier object. In order to create a rectifier object, a calibration file has to be created first  the code from dscalib. This is optional since other depthstream objects should be able to process stereo vision frames without a rectifier object.

DSRectifier rectifier = DSRectifier("calib.yml");

// Create a stereo camera stream object. The DSStream object encapsulates the stereo camera attached to the host PC.
DSStream stream = DSStream(2, 1, width, height, rectifier);

// Create a frame object. A frame encapsulates a pair of images captured from a stereo camera.
DSFrame frame;

// Create a matcher object. The DSMatcher object implements the stereo vision algorithm.
DSMatcher matcher = DSMatcher(width, height, disparities);

// Read a frame from the stream.
stream.read(frame);

// Compute the disparities. Parameters passed are explained in the IEEE paper.
matcher.compute(frame, rectangle, disparity, gamma, arm_length, max_arm_length, arm_threshold, strict_arm_threshold, region_voting_iterations, disparity_tolerance);
```
The code snippet above assumes usage with a stereo camera. If the disparities are to be computed directly from stereo images, one can skip the creation and use of DSStream object and use DSFrame directly. See /dsdemo for more thorough examples.
