MatlabSDK

This directory contains sample Matlab scripts for the Middlebury
stereo evaluation v. 3, graciously provided by Eric Psota and Jedrzej
Kowalczuk.

To use this code, download and unzip the MiddEval3-data and
Midd-Eval3-GT0 zip files, and place the resulting MiddEval3 folder
inside the MatlabSDK folder.  A sample stereo method is included,
which uses 9x7 census cost, uniform window aggregation, winner take
all, and consistency checking. It also calculates the error rates
using the ground truth images. The results are saved in
MiddEval3results.

Feel free to adapt this code to suit your needs.

*** The code is provide AS IS without any support or warranty
*** whatsoever. 
