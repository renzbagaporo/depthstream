disp('======= KITTI DevKit Demo =======');
clear all; close all; dbstop error;

% error threshold
tau = 3;

% stereo demo
disp('Load and show disparity map ... ');
D_est = disp_read('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\kitteval\results\OCVSGBMHH\000021_10.png');
D_gt  = disp_read('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\kitteval\data_stereo_flow\training\disp_noc\000021_10.png');
d_err = disp_error(D_gt,D_est,tau);
figure,imshow(disp_to_color([D_est;D_gt]));
title(sprintf('Error: %.2f %%',d_err*100));
