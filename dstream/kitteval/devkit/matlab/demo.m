disp('======= KITTI DevKit Demo =======');
clear all; close all; dbstop error;

% error threshold
tau = 3;

% stereo demo
disp('Load and show disparity map ... ');
D_est = disp_read('data/disp_est.png');
D_gt  = disp_read('data/disp_gt.png');
d_err = disp_error(D_gt,D_est,tau);
figure,imshow(disp_to_color([D_est;D_gt]));
title(sprintf('Error: %.2f %%',d_err*100));

% % flow demo
% disp('Load and show optical flow field ... ');
% F_est = flow_read('data/flow_est.png');
% F_gt  = flow_read('data/flow_gt.png');
% f_err = flow_error(F_gt,F_est,tau);
% F_err = flow_error_image(F_gt,F_est);
% figure,imshow([flow_to_color([F_est;F_gt]);F_err]);
% title(sprintf('Error: %.2f %%',f_err*100));
% figure,flow_error_histogram(F_gt,F_est);
