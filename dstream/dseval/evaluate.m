function [error, time] = evaluate( gt_path, est_path )

error_temp = zeros(194, 1);
time_temp = zeros(194, 1);
tau = 3;

for counter = 0 : 193
    filename = strcat('000', sprintf('%03d',counter) , '_10.png');    
    D_est = disp_read(strcat(est_path, '\', filename));
    D_gt  = disp_read(strcat(gt_path, '\', filename));
   
    d_err = disp_error(D_gt,D_est,tau) * 100;
    info = imfinfo(strcat(est_path, '\', filename));
    
    time_temp(counter + 1) = str2num(info.Comment);
    error_temp(counter + 1) = d_err;
end

error = error_temp;
time = time_temp;

end

