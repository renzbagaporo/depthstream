function results = evaluate( gt_path, est_path )

results_temp = zeros(194, 1);
tau = 3;

for counter = 0 : 193
    
    filename = strcat('000', sprintf('%03d',counter) , '_10.png');    
    D_est = disp_read(strcat(est_path, '\', filename));
    D_gt  = disp_read(strcat(gt_path, '\', filename));
    d_err = disp_error(D_gt,D_est,tau) * 100;
    results_temp(counter + 1) = d_err;
    
    %figure,imshow(disp_to_color([D_est;D_gt]));
    %title(sprintf('Error: %.2f %%',d_err*100));
end

results = results_temp;

end

