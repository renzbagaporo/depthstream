for counter = 0 : 193
    filename = strcat('000', sprintf('%03d',counter) , '_10.png');    
    left = disp_read(strcat('..\kitteval\data_stereo_flow\training\image_0', '\', filename));
    right  = disp_read(strcat('..\kitteval\data_stereo_flow\training\image_1', '\', filename));
    
    tic;
    bm =  disparity(left, right, 'Method', 'BlockMatching', 'DisparityRange', [0 256]);
    time = round(toc*1000);
    
    bm = uint16(bm * 256);
    imwrite(bm, strcat('..\kitteval\results\MATBM', '\', filename));
    cmd = strcat('exiftool.exe -Comment=', num2str(time) , strcat(' ..\kitteval\results\MATBM', '\', filename));
    system(cmd);
end

for counter = 0 : 193
    filename = strcat('000', sprintf('%03d',counter) , '_10.png');    
    left = disp_read(strcat('..\kitteval\data_stereo_flow\training\image_0', '\', filename));
    right  = disp_read(strcat('..\kitteval\data_stereo_flow\training\image_1', '\', filename));
    
    tic;
    sgbm = disparity(left, right, 'Method', 'SemiGlobal', 'DisparityRange', [0 256]);
    time = round(toc*1000);
    
    sgbm = uint16(sgbm * 256);
    imwrite(sgbm, strcat('..\kitteval\results\MATSGBM', '\', filename));
    cmd = strcat('exiftool.exe -Comment=', num2str(time) , strcat(' ..\kitteval\results\MATSGBM', '\', filename));
    system(cmd);
end

