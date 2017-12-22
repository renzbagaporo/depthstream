for counter = 0 : 193
    
    filename = strcat('000', sprintf('%03d',counter) , '_10.png');    
    left = disp_read(strcat('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\kitteval\data_stereo_flow\training\image_0', '\', filename));
    right  = disp_read(strcat('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\kitteval\data_stereo_flow\training\image_1', '\', filename));
    tic;
    bm =  disparity(left, right, 'Method', 'BlockMatching', 'DisparityRange', [0 256]);
    time = round(toc * 1000);
    bm = uint16(bm * 256);
    
    imwrite(bm, strcat('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\kitteval\results\MATBM', '\', filename));
    
end

for counter = 0 : 193
    
    filename = strcat('000', sprintf('%03d',counter) , '_10.png');    
    left = disp_read(strcat('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\kitteval\data_stereo_flow\training\image_0', '\', filename));
    right  = disp_read(strcat('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\kitteval\data_stereo_flow\training\image_1', '\', filename));
    tic;
    sgbm = disparity(left, right, 'Method', 'SemiGlobal', 'DisparityRange', [0 256]);
    time = round(toc * 1000);
    sgbm = uint16(sgbm * 256);
    

    imwrite(sgbm, strcat('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\kitteval\results\MATSGBM', '\', filename));
        
end
