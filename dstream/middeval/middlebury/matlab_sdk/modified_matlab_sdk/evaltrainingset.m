function evaltrainingset(path)

training_set_names = {'Adirondack', 'ArtL', 'Jadeplant', 'Motorcycle', ...
    'MotorcycleE', 'Piano', 'PianoL', 'Pipes', 'Playroom', 'Playtable', ... 
    'PlaytableP', 'Recycle', 'Shelves', 'Teddy', 'Vintage'};
training_set_disp = [290, 256, 640, 280, 280, 260, 260, 300, 330, 290, 290, 260, 240, 256, 760] * 0.25;

loop_number = 10;

for i = 1 : 15
    subpath = strcat(path, '\', training_set_names{i});
    left_im = imread(strcat(subpath, '\im0.png'));
    left_im = rgb2gray(left_im);
    right_im = imread(strcat(subpath, '\im1.png'));
    right_im = rgb2gray(right_im);
    
    times = [0 0 0 0 0 0 0 0 0 0];
    
    %BM
    for j = 1 : loop_number
            tic;
            bm_disp = disparity(left_im, right_im, 'Method', 'BlockMatching', 'DisparityRange', [int32(0)  int32(round_up_to_multiple(training_set_disp(i),16))]);
            times(j) = toc;   
    end
    
    bm_disp(bm_disp <= 0) = Inf;    
    pfmwrite(bm_disp, strcat('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\middeval\result-MATBM\trainingQ\', training_set_names{i}), strcat('disp0MATBM_s.pfm'));
    timewrite(mean(times), strcat('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\middeval\result-MATBM\trainingQ\', training_set_names{i}), strcat('timeMATBM.txt'));
    
    %SGBM
    for j = 1 : loop_number
        tic;
        sgbm_disp = disparity(left_im, right_im, 'Method', 'SemiGlobal', 'DisparityRange', [int32(0)  int32(round_up_to_multiple(training_set_disp(i),16))]);
        times(j) = toc;
    end
   
    sgbm_disp(sgbm_disp <= 0) = Inf; 
    pfmwrite(sgbm_disp, strcat('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\middeval\result-MATSGBM\trainingQ\', training_set_names{i}), strcat('disp0MATSGBM_s.pfm'));
    timewrite(mean(times), strcat('C:\Users\renzb\Documents\Visual Studio 2013\Projects\depthstream\dstream\middeval\result-MATSGBM\trainingQ\', training_set_names{i}), strcat('timeMATSGBM.txt'));

end

end



