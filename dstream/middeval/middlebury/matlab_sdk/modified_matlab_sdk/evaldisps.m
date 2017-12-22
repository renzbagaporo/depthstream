function evaldisps(path, bad_disp_thresh)

    %Enumerate set names
    dataset{1} = 'Adirondack';
    dataset{2} = 'ArtL';
    dataset{3} = 'Jadeplant';
    dataset{4} = 'Motorcycle';
    dataset{5} = 'MotorcycleE';
    dataset{6} = 'Piano';
    dataset{7} = 'PianoL';
    dataset{8} = 'Pipes';
    dataset{9} = 'Playroom';
    dataset{10} = 'Playtable';
    dataset{11} = 'PlaytableP';
    dataset{12} = 'Recycle';
    dataset{13} = 'Shelves';
    dataset{14} = 'Teddy';
    dataset{15} = 'Vintage';
        
    for set_iter = 1 : 15 
        subpath = strcat(path, '\', dataset(set_iter));
        
        %Read ground truth
        gt_path = char(strcat(subpath, '\disp0GT.pfm'));
        gt = readpfm(gt_path);
    
        %Read mask
        mask_path = char(strcat(subpath, '\mask0nocc.png'));
        mask_nocc = imread(mask_path);
        mask_nocc = mask_nocc == 255;
        mask_all = mask_nocc <= 255;
        
        %Scan the subpath for the disparity maps to evaluate
        [eval_size, file_names, alg_names] = scan_path(char(subpath));
        
        for alg_iter = 1 : eval_size
            %Load disp image
            %disp = imread(file_names(alg_iter));
            disp_path = char(file_names(alg_iter));
            disp = readpfm(disp_path);
            disp(disp <= 0) = Inf;
            
            error = abs(gt - disp) > bad_disp_thresh;
            error_nocc = error; error_all = error;
            
            error_nocc(~mask_nocc) = 0;
            error_all(~mask_all) = 0;
            
            %Evaluate
            bad_error_nocc = sum(error_nocc(:))/sum(mask_nocc(:));
            bad_error_all = sum(error_all(:))/sum(mask_all(:));
           
            %Write results
            fprintf([dataset{set_iter} , '(', alg_names{alg_iter}, ') : bad_nocc = %0.6f, bad_all = %0.6f\n'], bad_error_nocc, bad_error_all);
            
        end   
    end
end

