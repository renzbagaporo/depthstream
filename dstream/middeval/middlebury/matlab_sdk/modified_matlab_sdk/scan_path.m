function [eval_size, file_names, alg_names] = scan_path(path)
    files = dir(path);
    files_size = size(files, 1);
    
    file_names = {};
    alg_names = {};
    eval_size = 0;
    for file_iter = 1 : files_size
        if strfind(files(file_iter).name, 'disp0')
            eval_size = eval_size + 1;
            file_names{eval_size} = [path, '\', files(file_iter).name];
            [fpath, name, ext] = fileparts(files(file_iter).name(6:end));
            alg_names{eval_size} = name;
        end
    end
end

