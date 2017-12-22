function pfmwrite(D, path, filename)
%PFMREAD Write a PFM image file.
%   PFMREAD(D, filename) writes the contents of a floating-point,
%   single-channel image D into a file specified by filename.
%   
% Created: 11/13/2014 JK

assert(size(D, 3) == 1 & isa(D, 'single'));

filename = strcat(path, '\', filename);

[rows, cols] = size(D);
scale = -1.0;

fid = fopen(filename, 'wb');

fprintf(fid, ['Pf', char(10)]);
fprintf(fid, ['%d %d', char(10)], cols, rows);
fprintf(fid, ['%f', char(10)], scale);

fwrite(fid, D(end:-1:1, :)', 'single');
fclose(fid);

end

