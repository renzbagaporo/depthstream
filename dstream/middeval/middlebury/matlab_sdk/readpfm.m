function D = readpfm(filename_pfm)

fid = fopen(filename_pfm);

fscanf(fid,'%c',[1,3]);
cols = fscanf(fid,'%f',1);
rows = fscanf(fid,'%f',1);
fscanf(fid,'%f',1);
fscanf(fid,'%c',1);
D = fread(fid,[cols,rows],'single');
D(D == Inf) = 0;
D = rot90(D);
fclose(fid);