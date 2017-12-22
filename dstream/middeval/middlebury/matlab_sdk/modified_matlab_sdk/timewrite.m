function timewrite( time, path, filename )
    filename = strcat(path, '\', filename);
    file = fopen(filename, 'wb');
    fprintf(file, '%6.4f', time);
end

