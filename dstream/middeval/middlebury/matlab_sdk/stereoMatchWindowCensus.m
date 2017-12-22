% Usage: [D1,D2] = stereoMatchWindowCensus(I1, I2, window_radius, disparity_range)
%
% Match Stereo Images using Census Cost and Standard Uniform Window Aggregation
%
% Takes as input an image <I1>, an image <I2>, the radius of the window
% used for aggregation <window_radius>, and the range of disparities
% <disparity_range> as a 1x2 vector. This function returns the
% winner-take-all disparity maps going from <I1> to <I2> and vice versa.
%
% Author: Eric Psota
% Date: June 26, 2014

function [D1,D2] = stereoMatchWindowCensus(I1, I2, window_radius, disparity_range)

[~,cols,~] = size(I1);
[rows,cols2,~] = size(I2);
if cols > cols2
    I2(:,end+1:cols,:) = 0;
elseif cols2 > cols
    cols = cols2;
    I1(:,end+1:cols2,:) = 0;
end

IC1 = censusTransform9x7(I1);
IC2 = censusTransform9x7(I2);

C1min = single(Inf*ones(rows,cols));
C2min = single(Inf*ones(rows,cols));
D1 = single(zeros(rows,cols));
D2 = single(zeros(rows,cols));
for d = disparity_range(1):disparity_range(2)
    fprintf('Disparity = %04d', d);
    range1 = max(1,1+d):min(size(I1,2),size(I1,2)+d);
    range2 = max(1,1-d):min(size(I1,2),size(I1,2)-d);
    S = Inf(size(I1,1),length(range1));
    Sbit = zeros(size(S,1) + 2*window_radius,size(S,2) + 2*window_radius);
    Sbit(1+window_radius:end-window_radius,1+window_radius:end-window_radius) = bitxor(IC1(:,range1,:),IC2(:,range2,:));
    Shamming = zeros(size(Sbit));
    for k=1:64
        Shamming = Shamming + double(bitget(Sbit,k));
    end
    Sint = integralImage(Shamming);
    Sagg = Sint(2*window_radius+2:end,2*window_radius+2:end)...
        - Sint(1:end-(2*window_radius+1),2*window_radius+2:end)...
        - Sint(2*window_radius+2:end,1:end-(2*window_radius+1))...
        + Sint(1:end-(2*window_radius+1),1:end-(2*window_radius+1));
    S = Sagg;
    C1crop = C1min(:,range1);
    C2crop = C2min(:,range2);
    D1crop = D1(:,range1);
    D2crop = D2(:,range2);
    idx1 = single(S) < C1min(:,range1);
    idx2 = single(S) < C2min(:,range2);
    C1crop(idx1) = S(idx1);
    C2crop(idx2) = S(idx2);
    D1crop(idx1) = d;
    D2crop(idx2) = d;
    C1min(:,range1) = C1crop;
    C2min(:,range2) = C2crop;
    D1(:,range1) = D1crop;
    D2(:,range2) = D2crop;
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
end

end





function C = censusTransform9x7(I)

[rows, cols, channels] = size(I);
if channels == 3
    I = rgb2gray(I);
end

C = uint64(zeros(rows,cols));
pow = 63;
Ishifted = zeros(size(I));
for rs = -4:4
    for cs = -3:3
        Ishifted(:,:) = 0;
        if rs <= 0 && cs <= 0
            Ishifted(1:end+rs,1:end+cs) = I(1-rs:end,1-cs:end);
        elseif rs <= 0 && cs >= 0
            Ishifted(1:end+rs,1+cs:end) = I(1-rs:end,1:end-cs);
        elseif rs >= 0 && cs <= 0
            Ishifted(1+rs:end,1:end+cs) = I(1:end-rs,1-cs:end);
        elseif rs >= 0 && cs >= 0
            Ishifted(1+rs:end,1+cs:end) = I(1:end-rs,1:end-cs);
        end
        C(I < Ishifted) = bitset(C(I < Ishifted),pow);
        pow = pow - 1;
    end
end

end