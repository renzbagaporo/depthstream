% Usage: [D1,D2] = stereoConsistencyCheck(D1, D2)
%
% Check for Disparities that are Consistent Between Maps
%
% Takes as input two disparity maps <D1> and <D2>, checks to see if 
% disparities map back and forth between the two maps, and sets all those
% that do not map consistently to each other within <thresh>.
%
% Author: Jedrzej Kowalczuk
% Date: April 23, 2013

function [D1out,D2out] = stereoConsistencyCheck(D1, D2, thresh)

D1out = D1;
D2out = D2;
D1 = round(D1);
D2 = round(D2);

Consistent1 = zeros(size(D1));
Consistent2 = zeros(size(D2));

[I, J] = ind2sub(size(D1),1:numel(D1));
ind = sub2ind(size(D1),I(:),max(J(:)-D1(:),1));
Consistent1(:) = abs(D1(:) - D2(ind)) < thresh;
ind = sub2ind(size(D1),I(:),min(J(:)+D2(:),size(D1,2)));
Consistent2(:) = abs(D1(ind) - D2(:)) < thresh;
D1out(~Consistent1) = 0;
D2out(~Consistent2) = 0;