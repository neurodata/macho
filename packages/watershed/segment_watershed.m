%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) [2014] The Johns Hopkins University / Applied Physics Laboratory All Rights Reserved. Contact the JHU/APL Office of Technology Transfer for any additional rights.  www.jhuapl.edu/ott
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%    http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function S = segment_watershed(data, dilXY, dilZ, thresh)
% this function is intended to be called from python
%V2.0  W Gray Roncal

dilXY = double(dilXY);
dilZ = double(dilZ);
thresh = double(thresh);

minSize = 500; 
%Scale 1
%param.dilSize = [11,5];
%param.thresh = 10; 
%param.minSize = 500;
  
% %Scale 3
% param.dilSize = [3,3];
% param.thresh = 40;
% param.minSize = 50;
 
%% Watershed Code
Iout = uint8(round(255*data));
%fprintf('There are %d membrane values...\n',length(unique(Iout))-1)
fprintf('beginning watershed...\n')

for i = 1:size(Iout,3)
    Iout(:,:,i) = medfilt2(Iout(:,:,i),[9,9]);
end

% This is a way to mask out isolated membrane pixels
if dilXY ~= 0
    Iout2 = imdilate(Iout,strel('ball',dilXY, dilZ));
else
    Iout2 = Iout;
end

% TODO
%mm = bwareaopen(Iout2>(0.75*255), 1000, 4);
%Iout2(mm == 0) = 0;

zf = Iout2 < thresh;

zz = imimposemin(Iout, zf);% | zbg);

labelOut = watershed(zz,8);

labelOutZ = labelOut;

rp = regionprops(labelOutZ,'PixelIdxList','Area');
areaAll = [rp.Area];
toRemove = find(areaAll < minSize); %perhaps 500 pixels

% Need to do per slice and per object
% relabel...

for i = 1:length(toRemove)
    labelOutZ(rp(toRemove(i)).PixelIdxList) = 0;
end


labelOutZ = cleanup_speckle(labelOutZ);
[S,n] = relabel_id(labelOutZ);
