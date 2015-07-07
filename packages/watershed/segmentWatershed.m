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


function S = segmentWatershed(edata, dilXY, dilZ, thresh)
% This is the function for the server
%V1.0  W Gray Roncal / Dean Kleissas - March 2014

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

Iout = uint8(round(255*edata.data));
fprintf('There are %d membrane values...\n',length(unique(Iout))-1)

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
labelOut = watershed(zz,26);
%unique(labelOut)


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
[labelOutZ,n] = relabel_id(labelOutZ);


% %find small pixels
% rp = regionprops(labelOut,'Area');
% idx = ismember(labelOut,find([rp.Area]< minSize));
% 
% %set to zero for imfill
% labelOutZ(idx) = 0;
% 
% for i = 1:size(labelOutZ,3)
%     temp = labelOutZ(:,:,i);
%     temp = imfill(temp, 4, 'holes');
%     labelOutZ(:,:,i) = temp;
% end
% [labelOutZ,n] = relabel_id(labelOutZ);

n

S = RAMONVolume;
%Set upload type
S.setDataType(eRAMONDataType.anno32);
S.setResolution(edata.resolution);
S.setXyzOffset(edata.xyzOffset);
S.setCutout(labelOutZ);
