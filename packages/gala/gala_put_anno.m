function gala_put_anno(imgToken, annoToken, annoServiceLocation, annoMat, emCube, author, query_file, padX, padY, useSemaphore, labelOutFile, tokenFile)
% rhoana_put_anno - this function writes rhoana results to OCP
%
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

%% Flag to indicate upload method. "old" method uploads ramon objects to get Ids
% "new" method reserves ids
use_new_method = false;

%% Setup OCP
if useSemaphore == 1
    oo = OCP('semaphore');
else
    oo = OCP();
end

% set to server default
oo.setErrorPageLocation('/mnt/pipeline/errorPages');

% Set server details
oo.setServerLocation(annoServiceLocation);
oo.setImageToken(imgToken);
oo.setAnnoToken(annoToken);

%% Load gala results
disp(annoMat)
disp(emCube)
load(annoMat); % file has matrix labels
load(emCube); % load em cube for position info
labels = permute(labels, [2,3,1]);
% Load Query
load(query_file);

img_size = oo.imageInfo.DATASET.IMAGE_SIZE(query.resolution);

% If pad, need to reset XYZ Offset, trim block
% If pad, need to trim block

% Compute padding offsets
if padX ~= 0 || padY ~=0
    
    if (query.xRange(1) == 0) && (query.yRange(1) == 0)
        % No x or y start padding
        xstart = 1;
        ystart = 1;
        xend = size(labels,1) - padX;
        yend = size(labels,2) - padY;
        xyzOffset = '[query.xRange(1), query.yRange(1), query.zRange(1)]';
        
    elseif query.xRange(1) == 0
        % No x start padding
        xstart = 1;
        ystart = padY+1;
        xend = size(labels,1) - padX;
        yend = size(labels,2) - padY;
        xyzOffset = '[query.xRange(1), query.yRange(1) + padY, query.zRange(1)]';
        
    elseif query.yRange(1) == 0
        % No y start padding
        xstart = padX+1;
        ystart = 1;
        xend = size(labels,1) - padX;
        yend = size(labels,2) - padY;
        xyzOffset = '[query.xRange(1) + padX, query.yRange(1), query.zRange(1)]';
        
    elseif (query.xRange(2) == img_size(1)) && (query.yRange(2) == img_size(2))
        % No x or y end padding
        xstart = padX+1;
        ystart = padY+1;
        xend = size(labels,1);
        yend = size(labels,2);
        xyzOffset = '[query.xRange(1) + padX, query.yRange(1) + padY, query.zRange(1)]';
        
    elseif query.xRange(2) == img_size(1)
        % No x end padding
        xstart = padX+1;
        ystart = padY+1;
        xend = size(labels,1);
        yend = size(labels,2) - padY;
        xyzOffset = '[query.xRange(1) + padX, query.yRange(1) + padY, query.zRange(1)]';
        
    elseif query.yRange(2) == img_size(2)
        % No y end padding
        xstart = padX+1;
        ystart = padY+1;
        xend = size(labels,1) - padX;
        yend = size(labels,2);
        xyzOffset = '[query.xRange(1) + padX, query.yRange(1) + padY, query.zRange(1)]';
        
    else
        % Normal padding sitution
        xstart = padX+1;
        ystart = padY+1;
        xend = size(labels,1) - padX;
        yend = size(labels,2) - padY;
        xyzOffset = '[query.xRange(1) + padX, query.yRange(1) + padY, query.zRange(1)]';
    end
else
    xstart = 1;
    xend = size(labels,1);
    ystart = 1;
    yend = size(labels,2);
    xyzOffset = '[query.xRange(1),query.yRange(1),query.zRange(1)]';
end

%crop volume as necessary
xstart
xend
ystart
yend
size(labels)
labels = labels(xstart:xend,ystart:yend,:);
size(labels)
%below is original code...
%% Upload to OCP
% Get number of objects
ids = unique(labels);
ids(ids == 0) = [];
num_objs = length(ids)
whos labels

if use_new_method == true
    % Block reserve IDs
    ocp_ids = oo.reserve_ids(num_objs);
    [labelOut, nLabel] = relabel_id(labels);
    labelOut = uint32(labelOut);
    if nLabel ~= num_objs
        error('something went wrong during relabel');
    end
    labelOut = labelOut + ocp_ids(1) - 1;
    labelOut(labels == 0) = 0;
    
    % Create empty RAMON Objects
    seg = RAMONSegment();
    seg.setAuthor(author);
    seg_cell = cell(num_objs,1);
    for ii = 1:num_objs
        s = seg.clone();
        s.setId(ocp_ids(ii));
        seg_cell{ii} = s;
    end
    
    % Batch write RAMON Objects
    tic
    oo.setBatchSize(100);
    oo.createAnnotation(seg_cell);
    fprintf('Batch Metadata Upload: ');
    toc
else
    % relabel Paint
    fprintf('Relabling: ');
    labels = uint32(labels); %TODO - careful casting in this way - possible loss of precision
    
    [zz, n] = relabel_id(labels);
    
    % Create empty RAMON Objects
    seg = RAMONSegment();
    seg.setAuthor(author);
    seg_cell = cell(n,1);
    for ii = 1:n
        s = seg.clone();
        seg_cell{ii} = s;
    end
    
    % Batch write RAMON Objects
    tic
    oo.setBatchSize(100);
    ids = oo.createAnnotation(seg_cell);
    fprintf('Batch Metadata Upload: ');
    toc

    labelOut = zeros(size(zz));
    
    rp = regionprops(zz,'PixelIdxList');
    for ii = 1:length(rp)
        labelOut(rp(ii).PixelIdxList) = ids(ii);
    end
    
    clear zz
    toc

end

% Block write paint
tic
size(labelOut)
max(labelOut(:))
em_cube.resolution

paint = RAMONVolume();
paint.setCutout(labelOut);
paint.setDataType(eRAMONDataType.anno32);
paint.setResolution(em_cube.resolution);
paint.setXyzOffset(eval(xyzOffset));
oo.createAnnotation(paint);
fprintf('Block Write Upload: ');
paint.xyzOffset
annoToken
toc

%fprintf('@@!!@@ %s @@!!@@\n', annoToken)
%fprintf('@@start@@%s##end##\n', annoToken)

% save labelOut
%if ~isempty(varargin)
%    labelOutFile = varargin{1};
save(labelOutFile, 'labelOut')
save(tokenFile, 'annoToken')
%else
%    disp('skipping labelOut save')
end

