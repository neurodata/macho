function segmentWatershedWrapper(emCube,annoToken, dilXY, dilZ, thresh,useSemaphore, errorPageLocation, serviceLocation, labelOutFile, tokenFile)
% segmentWatershedWrapper - this function addes OCP annotation
% database upload to the detector
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


if ~exist('useSemaphore','var')
    useSemaphore = false;
end

%% Load cubes
em = load(emCube);
%vesicles = load(vesicleCube);

%% Load Classifier
%load(classifier_file)
%API change - this now happens as needed within contextSynDetect run
%% Run Detector
segCuboid = segmentWatershed(em.cube, dilXY, dilZ, thresh);

%% Upload Segment
if useSemaphore == 1
    ocp = OCP('semaphore');
else
    ocp = OCP();
end

if ~exist('errorPageLocation','var')
    % set to server default
    ocp.setErrorPageLocation('/mnt/pipeline/errorPages');
else
    ocp.setErrorPageLocation(errorPageLocation);
end

if exist('serviceLocation','var')
    ocp.setServerLocation(serviceLocation);
end

ocp.setAnnoToken(annoToken);

if isempty(segCuboid) || isempty(segCuboid.data)
    fprintf('No Segments Detected\n');
    
else
    fprintf('Uploading segments...\n');
    
    [zz, n] = relabel_id(segCuboid.data);
    
    % Old method
    % Create empty RAMON Objects
    seg = RAMONSegment();
    seg.setAuthor('apl_watershed');
    seg_cell = cell(n,1);
    for ii = 1:n
        s = seg.clone();
        seg_cell{ii} = s;
    end
    
    % Batch write RAMON Objects
    tic
    ocp.setBatchSize(100);
    ids = ocp.createAnnotation(seg_cell);
    fprintf('Batch Metadata Upload: ');
    toc
    
    % relabel Paint
    fprintf('Relabling: ');
    
    tic
    labelOut = zeros(size(zz));
    
    rp = regionprops(zz,'PixelIdxList');
    for ii = 1:length(rp)
        labelOut(rp(ii).PixelIdxList) = ids(ii);
    end
    
    clear zz
    toc
    
    % Block write paint
    tic
    paint = RAMONVolume();
    paint.setCutout(labelOut);
    paint.setDataType(eRAMONDataType.anno32);
    paint.setResolution(em.cube.resolution);
    paint.setXyzOffset(em.cube.xyzOffset);
    ocp.createAnnotation(paint);
    fprintf('Block Write Upload: ');
    toc
end

%Save out matrix
save(labelOutFile, 'labelOut')
save(tokenFile, 'annoToken')
%fprintf('@@start@@%s##end##\n', annoToken)

