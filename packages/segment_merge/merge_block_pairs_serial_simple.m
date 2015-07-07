function merge_block_pairs_serial_simple(cutoutFile, annoToken, overlap_threshold, dist_threshold,...
    useSemaphore, serviceLocation, start_index, stop_index)
% merge_block_pairs_serial - performs a serial merge of all blocks by
% operating on the merge cutouts one at a time.
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

%% Create Matlab Monitor instance
mon = MatlabMonitor();
mon.reset('serial_merge_simple');
mon.reset('tc_ids');

%% Setup OCP
if ~exist('useSemaphore','var')
    useSemaphore = false;
end
if ~exist('serviceLocation','var')
    serviceLocation = 'http://openconnecto.me/';
end

validateattributes(annoToken,{'char'},{'row'});
validateattributes(cutoutFile,{'char'},{'row'});

if useSemaphore == 1
    oo = OCP('semaphore');
else
    oo = OCP();
end

oo.setServerLocation(serviceLocation);
oo.setAnnoToken(annoToken);
oo.setDefaultResolution(1);

mon.update('merge_block_pairs_serial', 'OCP setup complete');

%% Prep
% Create merge tool instance
%m = BatchMergeTool();

if ~exist('overlap_threshold','var');
    overlap_threshold = 0.6;
end
if ~exist('dist_threshold','var');
    dist_threshold = 20;
end
mon.update('serial_merge_simple', sprintf('Correlation Threshold: %f',overlap_threshold));
mon.update('serial_merge_simple', sprintf('Distance Threshold: %f',dist_threshold));


% Load text file
fid = fopen(cutoutFile,'r');  % Open text file
queries = textscan(fid,'%s','delimiter','\n');
queries = queries{:};
fclose(fid);

if ~exist('start_index','var');
    start_index = 1;
end
if ~exist('stop_index','var');
    stop_index = length(queries);
end

%% Process each merge regions
for jj = start_index:stop_index
    fprintf('Beginning query index: %d...',jj);
    mon.update('serial_merge_simple', sprintf('Starting query index: %d',jj));
    
    % Load query
    load(queries{jj});
    query.setType(eOCPQueryType.annoDense);
    
    % Cutout Data
    cube = oo.query(query);
    
    % figure out orientation and split into planes
    s = size(cube);
    r = s(1);
    c = s(2);
    z = s(3);
    if r == 2
        % y plane
        A = squeeze(cube.data(1,:,:));
        B = squeeze(cube.data(2,:,:));
        clear cube;
    elseif c == 2
        % x plane
        A = squeeze(cube.data(:,1,:));
        B = squeeze(cube.data(:,2,:));
        clear cube;
    elseif z == 2
        % z plane
        A = squeeze(cube.data(:,:,1));
        B = squeeze(cube.data(:,:,2));
        clear cube;
    else
        error('merge_segments:InvalidInput','1 dimension of the input cube needs to be equal to 2.  Did you load a merge cube or cutout cube?');
    end
    
    % Make sure that one of the merge planes isn't all zero. if so
    % continue
    if sum(A(:)) == 0
        fprintf('merge plane all zeros!\n');
        mon.update('serial_merge_simple', 'merge plane all zeros!');
        continue;
    end
    if sum(B(:)) == 0
        fprintf('merge plane all zeros!\n');
        mon.update('serial_merge_simple', '!!!! merge plane all zeros !!!!');
        continue;
    end
    
    
    %% Simple Algorithm for block merging
    id_list = [];
    % Simply look for high dice overlap between A and B.
    % Assuming high values (>0.5), should only need to check A to B
    % Also, if strictly pairwise, no TC
    % This should be 100x faster than the more complex method
    
    %         %% Go from A to B
    
    rpA = regionprops(A,'PixelIdxList','Area');
    rpB = regionprops(B,'PixelIdxList','Area');
    c = 1;
    
    %% A to B
    for mm = 1:length(rpA)
        if rpA(mm).Area > 0
            
            bPix = B(rpA(mm).PixelIdxList);
            bPix(bPix == 0) = [];
            possTarget = mode(bPix);
            iCount = sum(bPix(:) == possTarget);
            
            if possTarget > 0 && rpB(possTarget).Area > 0
                diceCoeff = (2 * iCount) / (rpA(mm).Area + rpB(possTarget).Area);
                
                if (diceCoeff >= overlap_threshold) && (mm ~= possTarget)
                    % looks good! Merge it yo.
                    id_list{c} = [min(mm, possTarget), max(mm,possTarget)];
                    rpB(possTarget).Area = -1; %Used this ID!
                    rpA(mm).Area = -1; %Used this ID
                    c = c + 1;
                end
            end
        end
    end
    
    %% B to A
    % Going the other direction is especially important for branching
    
    for mm = 1:length(rpB)
        if rpB(mm).Area > 0
            
            aPix = A(rpB(mm).PixelIdxList);
            aPix(aPix == 0) = [];
            possTarget = mode(aPix);
            iCount = sum(aPix(:) == possTarget);
            
            if possTarget > 0 && rpA(possTarget).Area > 0
                diceCoeff = (2 * iCount) / (rpB(mm).Area + rpA(possTarget).Area);
                
                if (diceCoeff >= overlap_threshold) && (mm ~= possTarget)
                    % looks good! Merge it yo.
                    id_list{c} = [min(mm, possTarget), max(mm,possTarget)];
                    rpA(possTarget).Area = -1; %Used this ID!
                    rpB(mm).Area = -1; %Used this ID
                    
                    c = c + 1;
                end
            end
        end
    end
    
    if ~isempty(id_list)
        %% Merge and clean up lists
        id_list(cellfun(@isempty,id_list)) = [];
        
        %% Transitive Closure
        % Not required here
        %id_list = m.transitive_closure(id_list);
        
        %% Run merges
        mon.update('serial_merge_simple', sprintf('Requesting %d Merges',length(id_list)));
        for ii = 1:length(id_list)
            ids = id_list{ii};
            mon.update('tc_ids', sprintf('%d,',ids));
            
            cc = 0;
            dd = 0;
            try
                oo.mergeAnnotation(ids(1),ids(2:end));
                mon.update('serial_merge_simple', sprintf('Successfully Merged IDs...'));
            catch
                mon.update('serial_merge_simple', sprintf('Error in Merging IDs...'));
                cc = cc + 1;
            end
        end
        fprintf('done.\n');
        mon.update('serial_merge_simple', sprintf('Query Index %d Complete',jj));
    else
        mon.update('serial_merge_simple', sprintf('Requesting %d Merges',0));
        mon.update('serial_merge_simple', sprintf('Query Index %d Complete',jj));
        
    end
    
end

cc
