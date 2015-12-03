function merge_block_pairs_serial_sparse_wcentroid(cutoutFile, serviceLocation, annoToken, annoChannel, resolution, overlap_threshold,...
    useSemaphore, start_index, stop_index)
% merge_block_pairs_serial - performs a serial merge of all blocks by
% operating on the merge cutouts one at a time.
%

%% Setup OCP

if useSemaphore == 1
    oo = OCP('semaphore');
else
    oo = OCP();
end

oo.setServerLocation(serviceLocation);
oo.setAnnoToken(annoToken);
oo.setAnnoChannel(annoChannel);
oo.setDefaultResolution(resolution);


%% Prep

% Load text file
cutoutFile
fid = fopen(cutoutFile,'r')  % Open text file
queries = textscan(fid,'%s','delimiter','\n');
queries = queries{:};
fclose(fid);

if ~exist('start_index','var');
    start_index = 1;
end
if ~exist('stop_index','var');
    stop_index = length(queries);
end

f = OCPFields;
% count number of errors in merge
cc = 0;
%% Process each merge regions
for jj = start_index:stop_index
    fprintf('Beginning query index: %d...\n',jj);
    
    % Load query
    queries{jj}
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
        continue;
    end
    if sum(B(:)) == 0
        fprintf('merge plane all zeros!\n');
        continue;
    end
    
    
    %% Simple Algorithm for block merging
    id_list = [];
    % Simply look for high dice overlap between A and B.
    % Assuming high values (>0.5), should only need to check A to B
    % Also, if strictly pairwise, no TC
    % This should be 100x faster than the more complex method
    
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
        
        %% Run merges
        fprintf('serial_merge_simple', sprintf('Requesting %d Merges\n',length(id_list)));
        for ii = 1:length(id_list)
            ids = id_list{ii};
            %mon.update('tc_ids', sprintf('%d,',ids));
            
            try
                oo.mergeAnnotation(ids(1),ids(2:end));
                
                % recompute centroid for id1
                q = OCPQuery;
                q.setType(eOCPQueryType.RAMONVoxelList);
                q.setResolution(resolution);
                obj = oo.query(q);
                centroid = round(mean(obj.data));
                centroid = num2str(centroid);
                whos centroid
                centroid
                oo.setField(ids(1),'centroid',centroid)
                fprintf('Successfully Merged IDs...\n');

            catch
                fprintf('Error in Merging IDs...\n');
                cc = cc + 1;
            end
        end
        fprintf('done.\n');
        fprintf('Query Index %d Complete\n',jj);
    else
        fprintf('serial_merge_simple', sprintf('Requesting %d Merges\n',0));
        fprintf('serial_merge_simple', sprintf('Query Index %d Complete\n',jj));
        
    end
    
end

cc
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
