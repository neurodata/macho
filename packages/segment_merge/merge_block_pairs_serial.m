function merge_block_pairs_serial(cutoutFile, annoToken, overlap_threshold, dist_threshold,...
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
    mon.reset('serial_merge');
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
    m = BatchMergeTool();    

    if ~exist('overlap_threshold','var');
        overlap_threshold = 0.6;
    end
    if ~exist('dist_threshold','var');
        dist_threshold = 20;
    end
    mon.update('serial_merge', sprintf('Correlation Threshold: %f',overlap_threshold));
    mon.update('serial_merge', sprintf('Distance Threshold: %f',dist_threshold));


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
        mon.update('serial_merge', sprintf('Starting query index: %d',jj));
        
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
            mon.update('serial_merge', 'merge plane all zeros!');
            continue;
        end
        if sum(B(:)) == 0
            fprintf('merge plane all zeros!\n');
            mon.update('serial_merge', '!!!! merge plane all zeros !!!!');
            continue;
        end

        
        %% Go from A to B
        % Get all IDs in A
        A_ids = unique(A);
        A_ids(A_ids == 0) = [];
       
        % Allocate output
        id_list_A = cell(length(A_ids),1);
        
        % Loop for every ID in A
        for ii = 1:length(A_ids)
            if mod(ii,100) == 0
                fprintf('A-to-B: Checking ID %d of %d\n',ii,length(A_ids));
            end
            % Get overlap IDs
            B_overlap_ids = B(A == A_ids(ii));
            B_overlap_ids(B_overlap_ids == 0) = [];
            
            % Get the overlap IDs and counts
            B_overlap_counts = histc(B_overlap_ids(:),unique(B_overlap_ids));
            B_overlap_ids = unique(B_overlap_ids);
            [~,overlap_order] = sort(B_overlap_counts,'descend');
            
            % Loop through all overlap IDs trying to get a good match.
            % Start with the max overlap object and grow from there            
            XX = zeros(size(A));
            XX(A == A_ids(ii)) = 1;
            XX_bb = regionprops(XX,'BoundingBox');
            
            YY = zeros(size(B));
            for kk = 1:length(overlap_order)
                % Set B ID you are checking to 1 (agglomerating)
                YY(B == B_overlap_ids(overlap_order(kk))) = 1;
                
                % Compute cross-correlation            
                YY_bb = regionprops(YY,'BoundingBox');
                x_min = round(min(XX_bb.BoundingBox(1),YY_bb.BoundingBox(1)));
                x_max = x_min + round(max(XX_bb.BoundingBox(3),YY_bb.BoundingBox(3)));
                y_min = round(min(XX_bb.BoundingBox(2),YY_bb.BoundingBox(2)));
                y_max = y_min + round(max(XX_bb.BoundingBox(4),YY_bb.BoundingBox(4)));
                if x_max > size(XX,2)
                    x_max = size(XX,2);
                end
                if y_max > size(XX,1)
                    y_max = size(XX,1);
                end
                if all(all(XX(y_min:y_max,x_min:x_max),1)) && all(all(YY(y_min:y_max,x_min:x_max),1))
                    % Both are completely the same!
                    score = 1;
                    peak_dist = 0;
                else
                    % Use norm cross corr as a metric for similiarity
                    c = normxcorr2(XX(y_min:y_max,x_min:x_max),YY(y_min:y_max,x_min:x_max));   
                    score = max(c(:));

                    [ypeak, xpeak] = find(c==max(c(:)));
                    yoffSet = ypeak-size(XX(y_min:y_max,x_min:x_max),1);
                    xoffSet = xpeak-size(XX(y_min:y_max,x_min:x_max),2);
                    peak_dist = sqrt(mean(yoffSet)^2 + mean(xoffSet)^2);
                end
                
                % Check if match is good enough and not too far away.
                if score >= overlap_threshold
                    if peak_dist <= dist_threshold
                        % looks good! Merge it yo.
                        mergeTo = A_ids(ii); 
                        mergeFrom = B_overlap_ids(overlap_order(1:kk));
                        id_list_A{ii} = [mergeTo, mergeFrom'];

                        % Remove the object in A and B from consideration
                        A(A==mergeTo) = 0;
                        for gg = 1:length(mergeFrom)
                            B(B==mergeFrom(gg)) = 0;
                        end
                        break;
                    end
                end
            end % Overlap loop
        end % A loop

        %% Check if anything is left from B to A

        % Get all IDs in B
        B_ids = unique(B);
        B_ids(B_ids == 0) = [];
       
        % Allocate output
        id_list_B = cell(length(B_ids),1);
        
        % Loop for every ID in A
        for ii = 1:length(B_ids)            
            if mod(ii,100) == 0
                fprintf('B-to-A: Checking ID %d of %d\n',ii,length(A_ids));
            end
            % Get overlap IDs
            A_overlap_ids = A(B == B_ids(ii));
            A_overlap_ids(A_overlap_ids == 0) = [];
            
            % Get the overlap IDs and counts
            A_overlap_counts = histc(A_overlap_ids(:),unique(A_overlap_ids));
            A_overlap_ids = unique(A_overlap_ids);
            [~,overlap_order] = sort(A_overlap_counts,'descend');
            
            % Loop through all overlap IDs trying to get a good match.
            % Start with the max overlap object and grow from there            
            XX = zeros(size(B));
            XX(B == B_ids(ii)) = 1;
            XX_bb = regionprops(XX,'BoundingBox');
            
            YY = zeros(size(A));
            for kk = 1:length(overlap_order)
                % Set B ID you are checking to 1 (agglomerating)
                YY(A == A_overlap_ids(overlap_order(kk))) = 1;
                
                % Compute cross-correlation            
                YY_bb = regionprops(YY,'BoundingBox');
                x_min = round(min(XX_bb.BoundingBox(1),YY_bb.BoundingBox(1)));
                x_max = x_min + round(max(XX_bb.BoundingBox(3),YY_bb.BoundingBox(3)));
                y_min = round(min(XX_bb.BoundingBox(2),YY_bb.BoundingBox(2)));
                y_max = y_min + round(max(XX_bb.BoundingBox(4),YY_bb.BoundingBox(4)));
                if x_max > size(XX,2)
                    x_max = size(XX,2);
                end
                if y_max > size(XX,1)
                    y_max = size(XX,1);
                end
                c = normxcorr2(XX(y_min:y_max,x_min:x_max),YY(y_min:y_max,x_min:x_max));   
                score = max(c(:));
                [ypeak, xpeak] = find(c==max(c(:)));
                yoffSet = ypeak-size(XX(y_min:y_max,x_min:x_max),1);
                xoffSet = xpeak-size(XX(y_min:y_max,x_min:x_max),2);
                peak_dist = sqrt(mean(yoffSet)^2 + mean(xoffSet)^2);
                
                % Check if match is good enough and not too far away.
                if score >= overlap_threshold
                    if peak_dist <= dist_threshold
                        % looks good! Merge it yo.
                        mergeTo = B_ids(ii); 
                        mergeFrom = A_overlap_ids(overlap_order(1:kk));
                        id_list_B{ii} = [mergeTo, mergeFrom'];

                        % Remove the object in A and B from consideration
                        B(B==mergeTo) = 0;
                        for gg = 1:length(mergeFrom)
                            A(A==mergeFrom(gg)) = 0;
                        end
                        break;
                    end
                end
            end % Overlap loop
        end % B loop
        
        
        %% Merge and clean up lists
        id_list = cat(1,id_list_A,id_list_B);
        id_list(cellfun(@isempty,id_list)) = [];

        %% Transitive Closure
        id_list = m.transitive_closure(id_list);

        %% Run merges
        mon.update('serial_merge', sprintf('Requesting %d Merges',length(id_list)));
        for ii = 1:length(id_list)
            ids = id_list{ii};
            mon.update('tc_ids', sprintf('%d,',ids));
%            try
            oo.mergeAnnotation(ids(1),ids(2:end));
            mon.update('serial_merge', sprintf('Successfully Merged IDs...'));

 %           catch
  %            mon.update('serial_merge', sprintf('Error in Merging IDs, Skipping...'));
   
  %          end
            
        end
        fprintf('done.\n');
        mon.update('serial_merge', sprintf('Query Index %d Complete',jj));

    end
end

