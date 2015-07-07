function merge_segments(inputCube, threshold, ignore_zeros)
    % merge_segments - computes merges along a merge block
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
  
    %% Prep
    % Create merge tool instance
    m = BatchMergeTool();
    
    if ~exist('ignore_zeros','var');
        ignore_zeros = 1;
    end
    if ~exist('threshold','var');
        threshold = 0.9;
    end
         
    %% Load cube
    load(inputCube);
    
    % figure out orientation and split into planes
    s = size(cube); %#ok<NODEF>
    r = s(1);
    c = s(2);
    z = s(3);
    if r == 2
        % y plane
        A = cube.data(1,:,:);
        B = cube.data(2,:,:);
        clear cube;
    elseif c == 2
        % x plane
        A = cube.data(:,1,:);
        B = cube.data(:,2,:);
        clear cube;        
    elseif z == 2
        % z plane
        A = cube.data(:,:,1);
        B = cube.data(:,:,2);
        clear cube;        
    else
        error('merge_segments:InvalidInput','1 dimension of the input cube needs to be equal to 2.  Did you load a merge cube or cutout cube?');
    end
 
    %% Compute merges and add to merge table
    
    % From Will:
    % Think that we can simplify boundary merge objective fcn to the following:
    %
    % Assumptions - objects should be continuous at boundaries:
    % No boundary branching or crazy stuff
    % This means we only have to consider one direction and don't
    % have to worry about overlapping two totally different sized objects
    % Reasonable initial assumption - certainly for testing.
    % Since it's terrible anyway, this is unlikely to be the weak link

    % Only thing you'll need to do is pass in "A" and "B" and then
    % appropriately handle mergeTo and mergeFrom
     
    % Inputs:  1 plane is A, one is B.  Assume they are of identical dimension
    % need to do this to avoid huge RP structures if IDs are large
    A2 = relabel_id(A); %tools
    rpA = regionprops(A2,'PixelIdxList','Area');
    clear A2;

    for ii = 1:length(rpA)
        ids = B(rpA(ii).PixelIdxList);
        if ignore_zeros
            % should generally do this because of small object removal
            ids(ids==0) = []; %optionally remove 0s
        end

        mId = mode(double(ids));

        mergeScore = sum(ids==mId)/length(ids);

        if mergeScore > threshold
            mergeTo = A(rpA(ii).PixelIdxList(1));  %index back into original
            mergeFrom = mId;
            m.merge(mergeTo, mergeFrom);
        end
    end

   
    
    
end

