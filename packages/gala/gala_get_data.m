function gala_get_data(emToken,  emServiceLocation, membraneToken, membraneServiceLocation,...
    queryFile, emCube, emMat, membraneMat, wsMtx, dilXY, dilZ, wsThresh, useSemaphore)

% gala_get_data - this function pulls data for rhoana and saves it to
% mat files.
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

%% Download EM Cube
ws_raw = segmentWatershed2D(membrane_data,dilXY,dilZ,wsThresh);
membrane = permute(membrane, [3,1,2]);
im = permute(em_cube.data, [3,1,2]);
ws = permute(ws_raw.data,[3,1,2]);



%% Save Output Data

% Save EM Matrix
save(emMat,'im');

save(emCube,'em_cube');

save(wsMtx,'ws');

% Save Membrane Matrix
save(membraneMat,'membrane');


end

