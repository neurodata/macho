function axReticulaProcessing(imageToken, imageServer, uploadToken, uploadServer, ...
    upload_author, query_file, useSemaphore)
%
% Ayushi Sinha
% 12/8/2014
%
% This code shows how to use FindReticula. FindReticula annotates
% axoplasmic reticula (AR) in neural EM data.
% Wrapped and integrated into I2G by W. Gray Roncal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright Johns Hopkins Univeristy Applied Physics Laboratory
% Proprietary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check inputs
if ~exist('errorPage','var')
    % set to server default
    errorPage = '/mnt/pipeline/errorPages';
end

%% Create probability data access object
if useSemaphore == 1
    oo = OCP('semaphore');
else
    oo = OCP();
end

oo.setErrorPageLocation(errorPage);


%% Cutout Data

% Load query
load(query_file);

% Cutout image data
oo.setServerLocation(imageServer);
oo.setImageToken(imageToken);

query.setType(eOCPQueryType.imageDense);

im = oo.query(query);

% Set for upload
oo.setServerLocation(uploadServer);
oo.setAnnoToken(uploadToken);

%% Core Algorithm Code

fprintf('Allocating size for output... ' ) ;
% Allocate space for ourput cube
out = zeros(size(im.data));
fprintf('Done.\nFinding Reticula...\n');

% Find Reticula per slice
FindReticula( '--in', double(im.data), '--out', out, '--axoplasmicreticula');
fprintf('Done with MEX File.\n' ) ;

%% Upload Results

% Compute connected components to find Reticula strands
fprintf('Computing connected components... ' ) ;
CC = bwconncomp(out , 26 ) ;
rp = regionprops(CC,'PixelIdxList');
fprintf( 'Done. Number of connected components = %d.\n' , length(rp) ) ;

idOut = oo.reserve_ids(length(rp));

for i = 1:length(rp)
    out(rp(i).PixelIdxList) = idOut(i);
    
    AR = RAMONOrganelle;
    AR.setClass(eRAMONOrganelleClass.axoplasmicReticula);
    AR.setResolution(im.resolution);
    AR.setAuthor('Ayushi_ReticulaUpload_December16');
    
    ARALL{i} = AR;
    
end

oo.createAnnotation(ARALL);
fprintf('Done uploading empty AR objects.\n') ;

out = uint32(out);
ARDATA = RAMONVolume;
ARDATA.setResolution(im.resolution);
ARDATA.setXyzOffset(im.xyzOffset);
ARDATA.setCutout(out)
ARDATA.setDataType(eRAMONDataType.anno32);

ARDATA
oo.createAnnotation(ARDATA);
fprintf('Done uploading paint data.\n') ;
% fprintf('Uploading annotations...\n' ) ;
% for i = 1:length(rp)
%     if mod(i,1000) == 0
%         fprintf('%f%% Complete...\n' , i*100/length(rp) ) ;
%     end
%
%     clear pixLoc
%     q = rp(i).PixelIdxList ;
%     [pixLoc(:,2),pixLoc(:,1),pixLoc(:,3)] = ind2sub(CC.ImageSize,q);
%
%     AR = RAMONSynapse;
%     %AR = RAMONOrganelle;
%     %AR.setClass(eRAMONOrganelleClass.axoplasmicReticula);
%     AR.setResolution(1);
%
%     % Pick relevant author name for your upload (CHANGE THIS!!!)
%     AR.setAuthor('Ayushi_ReticulaUpload_December14');
%
%     AR.setVoxelList(im.local2Global(pixLoc));
% ARALL{i} = AR;
% end
%     oo.createAnnotation(ARALL);
%
% 