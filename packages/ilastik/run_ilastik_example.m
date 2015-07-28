function run_ilastik_example()
% Ilastik Example (based on WGR's https://github.com/openconnectome/manno/blob/master/code/)
% ilastik(headless) starter to demonstrate protocol functionality.  All required inputs
% are hardcoded for this demo.  Paths are hardcoded for Linux/Mac.

MACHO_PATH = '~/Documents/ocp/macho';
% Before running, set ILASTIK_PATH in ilastikRun.py


xstart = 7472;
xstop = xstart + 512;
ystart = 8712;
ystop = ystart + 512;
zstart = 1020;
zstop = zstart + 16;

resolution = 1;

query = OCPQuery;
query.setType(eOCPQueryType.imageDense);
query.setCutoutArgs([xstart, xstop], [ystart, ystop], [zstart, zstop], resolution);

%% Servers and tokens - alter appropriately
server = 'openconnecto.me';
token = 'kasthuri11cc';

serverUp = 'braingraph1dev.cs.jhu.edu';
tokenUp = 'temp2';

% Pull the image to annotate
ilastik_get_data(server, token, [MACHO_PATH '/code/packages/ilastik/queryFileTest'], 'tmp', 0)
RAMONtoTIFF('tmp', [MACHO_PATH '/code/packages/ilastik/kas11cc.tiff'])

% Run the annotations via a headless Ilastik instance
ilastikReturn = ilastik_runIlastik([MACHO_PATH '/code/data/ilastik_kas11_classifier.ilp'], 'kas11cc.tiff', './results');

% ilastik_runIlastik returns the system exit-code of the Ilastik call;
% we can check it here to see if everything exited correctly.

if ilastikReturn == 0
    % success!
    % Push the annotations back
    %ilastik_put_anno(server, token, queryFile, fileIn, protoRAMON, useSemaphore)
    ilastik_put_anno(serverUp, tokenUp, 'queryFileTest', [MACHO_PATH '/data/exampleAnno.nii.gz'], 'RAMONOrganelle', 0)
else
    ['Failure. Error Code ' ilastikReturn]
end


