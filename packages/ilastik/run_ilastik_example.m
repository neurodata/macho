function run_ilastik_example()
% Ilastik Example (based on WGR's https://github.com/openconnectome/manno/blob/master/code/)
% ilastik(headless) starter to demonstrate protocol functionality.  All required inputs
% are hardcoded for this demo.  Paths are hardcoded for Linux/Mac.

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

%% Pull the image to annotate
ilastik_get_data(server, token, 'queryFileTest', 'tmp', 0)
RAMONtoTIFF('tmp', 'testilk.tiff')

% Run the annotations via a headless Ilastik instance
ilastik_runIlastik('./example_classifier.ilp', './results', '*.tiff')

% Push the annotations back
ilastik_put_anno(serverUp, tokenUp, './results/', '~/data/exampleAnno.nii.gz',' RAMONOrganelle', 1, 0)
