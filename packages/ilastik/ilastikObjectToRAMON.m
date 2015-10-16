function ilastikObjectToRAMON(inFile, RAMONVolOutFile, queryFile, objChannel, padX, padY, padZ)

%For now, assumes only one dataset
h = h5info(inFile);
data = h5read(inFile, ['/', h.Datasets.Name]);
whos data
size(data)
unique(data)
data = squeeze(data(objChannel,:,:,:));
data = permute(data,[2,1,3]);
load(queryFile)

cube = RAMONVolume;
cube.setCutout(data);
cube.setResolution(query.resolution);
cube.setXyzOffset([query.xRange(1)+padX,query.yRange(1)+padY,query.zRange(1)+padZ]); %TODO
save(RAMONVolOutFile, 'cube') %-v7.3 TODO