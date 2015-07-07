function galaWatershed(membraneIn, outFile, dilXY, dilZ, thresh)

load(membraneIn)
labels = segmentWatershed2D(cube, dilXY, dilZ, thresh);
cube = labels.data;

save(outFile, 'cube')