function ilastik_put_anno(server, token, queryFile, fileIn, protoRAMON, useSemaphore)

load(queryFile)

ANNO = RAMONVolume;
ANNO.setCutout(anno);
ANNO.setResolution(query.resolution);
ANNO.setXyzOffset([query.xRange(1),query.yRange(1),query.zRange(1)]);


cubeUploadDense(server,token,ANNO,protoRAMON,useSemaphore)