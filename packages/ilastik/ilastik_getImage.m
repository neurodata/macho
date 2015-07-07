function ilastik_getImage(server, token, queryFile, fileOut, useSemaphore)
% J. Matelsky - adapted from mananno_getImage.m by W. Gray Roncal

if useSemaphore
    oo = OCP('semaphore');
else
    oo = OCP();
end

% Load query
load(queryFile)

oo.setServerLocation(server);
oo.setImageToken(token);
oo.setDefaultResolution(query.resolution);


im = oo.query(query);
for ii = 1:size(im.data,3)
    imwrite(im.data(:,:,ii), fileOut, 'writemode', 'append');
end