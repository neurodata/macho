function ilastik_get_data(server, token, queryFile, fileOut, useSemaphore)
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


cube = oo.query(query);
save(fileOut, 'cube');
end

put_data(type, format, ...):
    if (type != ramon):
        data = convert_(type)_to_ramon(data)
    
    upload(data)