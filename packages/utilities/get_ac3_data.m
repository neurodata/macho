function [eData, mData, sData, vData] = get_ac3_data(zSlice, zSliceEnd, padX, padY, padZ)
oo = OCP();
oo.setServerLocation('openconnecto.me')
oo.setImageToken('kasthuri11cc');

if nargin < 5
    pad = [0, 0, 0];
else
    pad = [padX, padY, padZ];
end

%AC3
xTrainExt = [5472-pad(1), 6496+pad(1)];
yTrainExt = [8712-pad(2), 9736+pad(2)];
zTrainExt = [zSlice-pad(3),zSliceEnd+pad(3)]; %1256

q = OCPQuery();
q.setResolution(1);
q.setType(eOCPQueryType.imageDense);
q.setXRange(xTrainExt);
q.setYRange(yTrainExt);
q.setZRange(zTrainExt);

try
    eData = oo.query(q);
    
catch
    eData = download_big_cuboid(oo,q);
end
%% Synapse Data

oo.setServerLocation('http://openconnecto.me/');
oo.setAnnoToken('ac3ac4');
oo.setAnnoChannel('ac3_synapse_truth');
q.setType(eOCPQueryType.annoDense);

try
    sData = oo.query(q);
catch
    sData = download_big_cuboid(oo,q);
end

%% Membranes
oo.setAnnoToken('cv_ac3_membrane_2014');
oo.setAnnoChannel('image');
q.setType(eOCPQueryType.probDense);

try
    mData = oo.query(q);
catch
    mData = download_big_cuboid(oo,q);
end
%% Vesicles

oo.setAnnoToken('cv_ac3_vesicle_2014');
oo.setAnnoChannel('annotation')
q.setType(eOCPQueryType.annoDense);

try
    vData = oo.query(q);
catch
    vData = download_big_cuboid(oo,q);
end
