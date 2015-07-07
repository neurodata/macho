%% Training for Gala for reals

ocp = OCP();
ocp.setServerLocation('http://braingraph1dev.cs.jhu.edu/');
ocp.setImageToken('kasthuri11cc');

xTestExt = [4400, 5424];
yTestExt = [5440, 6464];
zTestExt = [1100,1200];

q2 = OCPQuery();
q2.setResolution(1);
q2.setType(eOCPQueryType.imageDense);
q2.setXRange(xTestExt);
q2.setYRange(yTestExt);
q2.setZRange(zTestExt);

[eData, mData, sData, vData] = get_ac4_data(1100,1200);

%% Membranes

q4 = q2;
q4.setType(eOCPQueryType.probDense);

%mDataTest = ocpMembrane.query(q4);


q4.setType(eOCPQueryType.annoDense);
ocp.setServerLocation('www.openconnecto.me');
ocp.setAnnoToken('ac4');
segLabels = ocp.query(q4);

%ball: 9,5
%disk: 5

dilXY = 11;
dilZ = 5;
thresh = 10;

ws_raw = segmentWatershed2D(mData,dilXY,dilZ,thresh);
membrane = permute(single(mData.data), [3,1,2]);
im = permute(eData.data, [3,1,2]);
ws = permute(ws_raw.data,[3,1,2]);
truth = permute(segLabels.data, [3,1,2]);

save('em_ac4.mat','im');
save('membrane_ac4.mat','membrane');
save('labels_ac4.mat','truth');
save('ws_ac4.mat','ws');

im = im(1:50,:,:);
membrane = membrane(1:50,:,:);
truth = truth(1:50,:,:);
ws = ws(1:50,:,:);
save('em_ac4medium.mat','im');
save('membrane_ac4medium.mat','membrane');
save('labels_ac4medium.mat','truth');
save('ws_ac4medium.mat','ws');


im = im(1:10,:,:);
membrane = membrane(1:10,:,:);
truth = truth(1:10,:,:);
ws = ws(1:10,:,:);
save('em_ac4short.mat','im');
save('membrane_ac4short.mat','membrane');
save('labels_ac4short.mat','truth');
save('ws_ac4short.mat','ws');


%% 3D watershed option

dilXY = 11;
dilZ = 5;
thresh = 10;

ws_raw = segmentWatershed(mData,dilXY,dilZ,thresh);

ws = permute(ws_raw.data,[3,1,2]);
save('ws_ac4_3D.mat','ws');

ws = ws(1:50,:,:);
save('ws_ac4medium_3D.mat','ws');

ws = ws(1:10,:,:);
save('ws_ac4short_3D.mat','ws');
