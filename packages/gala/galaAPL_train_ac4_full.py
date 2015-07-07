#watershed example, based on code from Neal and Juan 

# imports
from gala import classify, features, agglo, evaluate as ev, optimized #imio
import scipy
import scipy.io
from gala import morpho
import scipy.ndimage as ndimage
import numpy as np
import scipy.signal as ssignal
import time
from gala import evaluate

start = time.time()
# read in OCP training data
inFileImage = '/mnt/pipeline/tools/i2g/packages/gala/em_ac4.mat'
inFileMembrane = '/mnt/pipeline/tools/i2g/packages/gala/membrane_ac4.mat'
inFileTruth = '/mnt/pipeline/tools/i2g/packages/gala/labels_ac4.mat'
inFileWatershed = '/mnt/pipeline/tools/i2g/packages/gala/ws_ac4.mat'

im = scipy.io.loadmat(inFileImage)['im']
im = im.astype('int32')
membraneTrain = scipy.io.loadmat(inFileMembrane)['membrane']
membraneTrain = membraneTrain.astype('float32')
	
gt_train = scipy.io.loadmat(inFileTruth)['truth']
gt_train = gt_train.astype('int64') #just in case!

ws_train = scipy.io.loadmat(inFileWatershed)['ws']
ws_train = ws_train.astype('int64') #just in case!
xdim, ydim, zdim = (im.shape)

fc = features.base.Composite(children=[features.moments.Manager(), features.histogram.Manager(25, 0, 1, [0.1, 0.5, 0.9]), 
    features.graph.Manager(), features.contact.Manager([0.1, 0.5, 0.9]) ])

print "Creating RAG..."
# create graph and obtain a training dataset
g_train = agglo.Rag(ws_train, membraneTrain, feature_manager=fc)
print 'Learning agglomeration...'
(X, y, w, merges) = g_train.learn_agglomerate(gt_train, fc,min_num_epochs=5)[0]
y = y[:, 0] # gala has 3 truth labeling schemes, pick the first one
print(X.shape, y.shape) # standard scikit-learn input format

print "Training classifier..."
# train a classifier, scikit-learn syntax
rf = classify.DefaultRandomForest().fit(X, y)
# a policy is the composition of a feature map and a classifier
learned_policy = agglo.classifier_probability(fc, rf)
classify.save_classifier(rf,'/mnt/pipeline/tools/i2g/packages/gala/ac4_full_classifier_v2.rf')