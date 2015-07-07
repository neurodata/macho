################################################################################
# (c) [2013] The Johns Hopkins University / Applied Physics Laboratory All Rights Reserved.
# Contact the JHU/APL Office of Technology Transfer for any additional rights.  www.jhuapl.edu/ott
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

""" Solves a membrane detection/classification problem.

This module provides the toplevel interface for solving a binary
"membrane vs non-membrane" classification problem for EM data sets
(e.g. [1]) using convolutional neural networks.

The overall approach is based on Dan Ciresan's paper [2] and the code
is derived from a LeNet example included in the Theano code base for
MNIST classification.

References:
  [1] http://brainiac2.mit.edu/isbi_challenge/
  [2] Ciresan, Dan, et al. "Deep neural networks segment neuronal membranes
      in electron microscopy images." Advances in neural information
      processing systems. 2012.

December 2013, mjp
"""

import os, os.path
import sys, time
import socket
import argparse

import numpy
from PIL import Image

import pdb

import theano
import theano.tensor as T

import em_networks as EMN
from em_utils import *
from tiles import *



def load_membrane_data(trainDataFile, trainLabelsFile, 
                       tileSize,
                       trainSlices, validSlices,
                       nZeeChannels=0):
    """Loads data set and creates corresponding tile managers.  
    """
    
    # load the volume and the labels
    if trainDataFile.endswith('.tif'):
        X = load_tiff_data(trainDataFile)
        # Assumes raw conference data (i.e. not preprocessed).
        #for ii in range(X.shape[0]):
        #    X[ii,:,:] = X[ii,:,:] - numpy.mean(X[ii,:,:])
        #X = X / numpy.max(numpy.abs(X))
        print '[%s]: Warning: no longer zero-meaning and scaling data' % __name__
    elif trainDataFile.endswith('.npz'):
        # assumes volume data is stored as the tensor X and is suitably preprocessed
        X = numpy.load(trainDataFile)['X']
    else:
        raise RuntimeError('unexpected data file extension')

    Y = load_tiff_data(trainLabelsFile)

    # mirror edges
    border = numpy.floor(tileSize/2.)
    X = mirror_edges_tensor(X, border)
    Y = mirror_edges_tensor(Y, border)

    # Use 0 and 1 as class labels.  This is actually important because
    # the neural network code will use class labels as indices into
    # the outputs of the last network layer.
    #
    # 0 := non-membrane
    # 1 := membrane
    Y[Y==0] = 1;  Y[Y==255] = 0
    assert(Y.max() == 1)

    X_train = X[trainSlices,:,:]
    Y_train = Y[trainSlices,:,:]
    X_valid = X[validSlices,:,:]
    Y_valid = Y[validSlices,:,:]

    # tile managers will put the images into GPU memory via Theano shared vars.
    train = TileManager(X_train, Y_train, tileSize=tileSize, nZeeChannels=nZeeChannels)
    valid = TileManager(X_valid, Y_valid, tileSize=tileSize, nZeeChannels=nZeeChannels)

    return (train, valid, (X, Y))



def random_image_modifiers(flipProb=.6, rotProb=.6):
    """Randomly applies certain transforms to a 2d image.
    As of this writing, these transforms are some
    combination of flips and rotations.
    """
    # clip probabilities to [0,1]
    flipProb = max(min(flipProb,1),0)
    rotProb = max(min(rotProb,1),0)
    
    flipDim = 0; rotDir = 0
    if numpy.random.rand() < flipProb:
        flipDim = numpy.sign(numpy.random.rand() - .5)
    if numpy.random.rand() < rotProb:
        rotDir = numpy.sign(numpy.random.rand() - .5)
    return flipDim, rotDir


    
def train_network(nn, trainMgr, validMgr,
                  nEpochs=30, learningRate=.001, decay=.995,
                  maxNumTilesPerEpoch=sys.maxint,
                  outDir="."):
    """Learns parameters for the given neural network.
    """

    p2 = int(numpy.floor(nn.p/2.0))
    
    # compute number of minibatches 
    nTrainBatches = int(numpy.ceil(trainMgr.batchSize / nn.miniBatchSize))
    nValidBatches = int(numpy.ceil(validMgr.batchSize / nn.miniBatchSize))
    print '[%s]: # of training batches is %d' % (__name__, nTrainBatches)

    # allocate symbolic variables 
    indexT = T.lscalar()         # index to a [mini]batch
    learningRateT = T.scalar()   # learning rate, theano variable
    
    print '[%s]: initializing Theano...' % __name__

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # functions for the validation data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    predict_validation_data = theano.function([indexT], nn.layers[-1].p_y_given_x,
             givens={
                nn.x: validMgr.X_batch_GPU[(indexT*nn.miniBatchSize):(indexT+1)*nn.miniBatchSize]})
                #nn.x: validMgr.X_batch_GPU[(indexT*nn.miniBatchSize):(indexT+1)*nn.miniBatchSize],
                #nn.y: validMgr.y_batch_int[(indexT*nn.miniBatchSize):(indexT+1)*nn.miniBatchSize]})

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # functions for the training data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The cost we minimize during training is the NLL of the model
    # Assumes the last layer is the logistic regression layer.
    cost = nn.layers[-1].negative_log_likelihood(nn.y)
    
    # create a list of all model parameters to be fit by gradient descent
    #params = layer3.params + layer2.params + layer1.params + layer0.params
    params = reduce(lambda a,b: a+b, [l.params for l in nn.layers])

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters via
    # SGD. Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learningRateT * grad_i))

    train_model = theano.function([indexT, learningRateT], [cost, nn.layers[-1].p_y_given_x], updates=updates,
          givens={
            nn.x: trainMgr.X_batch_GPU[(indexT*nn.miniBatchSize):(indexT+1)*nn.miniBatchSize],
            nn.y: trainMgr.y_batch_int[(indexT*nn.miniBatchSize):(indexT+1)*nn.miniBatchSize]})

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Do the training
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    startTime = time.clock()
    trainTime = 0
    validTime = 0
    lastChatter = -1
    nTilesProcessed = 0
    nTilesFlipped = 0
    nTilesRotated = 0

    print '[%s]: Training network.' % __name__
    for epoch in xrange(nEpochs):
        print '[%s]: Starting epoch %d / %d  (net time: %0.2f m)' % (__name__, epoch, nEpochs, (time.clock()-startTime)/60.)
        sys.stdout.flush()

        prevParams = EMN.save_network_parameters(nn, None)  # params just before learning
        predictions = numpy.zeros(trainMgr.y_batch_local.shape)
        nErrors = 0
        
        for slices,rows,cols,pct in trainMgr.make_balanced_pixel_generator():
            # reset predictions
            predictions[:] = -1;
            
            # transform images and udpate GPU memory
            flipDim,rotDir = random_image_modifiers()
            trainMgr.update_gpu(slices, rows, cols, flipDim=flipDim, rotDir=rotDir) 
            if flipDim != 0: nTilesFlipped += len(slices)
            if rotDir != 0: nTilesRotated += len(slices)
            
            # process all mini-batches
            for minibatchIdx in xrange(nTrainBatches):
                tic = time.clock()
                [costij, probij] = train_model(minibatchIdx, learningRate)
                trainTime += time.clock()-tic
                
                predij = numpy.argmax(probij,axis=1)
                predictions[(minibatchIdx*nn.miniBatchSize):(minibatchIdx+1)*nn.miniBatchSize] = predij
                
            nTilesProcessed += len(slices)
            nErrors = numpy.sum(predictions != trainMgr.y_batch_local)

            # periodically report progress (e.g. every 30 min)
            netTime = time.clock()-startTime
            if numpy.floor(netTime/1800) > lastChatter:
                print '[%s]: epoch %d; processed %0.2e tiles (%0.2f %%); net time %0.2f m' % (__name__, epoch, nTilesProcessed, pct, netTime/60.)
                lastChatter = numpy.floor(netTime/1800)
                sys.stdout.flush()
                
            # check for early epoch termination
            if nTilesProcessed >= maxNumTilesPerEpoch:
                print '[%s]:  epoch %d: quitting early after %d tiles processed (%0.2f %%)' % (__name__, epoch, nTilesProcessed, pct)
                break

        #----------------------------------------
        # update learning rate after each training epoch
        #----------------------------------------
        if decay < 1:
            learningRate *= decay
            
        #----------------------------------------
        # save result (even though it may just be an intermediate result)
        #----------------------------------------
        fn = 'params_epoch%02d' % epoch
        newParams = EMN.save_network_parameters(nn, os.path.join(outDir, fn), verbose=False)

        # report how much the network parameters changed
        keys = newParams.keys();  keys.sort()
        for key in keys:
            delta = numpy.ndarray.flatten(numpy.abs(newParams[key] - prevParams[key]))
            print '[%s]: %s  (%d params)\n                 %0.2e / %0.2e / %0.2e / %0.2e' % (__name__, key, len(delta), numpy.min(delta), numpy.max(delta), numpy.mean(delta), numpy.median(delta))
            
        #----------------------------------------
        # validation performance
        #----------------------------------------
        print '[%s]: validating performance ...' % __name__
        Y_hat = numpy.zeros(validMgr.Y_local.shape)
        for slices,rows,cols in validMgr.make_all_pixel_generator():
            # update tiles on the GPU
            validMgr.update_gpu(slices,rows,cols,flipDim=0,rotDir=0)
            
            for ii in range(nValidBatches):
                # predictions is a (nTiles x 2) matrix
                # grab the second output (y=1) 
                # (i.e. we store probability of membrane)
                tic = time.clock()
                pMembrane = predict_validation_data(ii)[:,1]
                validTime += time.clock() - tic

                # Be careful - on the last iteration, there may be
                # less than batchSize tiles remaining. 
                a = ii*nn.miniBatchSize
                b = min((ii+1)*nn.miniBatchSize, len(slices))
                if a > len(slices): break
                Y_hat[slices[a:b], rows[a:b], cols[a:b]] = pMembrane[0:b-a]
                
        # Validation statistics are based on a simple threshold
        # (without any other postprocessing).
        #
        # note: throw away the border before evaluating
        Y_true = validMgr.Y_local[:,p2:-p2,p2:-p2]
        Y_hat = Y_hat[:,p2:-p2,p2:-p2]
        eval_performance(Y_true, Y_hat, 0.5, verbose=True)
        eval_performance(Y_true, Y_hat, 0.7, verbose=True)

        # statistics for this epoch
        print '[%s]: epoch %d complete!' % (__name__, epoch)
        print '[%s]:    learning rate:       %0.2e' % (__name__, learningRate)
        print '[%s]:    # errors:            %d' % (__name__, nErrors)
        print '[%s]:    net elapsed time:    %0.2f m' % (__name__, ((time.clock() - startTime) / 60.))
        print '[%s]:    net gpu train time:  %0.2f m' % (__name__, (trainTime/60.))
        print '[%s]:    net validation time: %0.2f m' % (__name__, (validTime/60.))
        print '[%s]:    processed tiles:     %0.2e' % (__name__, nTilesProcessed)
        print '[%s]:    flipped tiles:       %0.2e' % (__name__, nTilesFlipped)
        print '[%s]:    rotated tiles:       %0.2e' % (__name__, nTilesRotated)


    endTime = time.clock()
    print('[%s]: Optimization complete.' % __name__)
    print '[%s]: The code for file "%s" ran for %0.2fm' % (__name__, os.path.split(__file__)[1], ((endTime - startTime) / 60.))
    print "[%s]: GPU train time: %0.2fm" % (__name__, (trainTime/60.0))


    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a neural network on the EM data set')

    #    
    # Parameters for defining and training the neural network
    #
    parser.add_argument('-n', dest='network', type=str, default='LeNetMembraneN3', 
                        help='neural network architecture (use a class name here)') 
    parser.add_argument('-e', dest='nEpochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('-r', dest='learnRate', type=float, default=0.001,
                        help='starting learning rate')
    parser.add_argument('-d', dest='decay', type=float, default=0.995,
                        help='learning rate decay')
    parser.add_argument('-m', dest='maxNumTilesPerEpoch', type=int, default=sys.maxint,
                        help='Maximum number of tiles used per epoch.  Use this if there are too many tiles to process them all each epoch.')

    # 
    # Data set parameters.  Assuming here a data cube, where each xy-plane is a "slice" of the cube.
    #
    parser.add_argument('-X', dest='trainFileName', type=str, default='train-volume-raw.npz',
                        help='Name of the file containing the membrane data (i.e. X)')
    parser.add_argument('-Y', dest='labelsFileName', type=str, default='train-labels.tif',
                        help='This is the file containing the class labels (i.e. Y)')
    parser.add_argument('--train-slices', dest='trainSlicesExpr', type=str, default='range(1,30)',
                        help='A python-evaluatable string indicating which slices should be used for training')
    parser.add_argument('--valid-slices', dest='validSliceExpr', type=str, default='range(27,30)',
                        help='A python-evaluatable string indicating which slices should be used for validation')

    #
    # Some special-case flags
    #
    parser.add_argument('--redirect-stdout', dest='redirectStdout', type=int, default=0,
                        help='set to 1 to send stdout to log.txt')
    parser.add_argument('-c', dest='nZeeChannels', type=int, default=0,
                        help='number of "mirror" channels') 
    args = parser.parse_args()


    # define and create output directory
    host = socket.gethostname() 
    deviceAndDate = theano.config.device + '_' + time.strftime('%d-%m-%Y')
    outDir = os.path.join(host, deviceAndDate, '%s_%03d_%0.4f_%0.4f' % (args.network, args.nEpochs, args.learnRate, args.decay))
    if not os.path.isdir(outDir): os.makedirs(outDir)
        
    # Redirect stdout, if asked to do so
    if args.redirectStdout:
        fn = os.path.join(outDir, 'log.txt')
        sys.stdout = open(fn, 'w')
            
    # Set up train/valid slices.  Using eval() might not be ideal, but
    # provides an easy way for the caller to define train/validation.
    trainSlices = eval(args.trainSlicesExpr)
    validSlices = eval(args.validSliceExpr)
    
    # create a neural network instance
    clazz = getattr(EMN, args.network)
    nn = clazz(nChannels=1+2*args.nZeeChannels)
    
    print '[%s]: Using the following parameters:' % __name__
    print '                 start time:   %s' % time.ctime()
    print '                 host:         %s' % host
    print '                 device:       %s' % theano.config.device
    print '                 pid:          %s' % os.getpid()
    print '                 train data:   %s' % args.trainFileName
    print '                 train labels: %s' % args.labelsFileName
    print '                 train slices: %s' % trainSlices
    print '                 valid slices: %s' % validSlices
    print '                 network:      %s' % nn.__class__.__name__
    print '                 # epochs:     %d' % args.nEpochs 
    print '                 max # tiles/epoch:   %d' % args.maxNumTilesPerEpoch
    print '                 learn rate:   %0.3f' % args.learnRate 
    print '                 decay:        %0.3f' % args.decay
    print '                 tile size:    %d' % nn.p
    for idx,l in enumerate(nn.layers): 
        print '                 layer %d: ' % idx,
        print str(l.W.get_value().shape)
    print '                 z-channels:   %d' % args.nZeeChannels
    print '                 output dir:   %s' % outDir

    print '[%s]: Loading data...' % __name__
    (train,valid,membraneData) = load_membrane_data(args.trainFileName, args.labelsFileName, 
                                                    tileSize=nn.p,
                                                    trainSlices=trainSlices,
                                                    validSlices=validSlices,
                                                    nZeeChannels=args.nZeeChannels)

    print '                 train dim:    %d x %d x %d' % (train.X_local.shape)
    print '                 valid dim:    %d x %d x %d' % (valid.X_local.shape)
    print '                 valid slices: %s' % (validSlices)

    #--------------------------------------------------
    # Do the work
    #--------------------------------------------------
    # train the neural network    
    train_network(nn, train, valid,
                  learningRate=args.learnRate,
                  decay=args.decay,
                  nEpochs=args.nEpochs,
                  maxNumTilesPerEpoch=args.maxNumTilesPerEpoch, 
                  outDir=outDir)

