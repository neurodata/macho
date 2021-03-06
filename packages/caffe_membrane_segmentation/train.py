#   Trains a Caffe [1] DNN for pixel-level classification of objects within
#   electron microscopy (EM) data.
#
#   This code is a newer version of a previous (Theano-based)
#   implementation using the approach of [2] to classify membranes in
#   EM data.  In this approach, to classify a pixel one first extracts
#   a tile of data centered on the pixel in question (provides
#   context).  The DNN is provided with the entire tile, whose label
#   is that of the center pixel.
#
#   The reason for this code (vs simply applying the command-line
#   version of Caffe) is that the images used for training/test are
#   square patches extracted from the orignal EM images.  One option
#   would be to extract all of these images up front, but this eats up
#   a lot of storage and also makes it a bit clunky to deal with class
#   imbalance (there are many more non-membrane pixels than membrane
#   pixels).  Another alternative would be to write a new Caffe data
#   layer that extracts these tiles on-the-fly; this is basically what
#   this code does (without being a formal Caffe layer object). The
#   downside of this approach is that we have to implement the solver
#   details locally.
#
#
#   Note: as of this writing, pycaffe does *not* support testing data.
#   So the protobuf files must not include any constructs relating to
#   test data (including parameters in the solver prototxt)
#
#
# Example usage: (see also the Makefile)
#     PYTHONPATH=~/Apps/caffe-master/python nohup ipython train.py &
#
# References:
#   [1] Jia, Y. et. al. "Caffe: Convolutional Architecture for Fast Feature Embedding
#       arXiv preprint, 2014.
#
#   [2] Ciresan, D., et al. "Deep neural networks segment neuronal membranes in
#       electron microscopy images." NIPS 2012.
#
#   [3] https://groups.google.com/forum/#!topic/caffe-users/NKsSbZ3boGg
#
# Feb 2015, Mike Pekala

################################################################################
# (c) [2014] The Johns Hopkins University / Applied Physics Laboratory All Rights Reserved.
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



import sys, os, argparse, time
import pdb

import numpy as np

import emlib



def get_args():
    """Command line parameters for the training procedure.
    """
    
    parser = argparse.ArgumentParser('Train a neural network on the EM data set')

    #----------------------------------------
    # Parameters for defining and training the neural network
    #----------------------------------------
    parser.add_argument('-s', dest='solver', type=str,
                        default=os.path.join('caffe_files', 'n3-solver.prototxt'), 
                        help='Caffe solver prototxt file to use for training')
 
    parser.add_argument('-gpu', dest='gpu', type=int, default=-1,
                        help='Run in GPU mode on given device ID')
 
    #----------------------------------------
    # Data set parameters.  Assuming here a data cube, where each xy-plane is a "slice" of the cube.
    #----------------------------------------
    parser.add_argument('-X', dest='trainFileName', type=str, required=True, 
                        help='Name of the file containing the EM data')
    parser.add_argument('-Y', dest='labelsFileName', type=str, required=True,
                        help='This is the file containing the class labels')
    
    parser.add_argument('-lb', dest='lb', type=float, default=-np.infty,
                        help='lower bound for pre-processing bandpass filter')
    parser.add_argument('-ub', dest='ub', type=float, default=np.infty,
                        help='upper bound for pre-processing bandpass filter')
    
    parser.add_argument('--train-slices', dest='trainSlicesExpr', type=str, default='range(0,20)',
                        help='A python-evaluatable string indicating which slices should be used for training')
    
    parser.add_argument('--valid-slices', dest='validSliceExpr', type=str, default='range(27,30)',
                        help='A python-evaluatable string indicating which slices should be used for validation')
    
    parser.add_argument('--snapshot-prefix', dest='snapPrefix', type=str, default='',
                        help='(optional) override the "snapshot_prefix" in the solver file')

    return parser.parse_args()



def _xform_minibatch(X):
    """Performs operations on the data tensor X that preserve the class label
    (used to synthetically increase size of data set on-the-fly).

    For now, these transforms consist of flips and/or transposes.
    
    Note: for some reason, some implementation of row and column reversals, e.g.
               X[:,:,::-1,:]
          break Caffe.  Numpy must be doing something under the hood (e.g. changing
          from C order to Fortran order) to implement this efficiently which is
          incompatible w/ Caffe.  Hence the explicit construction of X2 with order 'C'.

    X := a (# slices, # channels, rows, colums) tensor
    """
    X2 = np.zeros(X.shape, dtype=np.float32, order='C')

    toss = np.random.rand()
    if toss < .2:
        X2[:,:,:,:] = X[:,:,::-1,:]
    elif toss < .4:
        X2[:,:,:,:] = X[:,:,:,::-1]
    else:
        X2[...] = X[...]  # no transformation

    # XXX: add transposes to the mix
    
    return X2
 


def _training_loop(solver, X, Y, M, solverParam, batchDim, outDir):
    """Performs CNN training.
    """
    assert(batchDim[2] == batchDim[3])     # tiles must be square

    # Some variables and storage that we'll use in the loop below
    #
    tileRadius = int(batchDim[2]/2)
    Xi = np.zeros(batchDim, dtype=np.float32)
    yi = np.zeros((batchDim[0],), dtype=np.float32)
    losses = np.zeros((solverParam.max_iter,)) 
    acc = np.zeros((solverParam.max_iter,))
    currIter = 0
    currEpoch = 0

    # SGD parameters.  SGD with momentum is of the form:
    #
    #    V_{t+1} = \mu V_t - \alpha \nablaL(W_t)
    #    W_{t+1} = W_t + V_{t+1}
    #
    # where W are the weights and V the previous update.
    # Ref: http://caffe.berkeleyvision.org/tutorial/solver.html
    #
    alpha = solverParam.base_lr            # alpha := learning rate
    mu = solverParam.momentum              # mu := momentum
    gamma = solverParam.gamma              # gamma := step factor
    isModeStep = (solverParam.lr_policy == u'step')
    isTypeSGD = (solverParam.solver_type == solverParam.SolverType.Value('SGD'))
    Vall = {}                              # stores previous SGD steps (for all layers)

    if not (isModeStep and isTypeSGD):
        raise RuntimeError('Sorry - only support SGD "step" mode at the present')
 
    # TODO: weight decay
    # TODO: layer-specific weights
    # TODO: evaluate performance on valid slices instead of train slices?
    #       (complicated by the fact that pycaffe doesn't support test mode)
 
    cnnTime = 0.0                          # time spent doing core CNN operations
    tic = time.time()
    
    while currIter < solverParam.max_iter:
        # Each generator provides a single epoch's worth of data.
        # However, Caffe doesn't really recognize the notion of an epoch; instead,
        # they specify a number of training "iterations" (mini-batch evaluations, I assume).
        # So the inner loop below is for a single epoch, which we may terminate
        # early if the max # of iterations is reached.
        currEpoch += 1
        it = emlib.stratified_interior_pixel_generator(Y, tileRadius, batchDim[0], mask=M)
        for Idx, epochPct in it:
            # Map the indices Idx -> tiles Xi and labels yi
            # 
            # Note: if Idx.shape[0] < batchDim[0] (last iteration of an epoch) a few examples
            # from the previous minibatch will be "recycled" here. This is intentional
            # (to keep batch sizes consistent even if data set size is not a multiple
            #  of the minibatch size).
            #
            for jj in range(Idx.shape[0]):
                yi[jj] = Y[ Idx[jj,0], Idx[jj,1], Idx[jj,2] ]
                a = Idx[jj,1] - tileRadius
                b = Idx[jj,1] + tileRadius + 1
                c = Idx[jj,2] - tileRadius
                d = Idx[jj,2] + tileRadius + 1
                Xi[jj, 0, :, :] = X[ Idx[jj,0], a:b, c:d ]

            # label-preserving data transformation (synthetic data generation)
            Xi = _xform_minibatch(Xi)

            #----------------------------------------
            # one forward/backward pass and update weights
            # (SGD with momentum term)
            #----------------------------------------
            _tmp = time.time()
            solver.net.set_input_arrays(Xi, yi)
            # XXX: could call preprocess() here?
            rv = solver.net.forward()
            solver.net.backward()

            for lIdx, layer in enumerate(solver.net.layers):
                for bIdx, blob in enumerate(layer.blobs):
                    key = (lIdx, bIdx)
                    V = Vall.get(key, 0.0)
                    Vnext = mu*V - alpha * blob.diff
                    blob.data[...] += Vnext
                    Vall[key] = Vnext
            cnnTime += time.time() - _tmp
                    
            # update running list of losses with the loss from this mini batch
            losses[currIter] = np.squeeze(rv['loss'])
            acc[currIter] = np.squeeze(rv['accuracy'])
            currIter += 1

            #----------------------------------------
            # Some events occur on mini-batch intervals.
            # Deal with those now.
            #----------------------------------------
            if (currIter % solverParam.snapshot) == 0:
                fn = os.path.join(outDir, 'iter_%05d.caffemodel' % (currIter))
                solver.net.save(str(fn))

            if isModeStep and ((currIter % solverParam.stepsize) == 0):
                alpha *= gamma

            if (currIter % solverParam.display) == 1:
                elapsed = (time.time() - tic)/60.
                print "[train]: completed iteration %d (of %d; %0.2f min elapsed; %0.2f CNN min)" % (currIter, solverParam.max_iter, elapsed, cnnTime/60.)
                print "[train]:    epoch: %d (%0.2f), loss: %0.3f, acc: %0.3f, learn rate: %0.3e" % (currEpoch, 100*epochPct, np.mean(losses[max(0,currIter-10):currIter]), np.mean(acc[max(0,currIter-10):currIter]), alpha)
                sys.stdout.flush()
 
            if currIter >= solverParam.max_iter:
                break  # in case we hit max_iter on a non-epoch boundary
 
    return losses, acc

    


if __name__ == "__main__":
    import caffe
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format

    #----------------------------------------
    # parse information from the prototxt files
    #----------------------------------------
    args = get_args()
    trainDir, solverFn = os.path.split(args.solver)
    if len(trainDir):
        os.chdir(trainDir)

    solverParam = caffe_pb2.SolverParameter()
    text_format.Merge(open(solverFn).read(), solverParam)

    netFn = solverParam.net
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFn).read(), netParam)
    
    batchDim = emlib.infer_data_dimensions(netFn)
    print('[train]: batch shape: %s' % str(batchDim))

    if len(args.snapPrefix):
        outDir = args.snapPrefix
    else:
        outDir = str(solverParam.snapshot_prefix)   # unicode -> str
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    
    #----------------------------------------
    # Load and preprocess data set
    #----------------------------------------
    X = emlib.load_cube(args.trainFileName, np.float32)
    Y = emlib.load_cube(args.labelsFileName, np.float32)

    # Class labels must be natural numbers (contiguous integers starting at 0)
    # because they are mapped to indices at the output of the network.
    # This next bit of code remaps the native y values to these indices.
    yAll = np.sort(np.unique(Y))
    Yhat = np.zeros(Y.shape, dtype=Y.dtype)
    for yIdx, y in enumerate(yAll):
        Yhat[Y==y] = yIdx
    Y = Yhat

    # mirror edges of images so that every pixel in the original data set can act
    # as a center pixel of some tile    
    borderSize = int(batchDim[2]/2)
    X = emlib.mirror_edges(X, borderSize)
    Y = emlib.mirror_edges(Y, borderSize)

    # Identify the subset of the data to use for training; make a copy (assumes
    # data set is not prohibitively large to make a copy)
    trainIdx = eval(args.trainSlicesExpr)
    Xtrain = X[trainIdx,:,:]
    Ytrain = Y[trainIdx,:,:]
    print('[train]: training data shape: %s' % str(Xtrain.shape))

    # Some pixels are trivial to classify based on their intensity.
    # We don't need a CNN for these - skip them in training (and in deploy).
    Mask = np.ones(Xtrain.shape, dtype=bool)
    Mask[Xtrain > args.ub] = 0
    Mask[Xtrain < args.lb] = 0
    if np.any(Mask == 0):
        nz = np.sum(Mask==0)
        print('[train]: bandpass mask is omitting %0.2f%% of the raw data' % (100 * nz / np.prod(Mask.shape)))
        print('[train]:   (%0.2f%% of these pixels have label 0)' % (100* np.sum(Ytrain[~Mask]==0) / nz))

    #----------------------------------------
    # Create the Caffe solver
    #----------------------------------------
    solver = caffe.SGDSolver(solverFn)
    for name, blobs in solver.net.params.iteritems():
        for bIdx, b in enumerate(blobs):
            print("%s[%d] : %s" % (name, bIdx, b.data.shape))

    # specify training mode and CPU or GPU
    if args.gpu >= 0:
        isModeCPU = False   # command line overrides solver file
        gpuId = args.gpu
    else:
        isModeCPU = (solverParam.solver_mode == solverParam.SolverMode.Value('CPU'))
        gpuId = 0
        
    # Note that different Caffe APIs put functions in different places (module vs net object).
    # Hence the try/catch.
    try:
        if not isModeCPU:
            caffe.set_mode_gpu()
            caffe.set_device(gpuId)
        else:
            caffe.set_mode_cpu()
        caffe.set_phase_train()
    except AttributeError:
        if not isModeCPU:
            solver.net.set_mode_gpu()
            solver.net.set_device(gpuId)
        else:
            solver.net.set_mode_cpu()
        solver.net.set_phase_train()
 
    #----------------------------------------
    # Do training; save results
    #----------------------------------------
    losses, acc = _training_loop(solver, Xtrain, Ytrain, Mask, solverParam, batchDim, outDir)
    
    solver.net.save(str(os.path.join(outDir, 'final.caffemodel')))
    np.save(os.path.join(outDir, '%s_losses' % outDir), losses)
    np.save(os.path.join(outDir, '%s_acc' % outDir), acc)
    
    print('[train]: all done!')
