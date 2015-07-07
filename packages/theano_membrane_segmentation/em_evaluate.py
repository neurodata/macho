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


""" Evaluates an previously trained CNN on a data set.
"""

import os, os.path
import sys, time
import argparse

import numpy
from scipy import ndimage

import h5py

import em_networks as EMN
from em_utils import *
from tiles import *



def evaluate_network(nn, X, pp_func, selectedPixels, lowerPixels, upperPixels, outDir="."):
    """
    now uses evaluate_network(nn, X, pp_func_curried, outDir=args.outDir, selectedPixels, lowerPixels, upperPixels)
    before was: evaluate_network(nn, X, pp_func, lbX=float('-Inf'), ubX=float('Inf'), outDir=".")
    Runs a (previously trained) neural network against the test data set X.
    """
    
    tileSize = nn.p
    border = numpy.floor(tileSize/2.)
    X = mirror_edges_tensor(X, border)
    Y = numpy.zeros(X.shape) # we do not have labels for test data
    
    # create a tile manager
    testMgr = TileManager(X, Y, tileSize=tileSize)
    nTestBatches = int(numpy.ceil(testMgr.batchSize / nn.miniBatchSize))

    # set up Theano
    print '[em_evaluate]: initializing Theano (using device %s)...' % theano.config.device
    index = T.lscalar()

    # note: I threw away the nn.y parameter - I think it is unnecessary and actually 
    # causes problems in newer versions of Theano.
    predict_test_data = theano.function([index], nn.layers[-1].p_y_given_x,
             givens={
                nn.x: testMgr.X_batch_GPU[index*nn.miniBatchSize:(index+1)*nn.miniBatchSize]})

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # evaluation phase
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print '[em_evaluate]: beginning evaluation (using device %s)...' % theano.config.device
    Y_hat = numpy.zeros(X.shape)
    cnt = 0
    startTime = time.clock()
    #for slices,rows,cols in testMgr.make_all_pixel_generator():
    #for slices,rows,cols in testMgr.make_bandpass_pixel_generator(lbX, ubX):
    for slices,rows,cols in testMgr.make_selected_pixel_generator(selectedPixels):
        # update tiles on the GPU
        testMgr.update_gpu(slices,rows,cols,flipDim=0,rotDir=0)
        
        for ii in range(nTestBatches):
            # predictions is a (nTiles x 2) matrix
            # grab the second output (y=1) 
            # (i.e. we store probability of membrane)
            pMembrane = predict_test_data(ii)[:,1]
                
            # Be careful - on the last iteration, there may be
            # less than batchSize tiles remaining.
            a = ii*nn.miniBatchSize
            b = min((ii+1)*nn.miniBatchSize, len(slices))
            if a > len(slices): break
            Y_hat[slices[a:b], rows[a:b], cols[a:b]] = pMembrane[0:b-a]

        # report status every so often
        cnt += 1
        if numpy.mod(cnt,10)==1:
            print '[em_evaluate]: last processed (%d, %d, %d).  Net time: %0.2f m' % (slices[-1], rows[-1], cols[-1], (time.clock()-startTime)/60.)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # postprocessing
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # discard the border (optional)
    if True:
        p2 = int(numpy.floor(tileSize/2.0))
        X = X[:, p2:-p2, p2:-p2]
        Y_hat = Y_hat[:, p2:-p2, p2:-p2]

    print '[em_evaluate]: postprocessing...'
    Y_hat = pp_func(X, Y_hat)
    
    # apply "bandpass" classification values
    #Y_hat[X < lbX] = 0
    #Y_hat[X > ubX] = 0
    Y_hat[lowerPixels] = 1 # we actually made the classifier classify nonmembrane, so these pixels have probability 1 of being nonmembrane. (A temporary fix for now)
    Y_hat[upperPixels] = 1
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # save results
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print '[em_evaluate]: done processing.  saving results...'

    # numpy output
    numpy.savez(os.path.join(outDir, 'test-results'), X=X, Y_hat=Y_hat, P=selectedPixels)

    # hdf5 output
    f5 = h5py.File(os.path.join(outDir, 'test-results.hdf5'), 'w')
    f5.create_dataset('Y_hat', data=Y_hat)

    # also save as a .tif Unfortunately, it doesn't appear that
    # PIL has a good way of *creating* a multi-page tif.  So we'll
    # save each slice independently and rely on some outside tool
    # to merge them together (or leave them as separate files).
    for sliceId in range(X.shape[0]):
        X_i = X[sliceId,:,:]
        Y_hat_i = Y_hat[sliceId,:,:]

        # The ISBI conference wants the cube to represent probability
        # of non-membrane, so invert the probabilities before saving.
        fn = os.path.join(outDir, 'test_slice%02d_Yhat.tif' % sliceId)
        save_tiff_data(1.0 - Y_hat_i, fn)
        
        # Save a copy of the input data as well.
        fn = os.path.join(outDir, 'test_slice%02d_X.tif' % sliceId)
        save_tiff_data(X_i, fn, rescale=True)

    return Y_hat

 
def postprocess(X, Y_hat, cubicCoeffs=[]):
    Y_hatC = numpy.zeros(Y_hat.shape)

    # TODO: a better estimate for these coefficients
    #
    # NOTE: these were derived for ISBI 2012.  You'll need to
    # re-derive these for other data sets.
    #coeffs = [1.133, -0.843, 0.707]
    
    for ii in range(Y_hat.shape[0]):
        Yi = Y_hat[ii,:,:]
        
        # calibrate
        if len(cubicCoeffs) == 3:
            print '[em_evaluate]: performing cubic calibration'
            #Y_hatC[ii,:,:] = (Yi**3)*cubicCoeffs[0] + (Yi**2)*cubicCoeffs[1] + Yi*cubicCoeffs[2]
            Y_hatC[ii,:,:] = numpy.minimum(1,(Yi**3)*cubicCoeffs[0] + (Yi**2)*cubicCoeffs[1] + Yi*cubicCoeffs[2])
            Y_hatC[ii,:,:] = numpy.maximum(0,Y_hatC[ii,:,:])
        else:
            print '[em_evaluate]: omitting cubic calibration'
            Y_hatC[ii,:,:] = Yi
                
        # median filter
        # Y_hatC[ii,:,:] = median_filter(Y_hatC[ii,:,:],r=4)
        # Aurora took out median filtering, dean and will requested that they will do it themselves, we just do the coefficient calibration
    return Y_hatC


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parser = argparse.ArgumentParser('Evaluate a neural network on an EM data set')
    parser.add_argument('-n', dest='network', type=str, default='LeNetMembraneN3', 
                        help='neural network architecture') 
    parser.add_argument('-X', dest='volumeFileName', type=str, 
                        default=os.path.join('..', '..', 'Data', 'EM_2012', 'test-volume.tif'),
                        help='the data to evaluate')
    parser.add_argument('-Y', dest='labelsFileName', type=str, default=None,
                        help='ground truth labels (optional)')
    parser.add_argument('--eval-slices', dest='evalSliceExpr', type=str, default='',
                        help='A python-evaluatable string indicating which slices should be used for validation (or empty string to evaluate the whole stack)')
    parser.add_argument('-p', dest='paramFileName', type=str, default='params_epoch001.npz',
                        help='the neural network parameters to load')
    parser.add_argument('-o', dest='outDir', type=str, default='.',
                        help='output directory')

    # preprocessing arguments 
    parser.add_argument('--normalizeinputs', dest='normalizeinputs', type=bool, default=True,
                        help='this boolean input, if True, says that Input X should be renormalized before evaluation (default is true)')
    parser.add_argument('--intensity-lower-bound', dest='xLowerBound', type=float, default=float('-Inf'),
                        help='membrane pixel intensities less than this bound have membrane probability 0')
    parser.add_argument('--intensity-upper-bound', dest='xUpperBound', type=float, default=float('Inf'),
                        help='membrane pixel intensities greater than this bound have membrane probability 0')
    parser.add_argument('--thresh-dilation-kernel', dest='threshDilationKernel', type=int, default=0,
                        help='size of selected pixel dilation kernel (or 0 for no dilation)')
    parser.add_argument('--thresh-erosion-kernel', dest='threshErosionKernel', type=int, default=0,
                        help='size of selected pixel erosion kernel (or 0 for no erosion)')
    
    # postprocessing arguments
    parser.add_argument('--cubic-coeffs', dest='cubicCoeffs', type=str, default="[]",
                        help='coefficients to use with cubic postprocessing (or [] for none)')
    
    args = parser.parse_args()

    args.cubicCoeffs = eval(args.cubicCoeffs)

    # create a neural network instance
    clazz = getattr(EMN, args.network)
    nn = clazz()

    print '[em_evaluate]: Using the following parameters:'
    print '                 network:       %s' % nn.__class__.__name__
    for key, value in vars(args).iteritems():
        print '                 %s:    %s' % (key, value)
    print '\n'
    
    
    # load the volume
    if args.volumeFileName.endswith('.tif'):
        X = load_tiff_data(args.volumeFileName)
    elif args.volumeFileName.endswith('.npz'):
        # assumes volume data is stored as the tensor X
        X = numpy.load(args.volumeFileName)['X']
    else:
        raise RuntimeError('unexpected data file extension')

    # pare down to set of slices that are of interest (optional)
    if len(args.evalSliceExpr):
        evalSlices = eval(args.evalSliceExpr)
        X = X[evalSlices]

    # preprocessing.  This includes volume normalization (optional) and thresholding (optional)
    selectedPixels = numpy.logical_and(args.xLowerBound <= X, X <= args.xUpperBound)

    # Note: I observed strange behavior when running the erosion
    # operator on the entire tensor in a single call.  So for now,
    # I'll do this a slice at a time until I can figure out what the
    # situation is with the tensor.
    if args.threshDilationKernel > 0:
        kernel = ndimage.generate_binary_structure(2,1)
        kernel = ndimage.iterate_structure(kernel, args.threshDilationKernel).astype(int)
        for ii in range(selectedPixels.shape[0]):
            selectedPixels[ii,:,:] = ndimage.binary_dilation(selectedPixels[ii,:,:], structure=kernel, iterations=1)
    else:
        print '[em_evaluate]: no threshold dilation will be applied'
            
    if args.threshErosionKernel > 0:
        kernel = ndimage.generate_binary_structure(2,1)
        kernel = ndimage.iterate_structure(kernel, args.threshErosionKernel).astype(int)
        for ii in range(selectedPixels.shape[0]):
            selectedPixels[ii,:,:] = ndimage.binary_erosion(selectedPixels[ii,:,:], structure=kernel, iterations=1)
    else:
        print '[em_evaluate]: no threshold erosion will be applied'
            
    lowerPixels = numpy.logical_and(numpy.logical_not(selectedPixels), X < args.xLowerBound)
    upperPixels = numpy.logical_and(numpy.logical_not(selectedPixels), X > args.xUpperBound)
    
    if args.normalizeinputs : 
        for ii in range(X.shape[0]):
            X[ii,:,:] = X[ii,:,:] - numpy.mean(X[ii,:,:])
        X = X / numpy.max(numpy.abs(X))
        
    print '                volume dim:     %d x %d x %d' % (X.shape[0], X.shape[1], X.shape[2])
    print '            volume min/max:     %0.2f %0.2f' % (numpy.min(X), numpy.max(X))
    print '          # pixels to eval:     %d' % numpy.sum(selectedPixels)
    print ''

    if not os.path.exists(args.outDir): os.makedirs(args.outDir)

    # load the parameters and run it
    EMN.load_network_parameters(nn, args.paramFileName)
    pp_func_curried = lambda X, Yhat: postprocess(X, Yhat, args.cubicCoeffs)
    #Y_hat = evaluate_network(nn, X, pp_func_curried, outDir=args.outDir, lbX=args.xLowerBound, ubX=args.xUpperBound)
    # changed to now use the already found selectedPixels, rather than applying upper and lower bounds
    Y_hat = evaluate_network(nn, X, pp_func_curried, selectedPixels, lowerPixels, upperPixels, outDir=args.outDir)

    # generate performance metrics, if applicable
    if args.labelsFileName is not None:
        Y = load_tiff_data(args.labelsFileName)
        assert(numpy.all(numpy.logical_or(Y==0, Y==255)))

        # remap values to 1=membrane, 0=non-membrane            
        Y[Y==0] = 1;   Y[Y==255] = 0;
        
        for ii in range(Y_hat.shape[0]):
            print '[em_evaluate]: performance for slice %d:' % ii
            eval_performance(Y[ii,:,:], Y_hat[ii,:,:], .5, verbose=True)

