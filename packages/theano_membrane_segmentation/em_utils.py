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

""" Assorted functions, primarily for working with image data, that do
*not* require Theano.

Note that all "images" here are typically assumed to be numpy arrays
(e.g. vs PIL Image objects).

December 2013, mjp
"""

import sys, os, time
import itertools

import numpy
import pylab
import scipy.stats
import scipy.signal.signaltools as ST
from PIL import Image
from scipy import ndimage

import pdb


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for working with images
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_tiff_data(dataFile, dtype='float32'):
    """
    Loads data from a multilayer .tif file.  
    Returns result as a 3d numpy tensor.
    """
    if not os.path.isfile(dataFile):
        raise RuntimeError('could not find "%s"' % dataFile)
    
    # load the data from multi-layer TIF files
    dataImg = Image.open(dataFile)
    X = [];
    for ii in xrange(sys.maxint):
        Xi = numpy.array(dataImg, dtype=dtype)
        X.append(Xi)
        try:
            dataImg.seek(dataImg.tell()+1)
        except EOFError:
            break

    # Put data together into a tensor.
    #
    # Also must "permute" dimensions (via the transpose function) so
    # that the slice dimension is first (for some reason, dstack seems
    # to do something odd with the dimensions).
    X = numpy.dstack(X).transpose((2,0,1))

    return X



def save_tiff_data(X, fileName, rescale=False):
    """
    Unfortunately, it appears PIL can only load multi-page .tif files
    (i.e. it cannot create them).  So the approach is to create them a
    slice at a time, using this function, and then combine them after
    the fact using convert, e.g.

    convert slice??.tif -define quantum:format=floating-point combinedF.tif
    """
    if rescale:
        # rescale data to [0 255]
        X = X - numpy.min(X)
        X = 255*(X / numpy.max(X))
    
        # change type to uint8 and save
        im = Image.fromarray(X.astype('uint8'))
    else:
        # 32 bit values in [0,1] are assumed
        # (ISBI conference format)
        #
        # Note that for ISBI, values near 0 are membrane and those
        # near 1 are non-membrane.
        assert(numpy.min(X) >= 0)
        assert(numpy.max(X) <= 1.0)
        im = Image.fromarray(X.astype('float32'))
            
    im.save(fileName)

 

def mirror_edges_tensor(X, nPixels, zeroBorder=False):
    """
    Mirrors all edges of a 3d tensor.
    """
    X_mirror = [mirror_edges(X[ii,:,:], nPixels, zeroBorder=zeroBorder) 
                for ii in range(X.shape[0])]
    X = numpy.dstack(X_mirror).transpose((2,0,1))
    return X



def mirror_edges(X, nPixels, zeroBorder=False):
    """
    Given an mxn matrix X, generates a new (m+2*nPixels)x(n+2*nPixels)
    matrix with an "outer border" created by mirroring pixels along
    the outer border of X
    """
    assert(nPixels > 0)
    
    m,n = X.shape
    Xm = numpy.zeros((m+2*nPixels, n+2*nPixels), dtype=X.dtype)
    Xm[nPixels:m+nPixels, nPixels:n+nPixels] = X

    # Special case: sometimes (e.g. for class labels) we don't really
    # want to mirror the data, but just leave a blank border.
    if zeroBorder:
        return Xm

    # a helper function for dealing with corners
    flip_corner = lambda X : numpy.fliplr(numpy.flipud(X))

    # top left corner
    Xm[0:nPixels,0:nPixels] = flip_corner(X[0:nPixels,0:nPixels])

    # top right corner
    Xm[0:nPixels,n+nPixels:] = flip_corner(X[0:nPixels,n-nPixels:])

    # bottom left corner
    Xm[m+nPixels:,0:nPixels] = flip_corner(X[m-nPixels:,0:nPixels])

    # bottom right corner
    Xm[m+nPixels:,n+nPixels:] = flip_corner(X[m-nPixels:,n-nPixels:])

    # top border
    Xm[0:nPixels, nPixels:n+nPixels] = numpy.flipud(X[0:nPixels,:])

    # bottom border
    Xm[m+nPixels:, nPixels:n+nPixels] = numpy.flipud(X[m-nPixels:,:])

    # left border
    Xm[nPixels:m+nPixels,0:nPixels] = numpy.fliplr(X[:,0:nPixels])

    # right border
    Xm[nPixels:m+nPixels,n+nPixels:] = numpy.fliplr(X[:,n-nPixels:])

    return Xm



def local_contrast_normalization(X, kernel=7):
    '''
    Local contrast normalization divides each pixel by the standard
    deviation of its neighbors.
    
    See: Jarrett et. al. "High-Accuracy Object Recognition with a
    New Convolutional Net Architecture and Learning Algorithm"
    '''
    m,n = X.shape
    
    assert(numpy.mod(kernel,2)==1)
    k2 = int(numpy.floor(kernel/2.0))

    # so I don't have to deal with special case code at the edges.
    X_tmp = mirror_edges(X,k2)
    selfIdx = kernel*k2 + k2 - 1
    
    X_out = numpy.zeros(X.shape)
    for ii in range(m):
        for jj in range(n):
            nbrs = X_tmp[ii:ii+kernel, jj:jj+kernel].flatten()
            nbrs = numpy.concatenate((nbrs[0:selfIdx], nbrs[selfIdx+1:])) # drop center pixel
            X_out[ii,jj] = X_tmp[ii,jj] / numpy.std(nbrs)

    return X_out



def median_filter(Y, r=2):
    """
    Simple median filter. 

    Parameters:
      Y := the input image
      r := median filter "radius".  Filter will be 2*r+1 in width/height.
    """
    # For now, assume a single 2d image (vs a stack)
    assert(len(Y.shape)==2)
    assert(r >= 1)

    w = 2*r+1  # width of kernel
    
    # so we can run the filter on the interior and still be centered
    # on all pixels of Y.
    Ym = mirror_edges(Y,r)

    if True:
        # This is the built-in version
        return ST.medfilt2d(Ym,w)[r:-r, r:-r]
    else:
        # a hand-rolled implementation
        Z = numpy.zeros(Y.shape)
        for row in range(Z.shape[0]):
            for col in range(Z.shape[1]):
                tile = Ym[row:row+w, col:col+w]
                assert(tile.shape[0] == w); 
                assert(tile.shape[1] == w)
                Z[row,col] = numpy.median(tile)

        assert(not numpy.any(numpy.isnan(Z)))
        return Z



def xform(X, flipDim=0, rotDir=0):
    """
    Generates a flipped and/or rotated version of the input image X.

    NOTE: this should work if X is a 2d matrix OR and 3d tensor,
    assuming the first dimension is the # of images/tiles.
    """
    assert(X.ndim==2 or X.ndim==3)
    
    if flipDim > 0:
        Y = numpy.flipud(X)
    elif flipDim < 0:
        Y = numpy.fliplr(X)
    else:
        # avoid returning a view of X...
        Y = X.copy()

    # be careful with transposes here - if we have a 3d tensor, be
    # careful to transpose the latter two dimensions only
    if X.ndim==2:
        if rotDir > 0:
            # rotate 90 degrees
            Y = numpy.fliplr(Y.T)
        elif rotDir < 0:
            # rotate -90 degrees
            Y = numpy.flipud(Y.T)
    else:
        # ndim = 3
        if rotDir > 0:  
            # rotate 90 degrees
            Y = numpy.fliplr(numpy.transpose(Y,[0,2,1]))
        elif rotDir < 0:
            # rotate -90 degrees
            Y = numpy.flipud(numpy.transpose(Y,[0,2,1]))

    return Y



def z_slices(centerSlice, nNbrs, zRange=[0,30]):
    """
    Computes slice indices of neighbors 'above' and 'below' the center slice.
    The only trick here is to mirror when the number of slices hits the lower or upper bound on z.
    """
    rv = numpy.arange(centerSlice-nNbrs, centerSlice+nNbrs+1)
    rv[rv>=zRange[1]] = zRange[1] - (rv[rv>=zRange[1]] - zRange[1]) - 2
    rv[rv<zRange[0]] = zRange[0] - rv[rv<zRange[0]]
    return rv



def calibrate(Y, Yhat, showPlot=False):
    """
    Re-calibrates a membrane probability estimate as per section 2.3 in [1].
    
    This function based on Aurora's code from view_results.py
    
    References:
      [1] Ciresan et. al. "Deep Neural Networks Segment Neuronal 
          Membranes in Electron Microscopy Images," NIPS 2012.
    """
    # XXX: address multiple slices??  For now, caller has to concate
    # multiple slices into one big image before calling...
    assert(len(Y.shape) == 2)
    assert(len(Yhat.shape) == 2)

    # This function returns a logical array that counts the number of
    # points in an image that fall in the interval [a,b).
    # Might not be super efficient to do this way, but whatever.
    count = lambda Z, a, b: numpy.logical_and(Z >= a, Z < b)

    # Pick a set of probability intervals [a_i,b_i).  Then, for each
    # interval, count the number of pixels with this probability and
    # see what proportion of them correspond to actual membrane data
    # (Y==1).
    binCenters = numpy.arange(.05, 1, .1)
    d = 0.05  # delta = half bin width
    probCnt = [numpy.sum(numpy.logical_and(Y==1, count(Yhat, c-d, c+d))) / float(numpy.sum(count(Yhat, c-d, c+d)))
               for c in binCenters]

    # reshape into column vectors
    probCnt = numpy.array(probCnt)
    probCnt = numpy.reshape(probCnt, (len(probCnt),1))
    binCenters = numpy.reshape(binCenters, (len(binCenters),1))

    # least squares fit to data
    P = numpy.concatenate((binCenters**3, binCenters**2, binCenters),axis=1)
    rvLSest = scipy.linalg.lstsq(P, probCnt)
    coeffs = rvLSest[0]

    if showPlot:
        result = numpy.dot(P,coeffs)
        pylab.figure
        pylab.plot(binCenters, probCnt, 'bs', binCenters, result, 'g-')
        pylab.show()

    return coeffs



def interior_pixel_generator(Y, borderSize, batchSize, whichSlices=[]):
    """
    Iterates over all pixels in the interior of Y (i.e. not lying in
    the outer borderSize pixels).

    Technically this function only really needs the shape of y, but
    the balanced pixel generator needs the labels, so to keep things
    consistent this function asks for the full Y tensor as well.
    """
    [s,m,n] = Y.shape

    # default is to consider all slices
    if not whichSlices: whichSlices = range(s)
        
    # "it" generates all possible upper-left-hand corners
    # of tiles within the interior of the image.
    it = itertools.product(whichSlices, xrange(borderSize, m-borderSize), xrange(borderSize, n-borderSize))
    allDone = False

    slices = numpy.zeros((batchSize,), dtype='int32')
    rows = numpy.zeros((batchSize,), dtype='int32')
    cols = numpy.zeros((batchSize,), dtype='int32')
        
    # Return pixels in batches of batchSize (possibly fewer, on the
    # final iteration).
    while not allDone:
        for jj in range(batchSize):
            try:
                slices[jj], rows[jj], cols[jj] = it.next()
            except StopIteration: 
                allDone = True
                # truncate to proper size
                slices = slices[0:jj]
                rows = rows[0:jj]
                cols = cols[0:jj]
                break

        yield slices, rows, cols


        
def bandpass_interior_pixel_generator(X, lowerBound, upperBound, borderSize, batchSize):
    """
    Iterates over all pixels in X that are (a) not in the border
    defined by borderSize and (b) between the lower and upper bound.
    """
    
    [s,m,n] = X.shape
        
    # only pixels in the interior of the image can be used as tile centers.
    #bitMask = numpy.zeros(Y.shape)
    #bitMask[:, borderSize:-borderSize, borderSize:-borderSize] = 1
    borderMask = numpy.ones(X.shape)
    if borderSize > 0:
        borderMask[:, 0:borderSize, :] = 0
        borderMask[:, m-borderSize:m, :] = 0
        borderMask[:, :, 0:borderSize] = 0
        borderMask[:, :, n-borderSize:n] = 0

    # bandpass
    bandpass = numpy.logical_and(lowerBound <= X, X <= upperBound)
    slices, rows, cols = numpy.nonzero(numpy.logical_and(bandpass, borderMask==1))

    # return the pieces incrementally
    for ii in range(0, len(slices), batchSize):
        nRet = min(batchSize, len(slices)-ii)
        yield slices[ii:ii+nRet], rows[ii:ii+nRet], cols[ii:ii+nRet]
              

def selected_interior_pixel_generator(X, selectedPixels, borderSize, batchSize):
    """
    Iterates over all pixels in X that are (a) not in the border
    defined by borderSize and (b) within the set of selectedPixels
    """
    
    [s,m,n] = X.shape
    
    # only pixels in the interior of the image can be used as tile centers.
    #bitMask = numpy.zeros(Y.shape)
    #bitMask[:, borderSize:-borderSize, borderSize:-borderSize] = 1
    borderMask = numpy.ones(X.shape)
    if borderSize > 0:
        borderMask[:, 0:borderSize, :] = 0
        borderMask[:, m-borderSize:m, :] = 0
        borderMask[:, :, 0:borderSize] = 0
        borderMask[:, :, n-borderSize:n] = 0
    bandpassMask = numpy.zeros(X.shape)
    if 1: #borderSize > 0:
        bandpassMask[:, borderSize:m-borderSize, borderSize:n-borderSize] = selectedPixels

    # bandpass
    #bandpass = numpy.logical_and(lowerBound <= X, X <= upperBound)
    slices, rows, cols = numpy.nonzero(numpy.logical_and(bandpassMask==1, borderMask==1))

    # return the pieces incrementally
    for ii in range(0, len(slices), batchSize):
        nRet = min(batchSize, len(slices)-ii)
        yield slices[ii:ii+nRet], rows[ii:ii+nRet], cols[ii:ii+nRet]
        

         
def balanced_interior_pixel_generator(Y, borderSize, batchSize, whichSlices=[], stratified=False):
    """
    Iterates over a subset of all pixels such that there is an
    equal number of positive and negative examples across
    iterations.

    UPDATE: TODO: describe the stratified flag
    """
    # for now, only supports binary class labels encoded as {0,1}.
    vals = numpy.unique(Y); numpy.sort(vals)
    assert(vals[0] == 0 and vals[1] == 1)

    # batch size must be even
    assert(numpy.mod(batchSize,2)==0)
    
    [s,m,n] = Y.shape
        
    # Used to restrict the set of pixels under consideration.
    bitMask = numpy.zeros(Y.shape)
        
    # default is to consider all slices
    if len(whichSlices)==0: whichSlices = range(s)

    if stratified:
        # In the "stratified" sampling scheme, the row composition
        # determines which pixels can be selected.
        for kk in range(s):
            for ii in range(borderSize, m-borderSize):
                idxPos = numpy.nonzero(Y[kk,ii,borderSize:n-borderSize] == 1)[0]
                idxNeg = numpy.nonzero(Y[kk,ii,borderSize:n-borderSize] == 0)[0]
                idxPos += borderSize
                idxNeg += borderSize

                # rebalance (by trimming majority class)
                cnt = min(len(idxPos), len(idxNeg))
                if len(idxNeg) > cnt:
                    numpy.random.shuffle(idxNeg)
                    idxNeg = idxNeg[0:cnt]
                else:
                    numpy.random.shuffle(idxPos)
                    idxPos = idxPos[0:cnt]
                        
                # mark for selection
                if cnt > 0:
                    rowsToAdd = ii*numpy.ones((cnt,), dtype='int32')
                    slicesToAdd = kk*numpy.ones((cnt,), dtype='int32')
                    bitMask[slicesToAdd, rowsToAdd, idxPos] = 1
                    bitMask[slicesToAdd, rowsToAdd, idxNeg] = 1

    else:
        # In the "uniform" sampling scheme, any interior pixel within
        # any valid slice is ok.
        for sliceId in whichSlices:
            bitMask[sliceId,:,:] = 1

        # only pixels in the interior of the image can be used as tile centers.
        bitMask[:, 0:borderSize, :] = 0
        bitMask[:, m-borderSize:m, :] = 0
        bitMask[:, :, 0:borderSize] = 0;
        bitMask[:, :, n-borderSize:n] = 0;
    
    # Find locations of pixels in each class.
    # *** ASSUMES: negative examples have label <= 0
    negSlices, negRows, negCols = numpy.nonzero(numpy.logical_and(Y <= 0, bitMask==1))
    posSlices, posRows, posCols = numpy.nonzero(numpy.logical_and(Y > 0, bitMask==1))

    # randomize pixel visit order
    negSlices, negRows, negCols = shuffle_sync(negSlices, negRows, negCols)
    posSlices, posRows, posCols = shuffle_sync(posSlices, posRows, posCols)

    # rebalance.
    cnt = min(len(negRows), len(posRows))
    negSlices = negSlices[0:cnt]; negRows = negRows[0:cnt];  negCols = negCols[0:cnt];
    posSlices = posSlices[0:cnt]; posRows = posRows[0:cnt];  posCols = posCols[0:cnt];

    # interleave the two sequences.
    # do it with matrix ops.
    nTotal = len(posSlices) + len(negSlices)
    allSlices = numpy.reshape(numpy.vstack((posSlices, negSlices)).T, (nTotal,))
    allRows = numpy.reshape(numpy.vstack((posRows, negRows)).T, (nTotal,))
    allCols = numpy.reshape(numpy.vstack((posCols, negCols)).T, (nTotal,))

    # a quick sanity check
    assert(numpy.max(allRows) <= (m-borderSize))
    assert(numpy.min(allRows) >= borderSize)
    assert(numpy.max(allCols) <= (n-borderSize))
    assert(numpy.min(allCols) >= borderSize)

    # return the pieces incrementally
    for ii in range(0, len(allSlices), batchSize):
        nRet = min(batchSize, len(allSlices)-ii)
        yield allSlices[ii:ii+nRet], allRows[ii:ii+nRet], allCols[ii:ii+nRet], (100*(ii+1)/float(len(allSlices)))


        
def difficult_interior_pixel_generator(Y, X, borderSize, batchSize, pct=15):
    """
    """
    # for now, only supports binary class labels encoded as {0,1}.
    vals = numpy.unique(Y); numpy.sort(vals)
    assert(vals[0] == 0 and vals[1] == 1)

    # batch size must be even
    assert(numpy.mod(batchSize,2)==0)
    
    [s,m,n] = Y.shape
        
    # only pixels in the interior of the image can be used as tile centers.
    #bitMask = numpy.zeros(Y.shape)
    #bitMask[:, borderSize:-borderSize, borderSize:-borderSize] = 1
    bitMask = numpy.ones(Y.shape)
    if borderSize > 0:
        bitMask[:, 0:borderSize, :] = 0
        bitMask[:, m-borderSize:m, :] = 0
        bitMask[:, :, 0:borderSize] = 0
        bitMask[:, :, n-borderSize:n] = 0

    # Finds instances of the negative class with an x-value in the
    # bottom pct percent.  
    negVals = X[Y<=0].flatten()
    thresh = scipy.stats.scoreatpercentile(negVals, pct)
    negIdxLEQ = numpy.logical_and(numpy.logical_and(Y <= 0, bitMask==1), X <= thresh)
    negIdxGT = numpy.logical_and(numpy.logical_and(Y <= 0, bitMask==1), X > thresh)

    # positive and negative pixel locations.  Note we split the
    # negative pixel population into two subpopulations based on the
    # threshold.
    posSlices, posRows, posCols = numpy.nonzero(numpy.logical_and(Y>0, bitMask==1))
    negSlicesLEQ, negRowsLEQ, negColsLEQ = numpy.nonzero(negIdxLEQ)
    negSlicesGT, negRowsGT, negColsGT = numpy.nonzero(negIdxGT)

    # Randomize pixel visit order.  Note that this order puts all the
    # low negative indices first...a preferential sampling approach
    # might be better...
    posSlices, posRows, posCols = shuffle_sync(posSlices, posRows, posCols)
    negSlicesLEQ, negRowsLEQ, negColsLEQ = shuffle_sync(negSlicesLEQ, negRowsLEQ, negColsLEQ)
    negSlicesGT, negRowsGT, negColsGT = shuffle_sync(negSlicesGT, negRowsGT, negColsGT)
    negSlices = numpy.concatenate((negSlicesLEQ, negSlicesGT))
    negRows = numpy.concatenate((negRowsLEQ, negRowsGT))
    negCols = numpy.concatenate((negColsLEQ, negColsGT))

    # rebalance.
    cnt = min(len(negRows), len(posRows))
    negSlices = negSlices[0:cnt]; negRows = negRows[0:cnt];  negCols = negCols[0:cnt];
    posSlices = posSlices[0:cnt]; posRows = posRows[0:cnt];  posCols = posCols[0:cnt];

    # interleave the two sequences.
    # do it with matrix ops.
    nTotal = len(posSlices) + len(negSlices)
    allSlices = numpy.reshape(numpy.vstack((posSlices, negSlices)).T, (nTotal,))
    allRows = numpy.reshape(numpy.vstack((posRows, negRows)).T, (nTotal,))
    allCols = numpy.reshape(numpy.vstack((posCols, negCols)).T, (nTotal,))

    # a quick sanity check
    assert(numpy.max(allRows) <= (m-borderSize))
    assert(numpy.min(allRows) >= borderSize)
    assert(numpy.max(allCols) <= (n-borderSize))
    assert(numpy.min(allCols) >= borderSize)

    # return the pieces incrementally
    for ii in range(0, len(allSlices), batchSize):
        nRet = min(batchSize, len(allSlices)-ii)
        yield allSlices[ii:ii+nRet], allRows[ii:ii+nRet], allCols[ii:ii+nRet], (100*(ii+1)/float(len(allSlices)))

        

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for working with classification data sets
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def eval_performance(Y, Y_hat, thresh, verbose=False, doPlot=False):
    """
    Evaluates performance for a single tensor.
      Y := the true class labels
      Y_hat := the estimated labels (probabilities)
    """
    nTruePos = numpy.sum(numpy.logical_and(Y_hat >= thresh, Y==1))
    nTrueNeg = numpy.sum(numpy.logical_and(Y_hat < thresh, Y==0))
    nFalsePos = numpy.sum(numpy.logical_and(Y_hat >= thresh, Y==0))
    nFalseNeg = numpy.sum(numpy.logical_and(Y_hat < thresh, Y==1))
    fallout = nFalsePos / float(nTrueNeg+nFalsePos)
    recall = nTruePos / float(nTruePos+nFalseNeg)
    precision = nTruePos / float(nTruePos+nFalsePos)
    specificity = nTrueNeg / float(nTrueNeg + nFalsePos)
    f1 = 2*(precision*recall) / (precision+recall)
    f1Alt = 2*nTruePos / float(2*nTruePos + nFalsePos + nFalseNeg)

    # for predicted probabilities, see empirically how many are actually membrane.
    # See calibrate() for more details
    bins = numpy.logical_and(Y_hat >= (thresh-.05), Y_hat < (thresh+.05))
    if numpy.sum(bins) > 0:
        probT = numpy.sum(numpy.logical_and(Y==1, bins)) / float(numpy.sum(bins))
    else:
        probT = numpy.nan
            
    if verbose:
        print '[info]: for threshold: %0.2f' % thresh
        print '[info]:    p(%d%%):                  %0.3f' % (100*thresh,probT)
        print '[info]:    FP/N (fall-out):         %0.3f' % fallout
        print '[info]:    TN/N (specificity):      %0.3f' % specificity
        print '[info]:    TP/P (recall):           %0.3f' % recall
        print '[info]:    TP/(TP+FP) (precision):  %0.3f' % precision
        print '[info]:    F1:                      %0.3f' % f1

    if doPlot:
        tp = numpy.nonzero(numpy.logical_and(Y_hat >= thresh, Y==1))
        tn = numpy.nonzero(numpy.logical_and(Y_hat < thresh, Y==0))
        fp = numpy.nonzero(numpy.logical_and(Y_hat >= thresh, Y==0))
        fn = numpy.nonzero(numpy.logical_and(Y_hat < thresh, Y==1))
        Z = numpy.zeros(Y.shape)
        Z[tp] = 1
        Z[tn] = 0
        Z[fp] = .9
        Z[fn] = .5
        pylab.imshow(Z, cmap='spectral')
        pylab.show()
        

    return {'nTruePos' : nTruePos,
            'nTrueNeg' : nTrueNeg,
            'nFalsePos' : nFalsePos,
            'nFalseNeg' : nFalseNeg,
            'recall' : recall,
            'precision' : precision,
            'fallout' : fallout,
            'f1' : f1}


def view_data_set(X,Y,save=False,prefix="data"):
    """
    Incrementally visualizes slices of a data set.  This function is
    mainly for debugging.
    """
    for ii in range(X.shape[0]):
        pylab.subplot(2,2,1)
        pylab.imshow(X[ii,:,:], interpolation='nearest', cmap='bone')
        pylab.colorbar()
        pylab.title('Tile %d (of %d). Mean X=%0.2f' % (ii, X.shape[0], numpy.mean(X[ii,:,:])))
        
        pylab.subplot(2,2,2)
        pylab.hist(numpy.reshape(X[ii,:,:], (X.shape[1]*X.shape[2],1)), bins=50)
        
        pylab.subplot(2,2,3)
        pylab.imshow(Y[ii,:,:], interpolation='nearest', cmap='bone')
        pctTarget = numpy.sum(Y[ii,:,:]!=0) / (1.0*Y.shape[1]*Y.shape[2])
        pylab.title('%0.2f percent membrane' % pctTarget)
        
        pylab.subplot(2,2,4)
        pylab.hist(numpy.reshape(Y[ii,:,:], (Y.shape[1]*Y.shape[2],1)), bins=2)

        if save:
            outFileName = '%s_slice_%03d.eps' % (prefix,ii)
            pylab.savefig(outFileName)
            pylab.close()
        else:
            pylab.show()


                
def number_of_visible_contiguous_regions(Y, tileSize, membraneVal=0, verbose=False):
    """For a given tile size computes, for each
    possible tile, the number of contiguous compartments within that tile.

    Y        : a 3d tensor (membrane stack)
    tileSize : the tile width/height (assumes square tiles)
    """
    # tile size must be odd
    assert(numpy.mod(tileSize,2)==1)
    ts2 = int(numpy.floor(tileSize/2.0))

    # assuming a 3d tensor
    assert(Y.ndim==3)

    # preallocate return value 
    Z = numpy.zeros(Y.shape)

    # Go through each possible tile, and count the number of
    # contiguous compartments.  Another approach might be to count all
    # the compartment up a-priori, and then just run a unique()
    # against each tile.  However, this will not work in the case of
    # non-convex compartments that are "segmented" by a tile edge and
    # thus appear twice in the same tile as disjoint regions.
    batchSize=10000
    tic = time.clock()
    for slices,rows,cols in interior_pixel_generator(Y, ts2, batchSize):
        for slic,row,col in zip(slices,rows,cols):
            tile = Y[slic, row-ts2:row+ts2+1, col-ts2:col+ts2+1]
            tile = numpy.squeeze(tile)
            tileLabels = ndimage.measurements.label(tile)[0]
            Z[slic,row,col] = len(numpy.unique(tileLabels[tile!=membraneVal]))
        if verbose:
            print '[n_vis_cont_reg]: processed (%d,%d,%d) - (%d,%d,%d); runtime: %0.2fm' % (slices[0], rows[0], cols[0], slices[-1], rows[-1], cols[-1], (time.clock()-tic)/60.0)

    return Z


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Misc. functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def shuffle_sync(*seqs):
    """
    Given a list of sequences, all having the same length, returns a
    shuffled version where each sequence has been shuffled the same
    way to keep them all "in sync".
    """
    if len(seqs)==0: return
    
    n = len(seqs[0])
    idx = range(n);
    numpy.random.shuffle(idx)

    return [s[idx] for s in seqs]
