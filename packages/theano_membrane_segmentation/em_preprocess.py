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

"""
Preprocesses data set [1].
Only run this on the data (i.e. not the labels, ldo).

Example:
 python em_preprocess.py ../../Data/EM_2012/train-volume.tif

December 2013, mjp

References:
  [1] http://brainiac2.mit.edu/isbi_challenge/
"""


import sys
import os.path
import argparse

import pylab
from scipy import stats

from em_utils import *


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# helper function; col(x) is like x(:) in matlab
col = lambda X: numpy.reshape(X, (-1,))



#-------------------------------------------------------------------------------
# parse command line args
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser('Pre-process an EM data set')
parser.add_argument('-v', dest='inFile', type=str, 
                    default='test-volume.tif', 
                    help='the EM volume file to preprocess (.tif file)')
parser.add_argument('-d', dest='doProcessing', type=int, default=1, 
                    help='set to 0 to skip actual pre-processing and view existing data') 
args = parser.parse_args()


# output file name
dirName, fileName = os.path.split(args.inFile)
if fileName.endswith('.tif'):
    fileName = fileName[0:-4]

outFileLCN = os.path.join(dirName, fileName+'-lcn')
outFileRaw = os.path.join(dirName, fileName+'-raw')

# load the volume
X = load_tiff_data(args.inFile)

#-------------------------------------------------------------------------------
# Preprocessing (can turn off if you just want to look at data you've
# already processed)
#-------------------------------------------------------------------------------
if args.doProcessing:
    print '[em_preprocess]: preprocessing...'

    # Try a couple of preprocessing approaches.
    # Create a tensor for each.
    X_lcn = numpy.zeros(X.shape)
    X_raw = numpy.zeros(X.shape)

    for ii in range(X.shape[0]):
        print '[em_preprocess]: processing layer %d (of %d)' % (ii, X.shape[0])
        
        # zero mean & local contrast normalization
        V = X[ii,:,:] - numpy.mean(X[ii,:,:])
        V = local_contrast_normalization(V)
        # log transform extreme values (???)
        V[V>5] = 5 + numpy.log(V[V>5])
        V[V<-5] = -5 - numpy.log(numpy.abs(V[V<-5]))
        # rescale (???)
        V = V / numpy.max(numpy.abs(V))
        #
        X_lcn[ii,:,:] = V
        del V

        # This one is just zero mean and scaling
        W = X[ii,:,:] - numpy.mean(X[ii,:,:])
        W = W / numpy.max(numpy.abs(W))
        X_raw[ii,:,:] = W
        del W

    # save results
    numpy.savez(outFileLCN, X=X_lcn)
    numpy.savez(outFileRaw, X=X_raw)


#-------------------------------------------------------------------------------
# visualize (optional)
#-------------------------------------------------------------------------------
if True:
    print '[em_preprocess]: saving images...'
    
    # also test the load while we're at it
    blah = numpy.load(outFileLCN+'.npz')
    Z = blah['X']
    blah = numpy.load(outFileRaw+'.npz')
    W = blah['X']

    for ii in range(X.shape[0]):
        Xi = X[ii,:,:]
        Zi = Z[ii,:,:]
        Wi = W[ii,:,:]
        
        pylab.subplot(2,3,1)
        pylab.imshow(Xi, interpolation='nearest', cmap='gist_earth')
        pylab.title('(mu=%0.2f, std=%0.2f, sk=%0.2f)' % (numpy.mean(Xi), numpy.std(Xi), stats.skew(col(Xi))))
        pylab.colorbar()
        
        pylab.subplot(2,3,4)
        pylab.boxplot(col(Xi))
        
        pylab.subplot(2,3,2)
        pylab.imshow(Zi, interpolation='nearest', cmap='gist_earth')
        pylab.title('(mu=%0.2f, std=%0.2f, sk=%0.2f)' % (numpy.mean(Zi), numpy.std(Zi), stats.skew(col(Zi))))
        pylab.colorbar()
        
        pylab.subplot(2,3,5)
        pylab.boxplot(col(Zi))
        
        pylab.subplot(2,3,3)
        pylab.imshow(Wi, interpolation='nearest', cmap='gist_earth')
        pylab.title('(mu=%0.2f, std=%0.2f, sk=%0.2f)' % (numpy.mean(Wi), numpy.std(Wi), stats.skew(col(Wi))))
        pylab.colorbar()
        
        pylab.subplot(2,3,6)
        pylab.boxplot(col(Wi))

        
        pylab.savefig(fileName+'-page%02d.eps' % ii)
        pylab.close()
        
