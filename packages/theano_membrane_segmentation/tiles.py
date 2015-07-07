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

""" Code for loading image tiles (minibatches) onto the GPU.

December 2013, mjp
"""


import sys, os 
import time

import numpy
import math

import theano
import theano.tensor as T

from em_utils import *



class TileManager:
    """
    This object takes care of extracting subtensors (tiles) from an 
    image tensor and placing the extracted tiles into GPU memory via 
    Theano shared variables.

    A fundamental assumption here is that the class label of a tile is
    the class label of the pixel at the center of the tile.

    This object will place batchSize tiles into GPU memory each time it 
    is asked to update.  Thus, batchSize should be a multiple of the 
    mini-batch size (and we call the data loaded here a "batch" to
    indicate it is one or more mini-batches worth of data).

    I tried two strategies: 
      (1) tiling data locally on the CPU, then copying the GPU
      (2) tiling data remotely on the GPU
      
    I had thought (2) would be substantially faster than (1) since it avoids
    expensive memory copies.  However, it appears the memory copies aren't 
    that bad compared to the GPU processing time and since slicing is no faster
    on a GPU, method (1) is simpler and currently the default.
    """

    def __init__(self, X, Y, tileSize=65, batchSize=5000, tileOnGPU=False, nZeeChannels=0):
        """
        X         := a (#slices, #rows, #cols) tensor of membrane data
        Y         := a (#slices, #rows, #cols) tensor of class labels
        tileSize  := n, where tiles are of size (n x n)
        batchSize := the number of tiles to load each batch 
                     (should be a multiple of the mini-batch size)
        tileOnGPU := controls where tiles are created (CPU vs GPU)
        nZeeChannels := TBD
        """
        if X.ndim != 3:
            raise RuntimeError('unexpected input dimensions.  input should be a grayscale cube')
        if numpy.mod(tileSize,2) != 1:
            raise RuntimeError('tile size must be odd')
        if batchSize <= 0:
            raise RuntimeError('# tiles must be positive')
        if tileOnGPU:
            raise NotImplementedError("you don't want to use this feature right now...")
        if nZeeChannels < 0:
            raise RuntimeError('# tiles channels must be non-negative')

        self.batchSize=batchSize
        self.tileSize=tileSize
        self.tileOnGPU=tileOnGPU
        self.nZeeChannels=nZeeChannels
        
        self.ts2 = int(numpy.floor(self.tileSize/2.0))

        # put the "raw" (untiled) data onto the GPU via 
        # Theano shared variables.  Data must be stored as float32 
        # on the GPU (controlled by "floatX").
        if tileOnGPU:
            self.X = theano.shared(numpy.asarray(X,dtype=theano.config.floatX), 
                                   borrow=True)
            self.Y = theano.shared(numpy.asarray(Y,dtype=theano.config.floatX), 
                                   borrow=True)

        # Keep a local (resident on CPU) copy of the data set too.
        # This will make it easier to control the tiling process.
        # Also, it gives us the option of slicing on the GPU or the CPU.
        #
        # UPDATE: make a copy to avoid any possible side-effects.
        self.Y_local = Y.copy();  self.Y_local.flags.writeable = False;
        self.X_local = X.copy();  self.X_local.flags.writeable = False;

        # X/y_batch := subset of tiles that are active on the GPU
        #
        # Keep a local copy of the batch and also a GPU-side copy (via
        # Theano).  If it were very slow to sample the images to
        # create the batches, could possibly have one thread filling
        # up the *local copy for the next batch while the GPU works on
        # the current batch...
        #
        # UPDATE: make the X batches 4-d tensors.  Before, the input
        # to layer0 would take care of this in Theano. However, it's
        # easier to debug if we do this step here instead.
        nChannels = 1 + 2*self.nZeeChannels
        self.X_batch_local = numpy.zeros((batchSize, nChannels, self.tileSize, self.tileSize), dtype=theano.config.floatX)
        self.y_batch_local = numpy.zeros((batchSize,), dtype=theano.config.floatX)
        self.X_batch_GPU = theano.shared(self.X_batch_local, borrow=True)
        self.y_batch_GPU = theano.shared(self.y_batch_local, borrow=True)

        # During computations labels must be integers (they 
        # are used as indices); therefore cast to int.
        self.y_batch_int = T.cast(self.y_batch_GPU, 'int32')

        # create Theano functions for updating the mini-batch
        if tileOnGPU:
            self.__make_theano_functions()


    def __make_theano_functions(self):
        """
        Makes functions for tiling data remotely on the GPU.
        Called once, during object initialization.
        """

        # The next three symbolic variables are used to index into the 
        # raw image for the purpose of extracting subtensors (tiles).
        tSlices = theano.tensor.vector('slices',dtype='int32')
        tRows = theano.tensor.vector('rows',dtype='int32')
        tColumns = theano.tensor.vector('columns',dtype='int32')

        #--------------------------------------------------
        # create a function to grab new tiles from the image.
        #--------------------------------------------------
        get_tile = lambda s,r,c,X : X[s, (r-self.ts2):(r+self.ts2+1), (c-self.ts2):(c+self.ts2+1)]
        tiles, updates = theano.scan(fn=get_tile,
                                     outputs_info=None,
                                     sequences=[tSlices,tRows,tColumns],
                                     non_sequences=self.X)
        # The "tiles" return value will be a 3 dimensional tensor.
        # It seems the default behavor of scan is to store the iterates
        # into a tensor, which is exactly what we want (no need to do
        # an additional aggregation step).
    
        # Now, we need to construct a Theano function that uses the 
        # scan above to repopulate the batch data. We don't want
        # this function to generate an explicit output; instead, we
        # want the side-effect that the extracted tiles are used
        # to update the batch variable.
        update = (self.X_batch_GPU, tiles)
        self.__retile = theano.function(inputs=[tSlices,tRows,tColumns],
                                        updates=[update],
                                        outputs=[])

        #--------------------------------------------------
        # we also need a similar function for updating the ground truth.
        #--------------------------------------------------
        get_center_pixel = lambda s,r,c,X : X[s,r,c]
        labels, updates = theano.scan(fn=get_center_pixel,
                                     outputs_info=None,
                                     sequences=[tSlices,tRows,tColumns],
                                     non_sequences=self.Y)

        update = (self.y_batch_GPU, labels)
        self.__relabel = theano.function(inputs=[tSlices,tRows,tColumns],
                                         updates=[update],
                                         outputs=[])


    def update_gpu(self, slices, rows, columns, flipDim=0, rotDir=0):
        """
        Copies tiles from the image into the local buffer, then onto the GPU.

        Note: The rows and columns parameters specify the *center* of
              the tile to extract from the original image.
        """
        # if slices is of size (x,1), this turns it into something of shape (x,).
        # this is just precautionary (otherwise, tile sizes can get screwy)
        slices = numpy.squeeze(slices)
        
        if self.tileOnGPU:
            # Use Theano (hopefully all GPU-side)
            self.__retile(slices,rows,columns)
            self.__relabel(slices,rows,columns)

        else:
            # Chop data on CPU and copy to GPU.

            # XXX: could put everything before the host->device
            #      copy on a separate thread...maybe also "double buffer"
            #      training data set??
            for ii in range(len(slices)):
                # if generating "z-channels", change the slices accordingly.
                if self.nZeeChannels > 0:
                    slicesii = z_slices(slices[ii], self.nZeeChannels, [0, self.X_local.shape[0]])
                else:
                    slicesii = slices[ii]
                    
                # Get tile(s).  Adjust for fact that the caller is
                # providing the center pixel of the tile.
                tile = self.X_local[slicesii, (rows[ii]-self.ts2):(rows[ii]+self.ts2+1), (columns[ii]-self.ts2):(columns[ii]+self.ts2+1)]

                # apply transform, if any
                if (flipDim !=0) or (rotDir !=0):
                    tile = xform(tile, flipDim, rotDir)

                # copy over into buffer
                self.X_batch_local[ii,:,:,:] = tile
                self.y_batch_local[ii] = self.Y_local[slices[ii], rows[ii], columns[ii]]

            # Note: a consequence of using X_batch_local and
            # y_batch_local as a staging area is that if len(slices)
            # is less than the number of tiles in a batch, the end of
            # the batch will contain left-over tiles from last batch.
            # This is deliberate - it is better to have some viable
            # data in the buffer then some random stuff.

            # CPU -> GPU
            self.X_batch_GPU.set_value(self.X_batch_local)
            self.y_batch_GPU.set_value(self.y_batch_local)


    def make_all_pixel_generator(self):
        "helper method for creating a pixel generator (e.g. for testing)"
        return interior_pixel_generator(self.Y_local, self.ts2, self.batchSize)
    
    def make_bandpass_pixel_generator(self, lbX, ubX):
        "helper method for creating a pixel generator (e.g. for testing)"
        return bandpass_interior_pixel_generator(self.X_local, lbX, ubX, self.ts2, self.batchSize)

    def make_selected_pixel_generator(self, selectedPixels):
        "helper method for creating a pixel generator (e.g. for testing)"
        return selected_interior_pixel_generator(self.X_local, selectedPixels, self.ts2, self.batchSize)

    def make_balanced_pixel_generator(self):
        "helper method for creating a balanced pixel generator (e.g. for training)"
        return balanced_interior_pixel_generator(self.Y_local, self.ts2, self.batchSize, stratified=False)

    def make_difficult_pixel_generator(self):
        pct = 15
        print "[%s]: WARNING: using experimental pixel generator; pct=%d" % (__name__, pct)
        return difficult_interior_pixel_generator(self.Y_local, self.X_local, self.ts2, self.batchSize, pct=pct)
