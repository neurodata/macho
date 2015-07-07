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
Convolutional neural networks for solving classification problem [1] with Theano.

References:
  [1] http://brainiac2.mit.edu/isbi_challenge/
"""

import numpy

import theano
import theano.tensor as T

from layers import LogisticRegression, LeNetConvPoolLayer, HiddenLayer
import tiles




class LeNetMembraneN3:
    """
    Implements a guess as to the N3 network from [1].
    (we don't actually know the number of filters or filter parameters)

    References:
    [1] Ciresan et. al. "Deep Neural Networks Segment Neuronal 
        Membranes in Electron Microscopy Images," NIPS 2012.
    """
    def __init__(self, nkerns=[48,48,48], miniBatchSize=200, nHidden=200, 
                 nClasses=2, nMaxPool=2, nChannels=1):
        """
        nClasses : the number of target classes (e.g. 2 for binary classification)
        nMaxPool : number of pixels to max pool
        nChannels : number of input channels (e.g. 1 for single grayscale channel)
        """
        rng = numpy.random.RandomState(23455)

        self.p = 65
        self.miniBatchSize = miniBatchSize

        # Note: self.x and self.y will be re-bound to a subset of the
        # training/validation/test data dynamically by the update
        # stage of the appropriate function.
        self.x = T.tensor4('x')     # membrane mini-batch
        self.y = T.ivector('y')     # 1D vector of [int] labels

        # We now assume the input will already be reshaped to the
        # proper size (i.e. we don't need a theano resize op here).
        layer0_input = self.x

        #--------------------------------------------------
        # LAYER 0
        # layer0 convolution+max pool reduces image dimensions by:
        # 65 -> 62 -> 31
        #--------------------------------------------------
        fs0 = 4                          # conv. filter size, layer 0
        os0 = (self.p-fs0+1)/nMaxPool    # image out size 0
        assert(os0 == 31)
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                                    image_shape=(self.miniBatchSize, nChannels, self.p, self.p),
                                    filter_shape=(nkerns[0], nChannels, fs0, fs0), 
                                    poolsize=(nMaxPool, nMaxPool))

        #--------------------------------------------------
        # LAYER 1
        # layer1 convolution+max pool reduces image dimensions by:
        # 31 -> 28 -> 14
        #--------------------------------------------------
        fs1 = 4                     # filter size, layer 1
        os1 = (os0-fs1+1)/nMaxPool  # image out size 1
        assert(os1 == 14)
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                                    image_shape=(self.miniBatchSize, nkerns[0], os0, os0),
                                    filter_shape=(nkerns[1],nkerns[0],fs1,fs1), 
                                    poolsize=(nMaxPool, nMaxPool))

        #--------------------------------------------------
        # LAYER 2
        # layer2 convolution+max pool reduces image dimensions by:
        # 14 -> 10 -> 5
        #--------------------------------------------------
        fs2 = 5
        os2 = (os1-fs2+1)/nMaxPool
        assert(os2 == 5)
        layer2 = LeNetConvPoolLayer(rng, input=layer1.output,
                                    image_shape=(self.miniBatchSize, nkerns[1], os1, os1),
                                    filter_shape=(nkerns[2],nkerns[1],fs2,fs2), 
                                    poolsize=(nMaxPool, nMaxPool))

        #--------------------------------------------------
        # LAYER 3
        # Fully connected sigmoidal layer, goes from
        # 5*5*48  -> 200
        #--------------------------------------------------
        layer3_input = layer2.output.flatten(2)
        layer3 = HiddenLayer(rng, input=layer3_input, 
                             n_in=nkerns[2] * os2 * os2,
                             n_out=nHidden, activation=T.tanh)

        #--------------------------------------------------
        # LAYER 4
        # Classification via a logistic regression layer
        # 200 -> 2
        #--------------------------------------------------
        # classify the values of the fully-connected sigmoidal layer
        layer4 = LogisticRegression(input=layer3.output, 
                                    n_in=nHidden, n_out=nClasses)

        self.layers = (layer0, layer1, layer2, layer3, layer4)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LeNetMembraneN4:
    """
    Implements the N4 network from [1].
    Note that we do not implement all of the preprocessing 
    (e.g. foveation, nonlinear sampling) nor to we generate
    rotated and mirrored versions of the training data.  Thus,
    we do not expect to exactly re-create the reported best
    performance on this data set.

    References:
    [1] Ciresan et. al. "Deep Neural Networks Segment Neuronal 
        Membranes in Electron Microscopy Images," NIPS 2012.
    """
    def __init__(self, nkerns=[48,48,48,48], miniBatchSize=200):
        rng = numpy.random.RandomState(23455)
        nClasses = 2
        nMaxPool = 2
        nHidden = 200

        self.p = 95
        #self.x = T.tensor3('x')     # membrane data set
        self.x = T.tensor4('x')     # membrane mini-batch
        self.y = T.ivector('y')     # labels := 1D vector of [int] labels
        self.miniBatchSize = miniBatchSize

        # Reshape matrix of rasterized images # to a 4D tensor, 
        # compatible with the LeNetConvPoolLayer
        #layer0_input = self.x.reshape((self.miniBatchSize, 1, self.p, self.p))
        layer0_input = self.x

        #--------------------------------------------------
        # LAYER 0
        # layer0 convolution+max pool reduces image dimensions by:
        # 95 -> 92 -> 46
        #--------------------------------------------------
        fs0 = 4                     # filter size, layer 0
        os0 = (self.p-fs0+1)/nMaxPool    # image out size 0
        assert(os0 == 46)
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                                    image_shape=(self.miniBatchSize, 1, self.p, self.p),
                                    filter_shape=(nkerns[0], 1, fs0, fs0), 
                                    poolsize=(nMaxPool, nMaxPool))

        #--------------------------------------------------
        # LAYER 1
        # layer1 convolution+max pool reduces image dimensions by:
        # 46 -> 42 -> 21
        #--------------------------------------------------
        fs1 = 5                     # filter size, layer 1
        os1 = (os0-fs1+1)/nMaxPool  # image out size 1
        assert(os1 == 21)
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                                    image_shape=(self.miniBatchSize, nkerns[0], os0, os0),
                                    filter_shape=(nkerns[1],nkerns[0],fs1,fs1), 
                                    poolsize=(nMaxPool, nMaxPool))

        #--------------------------------------------------
        # LAYER 2
        # layer2 convolution+max pool reduces image dimensions by:
        # 21 -> 18 -> 9
        #--------------------------------------------------
        fs2 = 4
        os2 = (os1-fs2+1)/nMaxPool
        assert(os2 == 9)
        layer2 = LeNetConvPoolLayer(rng, input=layer1.output,
                                    image_shape=(self.miniBatchSize, nkerns[0], os1, os1),
                                    filter_shape=(nkerns[2],nkerns[1],fs2,fs2), 
                                    poolsize=(nMaxPool, nMaxPool))

        #--------------------------------------------------
        # LAYER 3
        # layer3 convolution+max pool reduces image dimensions by:
        # 9 -> 6 -> 3
        #--------------------------------------------------
        fs3 = 4
        os3 = (os2-fs3+1)/nMaxPool
        assert(os3 == 3)
        layer3 = LeNetConvPoolLayer(rng, input=layer2.output,
                                    image_shape=(self.miniBatchSize, nkerns[0], os2, os2),
                                    filter_shape=(nkerns[3],nkerns[2],fs3,fs3), 
                                    poolsize=(nMaxPool, nMaxPool))

        #--------------------------------------------------
        # LAYER 4
        # Fully connected sigmoidal layer, goes from
        # 3*3*48 ~ 450 -> 200
        #--------------------------------------------------
        layer4_input = layer3.output.flatten(2)
        layer4 = HiddenLayer(rng, input=layer4_input, 
                             n_in=nkerns[3] * os3 * os3,
                             n_out=nHidden, activation=T.tanh)

        #--------------------------------------------------
        # LAYER 5
        # Classification via a logistic regression layer
        # 200 -> 2
        #--------------------------------------------------
        # classify the values of the fully-connected sigmoidal layer
        layer5 = LogisticRegression(input=layer4.output, 
                                    n_in=nHidden, n_out=nClasses)

        self.layers = (layer0, layer1, layer2, layer3, layer4, layer5)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        

class LeNetMembraneN1:
    """
    Implements a guess as to the  N1 network from [1].
    
    This is mainly for faster unit testing, since we don't currently
    implement foviation or nonuniform sampling.

    References:
    [1] Ciresan et. al. "Deep Neural Networks Segment Neuronal 
        Membranes in Electron Microscopy Images," NIPS 2012.
    """
    def __init__(self, nkerns=[48], miniBatchSize=200):
        rng = numpy.random.RandomState(23455)
        nClasses = 2
        nMaxPool = 2
        nHidden = 200

        self.p = 65
        #self.x = T.tensor3('x')     # membrane data set
        self.x = T.tensor4('x')     # membrane mini-batch
        self.y = T.ivector('y')     # 1D vector of [int] labels
        self.miniBatchSize = miniBatchSize

        # Reshape matrix of rasterized images # to a 4D tensor, 
        # compatible with the LeNetConvPoolLayer
        #layer0_input = self.x.reshape((self.miniBatchSize, 1, self.p, self.p))
        layer0_input = self.x

        #--------------------------------------------------
        # LAYER 0
        # layer0 convolution+max pool reduces image dimensions by:
        # 65 -> 62 -> 31
        #--------------------------------------------------
        fs0 = 4                          # filter size, layer 0
        os0 = (self.p-fs0+1)/nMaxPool    # image out size 0
        assert(os0 == 31)
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                                    image_shape=(self.miniBatchSize, 1, self.p, self.p),
                                    filter_shape=(nkerns[0], 1, fs0, fs0), 
                                    poolsize=(nMaxPool, nMaxPool))

        #--------------------------------------------------
        # LAYER 1
        # Fully connected sigmoidal layer, goes from
        # X  -> 200
        #--------------------------------------------------
        layer1_input = layer0.output.flatten(2)
        layer1 = HiddenLayer(rng, input=layer1_input, 
                             n_in=nkerns[0] * os0 * os0,
                             n_out=nHidden, activation=T.tanh)

        #--------------------------------------------------
        # LAYER 2
        # Classification via a logistic regression layer
        # 200 -> 2
        #--------------------------------------------------
        # classify the values of the fully-connected sigmoidal layer
        layer2 = LogisticRegression(input=layer1.output, 
                                    n_in=nHidden, n_out=nClasses)

        self.layers = (layer0, layer1, layer2)

        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_network_parameters(nn, fileName, verbose=False):
    """
    Returns (and optionally saves) the weights and biases from all
    layers in the given network.

    Assumes all parameters are numpy objects.
    """

    allParameters = {}
    nParameters = 0

    for ii in range(len(nn.layers)):

        layer = nn.layers[ii]
        for jj in range(len(layer.params)):
            key = 'layer_%d_param_%d' % (ii,jj)
            value = layer.params[jj].get_value()
            allParameters[key] = value

            nParameters += numpy.prod(value.shape)

    if not fileName is None:
        numpy.savez(fileName, **allParameters)
        if verbose:
            print '[save_network_parameters]: Saved %d network parameters to file "%s" ' % (nParameters, fileName)

    return allParameters

 

def load_network_parameters(nn, fileName):
    """
    Loads the weights and biases from all layers in the given network.

    WARNING: mutates the nn object!
    """

    params = numpy.load(fileName)

    for ii in range(len(nn.layers)):
        layer = nn.layers[ii]
        for jj in range(len(layer.params)):
            key = 'layer_%d_param_%d' % (ii,jj)
            value = params[key]
            layer.params[jj].set_value(value)
