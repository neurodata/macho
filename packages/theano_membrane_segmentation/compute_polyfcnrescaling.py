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
VIEW_RESULTS  Post-processing analysis of membrane segmentation results.

example usage: 
  python compute_polyfcnrescaling.py evaltraindata_All/test-results.npz ../../Data/Kasthuri/ac4cc_membrane_truth_OM_03042014_fliprescale.tif 
  python compute_polyfcnrescaling.py evaltraindata_USFirst/test-results.npz ../../Data/Kasthuri/ac4cc_membrane_truth_OM_03042014_fliprescale.tif 

saves output to local file: coeffs.txt (last line has the coefficients for all training slices aggregated).

References:
  [1] http://brainiac2.mit.edu/isbi_challenge/
  [2] Ciresan et. al. "Deep Neural Networks Segment Neuronal 
        Membranes in Electron Microscopy Images"
"""
# November 2013


import sys, os.path

import copy
import numpy
import pylab
import pdb
import em_utils

from scipy.signal import convolve2d
import scipy.linalg


# Process command line arguments
inFile = sys.argv[1]        # the classification output results file.


# controls whether to make plots on the screen, or just to file.
#if len(sys.argv) > 2:
#    TO_FILE = int(sys.argv[2])
#    fargs = {'bbox_inches' : 'tight', 'pad_inches' : 0}
#else:
#    TO_FILE = 0
TO_FILE = 0

fargs = {'bbox_inches' : 'tight', 'pad_inches' : 0}

make_plots_flag = 0 # actually, suppress the plotting

# TODO: we might need to make the tile size an input argument as well 
# (or save it in the results file).

npf = numpy.load(inFile)
Xarray = npf['X']
#Y = npf['Y']
Y_hatarray = npf['Y_hat']

Yarray = em_utils.load_tiff_data(sys.argv[2])

intervals_to_use = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
midpoints_list = [0.5*intervals_to_use[x] + 0.5*(intervals_to_use[x+1]) for x in range(len(intervals_to_use)-1)]
All_counts_actual   = numpy.zeros((len(Yarray[:,1,1]),len(midpoints_list)));
All_counts_observed = numpy.zeros((len(Yarray[:,1,1]),len(midpoints_list)));
#counts_list_actual = [0 for x in range(len(intervals_to_use)-1)]
#counts_list_observed = [0 for x in range(len(intervals_to_use)-1)]

plot_each = 0

for stacknum in range(len(Yarray[:,1,1])):
    Y = Yarray[stacknum,:,:]
    Y_hat = Y_hatarray[stacknum,:,:]
    X = Xarray[stacknum,:,:]

    print "X is size", X.shape

    #for thresh in [.5, .55, .6, .65, .7, .75, .8]:
    for thresh in [.5]:
        Y_tmp = numpy.zeros(Y_hat.shape);
        Y_tmp[numpy.logical_and(Y_hat>thresh, Y==1)] = 1;
        Y_tmp[numpy.logical_and(Y_hat>thresh, Y==0)] = .9;
        #Y_tmp[numpy.logical_and(Y_hat<thresh, Y==1)] = .7;
        Y_tmp[numpy.logical_and(Y_hat<thresh, Y==1)] = .4;

    show_plots = 0

    list_thresh =  [.2, .3, .4, .45, .5, .52, .55, .6, .7, .8, .9]
    #list_thresh =  [.1, .15, .2, .25, .3, .4, .45, .5, .52, .55, .6]
    list_Fscores_nomedfilt = [0 for x in range(len(list_thresh))]
    list_Pd_nomedfilt = [0 for x in range(len(list_thresh))]
    list_Pfa_nomedfilt = [0 for x in range(len(list_thresh))]
    list_Fscores_medfilt = [0 for x in range(len(list_thresh))]
    list_Pd_medfilt = [0 for x in range(len(list_thresh))]
    list_Pfa_medfilt = [0 for x in range(len(list_thresh))]

    #Y_subset = Y[32:-32, 32:-32]
    #Y_hat_subset = Y_hat[32:-32, 32:-32]
    # we are no longer needing to take subset of the Y and Yhat. (saves only the original image size)
    Y_subset = Y
    Y_hat_subset = Y_hat
    Y_hat_subset_corrected = copy.deepcopy(Y_hat_subset)
    Y_hat_subset_filtered = copy.deepcopy(Y_hat_subset)

    if 1: # Before median filtering use a rectifier to correct the p_membrane outputs
        # This is done by fitting a cubic monotonic function (Section 2.3 of Ciresan et al)
        #intervals_to_use = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
        #midpoints_list = [0.5*intervals_to_use[x] + 0.5*(intervals_to_use[x+1]) for x in range(len(intervals_to_use)-1)]
        probcounts_list = [0 for x in range(len(intervals_to_use)-1)]
        counts_list_actual = [0 for x in range(len(intervals_to_use)-1)]
        counts_list_observed = [0 for x in range(len(intervals_to_use)-1)]
        midpoints = numpy.asarray(midpoints_list)
        #probcounts = numpy.asarray(probcounts_list)
        for x in range(len(midpoints)):
            # Count number of Y_hats in each interval
            loginds = numpy.logical_and(Y_hat_subset <= intervals_to_use[x+1], Y_hat_subset > intervals_to_use[x])
            detectedcount = numpy.sum(loginds)
            cury = Y_subset[loginds]
            actualmembcount = numpy.sum(cury)
            if 0:
                pylab.imshow(loginds,cmap='bone',interpolation='nearest')
                pylab.title(['y values of loginds, midpoint = ',midpoints[x]])
                pylab.show()
            if detectedcount > 0:
                probcounts_list[x] = actualmembcount/detectedcount
            else:
                probcounts_list[x] = 1
            counts_list_actual[x]   = actualmembcount
            counts_list_observed[x] = detectedcount
            All_counts_actual[stacknum,x]   = actualmembcount
            All_counts_observed[stacknum,x] = detectedcount

        probcounts = numpy.asarray(probcounts_list)
        print "midpts = ", midpoints
        print "probcts = ", probcounts
    
        midpointstrans = midpoints.transpose()
        Pmatrix_list = [midpointstrans**3, midpointstrans**2, midpointstrans]
        #Pmatrix = numpy.concatenate(midpoints.transpose**2,Pmatrix)
        #Pmatrix = numpy.concatenate(midpoints.transpose**3,Pmatrix)
        Pmatrix_trans = numpy.asarray(Pmatrix_list)
        Pmatrix = Pmatrix_trans.transpose()
        print "Pmatrix is ", len(Pmatrix[:,1])," by ", len(Pmatrix[1,:])
        probcountstrans = probcounts.transpose()
        print "b is ", probcountstrans.shape
    
        linalgoutput= scipy.linalg.lstsq(Pmatrix,probcountstrans)
        coeffs = linalgoutput[0]
        #coeffs = numpy.polyfit(midpoints,probcounts,3)
        print "coeffs = ", coeffs
        print "coeffs is size", coeffs.shape
    
        # Write to a coeffs file
        if 1:
            f = open('coeffs.txt', 'a')

        strlt = ""
        strlt += str(coeffs[0])
        for i in range(len(coeffs)-1):
            strlt += ", "
            strlt += str(coeffs[i+1])
        print "Coeffs =  [%s]" % strlt
        s = "Coeffs =  [%s] " % strlt
        f.write(s+"\n")

        result = numpy.dot(Pmatrix,coeffs)
        #result_list = [coeffs[0]*midpoints[x]**3 for x in range(len(midpoints))]
    
        if plot_each:
            pylab.figure
            pylab.plot(midpoints,probcounts)
            pylab.plot(midpoints, probcounts, 'bs', midpoints, result, 'g^')
            pylab.title('actual prob of membrane versus report probmembrane')
            pylab.savefig('Y_hat_probmemb_correctionfit.eps')
            pylab.show()
        array_rows = len(Y_hat_subset[:,1])
        array_cols = len(Y_hat_subset[1,:])
        for x in range(array_cols):
            for p in range(array_rows):
                curp = Y_hat_subset[x,p]
                newval = numpy.power(curp,3)*coeffs[0] + numpy.power(curp,2)*coeffs[1] + curp*coeffs[2]
                Y_hat_subset_corrected[x,p] = newval
        if 0:
            pylab.imshow(Y_hat_subset,cmap='bone',interpolation='nearest')
            pylab.title('Y_hat_subset')
            pylab.colorbar()
            pylab.savefig('Y_hat_subset.eps')
            pylab.show()
            
            pylab.imshow(Y_hat_subset_corrected,cmap='bone',interpolation='nearest')
            pylab.title('Y_hat_subset_corrected')
            pylab.colorbar()
            pylab.savefig('Y_hat_subset_corrected.eps')
            pylab.show()

# Now compute the coefficients using all the data in the validation tiff stack.
for x in range(len(midpoints)):
    cursum_actual = 0
    cursum_observed = 0
    for t in range(len(Yarray[:,1,1])):
        cursum_actual = cursum_actual + All_counts_actual[t,x]
        cursum_observed = cursum_observed + All_counts_observed[t,x]
    #probcounts_list[x] = cursum_actual/cursum_observed
    if cursum_observed > 0:
        #probcounts_list[x] = actualmembcount/detectedcount
        probcounts_list[x] = cursum_actual/cursum_observed
    else:
        probcounts_list[x] = 1


probcounts = numpy.asarray(probcounts_list)
print "midpts = ", midpoints
print "probcts = ", probcounts

midpointstrans = midpoints.transpose()
Pmatrix_list = [midpointstrans**3, midpointstrans**2, midpointstrans]
Pmatrix_trans = numpy.asarray(Pmatrix_list)
Pmatrix = Pmatrix_trans.transpose()
#print "Pmatrix is ", len(Pmatrix[:,1])," by ", len(Pmatrix[1,:])
probcountstrans = probcounts.transpose()
#print "b is ", probcountstrans.shape
    
linalgoutput= scipy.linalg.lstsq(Pmatrix,probcountstrans)
coeffs = linalgoutput[0]

print "Total coeffs = ", coeffs
print "coeffs is size", coeffs.shape
    
# Write to a coeffs file
if 1:
    f = open('coeffs.txt', 'a')
    
strlt = ""
strlt += str(coeffs[0])
for i in range(len(coeffs)-1):
    strlt += ", "
    strlt += str(coeffs[i+1])
print "Total Coeffs =  [%s]" % strlt
s = "Total Coeffs =  [%s] " % strlt
f.write(s+"\n")

result = numpy.dot(Pmatrix,coeffs)
    
if 1:
    pylab.figure
    pylab.plot(midpoints,probcounts)
    pylab.plot(midpoints, probcounts, 'bs', midpoints, result, 'g^')
    pylab.title('Total stack: actual prob of membrane versus report probmembrane')
    pylab.savefig('Total_Y_hat_probmemb_correctionfit.eps')
    #pylab.savefig('Y_hat_subset_corrected.eps')
    pylab.show()
