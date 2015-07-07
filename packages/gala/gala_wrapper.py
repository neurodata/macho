#!/usr/bin/python
import sys
print sys.version_info
## Wrapper For rhoana workflow##

## This wrapper exists to facilitate workflow level parallelization inside the LONI pipeline until
## it is properly added to the tool.  It is important for this step to do workflow level parallelization
## because of the order of processing.
##
## Make sure that you specify the environment variable MATLAB_EXE_LOCATION inside the LONI module.  This can be
## set under advanced options on the 'Execution' tab in the module set up.

# (c) [2014] The Johns Hopkins University / Applied Physics Laboratory All Rights Reserved. Contact the JHU/APL Office of Technology Transfer for any additional rights.  www.jhuapl.edu/ott
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

from sys import argv
from sys import exit
import sys
import re
import os
from subprocess import Popen,PIPE

# read in command line args
params = list(argv)
del params[0]


emToken = params[0:2]
emServiceLocation = params[2:4]
membraneToken = params[4:6]
membraneServiceLocation = params[6:8]
annoToken = params[8:10]
annoServiceLocation = params[10:12]
queryFile = params[12:14]
author = params[14:16]
thresh = params[16:18]
dilateXY = params[18:20]
dilateZ = params[20:22]
algo = params[22:24]
algoFile = params[24:26]
padX = params[26:28]
padY = params[28:30]
wsThresh = params[30:32]
useSemaphore = params[32:34]
emCube = params[34:36]
emMat = params[36:38]
membraneMat = params[38:40]
wsMat = params[40:42]
tokenMat = params[42:44]
annoMat = params[44:46]
labelMat = params[46:48]

# get root directory of framework
frameworkRootCAJAL3D = os.getenv("CAJAL3D_LOCATION")
if frameworkRootCAJAL3D is None:
    raise Exception('You must set the CAJAL3D_LOCATION environment variable so the wrapper knows where the framework is!')


frameworkRootI2G = os.getenv("I2G_LOCATION")
if frameworkRootI2G is None:
    raise Exception('You must set the I2G_LOCATION environment variable so the wrapper knows where the framework is!')

# Gen path of matlab wrapper
wrapper = os.path.join(frameworkRootI2G, 'packages', 'utilities','basicWrapperI2G.py')

# Build call to Rhoana Data Pull
args = [wrapper] + [os.path.join(frameworkRootI2G, 'packages', 'gala', 'gala_get_data.m')] + emToken + emServiceLocation + membraneToken + membraneServiceLocation + queryFile + emCube + emMat + membraneMat + wsMat + dilateXY + dilateZ + wsThresh + useSemaphore
print args
# Call Cube Cutout
process = Popen(args, stdout=PIPE, stderr=PIPE)
output = process.communicate()
proc_error = output[1]
proc_output = output[0]
exit_code = process.wait()

# Write std out stream
print "#######################\n"
print "Output From  Rhoana Data Pull\n"
print "#######################\n\n\n"
print proc_output

# If exit code != 0 exit
if exit_code != 0:
    # it bombed.  Write out matlab errors and return error code
    sys.stderr.write("Error from Rhoana Data Pull:\n\n")
    sys.stderr.write(proc_error)
    exit(exit_code)

print 'calling gala'
# Build call to Gala
gala = os.path.join(frameworkRootI2G, 'packages', 'gala','galaRun.py')
args = ['/usr/bin/python2.7'] + [gala] + [emMat[1]] + [membraneMat[1]] + [annoMat[1]] + [thresh[1]] + [algo[1]] + [algoFile[1]] + [wsMat[1]]

print args

print "########################################\n"
print "Output From Rhoana\n"
print "########################################\n\n\n"

# Call Rhoana Segmentation Algorithm
process = Popen(args)
output = process.communicate()
exit_code2 = process.wait()

if exit_code2 != 0:
    # it bombed.  Write out matlab errors and return error code
    exit(exit_code2)


 # Build call to Gala Result Push
args = [wrapper] + [os.path.join(frameworkRootI2G, 'packages', 'gala', 'gala_put_anno.m')] + emToken + annoToken + annoServiceLocation + annoMat + emCube + author + queryFile + padX + padY + useSemaphore + labelMat + tokenMat + ["-b", "0"]
print args
# Call Cube Cutout
process = Popen(args, stdout=PIPE, stderr=PIPE)
output = process.communicate()
proc_error = output[1]
proc_output = output[0]
exit_code3 = process.wait()

# Write std out stream
print "#######################\n"
print "Output From  Rhoana Data Pull\n"
print "#######################\n\n\n"
print proc_output

# If exit code != 0 exit
if exit_code3 != 0:
    # it bombed.  Write out matlab errors and return error code
    sys.stderr.write("Error from Rhoana Data Pull:\n\n")
    sys.stderr.write(proc_error)
    exit(exit_code3
    	)