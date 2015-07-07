# Generic variables for use by all experiment-specific makefiles.
#
# May 2014, mjp


#-------------------------------------------------------------------------------
# Flags for running on different devices.
#
# Note: you can override the GPU variable at the command line to put
# the run on a specific device.  By default, runs go on GPU 0
#-------------------------------------------------------------------------------
CPUFLAGS = THEANO_FLAGS='floatX=float32,device=cpu,nvcc.fastmath=True' 

GPU = 0
GPUFLAGS = THEANO_FLAGS='floatX=float32,device=gpu$(GPU),nvcc.fastmath=True' 
