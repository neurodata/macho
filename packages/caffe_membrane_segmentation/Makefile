# Demonstrates running the Caffe CNN code on various data sets of interest.
# 
# Assumes code will be running remotely (e.g. on our gpu cluster); hence
# the use of nohup below
#
# To run on CPU (vs GPU) remove the "-gpu X" from the command line (and make sure the
# default in the solver prototxt is CPU)
#
# February 2015, Mike Pekala


# Change this to wherever you have pycaffe installed
CAFFE_DIR=~/Apps/caffe/python

# The remaining macros should be system-independent
ISBI2012_DIR  = "Caffe-N3-ISBI2012"


#-------------------------------------------------------------------------------
# ISBI 2012
#-------------------------------------------------------------------------------
train-isbi2012 :
	PYTHONPATH=$(CAFFE_DIR) nohup python train.py  \
		-X isbi2012/train-volume.tif \
		-Y isbi2012/train-labels.tif  \
		-ub 150 \
		--train-slices "range(0,27)" --valid-slices "range(27,31)" \
		--snapshot-prefix $(ISBI2012_DIR) \
		-s caffe_files/n3-solver.prototxt  > nohup.train.isbi2012 -gpu 0 &

deploy-isbi2012 :
	PYTHONPATH=$(CAFFE_DIR) nohup python deploy.py \
		-X isbi2012/test-volume.tif \
		-ub 150 \
		-s caffe_files/n3-solver.prototxt  \
		-m $(ISBI2012_DIR)/iter_20000.caffemodel \
		--output-file isbi2012/Yhat_test \
		-gpu 1 > nohup.deploy.isbi2012 &


#-------------------------------------------------------------------------------
# Misc targets
#
# Use "make tar" to create a code-only tarball or "tar-all" to include data sets.
#-------------------------------------------------------------------------------
tar :
	pushd .. && tar cvf ./tocluster.tar `find ./caffe_membrane_segmentation -name \*.py -print`
	pushd .. && tar rvf ./tocluster.tar `find ./caffe_membrane_segmentation -name \*.prototxt -print`
	pushd .. && tar rvf ./tocluster.tar `find ./caffe_membrane_segmentation -name Makefile -print`

tar-all : tar
	pushd .. && tar rvf ./tocluster.tar `find ./caffe_membrane_segmentation -name \*.tif -print`
