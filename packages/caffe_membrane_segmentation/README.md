Overview
-------

This package contains software for classifying electron microscopy (EM) data as either "membrane" or "non-membrane" at the pixel level.  The basic approach follows that of [Ciresan] and uses convolutional neural networks (CNNs) to solve the classification problem.
Note that this package is a newer implementation; our previous approach used Theano as the underlying CNN framework.  Switching to Caffe simplifies the code and makes it a bit easier to experiment with different network configurations.


**Author:** Mike Pekala (mike.pekala@jhuapl.edu)

Quick Start
-------
There are two main tasks: training a classifier and evaluating the classifier on new data (termed "deploying" the classifer).  The Makefile provides examples of both.  Note that, as of this writing, both training and deploying are computationally intensive (e.g. training on [ISBI2012] can take order of days and classifying a 30x512x512 cube takes about 1.5 hours using the brute-force approach of extracting and processing all tiles separately).  Accordingly, we usually run these tasks remotely on our GPU cluster.  You will need to make the necessary adjustments to the commands below for your system configuration.

### Step 0: Obtain ISBI 2012 data

    pushd caffe_files/isbi2012
    ./getdata.sh
    popd

This should create three .tif files within caffe_files/isbi2012

### Step 1: Copy data and code to GPU cluster

    make tar-all
    scp ../tocluster.tar gpucluster0:/home/pekalmj1


### Step 2: Training
Note: you may want to run nvidia-smi on the cluster first to make sure the GPU device(s) you wish to use are free.  Also, it's a good idea to put everything on /scratch as this is locally mounted (e.g. vs /home, which is nfs mounted).

    ssh gpucluster0
    cd /scratch/pekalmj1/isbi2012-demo
    tar xvf ~/tocluster.tar
    cd caffe_membrane_segmentation
    make train-isbi2012
    tail -f nohup.train.isbi2012

If all goes well, you should see a lot of debugging info being sent to the output file, then eventually CNN training status updates of the form:


    [train]: completed iteration 1 (of 200000; 0.20 min elapsed; 0.01 CNN min)
    [train]:    epoch: 1 (0.00), loss: 0.693, acc: 0.500, learn rate: 1.000e-03
    [train]: completed iteration 201 (of 200000; 2.37 min elapsed; 2.17 CNN min)
    [train]:    epoch: 1 (0.66), loss: 0.693, acc: 0.566, learn rate: 1.000e-03
    ...

Output (e.g. model snapshots) will appear in the specified output directory (defaults to "caffe_files/Caffe-N3-ISBI2012" for ISBI 2012)

Note: initially you may want to set up a smaller run (e.g. reduced number of slices/epochs) and let it run end-to-end to make sure there are no issues prior to the full up training run.  It is a bummer to have a training run complete and then crash due to a path-related issue with the output file...


### Step 3: Deploying
Assumes you have already copied the data and code to the cluster as per the previous step.

    cd /scratch/pekalmj1/isbi2012-demo/caffe_membrane_segmentation
    make deploy-isbi2012
    tail -f nohup.deploy.isbi2012


References
-----------
[Ciresan] Ciresan, D. et. al. "Deep Nerual networks segment neuronal membranes in electron microscopy images." NIPS 2012.

[ISBI2012] http://brainiac2.mit.edu/isbi_challenge/home
