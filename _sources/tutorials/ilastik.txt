ilastik
***********

ilastik is a tool developed to do machine annotation and is widely used by the connectomics community.  Here, we developed a lightweight protocol and data exchange format to allow users to use ilastik and OCP together.  This provides an easy method for users to quickly prototype, build, and deploy machine learning solutions to neuroscience problems.  The figure below outlines a sample workflow.  The initial version of this workflow only supports pixel classification.  

.. figure:: ../images/ocpilastik_intro.png
    :align: center

Train
-----

- Identify a region of interest in OCP, and note the data server, token, resolution, and coordinates
- Create a query, using the instructions for CAJAL
- Run *ilastik_getImage.m* to generate an image volume suitable for annotation in ilastik
- Using ilastik, build a pixel classifier, following the `Pixel Classification instructions in Ilastik <http://ilastik.org/documentation/pixelclassification/pixelclassification.html>`_
- Save your ilastik project, containing your trained classifier

Deploy
------

- Identify a region of interest in OCP, and note the data server, token, resolution, and coordinates
- Create a query, using the instructions for CAJAL
- Run *ilastik_getImage.m* to generate an image volume suitable for annotation in ilastik.  This can be batched using our LONI framework tools
- Classify the volume of interest using `ilastik in headless mode <http://ilastik.org/documentation/pixelclassification/headless.html>

**Choose from one of the following options to post-process and upload your data**

- Upload raw probabilities:  *put_anno_probs.m*, specifying the server, token, annotation file, and query used to download the underlying image. This uploads a probability map of your annotations to the server

- Convert into objects and upload as a volume.  (We recommend thresholding, grouping objects by connected components, and size filtering, but this is outside the scope of a basic tutorial.)  If objects are created, they can be uploaded using:  ocp_upload_dense.m (see CAJAL documentation).

Advanced Topics/Future Functionality
------------------------------------

- When uploading annotations processed as many small cubes, often some sort of padding or stitching operation is required.  These will differ slightly depending on use cases.  Examples exist (e.g., i2g, vesicle) to use as a starting point

- ilastik also supports object detection classifiers and manual labeling.  Our wrappers should work unchanged.  The only changes expected are with the users' ilastik application.  Please let us know if you would like us to develop these protocols.