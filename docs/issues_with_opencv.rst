Issues with OpenCV
==================

In some systems, in the multiple GPU regime PyTorch may deadlock the DataLoader if `OpenCV` was compiled with `OpenCL` optimizations. Adding the following two lines before the library import may help. For more details https://github.com/pytorch/pytorch/issues/1355


.. code-block:: python

   cv2.setNumThreads(0)
   cv2.ocl.setUseOpenCL(False)


Similar trick may help to solve `OpenCV` + `Python` multiprocessing breaks on OSX. https://github.com/opencv/opencv/issues/5150
