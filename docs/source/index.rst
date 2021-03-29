yolort - YOLOv5 Runtime Stack
=======================================

yolort is another implementation of Ultralytics's [yolov5](https://github.com/ultralytics/yolov5),
and with modules refactoring to make it available in deployment backends such as `libtorch`,
`onnxruntime`, `tvm` and so on.

.. toctree::
   :maxdepth: 2
   :caption: Get started:

   installation

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   notebooks/load_model_as_ultralytics
   notebooks/visualize_jit_models

.. toctree::
   :maxdepth: 2
   :caption: Deployment:

   notebooks/inference_pytorch_export_libtorch
   notebooks/export_onnx_inference_onnxruntime
   notebooks/export_relay_inference_tvm

.. toctree::
   :maxdepth: 2
   :caption: yolort API:

   datasets
   models
   utils


Authors
=======

The yolort Team: Zhiqiang Wang

Acknowledgements
================

We were inspired by

- The implementation of `yolov5` borrow the code from [ultralytics](https://github.com/ultralytics/yolov5).
- This repo borrows the architecture design and part of the code from [torchvision](https://github.com/pytorch/vision).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
