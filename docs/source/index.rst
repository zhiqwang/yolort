Welcome to yolort's documentation!
==================================

Runtime stack for YOLOv5 on specialized accelerators such as ``libtorch``,
``onnxruntime``, ``tvm`` and ``ncnn``.

.. image:: _static/yolort_logo.png
   :width: 400px
   :align: center


.. _what-is-yolort:

**What is yolort?**

``yolort`` focus on making the training and inference of the object detection task
integrate more seamlessly together. ``yolort`` now adopts the same model
structure as the official YOLOv5. The significant difference is that we adopt
the dynamic shape mechanism, and within this, we can embed both pre-processing
(``letterbox``) and post-processing (``nms``) into the model graph, which
simplifies the deployment strategy. In this sense, ``yolort`` makes it possible
to be deployed more friendly on ``LibTorch``, ``ONNXRuntime``, ``TVM`` and so on.

.. _about-the-code:

**About the code**

Follow the design principle of `detr <https://github.com/facebookresearch/detr>`_:

..

   object detection should not be more difficult than classification, and should
   not require complex libraries for training and inference.

``yolort`` is very simple to implement and experiment with. You like the implementation
of torchvision's faster-rcnn, retinanet or detr? You like yolov5? You love ``yolort``!

.. _quick-get-stated:

**Quick Get started**

Read a source of image(s) and detect its objects:

.. code:: python

   from yolort.models import yolov5s

   # Load model
   model = yolov5s(pretrained=True, score_thresh=0.45)
   model.eval()

   # Perform inference on an image file
   predictions = model.predict("bus.jpg")
   # Perform inference on a list of image files
   predictions = model.predict(["bus.jpg", "zidane.jpg"])

.. toctree::
   :maxdepth: 2
   :caption: Get started

   installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/inference-pytorch-export-libtorch
   notebooks/how-to-align-with-ultralytics-yolov5
   notebooks/anchor-label-assignment-visualization
   notebooks/model-graph-visualization
   notebooks/export-onnx-inference-onnxruntime
   notebooks/onnx-graphsurgeon-inference-tensorrt
   notebooks/export-relay-inference-tvm

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   models
   yolov5
