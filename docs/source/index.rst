Welcome to yolort's documentation!
==================================

yolort is a runtime stack for yolov5 on specialized accelerators such as ``libtorch``,
``onnxruntime``, ``tvm`` and ``ncnn``.

.. image:: _static/yolort_logo.png
   :width: 500px
   :align: center

.. toctree::
   :maxdepth: 2
   :caption: Get started:

   what_is_yolort
   installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   notebooks/inference_pytorch_export_libtorch
   notebooks/how_to_align_with_ultralytics_yolov5
   notebooks/anchor_label_assignment_visualization
   notebooks/model_graph_visualization
   notebooks/export_onnx_inference_onnxruntime
   notebooks/export_relay_inference_tvm

.. toctree::
   :maxdepth: 2
   :caption: Deployment:

   deployment/libtorch
   deployment/onnxruntime
   deployment/ncnn

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   models
   runtime
