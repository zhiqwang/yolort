Install yolort
============

.. _required:

**Required**

Above all, follow the `official instructions <https://pytorch.org/get-started/locally/>`_ to
install PyTorch 1.8.0+ and torchvision 0.9.0+.

To install yolort, you may either use the repository

   https://github.com/zhiqwang/yolov5-rt-stack/

and ``setup.py`` from there or use

::

   pip install -U yolort

to get the latest release and

::

   pip install 'git+https://github.com/zhiqwang/yolov5-rt-stack.git'

to install the development version.

.. _optional:

**Optional**

Install pycocotools (for evaluation on COCO):

::

  pip install -U 'git+https://github.com/ppwwyyxx/cocoapi.git#subdirectory=PythonAPI'
