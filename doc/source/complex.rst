.. _Complex:

Multi-threaded processing pipelines
===================================

This example shows how GIFT-Grab can be used for running complex pipelines with multiple intermediate
processing nodes and threads.
The intermediate processing nodes are built on the same principles as in the SciPy_ example.
Running the example requires an `HEVC-encoded MP4 file`_, an `NVENC-capable GPU`_, and `NumPy support`_.

.. _`HEVC-encoded MP4 file`: https://github.com/gift-surg/GIFT-Grab/blob/master/doc/build.md#reading-video-files
.. _`NVENC-capable GPU`: https://github.com/gift-surg/GIFT-Grab/blob/master/doc/build.md#hevc
.. _`NumPy support`: https://github.com/gift-surg/GIFT-Grab/blob/master/doc/build.md#python-api

Below is the commented full source code:

.. literalinclude:: ../../src/tests/pipeline/complex_pipeline.py
   :language: python
   :linenos:
