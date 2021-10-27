(Py)TOD: Tensor-based Outlier Detection, A General GPU-Accelerated Framework
============================================================================

----

**Background**: Outlier detection (OD) is a key data mining task for identifying abnormal objects from general samples with numerous high-stake applications including fraud detection and intrusion detection.

To scale outlier detection (OD) to large-scale, high-dimensional datasets, we propose **TOD**, a novel system that abstracts OD algorithms into basic tensor operations for efficient GPU acceleration.

`The corresponding paper <https://www.andrew.cmu.edu/user/yuezhao2/papers/21-preprint-tod.pdf>`_.
**The code is being cleaned up and released. Please watch and star!**


One reason to use it:
^^^^^^^^^^^^^^^^^^^^^

On average, **TOD is 11 times faster than PyOD**!

If you need another reason: it can handle much larger datasets:more than **a million sample** OD within an hour!


----

TOD is featured for:

* **Unified APIs, detailed documentation, and examples** for the easy use (under construction)
* Supports more than 10 different OD algorithms and more are being added
* **TOD** supports multi-GPU acceleration
* **Advanced techniques** like provable quantization


Programming Model Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Complex OD algorithms can be abstracted into common tensor operators.

.. image:: https://raw.githubusercontent.com/yzhao062/pytod/master/figs/abstraction.png
   :target: https://raw.githubusercontent.com/yzhao062/pytod/master/figs/abstraction.png


For instance, ABOD and COPOD can be assembled by the basic tensor operators.

.. image:: https://raw.githubusercontent.com/yzhao062/pytod/master/figs/abstraction_example.png
   :target: https://raw.githubusercontent.com/yzhao062/pytod/master/figs/abstraction_example.png


End-to-end Performance Comparison with PyOD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overall, it is much (on avg. 11 times) faster than PyOD takes way less run time.

.. image:: https://raw.githubusercontent.com/yzhao062/pytod/master/figs/run_time.png
   :target: https://raw.githubusercontent.com/yzhao062/pytod/master/figs/run_time.png


----

Code is being released. Watch and star for the latest news!

