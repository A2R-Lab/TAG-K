Estimators
==========

All estimators inherit from :class:`~online_estimators.estimators.base.BaseEstimator`
and expose a single ``iterate(A, b) → x`` interface.

.. contents:: Estimator Classes
   :local:
   :depth: 1

Base Class
----------

.. automodule:: online_estimators.estimators.base
   :members:
   :undoc-members:
   :show-inheritance:

Recursive Least Squares
-----------------------

.. automodule:: online_estimators.estimators.rls
   :members:
   :undoc-members:
   :show-inheritance:

Kalman Filter
-------------

.. automodule:: online_estimators.estimators.kf
   :members:
   :undoc-members:
   :show-inheritance:

Randomized Kaczmarz Family
--------------------------

.. automodule:: online_estimators.estimators.rk
   :members:
   :undoc-members:
   :show-inheritance:

Greedy Randomized Kaczmarz Family
---------------------------------

.. automodule:: online_estimators.estimators.grk
   :members:
   :undoc-members:
   :show-inheritance:

Block Methods
-------------

.. automodule:: online_estimators.estimators.block
   :members:
   :undoc-members:
   :show-inheritance:

C++ Backend Loader
------------------

.. automodule:: online_estimators.estimators._backend
   :members:
   :undoc-members:
