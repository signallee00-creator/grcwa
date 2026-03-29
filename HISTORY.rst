=======
History
=======

0.2.0 (2026-03-29)
------------------

* Replace the old numpy/backend/autograd stack with a torch-only implementation.
* Add explicit ``device``, ``dtype_f``, and ``dtype_c`` controls to ``grcwa.obj``.
* Port FFT, lattice, RCWA, examples, and tests to PyTorch.
* Refresh README and Sphinx docs for the torch-based workflow.
* Add pytest configuration to ignore the local snapshot comparison folder.

0.1.2 (2020-11-01)
------------------

* Add example for hexagonal lattice

0.1.1 (2020-05-18)
------------------

* Fix license
  
0.1 (2020-05-12)
------------------

* First release on PyPI.
