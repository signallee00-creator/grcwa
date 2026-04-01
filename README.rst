=====
grcwa
=====
.. image:: https://img.shields.io/pypi/v/grcwa.svg
        :target: https://pypi.python.org/pypi/grcwa

..
   .. image:: https://img.shields.io/travis/weiliangjinca/grcwa.svg
	   :target: https://travis-ci.org/weiliangjinca/grcwa

.. image:: https://readthedocs.org/projects/grcwa/badge/?version=latest
        :target: https://grcwa.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

grcwa (autoGradable RCWA) is a python implementation of rigorous
coupled wave analysis (RCWA) for arbitrarily shaped photonic crystal
slabs, supporting automatic differentiation with PyTorch

* Free software: GPL license
* Documentation: https://grcwa.readthedocs.io.
* Quick docs in this repo:
  * English overview: `USAGE.md <./USAGE.md>`_
  * Korean overview: `README.ko.md <./README.ko.md>`_
  * Korean usage guide: `사용법.md <./사용법.md>`_
  * Update log: `update.md <./update.md>`_ / `update.ko.md <./update.ko.md>`_

Citing
-------

If you find **grcwa** useful for your research, please cite the
following paper:
::

   @article{Jin2020,
     title = {Inverse design of lightweight broadband reflector for relativistic lightsail propulsion},
     author ={Jin, Weiliang and Li, Wei and Orenstein, Meir and Fan, Shanhui},
     year = {2020},
     journal = {ACS Photonics},
     volume = {7},
     number = {9},
     pages = {2350--2355},
     year = {2020},
     publisher = {ACS Publications}
   }
  

Features
---------
.. image:: imag/scheme.png

RCWA solves EM-scattering problems of stacked photonic crystal
slabs. As illustrated in the above figure, the photonic structure can
have *N* layers of different thicknesses and independent spatial
dielectric profiles. All layers are periodic in the two lateral
directions, and invariant along the vertical direction.

* Each photonic crystal layer can have arbitrary dielectric profile on
  the *2D* grids.
* **PyTorch autograd** is integrated into the package, allowing for
  automated and fast gradient evaluations for the sake of large-scale
  optimizations. Differentiable parameters include dielectric constant
  on every grid, frequency, angles, thickness of each layer, and
  periodicity (however the ratio of periodicity along the two lateral
  directions must be fixed).
* Cached scattering/amplitude sweeps and selective field reconstruction
  reduce repeated work for multi-layer queries, line cuts, and
  partial single-layer updates.


Quick Start
-----------
* Installation:

  .. code-block:: console
		  
		  $ pip install grcwa

  Or,

  .. code-block:: console

		  $ git clone git://github.com/weiliangjinca/grcwa
		  $ pip install .


* Example 1: transmission and reflection (sum or by order) of a square lattice of a hole: `ex1.py <./example/ex1.py>`_

* Example 2: Transmission and reflection of two patterned layers: `ex2.py <./example/ex2.py>`_, as illustrated in the figure below (only a **unit cell** is plotted)

  .. image:: imag/ex.png
	     
  * *Periodicity* in the lateral direction is  *L*\ :sub:`x` = *L*\ :sub:`y` = 0.2, and *frequency* is 1.0.

  * The incident light has an angel *pi*/10.

    .. code-block:: python
		  
		    import grcwa
		    import torch
		    
		     # lattice constants
		     L1 = [0.2,0]
		     L2 = [0,0.2]
		     # Truncation order (actual number might be smaller)
		     nG = 101
		     # frequency
		     freq = 1.
		     # angle
		     theta = torch.pi/10
		     phi = 0.

		     # setup RCWA
		     obj = grcwa.obj(nG,L1,L2,freq,theta,phi,verbose=1,
		                     dtype_f=torch.float64,dtype_c=torch.complex128)

  * Geometry: the thicknesses of the four layers are 0.1,0.2,0.3, and 0.4. For patterned layers, we consider total grid points *N*\ :sub:`x` \* *N*\ :sub:`y` = 100\*100 within the unit cell.
    
  * Dielectric constant: 2.0 for the 0-th layer; 4.0 (1.0) for the 1st layer in the orange (void) region; 6.0 (1.0) for the 2nd layer in the bule (void) region; and 3.0 for the last layer.

    .. code-block:: python

		    Np = 2 # number of patterned layers
		    Nx = 100
		    Ny = 100
		    
		    thick0 = 0.1
		    pthick = [0.2,0.3]
		    thickN = 0.4

		    ep0 = 2.
		    epN = 3.
		    
		    obj.Add_LayerUniform(thick0,ep0)
		    for i in range(Np):
		        obj.Add_LayerGrid(pthick[i],Nx,Ny)
		    obj.Add_LayerUniform(thickN,epN)

		    obj.Init_Setup()

  * Patterned layer: the 1-th layer a circular hole of radius 0.5 *L*\ :sub:`x`, and the 2-nd layer has a square hole of 0.5 *L*\ :sub:`x`
  
    .. code-block:: python

		    radius = 0.5
		    a = 0.5

		    ep1 = 4.
		    ep2 = 6.
		    epbkg = 1.

		    # coordinate
		    x0 = torch.linspace(0,1.,Nx,dtype=torch.float64)
		    y0 = torch.linspace(0,1.,Ny,dtype=torch.float64)
		    x, y = torch.meshgrid(x0,y0,indexing='ij')

		    # layer 1
		    epgrid1 = torch.ones((Nx,Ny),dtype=torch.float64)*ep1
		    ind = (x-.5)**2+(y-.5)**2<radius**2
		    epgrid1[ind]=epbkg

		    # layer 2
		    epgrid2 = torch.ones((Nx,Ny),dtype=torch.float64)*ep2
		    ind = torch.logical_and(torch.abs(x-.5)<a/2, torch.abs(y-.5)<a/2)
		    epgrid2[ind]=epbkg		    
		    
		    # combine epsilon of all layers
		    epgrid = torch.concatenate((epgrid1.flatten(),epgrid2.flatten()))
		    obj.GridLayer_geteps(epgrid)

  * Incident light is *s*-polarized

    .. code-block:: python

		     planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
		     obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

		     # solve for R and T
		     R,T= obj.RT_Solve(normalize=1)

  Here ``normalize=1`` uses the actual incident Poynting flux of the
  current excitation. This is more robust than a fixed ``n/cos(theta)``
  factor when amplitudes are rescaled or the incident medium is lossy.
  ``obj.normalization`` now stores this same incident-power value, so the
  attribute and the runtime normalization path stay consistent.

* Example 3: topology optimization of reflection of a single patterned layer, `ex3.py <./example/ex3.py>`_

* Example 4: transmission and reflection (sum or by order) of a hexagonal lattice of a hole: `ex4.py <./example/ex4.py>`_

* Example 5: cache-aware field reconstruction, line cuts, and state reuse: `ex5_cache_and_fields.py <./example/ex5_cache_and_fields.py>`_

Fast-path notes
---------------

* For repeated field queries, build caches once and reuse them::

    obj.BuildSMatrixCache()
    obj.BuildAmplitudeCache()

  If memory becomes the main constraint after amplitudes are built, drop
  the heavy S-matrix sweep and keep only the lightweight amplitude and
  exterior caches::

    obj.ClearSMatrixCache()

* Reconstruct only what you need::

    E, H = obj.Solve_FieldXY(which_layer, z_list,
                             components=('Ex',))

* Use dedicated line-cut APIs instead of reconstructing full 2D fields
  when you only need *XZ* or *YZ* views::

    E, H, x_coords, z_coords, layer_ranges, layer_edges, z_step = obj.Solve_FieldXZ(
        y0=0.0, znum=4, components=('Ex',)
    )
    E, H, x_coords, z_coords, layer_ranges, layer_edges, z_step = obj.Solve_FieldXZ(
        y0=0.0, znum=3, z_step=0.01, components=('Ex',)
    )

  `Solve_FieldXZ` and `Solve_FieldYZ` return whole-structure cuts. For a
  single-layer line cut, use `Solve_FieldXZLayer` or `Solve_FieldYZLayer`.
  Integer ``znum`` means the minimum samples per layer; thicker layers are
  sampled on the same approximate ``z_step`` mesh. Pass ``z_step`` explicitly
  when you want to control the structure-wide spacing directly.

  `Solve_FieldXY` returns ``E, H`` where ``E = [Ex, Ey, Ez]`` and
  ``H = [Hx, Hy, Hz]``. Unrequested components are returned as ``None``.
  If only electric components are requested, ``H`` is returned as ``None``.

  To post-process absorption from user-supplied Im(eps) patterns::

    abs_3d = obj.Solve_AbsorptionLayer(which_layer, [imag_eps_grid1, imag_eps_grid2],
                                       z_step=0.01, z_offset=0.0, z_min=5)
    abs_z = obj.Solve_AbsorptionLayer(which_layer, [imag_eps_grid1, imag_eps_grid2],
                                      z_step=0.01, z_offset=0.0, z_min=5, avg='XY')
    abs_xy = obj.Solve_AbsorptionLayer(which_layer, [imag_eps_grid1, imag_eps_grid2],
                                       z_step=0.01, z_offset=0.0, z_min=5, avg='Z')
    layer_list = [1, 3]
    pattern_list = [
        [layer1_imag_eps_a, layer1_imag_eps_b],  # patterns for layer 1
        [layer3_imag_eps_a],                     # patterns for layer 3
    ]
    abs_all = obj.Solve_Absorption(layer_list, pattern_list,
                                   z_step=0.01, z_min=5, avg='XY')
    abs_tot = obj.Solve_AbsorptionLayer(which_layer, [imag_eps_grid1, imag_eps_grid2],
                                        z_step=0.01, z_offset=0.0, z_min=5, avg='tot')

  `Solve_AbsorptionLayer` returns one clean structure:
  ``absorption`` for the requested average mode, ``pattern_absorption`` as a
  list for each input pattern, and ``total`` as the integrated absorption.
  In `Solve_Absorption(layer_list, pattern_list, ...)`, the outer list follows
  the same order as `layer_list`, so each layer can use a different grid shape.

Root helper packages
--------------------

This fork also provides two lightweight root-level helper packages outside
`grcwa/`.

* `patterns/`

  Use reusable geometry builders and flatten the result directly into
  `GridLayer_geteps(...)`::

    from patterns import pattern_xx, nm
    import torch

    pat = pattern_xx(
        size_x=400 * nm, size_y=400 * nm,
        Nx=128, Ny=128,
        lamb0=1550 * nm, CRA=0.0, azimuth=0.0,
        eps_bg=1.0, eps_obj=3.48**2,
        radius=90 * nm,
        dtype_f=torch.float64, dtype_c=torch.complex128,
        device='cpu',
    )
    obj.GridLayer_geteps(pat.flatten())

  For multiple patterned layers, concatenate the flattened tensors::

    obj.GridLayer_geteps(torch.cat([pat1.flatten(), pat2.flatten()]))

* `materials/`

  Use the current wavelength as the default query point and access
  tabulated data by material/deck name::

    from materials import material
    import torch

    mat = material(
        1550e-9,
        dtype_f=torch.float64,
        dtype_c=torch.complex128,
        device='cpu',
    )
    eps_sio2 = mat.eps_SiO2_xx()
    eps_old = mat.eps_SiO2_xx(filename='SiO2_index.txt')

  Rules:

  - files use `WL n k` format
  - `WL` is interpreted in `nm`
  - query wavelengths are interpreted in `m`
  - `n` and `k` use linear interpolation with end-segment extrapolation
  - `eps = (n + i k)^2`
  - if both `SiO2_index.txt` and dated files such as
    `SiO2_index_260401.txt` exist, the latest dated file is used by default

* Replace a single grid layer and update only the affected scattering cache::

    obj.GridLayer_updateeps(which_layer, new_ep_grid, update_cache=False)
    obj.UpdateSMatrixCache()

* Save a baseline object for repeated what-if runs::

    obj.SaveState('baseline.pt', include_caches=False)
    restored = grcwa.obj.LoadState('baseline.pt')

  To save a lighter cached state for repeated field solves, keep the
  amplitude/exterior cache and drop the S-matrix sweep before saving::

    obj.BuildSMatrixCache()
    obj.BuildAmplitudeCache()
    obj.ClearSMatrixCache()
    obj.SaveState('baseline_light.pt', include_caches=True)

  For iterative design loops, the practical pattern is often::

    obj.BuildSMatrixCache()
    obj.BuildAmplitudeCache()
    obj.ClearSMatrixCache()

  This keeps field queries fast while releasing the large prefix/suffix
  S-matrix cache.
  
Note on conventions
-------------------

* The vacuum permittivity, permeability, and speed of light are *1*.
* The time harmonic convention is *exp(-i omega t)*.

Acknowledgements
----------------

My implementation of RCWA received helpful discussions from `Dr. Zin
Lin
<https://scholar.google.com/citations?user=3ZgzHLYAAAAJ&hl=en>`_. Many
details of implementations were referred to a RCWA package implemented
in c called `S4 <https://github.com/victorliu/S4>`_. The idea of
integrating **Autograd** into RCWA package rather than deriving
adjoint-variable gradient by hand was inspired by a discussion with
Dr. Ian Williamson and Dr. Momchil Minkov. Many implementation styles
follow their implementation in `legume
<https://github.com/fancompute/legume>`_. Haiwen Wang and Cheng Guo
provided useful feedback. Lastly, the template was credited to
Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_.


.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
