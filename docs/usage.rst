=====
Usage
=====

Repository quick guides:

- English Markdown guide: ``USAGE.md``
- Korean overview: ``README.ko.md``
- Korean usage guide: ``사용법.md``
- Update log: ``update.md`` / ``update.ko.md``

To use grcwa in a project::

    import grcwa
    import torch

To initialize the RCWA::

  obj = grcwa.obj(nG,L1,L2,freq,theta,phi,verbose=0,
                  dtype_f=torch.float64,dtype_c=torch.complex128)
  # verbose=1 prints the actual nG

To add layers, the order of adding will determine the layer order (1st added layer is 0-th layer, 2nd to be 1st layer, and so forth)::
  
  obj.Add_LayerUniform(thick0,ep0) # uniform slab
  obj.Add_LayerGrid(thickp,Nx,Ny) # patterned layer

  # after add all layers:
  obj.Init_Setup()

To feed the epsilon profile for patterned layer::

  # x is a 1D array: np.concatenate((epgrid1.flatten(),epgrid2.flatten(),...))
  obj.GridLayer_geteps(x)

To update only one grid-patterned layer without rebuilding every patterned layer::

  obj.GridLayer_updateeps(which_layer, ep_grid, update_cache=False)
  obj.UpdateSMatrixCache()

To scale the periodicity in both lateral directions simultaneously (as a differentiable parameter)::

  obj.Init_Setup(Pscale=scale) # period will be scale*Lx and scale*Ly

Fourier space truncation options ::

  obj.Init_Setup(Gmethod=0) # 0 for circular, 1 for rectangular

To define planewave excitation::

  obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0)

To define incidence light other than planewave::

  obj.a0 = ... # forward
  obj.bN = ... # backward, each have a length 2*obj.nG, for the 2 lateral directions
  
To normalize output when the 0-th media is not vacuum, or for oblique incidence::
  
  R, T = obj.RT_Solve(normalize = 1)

To get Poynting flux by order::
  
  Ri, Ti = obj.RT_Solve(byorder=1) # Ri(Ti) has length obj.nG, too see which order, check obj.G; too see which kx,ky, check obj.kx obj.ky

To get amplitude of eigenvectors at some layer at some zoffset ::

  ai,bi = obj.GetAmplitudes(which_layer,z_offset)

To reuse structure and excitation data across many solves::

  obj.BuildSMatrixCache()
  obj.BuildAmplitudeCache()
  obj.ClearSMatrixCache() # optional: drop heavy prefix/suffix matrices and keep lightweight amplitude/exterior caches


To get real-space epsilon profile reconstructured from the truncated Fourier orders ::
  
  ep = obj.Return_eps(which_layer,Nx,Ny,component='xx') # For patterned layer component = 'xx','xy','yx','yy','zz'; For uniform layer, currently it's assumed to be isotropic        
        
To get Fourier amplitude of fields at some layer at some zoffset ::

  E,H = obj.Solve_FieldFourier(which_layer,z_offset) #E = [Ex,Ey,Ez], H = [Hx,Hy,Hz]

To get fields in real space on grid points ::
  
  E,H = obj.Solve_FieldOnGrid(which_layer,z_offset) # E = [Ex,Ey,Ez], H = [Hx,Hy,Hz]
  E,H = obj.Solve_FieldOnGrid(which_layer,z_offset,components=('Ex','Hy'))

To get simplified field outputs and reconstruct only the requested components::

  E,H = obj.Solve_FieldXY(which_layer,z_offset,components=('Ex','Hy'))
  E,H,x_coords,z_coords,layer_ranges,layer_edges,z_step = obj.Solve_FieldXZ(y0=0.0,znum=4,components=('Ex',))
  E,H,y_coords,z_coords,layer_ranges,layer_edges,z_step = obj.Solve_FieldYZ(x0=0.0,znum=4,components=('Ex',))
  E,H,x_coords,z_coords,layer_ranges,layer_edges,z_step = obj.Solve_FieldXZ(y0=0.0,znum=3,z_step=0.01,components=('Ex',))
  E,H,x_coords,z_coords = obj.Solve_FieldXZLayer(which_layer,z_list,y0=0.0,components=('Ex',))

  # E = [Ex,Ey,Ez], H = [Hx,Hy,Hz]
  # unrequested components are None
  # if only electric components are requested, H is returned as None
  # xz/yz return structure-wide cuts with x_coords/y_coords, z_coords, layer_ranges, and layer_edges
  # integer znum means the minimum samples per layer; thicker layers use the same approximate z_step mesh
  # z_step can be passed explicitly for structure-wide xz/yz cuts
  # automatic x/y coordinates are only provided for axis-aligned orthogonal lattices

To save and restore a baseline object state::

  state = obj.ExportState(include_caches=False)
  obj.SaveState('baseline.pt', include_caches=False)
  restored = grcwa.obj.LoadState('baseline.pt')

  # for a lightweight cached state, build amplitudes first and then drop the heavy S-matrix cache
  obj.BuildSMatrixCache()
  obj.BuildAmplitudeCache()
  obj.ClearSMatrixCache()
  obj.SaveState('baseline_light.pt', include_caches=True)

For repeated z-scans, pass the whole z list in one call instead of looping Solve_FieldOnGrid externally::

  E,H = obj.Solve_FieldXY(which_layer,z_list,components=('Ex',))
  
To get volume integration with respect to some convolution matrix *M* defined for 3 directions, respectively::
  
  val = obj.Volume_integral(which_layer,Mx,My,Mz,normalize=1)

To compute Maxwell stress tensor, integrated over the *z*-plane::

  Tx,Ty,Tz = obj.Solve_ZStressTensorIntegral(which_layer)

To estimate absorption from user-supplied Im(eps) patterns in real space::

  abs_layer = obj.Solve_AbsorptionLayer(which_layer, imag_eps_grid, min_znum=5)
  abs_z = obj.Solve_AbsorptionLayerZ(which_layer, imag_eps_grid, min_znum=5, z_step=0.01)
  abs_xy = obj.Solve_AbsorptionLayerXY(which_layer, imag_eps_grid, min_znum=5, z_step=0.01)
  abs_all = obj.Solve_Absorption({1: imag_eps_grid1, 3: imag_eps_grid2}, min_znum=5)

  # imag_eps_grid can be an arbitrary Nx x Ny tensor for isotropic loss
  # or a length-3 tuple/list of Nx x Ny tensors for diagonal anisotropic loss
  # one layer can also receive multiple named patterns as a dict, e.g. {'core': grid1, 'clad': grid2}
  # Solve_AbsorptionLayerZ returns absorption_z, Solve_AbsorptionLayerXY returns absorption_xy
