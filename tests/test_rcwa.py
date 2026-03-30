import torch
import grcwa
from grcwa.rcwa import SolveExterior, SolveInterior, build_layer_z_points

from .utils import t_grad

tol = 1e-2
tolS4 = 1e-3  # error tolerance for S4 v.s. this code

Nlayer = 1
nG = 101
L1 = [0.1, 0]
L2 = [0, 0.1]
Nx = 100
Ny = 100

epsuniform0 = 1.0
epsuniformN = 1.0

thick0 = 1.0
thickN = 1.0

freq = 1.0
theta = torch.pi / 18
phi = torch.pi / 9
Pscale = 1.0

pthick = [0.2]
radius = 0.4
epgrid = torch.ones((Nx, Ny), dtype=torch.float64)
x0 = torch.linspace(0, 1.0, Nx, dtype=torch.float64)
y0 = torch.linspace(0, 1.0, Ny, dtype=torch.float64)
x, y = torch.meshgrid(x0, y0, indexing='ij')
sphere = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius ** 2
epgrid[sphere] = 12.0

planewave = {'p_amp': 1, 's_amp': 0, 'p_phase': 0, 's_phase': 0}


def rcwa_assembly(epgrid_value, freq_value, theta_value, phi_value, planewave_value, pthick_value, Pscale=1.0):
    obj = grcwa.obj(nG, L1, L2, freq_value, theta_value, phi_value, verbose=0)
    obj.Add_LayerUniform(thick0, epsuniform0)
    for i in range(Nlayer):
        obj.Add_LayerGrid(pthick_value[i], Nx, Ny)
    obj.Add_LayerUniform(thickN, epsuniformN)

    obj.Init_Setup(Pscale=Pscale, Gmethod=0)
    obj.MakeExcitationPlanewave(
        planewave_value['p_amp'],
        planewave_value['p_phase'],
        planewave_value['s_amp'],
        planewave_value['s_phase'],
        order=0,
    )
    obj.GridLayer_geteps(epgrid_value)
    return obj


def _small_grid(kind, Nx_small, Ny_small):
    x_axis = torch.linspace(0.0, 1.0, Nx_small, dtype=torch.float64)
    y_axis = torch.linspace(0.0, 1.0, Ny_small, dtype=torch.float64)
    xx, yy = torch.meshgrid(x_axis, y_axis, indexing='ij')

    if kind == 0:
        out = torch.ones((Nx_small, Ny_small), dtype=torch.float64) * 3.0
        out[(xx - 0.35) ** 2 + (yy - 0.55) ** 2 < 0.18 ** 2] = 1.4
        return out
    if kind == 1:
        out = torch.ones((Nx_small, Ny_small), dtype=torch.float64) * 4.5
        out[torch.logical_and(xx > 0.25, xx < 0.65)] = 1.8
        return out
    raise ValueError("Unknown grid kind")


def rcwa_small_assembly(grid_override=None):
    nG_small = 31
    L1_small = [0.2, 0.0]
    L2_small = [0.0, 0.3]
    Nx_small = 18
    Ny_small = 14
    obj = grcwa.obj(nG_small, L1_small, L2_small, 1.2, torch.pi / 12, torch.pi / 15, verbose=0)
    obj.Add_LayerUniform(0.15, 1.0)
    obj.Add_LayerGrid(0.18, Nx_small, Ny_small)
    obj.Add_LayerUniform(0.08, 2.2)
    obj.Add_LayerGrid(0.21, Nx_small, Ny_small)
    obj.Add_LayerUniform(0.12, 1.3)
    obj.Init_Setup(Pscale=1.0, Gmethod=0)
    obj.MakeExcitationPlanewave(0.7, 0.1, 0.4, -0.2, order=0, direction='forward')

    grid0 = _small_grid(0, Nx_small, Ny_small)
    grid1 = _small_grid(1, Nx_small, Ny_small)
    if grid_override is not None:
        if 1 in grid_override:
            grid0 = grid_override[1]
        if 3 in grid_override:
            grid1 = grid_override[3]

    obj.GridLayer_geteps(torch.concatenate((grid0.flatten(), grid1.flatten())))
    return obj, {1: grid0, 3: grid1}


def _assert_close(lhs, rhs, atol=1e-7, rtol=1e-6):
    assert torch.allclose(lhs, rhs, atol=atol, rtol=rtol), f"max error={torch.max(torch.abs(lhs-rhs)).item()}"


def _assert_field_list_close(lhs, rhs):
    if lhs is None or rhs is None:
        assert lhs is None and rhs is None
        return
    for left, right in zip(lhs, rhs):
        if left is None or right is None:
            assert left is None and right is None
        else:
            _assert_close(left, right)


def test_rcwa():
    obj = rcwa_assembly(epgrid, freq, theta, phi, {'p_amp': 1, 's_amp': 0, 'p_phase': 0, 's_phase': 0}, pthick, Pscale=1.0)
    R, T = obj.RT_Solve(normalize=0)
    assert abs(T.item() - 0.85249901083265) < tolS4 * T.item()

    obj = rcwa_assembly(epgrid, freq, theta, phi, {'p_amp': 0, 's_amp': 1, 'p_phase': 0, 's_phase': 0}, pthick, Pscale=1.0)
    R, T = obj.RT_Solve(normalize=0)
    assert abs(T.item() - 0.83900479939861) < tolS4 * T.item()

    ai, bi = obj.GetAmplitudes(1, 0.0)
    assert len(ai) == obj.nG * 2

    e, h = obj.Solve_FieldOnGrid(1, 0.0)
    assert e[0].shape == (Nx, Ny)

    Mx = torch.real(obj.Patterned_epinv_list[0])
    val = obj.Volume_integral(1, Mx, Mx, Mx, normalize=1)
    assert torch.real(val).item() > 0

    Tx, Ty, Tz = obj.Solve_ZStressTensorIntegral(0)
    assert Tz.item() < 0


def test_epsgrad():
    def fun(x_value):
        obj = rcwa_assembly(x_value, freq, theta, phi, planewave, pthick, Pscale=1.0)
        R, _ = obj.RT_Solve(normalize=1)
        return R

    x_value = epgrid.flatten()
    ind = torch.randint(Nx * Ny * Nlayer, size=(1,)).item()
    FD, AD = t_grad(fun, x_value, 1e-3, ind)
    ref = max(torch.abs(FD).item(), 1e-12)
    assert torch.abs(FD - AD).item() < ref * tol, 'wrong epsgrid gradient'


def test_thickgrad():
    def fun(x_value):
        obj = rcwa_assembly(epgrid.flatten(), freq, theta, phi, planewave, x_value, Pscale=1.0)
        R, _ = obj.RT_Solve(normalize=1)
        return R

    x_value = torch.tensor([0.1], dtype=torch.float64)
    FD, AD = t_grad(fun, x_value, 1e-3, 0)
    ref = max(torch.abs(FD).item(), 1e-12)
    assert torch.abs(FD - AD).item() < ref * tol, 'wrong thickness gradient'


def test_periodgrad():
    def fun(x_value):
        obj = rcwa_assembly(epgrid.flatten(), freq, theta, phi, planewave, pthick, Pscale=x_value)
        R, _ = obj.RT_Solve(normalize=1)
        return R

    x_value = torch.tensor(1.0, dtype=torch.float64)
    FD, AD = t_grad(fun, x_value, 1e-3, 0)
    ref = max(torch.abs(FD).item(), 1e-12)
    assert torch.abs(FD - AD).item() < ref * tol, 'wrong thickness gradient'


def test_freqgrad():
    def fun(x_value):
        obj = rcwa_assembly(epgrid.flatten(), x_value, theta, phi, planewave, pthick, Pscale=1.0)
        R, _ = obj.RT_Solve(normalize=1)
        return R

    x_value = torch.tensor(1.0, dtype=torch.float64)
    FD, AD = t_grad(fun, x_value, 1e-3, 0)
    ref = max(torch.abs(FD).item(), 1e-12)
    assert torch.abs(FD - AD).item() < ref * tol, 'wrong thickness gradient'


def test_thetagrad():
    def fun(x_value):
        obj = rcwa_assembly(epgrid.flatten(), freq, x_value, phi, planewave, pthick, Pscale=1.0)
        R, _ = obj.RT_Solve(normalize=1)
        return R

    x_value = torch.tensor(torch.pi / 10, dtype=torch.float64)
    FD, AD = t_grad(fun, x_value, 1e-3, 0)
    ref = max(torch.abs(FD).item(), 1e-12)
    assert torch.abs(FD - AD).item() < ref * tol, 'wrong thickness gradient'


def test_cached_interior_matches_uncached():
    obj, _ = rcwa_small_assembly()
    obj.BuildSMatrixCache()

    for which_layer in range(obj.Layer_N):
        ref_ai, ref_bi = SolveInterior(which_layer, obj.a0, obj.bN, obj.q_list, obj.phi_list, obj.kp_list, obj.thickness_list, dtype_c=obj.dtype_c)
        ai, bi = obj.GetAmplitudes_noTranslate(which_layer)
        _assert_close(ai, ref_ai)
        _assert_close(bi, ref_bi)


def test_cached_exterior_matches_uncached():
    obj, _ = rcwa_small_assembly()
    obj.BuildSMatrixCache()

    ref_aN, ref_b0 = SolveExterior(obj.a0, obj.bN, obj.q_list, obj.phi_list, obj.kp_list, obj.thickness_list, dtype_c=obj.dtype_c)
    aN, b0 = obj.GetExteriorAmplitudesCached()
    _assert_close(aN, ref_aN)
    _assert_close(b0, ref_b0)


def test_amplitude_cache_matches_per_layer_solves():
    obj, _ = rcwa_small_assembly()
    obj.BuildSMatrixCache()
    refs = [obj.GetAmplitudes_noTranslate(layer) for layer in range(obj.Layer_N)]

    obj.BuildAmplitudeCache()
    for layer in range(obj.Layer_N):
        ai, bi = obj.GetAmplitudes_noTranslate(layer)
        _assert_close(ai, refs[layer][0])
        _assert_close(bi, refs[layer][1])


def test_partial_update_matches_full_rebuild():
    obj_partial, base_grids = rcwa_small_assembly()
    state = obj_partial.ExportState(include_caches=True)

    new_grid1 = torch.roll(base_grids[1], shifts=2, dims=0)
    new_grid3 = torch.flip(base_grids[3], dims=[1])
    obj_partial.BuildSMatrixCache()
    obj_partial.GridLayer_updateeps(1, new_grid1, update_cache=False)
    obj_partial.GridLayer_updateeps(3, new_grid3, update_cache=False)
    obj_partial.UpdateSMatrixCache()

    obj_full, _ = rcwa_small_assembly({1: new_grid1, 3: new_grid3})

    R_partial, T_partial = obj_partial.RT_Solve(normalize=1)
    R_full, T_full = obj_full.RT_Solve(normalize=1)
    _assert_close(R_partial, R_full)
    _assert_close(T_partial, T_full)

    field_partial = obj_partial.Solve_FieldXY(1, [0.0, 0.05], components=('Ex', 'Hy'))
    field_full = obj_full.Solve_FieldXY(1, [0.0, 0.05], components=('Ex', 'Hy'))
    _assert_field_list_close(field_partial[0], field_full[0])
    _assert_field_list_close(field_partial[1], field_full[1])

    restored, _ = rcwa_small_assembly()
    restored.RestoreState(state, restore_caches=True)
    restored.GridLayer_updateeps(1, new_grid1, update_cache=False)
    restored.GridLayer_updateeps(3, new_grid3, update_cache=False)
    restored.UpdateSMatrixCache()
    R_restored, T_restored = restored.RT_Solve(normalize=1)
    _assert_close(R_restored, R_full)
    _assert_close(T_restored, T_full)


def test_same_eps_update_after_load_is_invariant(tmp_path):
    obj, base_grids = rcwa_small_assembly()
    obj.BuildSMatrixCache()
    obj.BuildAmplitudeCache()
    R0, T0 = obj.RT_Solve(normalize=1)

    state_path = tmp_path / "same_eps_state.pt"
    obj.SaveState(state_path, include_caches=True)
    loaded = grcwa.obj.LoadState(state_path, map_location='cpu')

    loaded.GridLayer_updateeps(1, base_grids[1], update_cache=True)
    R1, T1 = loaded.RT_Solve(normalize=1)
    _assert_close(R1, R0)
    _assert_close(T1, T0)

    changed_grid = torch.roll(base_grids[1], shifts=2, dims=0)
    changed = grcwa.obj.LoadState(state_path, map_location='cpu')
    changed.GridLayer_updateeps(1, changed_grid, update_cache=True)
    R2, T2 = changed.RT_Solve(normalize=1)

    assert not torch.allclose(R2, R0, atol=1e-10, rtol=1e-8)
    assert not torch.allclose(T2, T0, atol=1e-10, rtol=1e-8)


def test_clear_smatrix_cache_keeps_lightweight_cached_solves(tmp_path):
    obj, _ = rcwa_small_assembly()
    obj.BuildSMatrixCache()
    obj.BuildAmplitudeCache()

    rt_ref = obj.RT_Solve(normalize=1)
    field_ref = obj.Solve_FieldXY(1, [0.0, 0.05], components=('Ex', 'Hy'))

    obj.ClearSMatrixCache()
    assert obj._prefix_smat is None
    assert obj._suffix_smat is None

    rt_cleared = obj.RT_Solve(normalize=1)
    field_cleared = obj.Solve_FieldXY(1, [0.0, 0.05], components=('Ex', 'Hy'))

    _assert_close(rt_cleared[0], rt_ref[0])
    _assert_close(rt_cleared[1], rt_ref[1])
    _assert_field_list_close(field_cleared[0], field_ref[0])
    _assert_field_list_close(field_cleared[1], field_ref[1])

    state_path = tmp_path / "lightweight_cache_state.pt"
    obj.SaveState(state_path, include_caches=True)
    loaded = grcwa.obj.LoadState(state_path, map_location='cpu')

    assert loaded._prefix_smat is None
    assert loaded._suffix_smat is None
    rt_loaded = loaded.RT_Solve(normalize=1)
    field_loaded = loaded.Solve_FieldXY(1, [0.0, 0.05], components=('Ex', 'Hy'))

    _assert_close(rt_loaded[0], rt_ref[0])
    _assert_close(rt_loaded[1], rt_ref[1])
    _assert_field_list_close(field_loaded[0], field_ref[0])
    _assert_field_list_close(field_loaded[1], field_ref[1])


def test_absorption_layer_z_matches_direct_field_sum():
    obj, base_grids = rcwa_small_assembly()
    loss_grid = torch.where(base_grids[1] > 2.0, 0.04, 0.01)

    result = obj.Solve_AbsorptionLayerZ(1, loss_grid, min_znum=5, normalize=0)

    z_coords = result['z_coords']
    field, _ = obj.Solve_FieldXY(1, z_coords, Nxy=loss_grid.shape, components=('Ex', 'Ey', 'Ez'))
    e2_loss = loss_grid.unsqueeze(0) * (
        torch.abs(field[0]) ** 2 + torch.abs(field[1]) ** 2 + torch.abs(field[2]) ** 2
    )
    cell_area = torch.abs(obj.L1[0] * obj.L2[1] - obj.L1[1] * obj.L2[0]) * obj.Pscale ** 2
    dA = cell_area / (loss_grid.shape[0] * loss_grid.shape[1])
    expected_z = 0.5 * torch.real(obj.omega).to(dtype=obj.dtype_f) * dA * torch.sum(e2_loss, dim=(-2, -1))
    expected_total = torch.trapz(expected_z, z_coords)

    _assert_close(result['absorption_z'], expected_z)
    _assert_close(result['total'], expected_total)


def test_absorption_layer_xy_matches_direct_field_sum():
    obj, base_grids = rcwa_small_assembly()
    loss_grid = torch.where(base_grids[1] > 2.0, 0.04, 0.01)

    result = obj.Solve_AbsorptionLayerXY(1, loss_grid, min_znum=5, normalize=0)

    z_coords = build_layer_z_points(obj, obj.thickness_list[1], count=5, z_step=None)
    field, _ = obj.Solve_FieldXY(1, z_coords, Nxy=loss_grid.shape, components=('Ex', 'Ey', 'Ez'))
    density = 0.5 * torch.real(obj.omega).to(dtype=obj.dtype_f) * loss_grid.unsqueeze(0) * (
        torch.abs(field[0]) ** 2 + torch.abs(field[1]) ** 2 + torch.abs(field[2]) ** 2
    )
    expected_xy = torch.trapz(density, z_coords, dim=0)
    cell_area = torch.abs(obj.L1[0] * obj.L2[1] - obj.L1[1] * obj.L2[0]) * obj.Pscale ** 2
    dA = cell_area / (loss_grid.shape[0] * loss_grid.shape[1])
    expected_total = dA * torch.sum(expected_xy)

    _assert_close(result['absorption_xy'], expected_xy)
    _assert_close(result['total'], expected_total)


def test_absorption_multiple_patterns_and_layers_sum():
    obj, base_grids = rcwa_small_assembly()
    loss1a = torch.where(base_grids[1] > 2.0, 0.03, 0.00)
    loss1b = torch.where(base_grids[1] <= 2.0, 0.01, 0.00)
    loss3 = torch.where(base_grids[3] > 3.0, 0.02, 0.01)

    layer1 = obj.Solve_AbsorptionLayerZ(1, {'core': loss1a, 'clad': loss1b}, min_znum=5)
    layer3 = obj.Solve_AbsorptionLayerZ(3, loss3, min_znum=5)
    total = obj.Solve_Absorption({1: {'core': loss1a, 'clad': loss1b}, 3: loss3}, min_znum=5)

    _assert_close(layer1['total'], layer1['per_pattern']['core']['total'] + layer1['per_pattern']['clad']['total'])
    _assert_close(total['per_layer'][1], layer1['total'])
    _assert_close(total['per_layer'][3], layer3['total'])
    _assert_close(total['total'], layer1['total'] + layer3['total'])


def test_field_component_selection_and_new_views(tmp_path):
    obj, _ = rcwa_small_assembly()

    legacy_full = obj.Solve_FieldOnGrid(1, 0.0)
    legacy_ex = obj.Solve_FieldOnGrid(1, 0.0, components=('Ex',))
    _assert_close(legacy_ex[0][0], legacy_full[0][0])
    assert legacy_ex[0][1] is None
    assert legacy_ex[1] is None

    xy_e, xy_h = obj.Solve_FieldXY(1, 0.0, components=('Ex', 'Hy'))
    _assert_close(xy_e[0], legacy_full[0][0])
    _assert_close(xy_h[1], legacy_full[1][1])
    assert xy_e[1] is None
    assert xy_h[0] is None

    xy_only_e, xy_only_h = obj.Solve_FieldXY(1, 0.0, components=('Ex', 'Ey', 'Ez'))
    _assert_close(xy_only_e[0], legacy_full[0][0])
    assert xy_only_h is None

    Nx_small, Ny_small = obj.GridLayer_Nxy_list[obj.id_list[1][3]]
    iy = Ny_small // 3
    ix = Nx_small // 4
    y0_value = torch.as_tensor(obj.L2[1], dtype=obj.dtype_f, device=obj.device) * iy / Ny_small
    x0_value = torch.as_tensor(obj.L1[0], dtype=obj.dtype_f, device=obj.device) * ix / Nx_small

    xy_stack_e, xy_stack_h = obj.Solve_FieldXY(1, [0.0, 0.04], components=('Ex', 'Hy'))
    xz_layer_e, xz_layer_h, _, _ = obj.Solve_FieldXZLayer(1, [0.0, 0.04], y0=y0_value, components=('Ex', 'Hy'))
    yz_layer_e, yz_layer_h, _, _ = obj.Solve_FieldYZLayer(1, [0.0, 0.04], x0=x0_value, components=('Ex', 'Hy'))

    _assert_close(xz_layer_e[0], xy_stack_e[0][:, :, iy])
    _assert_close(xz_layer_h[1], xy_stack_h[1][:, :, iy])
    _assert_close(yz_layer_e[0], xy_stack_e[0][:, ix, :])
    _assert_close(yz_layer_h[1], xy_stack_h[1][:, ix, :])

    xz_e, xz_h, _, xz_z, xz_ranges, _, _ = obj.Solve_FieldXZ(y0=y0_value, znum=[2] * obj.Layer_N, components=('Ex', 'Hy'))
    yz_e, yz_h, _, yz_z, yz_ranges, _, _ = obj.Solve_FieldYZ(x0=x0_value, znum=[2] * obj.Layer_N, components=('Ex', 'Hy'))

    start1, stop1 = xz_ranges[1]
    start2, stop2 = yz_ranges[1]
    xz_layer_ref_e, _, _, _ = obj.Solve_FieldXZLayer(1, [0.0, obj.thickness_list[1]], y0=y0_value, components=('Ex',))
    yz_layer_ref_e, _, _, _ = obj.Solve_FieldYZLayer(1, [0.0, obj.thickness_list[1]], x0=x0_value, components=('Ex',))
    _assert_close(xz_e[0][start1:stop1], xz_layer_ref_e[0])
    _assert_close(yz_e[0][start2:stop2], yz_layer_ref_e[0])
    assert torch.isclose(xz_z[xz_ranges[0][1] - 1], xz_z[xz_ranges[1][0]])
    assert torch.isclose(yz_z[yz_ranges[0][1] - 1], yz_z[yz_ranges[1][0]])

    state_path = tmp_path / "baseline_state.pt"
    obj.BuildSMatrixCache()
    obj.BuildAmplitudeCache()
    obj.SaveState(state_path, include_caches=True)
    loaded = grcwa.obj.LoadState(state_path, map_location='cpu')
    loaded_e, loaded_h, _, _ = loaded.Solve_FieldXZLayer(1, [0.0, 0.04], y0=y0_value, components=('Ex', 'Hy'))
    _assert_close(loaded_e[0], xz_layer_e[0])
    _assert_close(loaded_h[1], xz_layer_h[1])


def test_structure_xz_uses_mesh_like_z_sampling():
    obj, _ = rcwa_small_assembly()
    _, _, _, _, xz_ranges, _, xz_step = obj.Solve_FieldXZ(y0=0.0, znum=3, components=('Ex',))

    expected_z_step = float(torch.min(torch.stack([torch.as_tensor(t, dtype=obj.dtype_f) for t in obj.thickness_list])).item()) / 2.0
    assert xz_step == expected_z_step

    counts = [stop - start for start, stop in xz_ranges]
    assert counts == [5, 6, 3, 7, 4]

    _, _, _, xz_z, xz_ranges, xz_edges, _ = obj.Solve_FieldXZ(y0=0.0, znum=3, components=('Ex',))
    for layer_index, (start, stop) in enumerate(xz_ranges):
        local_z = xz_z[start:stop] - xz_edges[layer_index]
        dz = local_z[1:] - local_z[:-1]
        if counts[layer_index] == 3:
            _assert_close(dz, torch.full_like(dz, expected_z_step))
        else:
            _assert_close(dz[:-1], torch.full_like(dz[:-1], expected_z_step))
            assert dz[-1] <= expected_z_step + 1e-12

    _, _, _, _, xz_fixed_ranges, _, xz_fixed_step = obj.Solve_FieldXZ(y0=0.0, znum=3, z_step=0.05, components=('Ex',))
    fixed_counts = [stop - start for start, stop in xz_fixed_ranges]
    assert fixed_counts == [4, 5, 3, 6, 4]
    assert xz_fixed_step == 0.05


def test_structure_xz_mesh_matches_layerwise_field_xy():
    obj, _ = rcwa_small_assembly()
    Nx_small, Ny_small = obj.GridLayer_Nxy_list[obj.id_list[1][3]]
    iy = Ny_small // 3
    y0_value = torch.as_tensor(obj.L2[1], dtype=obj.dtype_f, device=obj.device) * iy / Ny_small

    xz_e, xz_h, _, xz_z, xz_ranges, xz_edges, _ = obj.Solve_FieldXZ(y0=y0_value, znum=3, components=('Ex', 'Hy'))
    layer_index = 1
    start, stop = xz_ranges[layer_index]
    local_z = xz_z[start:stop] - xz_edges[layer_index]

    xz_layer_e, xz_layer_h, _, _ = obj.Solve_FieldXZLayer(layer_index, local_z, y0=y0_value, components=('Ex', 'Hy'))
    xy_layer_e, xy_layer_h = obj.Solve_FieldXY(layer_index, local_z, components=('Ex', 'Hy'))

    _assert_close(xz_e[0][start:stop], xz_layer_e[0])
    _assert_close(xz_h[1][start:stop], xz_layer_h[1])
    _assert_close(xz_layer_e[0], xy_layer_e[0][:, :, iy])
    _assert_close(xz_layer_h[1], xy_layer_h[1][:, :, iy])
