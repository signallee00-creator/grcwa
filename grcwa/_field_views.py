import numbers

import torch

from ._field_helpers import CORE_FIELD_NAMES, build_derived_fields
from .fft_funs import get_ifft_batch, get_ifft_xline_batch, get_ifft_yline_batch


def _as_z_tensor(z_offset, dtype_f, device):
    z = torch.as_tensor(z_offset, dtype=dtype_f, device=device)
    scalar_input = z.ndim == 0
    if scalar_input:
        z = z.reshape(1)
    return z, scalar_input


def field_coefficients_batched(obj, which_layer, z_offset):
    which_layer = obj._validate_layer_index(which_layer)
    z_tensor, scalar_input = _as_z_tensor(z_offset, obj.dtype_f, obj.device)
    ai0, bi0 = obj.GetAmplitudes_noTranslate(which_layer)

    q = obj.q_list[which_layer]
    phi = obj.phi_list[which_layer]
    kp = obj.kp_list[which_layer]
    thickness = torch.as_tensor(obj.thickness_list[which_layer], dtype=obj.dtype_f, device=obj.device)

    dz = z_tensor.reshape(-1, 1)
    q_row = q.reshape(1, -1)
    ai = ai0.reshape(1, -1) * torch.exp(1j * q_row * dz)
    bi = bi0.reshape(1, -1) * torch.exp(1j * q_row * (thickness - dz))

    fhxy = torch.matmul(ai + bi, torch.transpose(phi, 0, 1))
    fhx = fhxy[:, :obj.nG]
    fhy = fhxy[:, obj.nG:]

    tmp1 = (ai - bi) / (obj.omega * q_row)
    tmp2 = torch.matmul(tmp1, torch.transpose(phi, 0, 1))
    fexy = torch.matmul(tmp2, torch.transpose(kp, 0, 1))
    fey = -fexy[:, :obj.nG]
    fex = fexy[:, obj.nG:]

    fhz = (obj.kx.reshape(1, -1) * fey - obj.ky.reshape(1, -1) * fex) / obj.omega
    fez = (obj.ky.reshape(1, -1) * fhx - obj.kx.reshape(1, -1) * fhy) / obj.omega
    if obj.id_list[which_layer][0] == 0:
        fez = fez / obj.Uniform_ep_list[obj.id_list[which_layer][2]]
    else:
        epinv = obj.Patterned_epinv_list[obj.id_list[which_layer][2]]
        fez = torch.matmul(fez, torch.transpose(epinv, 0, 1))

    coeffs = torch.stack((fex, fey, fez, fhx, fhy, fhz), dim=1)
    return coeffs, scalar_input


def finalize_dict_output(out, scalar_input):
    if scalar_input:
        return {key: value[0] for key, value in out.items()}
    return out


def _default_layer_line_coords(obj, axis, which_layer):
    if not obj._layer_is_axis_aligned():
        raise ValueError('Auto-generated x/y coordinates are only available for axis-aligned orthogonal lattices')

    Nx, Ny = obj._default_grid_shape(which_layer)
    if axis == 'x':
        return torch.arange(Nx, dtype=obj.dtype_f, device=obj.device) / Nx * obj.L1[0]
    return torch.arange(Ny, dtype=obj.dtype_f, device=obj.device) / Ny * obj.L2[1]


def line_reduced_coords(obj, axis, which_layer, coords, fixed_coord):
    length_x, length_y = obj._basis_lengths()
    if coords is None:
        coords = _default_layer_line_coords(obj, axis, which_layer)

    coords = torch.as_tensor(coords, dtype=obj.dtype_f, device=obj.device)
    fixed_coord = torch.as_tensor(fixed_coord, dtype=obj.dtype_f, device=obj.device)
    if axis == 'x':
        return coords / length_x, fixed_coord / length_y
    return fixed_coord / length_x, coords / length_y


def _default_structure_line_coords(obj, axis):
    if not obj._layer_is_axis_aligned():
        raise ValueError('Auto-generated x/y coordinates are only available for axis-aligned orthogonal lattices')

    if axis == 'x':
        if obj.GridLayer_Nxy_list:
            count = max(shape[0] for shape in obj.GridLayer_Nxy_list)
        else:
            count = max(obj.nG, 64)
        return torch.arange(count, dtype=obj.dtype_f, device=obj.device) / count * obj.L1[0]

    if obj.GridLayer_Nxy_list:
        count = max(shape[1] for shape in obj.GridLayer_Nxy_list)
    else:
        count = max(obj.nG, 64)
    return torch.arange(count, dtype=obj.dtype_f, device=obj.device) / count * obj.L2[1]


def coerce_structure_line_coords(obj, axis, coords, fixed_coord):
    length_x, length_y = obj._basis_lengths()
    if coords is None:
        coords = _default_structure_line_coords(obj, axis)

    coords = torch.as_tensor(coords, dtype=obj.dtype_f, device=obj.device)
    fixed_coord = torch.as_tensor(fixed_coord, dtype=obj.dtype_f, device=obj.device)
    if axis == 'x':
        return coords, coords / length_x, fixed_coord / length_y
    return coords, coords / length_y, fixed_coord / length_x


def resolve_structure_z_sampling(obj, znum, z_step):
    if torch.is_tensor(znum):
        znum = znum.tolist()

    if isinstance(znum, numbers.Integral):
        min_znum = max(int(znum), 2)

        if z_step is None:
            positive_thicknesses = []
            for thickness in obj.thickness_list:
                thickness_value = float(torch.as_tensor(thickness, dtype=obj.dtype_f, device=obj.device).item())
                if thickness_value > 0.0:
                    positive_thicknesses.append(thickness_value)

            if positive_thicknesses and min_znum > 1:
                z_step_value = min(positive_thicknesses) / (min_znum - 1)
            else:
                z_step_value = None
        else:
            z_step_value = float(torch.as_tensor(z_step, dtype=obj.dtype_f, device=obj.device).item())
            if z_step_value <= 0.0:
                raise ValueError('z_step must be positive')

        return None, min_znum, z_step_value

    if z_step is not None:
        raise ValueError('z_step cannot be combined with a per-layer znum sequence')

    counts = [max(int(value), 2) for value in znum]
    if len(counts) != obj.Layer_N:
        raise ValueError('znum sequence must have one entry per layer')
    return counts, None, None


def build_layer_z_points(obj, thickness, count=None, z_step=None):
    thickness = torch.as_tensor(thickness, dtype=obj.dtype_f, device=obj.device)

    if z_step is None:
        return torch.linspace(0.0, thickness, count, dtype=obj.dtype_f, device=obj.device)

    if thickness.item() == 0.0:
        return torch.zeros((count,), dtype=obj.dtype_f, device=obj.device)

    if thickness.item() <= z_step * (count - 1):
        return torch.linspace(0.0, thickness, count, dtype=obj.dtype_f, device=obj.device)

    local_z = torch.arange(0.0, float(thickness.item()), z_step, dtype=obj.dtype_f, device=obj.device)
    if local_z.numel() == 0 or not torch.isclose(local_z[0], thickness.new_zeros(())):
        local_z = thickness.new_zeros((1,))

    if not torch.isclose(local_z[-1], thickness):
        local_z = torch.concatenate((local_z, thickness.reshape(1)))

    if local_z.numel() < count:
        return torch.linspace(0.0, thickness, count, dtype=obj.dtype_f, device=obj.device)

    return local_z


def build_structure_z_segments(obj, znum=32, z_step=None, duplicate_interfaces=True):
    z_counts, min_znum, mesh_z_step = resolve_structure_z_sampling(obj, znum, z_step)
    z_pieces = []
    segments = []
    layer_ranges = []
    layer_edges = [torch.zeros((), dtype=obj.dtype_f, device=obj.device)]
    z_offset = torch.zeros((), dtype=obj.dtype_f, device=obj.device)
    cursor = 0

    for layer_index, thickness in enumerate(obj.thickness_list):
        thickness = torch.as_tensor(thickness, dtype=obj.dtype_f, device=obj.device)
        if z_counts is None:
            local_z = build_layer_z_points(obj, thickness, count=min_znum, z_step=mesh_z_step)
        else:
            local_z = build_layer_z_points(obj, thickness, count=z_counts[layer_index], z_step=None)

        global_z = z_offset + local_z
        if layer_index > 0 and not duplicate_interfaces:
            local_z = local_z[1:]
            global_z = global_z[1:]

        start = cursor
        stop = cursor + local_z.numel()
        layer_ranges.append((start, stop))
        segments.append((layer_index, local_z))
        z_pieces.append(global_z)
        cursor = stop
        z_offset = z_offset + thickness
        layer_edges.append(z_offset)

    return torch.concatenate(z_pieces), segments, layer_ranges, torch.stack(layer_edges), mesh_z_step


def reconstruct_xy_core_fields(obj, coeffs, Nx, Ny, required_core):
    if not required_core:
        return {}

    indices = [CORE_FIELD_NAMES.index(name) for name in required_core]
    spatial = get_ifft_batch(Nx, Ny, coeffs[:, indices, :], obj.G, dtype_c=obj.dtype_c, device=obj.device)
    return {name: spatial[:, idx] for idx, name in enumerate(required_core)}


def reconstruct_line_core_fields(obj, axis, coeffs, scan_coords, fixed_coord, required_core):
    if not required_core:
        return {}

    indices = [CORE_FIELD_NAMES.index(name) for name in required_core]
    if axis == 'x':
        spatial = get_ifft_xline_batch(scan_coords, fixed_coord, coeffs[:, indices, :], obj.G, dtype_c=obj.dtype_c, device=obj.device)
    else:
        spatial = get_ifft_yline_batch(fixed_coord, scan_coords, coeffs[:, indices, :], obj.G, dtype_c=obj.dtype_c, device=obj.device)
    return {name: spatial[:, idx] for idx, name in enumerate(required_core)}


def build_spatial_output(core_fields, requested_core, requested_derived, scalar_input):
    out = {name: core_fields[name] for name in requested_core}
    out.update(build_derived_fields(core_fields, requested_derived))
    return finalize_dict_output(out, scalar_input)
