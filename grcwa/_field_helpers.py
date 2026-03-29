import torch


CORE_FIELD_NAMES = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
E_FIELD_NAMES = ('Ex', 'Ey', 'Ez')
H_FIELD_NAMES = ('Hx', 'Hy', 'Hz')
DERIVED_FIELD_NAMES = ('Px', 'Py', 'Pz', 'E2norm', 'Pnorm')


def normalize_field_requests(components, derived):
    if components is None:
        components = CORE_FIELD_NAMES
    if derived is None:
        derived = ()

    requested_core = []
    seen_core = set()
    for name in components:
        if name not in CORE_FIELD_NAMES:
            raise ValueError(f'Unknown field component: {name}')
        if name not in seen_core:
            requested_core.append(name)
            seen_core.add(name)

    requested_derived = []
    seen_derived = set()
    for name in derived:
        if name not in DERIVED_FIELD_NAMES:
            raise ValueError(f'Unknown derived field: {name}')
        if name not in seen_derived:
            requested_derived.append(name)
            seen_derived.add(name)

    required_core = set(requested_core)
    if 'E2norm' in requested_derived:
        required_core.update(E_FIELD_NAMES)
    if any(name in requested_derived for name in ('Px', 'Py', 'Pz', 'Pnorm')):
        required_core.update(CORE_FIELD_NAMES)

    required_core = [name for name in CORE_FIELD_NAMES if name in required_core]
    return requested_core, requested_derived, required_core


def build_derived_fields(core_fields, requested_derived):
    out = {}
    if not requested_derived:
        return out

    if 'E2norm' in requested_derived:
        ex = core_fields['Ex']
        ey = core_fields['Ey']
        ez = core_fields['Ez']
        out['E2norm'] = torch.abs(ex) ** 2 + torch.abs(ey) ** 2 + torch.abs(ez) ** 2

    p_components = None
    if any(name in requested_derived for name in ('Px', 'Py', 'Pz', 'Pnorm')):
        ex = core_fields['Ex']
        ey = core_fields['Ey']
        ez = core_fields['Ez']
        hx = core_fields['Hx']
        hy = core_fields['Hy']
        hz = core_fields['Hz']

        px = 0.5 * torch.real(ey * torch.conj(hz) - ez * torch.conj(hy))
        py = 0.5 * torch.real(ez * torch.conj(hx) - ex * torch.conj(hz))
        pz = 0.5 * torch.real(ex * torch.conj(hy) - ey * torch.conj(hx))
        p_components = {'Px': px, 'Py': py, 'Pz': pz}

        for name in ('Px', 'Py', 'Pz'):
            if name in requested_derived:
                out[name] = p_components[name]

    if 'Pnorm' in requested_derived:
        if p_components is None:
            raise RuntimeError('Pnorm requested without Poynting components being computed')
        out['Pnorm'] = torch.sqrt(
            p_components['Px'] ** 2 + p_components['Py'] ** 2 + p_components['Pz'] ** 2
        )

    return out
