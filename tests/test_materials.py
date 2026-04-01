from pathlib import Path

import torch

from materials import material


def write_table(path, rows):
    lines = ['WL n k']
    for wl, n_value, k_value in rows:
        lines.append(f'{wl} {n_value} {k_value}')
    path.write_text('\n'.join(lines), encoding='utf-8')


def test_material_uses_latest_dated_file(tmp_path):
    root = tmp_path
    deck = root / 'nk_xx'
    deck.mkdir()

    write_table(deck / 'SiO2_index.txt', [(100, 1.40, 0.0), (200, 1.50, 0.0)])
    write_table(deck / 'SiO2_index_260401.txt', [(100, 1.45, 0.0), (200, 1.55, 0.0)])
    write_table(deck / 'SiO2_index_260315.txt', [(100, 1.43, 0.0), (200, 1.53, 0.0)])

    mat = material(150e-9, data_root=root)

    assert Path(mat.summary('SiO2', deck='nk_xx')['file']).name == 'SiO2_index_260401.txt'
    assert torch.allclose(mat.n('SiO2', deck='nk_xx', wavelength=torch.tensor(150e-9)), torch.tensor(1.50, dtype=torch.float64))


def test_material_interpolates_nk_and_builds_eps(tmp_path):
    root = tmp_path
    deck = root / 'nk_xx'
    deck.mkdir()

    write_table(
        deck / 'Test_index.txt',
        [
            (100, 1.0, 0.0),
            (200, 2.0, 0.2),
            (300, 3.0, 0.4),
        ],
    )

    mat = material(150e-9, data_root=root)

    n_value = mat.n('Test', deck='nk_xx')
    k_value = mat.k('Test', deck='nk_xx')
    eps_value = mat.eps('Test', deck='nk_xx')

    assert torch.allclose(n_value, torch.tensor(1.5, dtype=torch.float64))
    assert torch.allclose(k_value, torch.tensor(0.1, dtype=torch.float64))
    assert torch.allclose(eps_value, (torch.tensor(1.5, dtype=torch.complex128) + 1j * torch.tensor(0.1, dtype=torch.complex128)) ** 2)


def test_material_extrapolates_from_end_segments(tmp_path):
    root = tmp_path
    deck = root / 'nk_xx'
    deck.mkdir()

    write_table(
        deck / 'Al2O3_index.txt',
        [
            (100, 1.0, 0.0),
            (200, 2.0, 0.2),
            (300, 3.0, 0.4),
        ],
    )

    mat = material(150e-9, data_root=root)

    n_low = mat.n('Al2O3', deck='nk_xx', wavelength=torch.tensor(50e-9))
    k_high = mat.k('Al2O3', deck='nk_xx', wavelength=torch.tensor(350e-9))

    assert torch.allclose(n_low, torch.tensor(0.5, dtype=torch.float64))
    assert torch.allclose(k_high, torch.tensor(0.5, dtype=torch.float64))


def test_available_materials_lists_unique_names(tmp_path):
    deck = tmp_path / 'nk_xx'
    deck.mkdir()

    write_table(deck / 'SiO2_index.txt', [(100, 1.4, 0.0), (200, 1.5, 0.0)])
    write_table(deck / 'SiO2_index_260401.txt', [(100, 1.45, 0.0), (200, 1.55, 0.0)])
    write_table(deck / 'SiN_index.txt', [(100, 2.0, 0.0), (200, 2.1, 0.0)])

    names = material.available_materials(data_root=tmp_path)
    assert names == ['SiN', 'SiO2']


def test_material_dynamic_methods_use_lamb0_by_default(tmp_path):
    root = tmp_path
    deck = root / 'nk_xx'
    deck.mkdir()

    write_table(
        deck / 'SiO2_index.txt',
        [
            (100, 1.0, 0.0),
            (200, 2.0, 0.2),
        ],
    )

    mat = material(150e-9, data_root=root)

    assert torch.allclose(mat.n_SiO2_xx(), torch.tensor(1.5, dtype=torch.float64))
    assert torch.allclose(mat.k_SiO2_xx(), torch.tensor(0.1, dtype=torch.float64))
    assert torch.allclose(mat.eps_SiO2_xx(), (torch.tensor(1.5, dtype=torch.complex128) + 1j * torch.tensor(0.1, dtype=torch.complex128)) ** 2)
