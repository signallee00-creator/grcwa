from pathlib import Path
import re

import torch

nm = 1e-9
um = 1e-6


class material:
    def __init__(
        self,
        lamb0,
        data_root=None,
        dtype_f=torch.float64,
        dtype_c=torch.complex128,
        device='cpu',
    ):
        self.lamb0 = torch.as_tensor(lamb0, dtype=dtype_f, device=device)
        self.dtype_f = dtype_f
        self.dtype_c = dtype_c
        self.device = torch.device(device)

        if data_root is None:
            self.data_root = Path(__file__).resolve().parent / 'nk'
        else:
            self.data_root = Path(data_root)

        self._cache = {}

    @classmethod
    def available_materials(cls, deck='nk_xx', data_root=None):
        if data_root is None:
            deck_dir = Path(__file__).resolve().parent / 'nk' / str(deck)
        else:
            deck_dir = Path(data_root) / str(deck)

        if not deck_dir.exists():
            return []

        names = set()
        for path in deck_dir.glob('*_index*.txt'):
            parsed = cls._parse_material_filename(path.name)
            if parsed is not None:
                names.add(parsed['name'])
        return sorted(names)

    @staticmethod
    def _parse_material_filename(filename):
        match = re.match(r'^(?P<name>.+)_index(?:_(?P<date>\d{6}))?\.txt$', filename)
        if match is None:
            return None
        return {
            'name': match.group('name'),
            'date': match.group('date'),
        }

    def _deck_dir(self, deck):
        return self.data_root / str(deck)

    def _resolve_file(self, name, deck, filename=None):
        deck_dir = self._deck_dir(deck)

        if filename is not None:
            path = Path(filename)
            if not path.is_absolute():
                path = deck_dir / path
            if not path.exists():
                raise FileNotFoundError(f'Material file not found: {path}')
            return path

        if not deck_dir.exists():
            raise FileNotFoundError(f'Material deck folder not found: {deck_dir}')

        dated = []
        plain = []
        for path in deck_dir.glob(f'{name}_index*.txt'):
            parsed = self._parse_material_filename(path.name)
            if parsed is None or parsed['name'] != name:
                continue
            if parsed['date'] is None:
                plain.append(path)
            else:
                dated.append((parsed['date'], path))

        if dated:
            dated.sort(key=lambda item: item[0])
            return dated[-1][1]

        if plain:
            plain.sort()
            return plain[-1]

        raise FileNotFoundError(f'No material file found for {name!r} in {deck_dir}')

    def _load_nk_table(self, filepath):
        wavelength = []
        n_values = []
        k_values = []

        for raw_line in filepath.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('#') or line.startswith('//') or line.startswith(';'):
                continue

            parts = re.split(r'[\s,]+', line)
            if len(parts) < 3:
                continue

            try:
                wl_value = float(parts[0])
                n_value = float(parts[1])
                k_value = float(parts[2])
            except ValueError:
                continue

            wavelength.append(wl_value)
            n_values.append(n_value)
            k_values.append(k_value)

        if not wavelength:
            raise ValueError(f'No numeric WL/n/k rows found in {filepath}')

        wavelength = torch.as_tensor(wavelength, dtype=self.dtype_f, device=self.device)
        n_values = torch.as_tensor(n_values, dtype=self.dtype_f, device=self.device)
        k_values = torch.as_tensor(k_values, dtype=self.dtype_f, device=self.device)

        order = torch.argsort(wavelength)
        return wavelength[order], n_values[order], k_values[order]

    def _dataset(self, name, deck='nk_xx', filename=None):
        key = (str(name), str(deck), None if filename is None else str(filename))
        if key not in self._cache:
            filepath = self._resolve_file(name, deck, filename=filename)
            wavelength_nm, n_data, k_data = self._load_nk_table(filepath)
            self._cache[key] = {
                'name': str(name),
                'deck': str(deck),
                'file': filepath,
                'wavelength_nm': wavelength_nm,
                'n': n_data,
                'k': k_data,
            }
        return self._cache[key]

    def _wavelength_to_nm(self, wavelength):
        if wavelength is None:
            wavelength = self.lamb0
        wavelength = torch.as_tensor(wavelength, dtype=self.dtype_f, device=self.device)
        return wavelength / nm

    def _interp_linear_extrap(self, x_data, y_data, x_query):
        if x_data.numel() == 1:
            return torch.ones_like(x_query, dtype=self.dtype_f, device=self.device) * y_data[0]

        x_query_flat = x_query.reshape(-1)
        index = torch.searchsorted(x_data, x_query_flat, right=True)
        index = torch.clamp(index, 1, x_data.numel() - 1)

        x0 = x_data[index - 1]
        x1 = x_data[index]
        y0 = y_data[index - 1]
        y1 = y_data[index]

        slope = (y1 - y0) / (x1 - x0)
        y_query = y0 + slope * (x_query_flat - x0)
        return y_query.reshape(x_query.shape)

    def n(self, name, deck='nk_xx', wavelength=None, filename=None):
        data = self._dataset(name, deck=deck, filename=filename)
        wavelength_nm = self._wavelength_to_nm(wavelength)
        return self._interp_linear_extrap(data['wavelength_nm'], data['n'], wavelength_nm)

    def k(self, name, deck='nk_xx', wavelength=None, filename=None):
        data = self._dataset(name, deck=deck, filename=filename)
        wavelength_nm = self._wavelength_to_nm(wavelength)
        return self._interp_linear_extrap(data['wavelength_nm'], data['k'], wavelength_nm)

    def nk(self, name, deck='nk_xx', wavelength=None, filename=None):
        n_value = self.n(name, deck=deck, wavelength=wavelength, filename=filename)
        k_value = self.k(name, deck=deck, wavelength=wavelength, filename=filename)
        return n_value.to(dtype=self.dtype_c) + 1j * k_value.to(dtype=self.dtype_c)

    def eps(self, name, deck='nk_xx', wavelength=None, filename=None):
        index = self.nk(name, deck=deck, wavelength=wavelength, filename=filename)
        return index * index

    def summary(self, name, deck='nk_xx', filename=None):
        data = self._dataset(name, deck=deck, filename=filename)
        return {
            'name': data['name'],
            'deck': data['deck'],
            'file': str(data['file']),
            'wavelength_nm_min': float(data['wavelength_nm'][0].item()),
            'wavelength_nm_max': float(data['wavelength_nm'][-1].item()),
            'num_rows': int(data['wavelength_nm'].numel()),
        }

    def __getattr__(self, attr):
        for prefix in ('n_', 'k_', 'nk_', 'eps_'):
            if attr.startswith(prefix):
                quantity = prefix[:-1]
                payload = attr[len(prefix):]
                if '_' not in payload:
                    break
                name, suffix = payload.rsplit('_', 1)
                deck = f'nk_{suffix}'

                def getter(wavelength=None, filename=None, _quantity=quantity, _name=name, _deck=deck):
                    method = getattr(self, _quantity)
                    return method(_name, deck=_deck, wavelength=wavelength, filename=filename)

                return getter
        raise AttributeError(f'{type(self).__name__!s} object has no attribute {attr!r}')
