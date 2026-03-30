"""Top-level package for grcwa."""
from .fft_funs import (
    Epsilon_fft,
    get_fft,
    get_ifft,
    get_ifft_batch,
    get_ifft_xline,
    get_ifft_xline_batch,
    get_ifft_yline,
    get_ifft_yline_batch,
)
from .kbloch import Lattice_Reciprocate, Lattice_getG, Lattice_SetKs
from .rcwa import obj

__author__ = """Weiliang Jin"""
__email__ = 'jwlaaa@gmail.com'
__version__ = '0.2.0'
