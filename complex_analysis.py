import sys
import numpy as np
from calculate_optical_flow import *

def compute_series_vectors(counter_series, image, view):
    
    rgb, optical_flow_derivatives, uv = compute_flow(image, view)

    def apply_complex(a):
        return np.complex(a[0], a[1])
    def apply_complex_conjugate(a):
        return np.complex(a[0], -a[1])

    optical_flow_derivatives /= np.expand_dims(counter_series,2)

    # flow derivative
    du_dv_complex_value = np.apply_along_axis(apply_complex, 2, optical_flow_derivatives)
    uv_complex = np.apply_along_axis(apply_complex, 2, uv)
    uv_complex_conjugate = np.apply_along_axis(apply_complex_conjugate, 2, uv)

    fl_ci = uv_complex_conjugate * du_dv_complex_value
    amplitude = fl_ci.imag
    real_part = fl_ci.real
    phase = np.angle(uv_complex)
    temporal = np.angle(fl_ci)

    return amplitude, real_part, phase, temporal, optical_flow_derivatives, rgb
