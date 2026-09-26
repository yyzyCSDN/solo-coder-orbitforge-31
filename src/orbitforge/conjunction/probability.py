from __future__ import annotations
import math
from .covariance import inverse, determinant, validate_covariance
from .encounter import project_to_plane
from .propagation import covariance_2d_at_tca

def collision_probability_2d(miss_x_km: float, miss_y_km: float, cov, hard_body_radius_km: float, samples: int=4000):
    validate_covariance(cov, size=2, name='b-plane covariance')
    inv = inverse(cov)
    det = determinant(cov)
    r = hard_body_radius_km
    total = 0.0
    n = max(20, int(math.sqrt(samples)))
    for i in range(n):
        x = -r + 2 * r * (i + 0.5) / n
        for j in range(n):
            y = -r + 2 * r * (j + 0.5) / n
            if x * x + y * y > r * r:
                continue
            dx = x - miss_x_km
            dy = y - miss_y_km
            q = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy)
            total += math.exp(-0.5 * q) / (2 * math.pi * math.sqrt(det))
    return total * (2 * r / n) ** 2

def collision_probability_at_tca(rel_r, rel_v, primary_cov6, secondary_cov6, hard_body_radius_km: float, samples: int=4000, dt_to_tca=None):
    # full chain: propagate both 6x6 covariances to TCA, combine, project to B-plane, integrate
    cov2, r_tca, dt = covariance_2d_at_tca(rel_r, rel_v, primary_cov6, secondary_cov6, dt_to_tca=dt_to_tca)
    miss_y, miss_z = project_to_plane(r_tca, rel_v)
    pc = collision_probability_2d(miss_y, miss_z, cov2, hard_body_radius_km, samples)
    return {
        'dt_to_tca_s': dt,
        'miss_distance_km': math.hypot(miss_y, miss_z),
        'bplane_covariance': cov2,
        'collision_probability': pc,
    }

def mahalanobis2(x, y, cov):
    inv = inverse(cov)
    return x * (inv[0][0] * x + inv[0][1] * y) + y * (inv[1][0] * x + inv[1][1] * y)
