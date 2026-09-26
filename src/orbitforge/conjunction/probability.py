from __future__ import annotations
import math
from .covariance import inverse, determinant, validate_covariance
from .encounter import time_of_closest_approach
from .bplane import bplane_coordinates
from .propagation import bplane_covariance_at_tca

def collision_probability_2d(miss_x_km: float, miss_y_km: float, cov, hard_body_radius_km: float, samples: int=4000):
    cov = validate_covariance(cov, size=2)
    inv = inverse(cov)
    det = determinant(cov)
    r = hard_body_radius_km
    area = math.pi * r * r
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

def collision_probability_from_state_covariance(rel_position_km, rel_velocity_km_s, cov6, hard_body_radius_km: float, tca_s: float=None, samples: int=4000):
    if tca_s is None:
        tca_s = time_of_closest_approach(rel_position_km, rel_velocity_km_s)
    cov_b = bplane_covariance_at_tca(cov6, tca_s, rel_velocity_km_s)
    miss = rel_position_km + rel_velocity_km_s * tca_s
    miss_x, miss_y = bplane_coordinates(miss, rel_velocity_km_s)
    return collision_probability_2d(miss_x, miss_y, cov_b, hard_body_radius_km, samples)

def mahalanobis2(x, y, cov):
    inv = inverse(cov)
    return x * (inv[0][0] * x + inv[0][1] * y) + y * (inv[1][0] * x + inv[1][1] * y)
