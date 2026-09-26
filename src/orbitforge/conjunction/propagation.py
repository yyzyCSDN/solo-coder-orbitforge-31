from __future__ import annotations
from orbitforge.core.errors import ValidationError
from orbitforge.core.vector import Vec3
from orbitforge.analysis.uncertainty import linear_covariance
from .covariance import validate_covariance, matrix_add
from .encounter import encounter_frame, time_of_closest_approach

def linear_state_transition(dt_s):
    # straight-line relative motion: r(t) = r0 + v0*dt, v(t) = v0
    m = [[0.0] * 6 for _ in range(6)]
    for i in range(6):
        m[i][i] = 1.0
    for i in range(3):
        m[i][i + 3] = float(dt_s)
    return m

def propagate_covariance(cov6, dt_s):
    # Phi P Phi^T with the linear-motion STM; input must be a valid 6x6 covariance
    cov = validate_covariance(cov6, size=6, name='6d covariance')
    return linear_covariance(linear_state_transition(dt_s), cov)

def bplane_projection_matrix(rel_r: Vec3, rel_v: Vec3):
    # rows are the in-plane B-plane axes (encounter frame y, z) in inertial components
    _, y, z = encounter_frame(rel_r, rel_v)
    return [list(y.as_tuple()), list(z.as_tuple())]

def project_to_bplane(cov6_at_tca, rel_r: Vec3, rel_v: Vec3):
    # position block of the 6x6 at TCA, projected onto the B-plane -> 2x2
    cov = validate_covariance(cov6_at_tca, size=6, name='6d covariance at TCA')
    position_block = [row[:3] for row in cov[:3]]
    out = linear_covariance(bplane_projection_matrix(rel_r, rel_v), position_block)
    return ((out[0][0], out[0][1]), (out[1][0], out[1][1]))

def covariance_2d_at_tca(rel_r: Vec3, rel_v: Vec3, *covariances_6d, dt_to_tca=None):
    # propagate each object's 6x6 to TCA (independent errors -> sum), project to B-plane
    if not covariances_6d:
        raise ValidationError('at least one 6d covariance is required')
    dt = time_of_closest_approach(rel_r, rel_v) if dt_to_tca is None else float(dt_to_tca)
    combined = None
    for cov in covariances_6d:
        propagated = propagate_covariance(cov, dt)
        combined = propagated if combined is None else matrix_add(combined, propagated)
    r_tca = rel_r + rel_v * dt
    return project_to_bplane(combined, r_tca, rel_v), r_tca, dt
