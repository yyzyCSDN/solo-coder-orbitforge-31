from __future__ import annotations
from orbitforge.analysis.uncertainty import linear_covariance
from .bplane import bplane_basis
from .covariance import validate_covariance

def rectilinear_stm(dt_s):
    d = float(dt_s)
    return ((1.0, 0.0, 0.0, d, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, d, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0, d),
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

def propagate_covariance(cov6, dt_s):
    p = validate_covariance(cov6, size=6)
    return validate_covariance(linear_covariance(rectilinear_stm(dt_s), p), size=6)

def bplane_projection(relative_velocity):
    _, t, r = bplane_basis(relative_velocity)
    return ((t.x, t.y, t.z, 0.0, 0.0, 0.0),
            (r.x, r.y, r.z, 0.0, 0.0, 0.0))

def project_to_bplane(cov6, relative_velocity):
    p = validate_covariance(cov6, size=6)
    return validate_covariance(linear_covariance(bplane_projection(relative_velocity), p), size=2)

def bplane_covariance_at_tca(cov6, dt_to_tca_s, relative_velocity):
    return project_to_bplane(propagate_covariance(cov6, dt_to_tca_s), relative_velocity)
