from __future__ import annotations
import math
from orbitforge.core.errors import ValidationError

def validate_covariance(cov, size=None, sym_tol=1e-10, pd_tol=1e-12):
    rows = tuple(tuple(float(v) for v in row) for row in cov)
    n = len(rows)
    if n == 0:
        raise ValidationError('covariance matrix is empty')
    if size is not None and n != size:
        raise ValidationError(f'covariance must be {size}x{size}, got {n} rows')
    if any(len(row) != n for row in rows):
        raise ValidationError('covariance must be square')
    scale = max(1.0, max(abs(v) for row in rows for v in row))
    for i in range(n):
        if rows[i][i] <= 0.0:
            raise ValidationError(f'covariance has non-positive variance at index {i}')
        for j in range(i + 1, n):
            if abs(rows[i][j] - rows[j][i]) > sym_tol * scale:
                raise ValidationError(f'covariance is asymmetric at ({i}, {j})')
    _cholesky_check(rows, pd_tol * scale)
    return rows

def _cholesky_check(cov, min_pivot):
    n = len(cov)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = cov[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if s <= min_pivot:
                    raise ValidationError(f'covariance is not positive definite (pivot {i}={s:g})')
                lower[i][j] = math.sqrt(s)
            else:
                lower[i][j] = s / lower[j][j]

def rotate_covariance_2d(cov, angle):
    c, s = (math.cos(angle), math.sin(angle))
    r = ((c, -s), (s, c))
    return tuple((tuple((sum((r[i][k] * sum((cov[k][m] * r[j][m] for m in range(2))) for k in range(2))) for j in range(2))) for i in range(2)))

def combine_covariance(a, b):
    return ((a[0][0] + b[0][0], a[0][1] + b[0][1]), (a[1][0] + b[1][0], a[1][1] + b[1][1]))

def determinant(c):
    return c[0][0] * c[1][1] - c[0][1] * c[1][0]

def inverse(c):
    d = determinant(c)
    if d <= 0:
        raise ValueError('covariance must be positive definite')
    return ((c[1][1] / d, -c[0][1] / d), (-c[1][0] / d, c[0][0] / d))
