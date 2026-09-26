from __future__ import annotations
import math
from orbitforge.core.errors import ValidationError

def rotate_covariance_2d(cov, angle):
    c, s = (math.cos(angle), math.sin(angle))
    r = ((c, -s), (s, c))
    return tuple((tuple((sum((r[i][k] * sum((cov[k][m] * r[j][m] for m in range(2))) for k in range(2))) for j in range(2))) for i in range(2)))

def combine_covariance(a, b):
    return ((a[0][0] + b[0][0], a[0][1] + b[0][1]), (a[1][0] + b[1][0], a[1][1] + b[1][1]))

def matrix_add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a))] for i in range(len(a))]

def is_symmetric(cov, tol=1e-10):
    n = len(cov)
    for i in range(n):
        for j in range(i + 1, n):
            scale = max(1.0, abs(cov[i][j]), abs(cov[j][i]))
            if abs(cov[i][j] - cov[j][i]) > tol * scale:
                return False
    return True

def is_positive_definite(cov):
    # Cholesky; assumes symmetric (use validate_covariance for full checking)
    n = len(cov)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum((lower[i][k] * lower[j][k] for k in range(j)))
            if i == j:
                d = cov[i][i] - s
                if d <= 0.0:
                    return False
                lower[i][j] = math.sqrt(d)
            else:
                lower[i][j] = (cov[i][j] - s) / lower[j][j]
    return True

def validate_covariance(cov, size=None, tol=1e-10, name='covariance'):
    try:
        rows = [[float(v) for v in row] for row in cov]
    except (TypeError, ValueError):
        raise ValidationError(f'{name}: non-numeric element') from None
    n = len(rows)
    if n == 0 or any(len(row) != n for row in rows):
        raise ValidationError(f'{name}: not a non-empty square matrix')
    if size is not None and n != size:
        raise ValidationError(f'{name}: expected {size}x{size}, got {n}x{n}')
    if not all(math.isfinite(v) for row in rows for v in row):
        raise ValidationError(f'{name}: non-finite element')
    for i in range(n):
        for j in range(i + 1, n):
            scale = max(1.0, abs(rows[i][j]), abs(rows[j][i]))
            if abs(rows[i][j] - rows[j][i]) > tol * scale:
                raise ValidationError(f'{name}: asymmetric at ({i},{j})')
    if not is_positive_definite(rows):
        raise ValidationError(f'{name}: not positive definite')
    return rows

def determinant(c):
    return c[0][0] * c[1][1] - c[0][1] * c[1][0]

def inverse(c):
    validate_covariance(c, size=2, name='2d covariance')
    d = determinant(c)
    return ((c[1][1] / d, -c[0][1] / d), (-c[1][0] / d, c[0][0] / d))
