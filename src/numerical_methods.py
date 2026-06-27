import numpy as np

def _result(root, iterations, error, converged):
    return {
        "root":       root,
        "iterations": iterations,
        "error":      error,
        "converged":  converged,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROOT FINDING
# ─────────────────────────────────────────────────────────────────────────────

def bisection(f, a, b, tol=1e-6, max_iter=100):

    if f(a) * f(b) >= 0:
        raise ValueError(
            f"Root not bracketed: f({a})={f(a):.4f}, f({b})={f(b):.4f}. "
            "f(a) and f(b) must have opposite signs."
        )

    c = a  # ensure c is always defined even if max_iter=0
    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        error = (b - a) / 2.0

        if abs(fc) < tol or error < tol:
            return _result(c, i + 1, error, True)

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return _result(c, max_iter, (b - a) / 2.0, False)


def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):

    x = float(x0)

    for i in range(max_iter):
        fx  = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-14:
            raise ValueError(
                f"Derivative too close to zero at x={x:.6f} (df={dfx:.2e}). "
                "Choose a better initial guess."
            )

        x_new = x - fx / dfx
        error = abs(x_new - x)

        if error < tol:
            return _result(x_new, i + 1, error, True)

        x = x_new

    return _result(x, max_iter, abs(f(x)), False)


def secant(f, x0, x1, tol=1e-6, max_iter=100):

    x0, x1 = float(x0), float(x1)

    if abs(x1 - x0) < 1e-14:
        raise ValueError(
            "x0 and x1 must be distinct initial guesses."
        )

    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        denom = fx1 - fx0

        if abs(denom) < 1e-14:
            raise ValueError(
                f"Secant denominator too close to zero at iteration {i+1}. "
                "Try different initial guesses."
            )

        x2    = x1 - fx1 * (x1 - x0) / denom
        error = abs(x2 - x1)

        if error < tol:
            return _result(x2, i + 1, error, True)

        x0, x1 = x1, x2

    return _result(x1, max_iter, abs(f(x1)), False)


# ─────────────────────────────────────────────────────────────────────────────
# NUMERICAL DIFFERENTIATION
# ─────────────────────────────────────────────────────────────────────────────

def forward_difference(f, x, h=1e-5):

    return (f(x + h) - f(x)) / h


def central_difference(f, x, h=1e-5):

    return (f(x + h) - f(x - h)) / (2.0 * h)


# ─────────────────────────────────────────────────────────────────────────────
# NUMERICAL INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def trapezoidal(x, y):

    x = list(x)
    y = list(y)

    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length. "
            f"Got len(x)={len(x)}, len(y)={len(y)}."
        )
    if len(x) < 2:
        raise ValueError("Need at least 2 points to integrate.")

    area = 0.0
    for i in range(len(x) - 1):
        width  = x[i + 1] - x[i]
        height = (y[i] + y[i + 1]) / 2.0
        area  += width * height

    return area


def simpson(x, y):

    x = list(x)
    y = list(y)

    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length. "
            f"Got len(x)={len(x)}, len(y)={len(y)}."
        )
    if len(x) < 3:
        raise ValueError("Simpson's rule requires at least 3 points.")

    n = len(x) - 1  # number of intervals
    h = (x[-1] - x[0]) / n

    if n % 2 == 1:
        # Odd number of intervals: apply Simpson to all but the last,
        # then add the last interval via trapezoidal.
        core  = simpson(x[:-1], y[:-1])
        tail  = 0.5 * (y[-2] + y[-1]) * (x[-1] - x[-2])
        return core + tail

    total = y[0] + y[-1]
    for i in range(1, n):
        total += (4 if i % 2 == 1 else 2) * y[i]

    return total * h / 3.0


# ─────────────────────────────────────────────────────────────────────────────
# LINEAR SYSTEMS
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_elimination(A, b):

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Build augmented matrix [A | b]
    M = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find the row with the largest absolute value in this column
        pivot = col + np.argmax(np.abs(M[col:, col]))

        if abs(M[pivot, col]) < 1e-12:
            raise ValueError(
                f"Matrix is singular or nearly singular at column {col}."
            )

        # Swap current row with pivot row
        M[[col, pivot]] = M[[pivot, col]]

        # Eliminate entries below the pivot
        for row in range(col + 1, n):
            factor      = M[row, col] / M[col, col]
            M[row, col:] -= factor * M[col, col:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]

    return x
