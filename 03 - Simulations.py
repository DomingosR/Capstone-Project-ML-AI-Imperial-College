import numpy as np

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

from scipy.optimize import minimize
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


AcqType = Literal["ucb", "ei", "pi"]

@dataclass
class BOConfig:
    acq: AcqType = "ucb"
    # UCB: bigger kappa -> more exploration
    kappa: float = 2.0
    # EI/PI: exploration parameter (sometimes called xi)
    xi: float = 1e-3

    # Optimization of acquisition:
    n_restarts: int = 32          # multi-start runs
    n_raw_samples: int = 4096     # random points used to seed restarts
    maxiter: int = 200

    # GP settings:
    alpha: float = 1e-10          # added to diagonal for numerical stability (if no WhiteKernel)
    normalize_y: bool = True
    random_state: Optional[int] = None


def _fit_gp(X: np.ndarray, y: np.ndarray, cfg: BOConfig) -> GaussianProcessRegressor:
    """
    ARD RBF + noise, fitted by maximizing log-marginal likelihood.
    X must be in [0,1]^n.
    """
    n = X.shape[1]

    # ARD length-scales: one per dimension
    kernel = (
        C(1.0, (1e-3, 1e3)) *
        RBF(length_scale=np.ones(n), length_scale_bounds=(1e-3, 1e3)) +
        WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=cfg.alpha,
        normalize_y=cfg.normalize_y,
        n_restarts_optimizer=5,
        random_state=cfg.random_state,
    )
    gp.fit(X, y)
    return gp


def _predict(gp: GaussianProcessRegressor, Xcand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu, std = gp.predict(Xcand, return_std=True)
    std = np.maximum(std, 1e-12)  # guard against zero std
    return mu, std


def acquisition(
    gp: GaussianProcessRegressor,
    Xcand: np.ndarray,
    y_best: float,
    cfg: BOConfig,
) -> np.ndarray:
    """
    Returns acquisition values to MAXIMIZE.
    """
    mu, std = _predict(gp, Xcand)

    if cfg.acq == "ucb":
        return mu + cfg.kappa * std

    # For EI / PI we assume MAXIMIZATION of f.
    imp = mu - y_best - cfg.xi
    Z = imp / std

    if cfg.acq == "ei":
        return imp * norm.cdf(Z) + std * norm.pdf(Z)

    if cfg.acq == "pi":
        return norm.cdf(Z)

    raise ValueError(f"Unknown acquisition: {cfg.acq}")


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def suggest_next_point(
    X: np.ndarray,
    y: np.ndarray,
    cfg: BOConfig = BOConfig(),
) -> Tuple[np.ndarray, GaussianProcessRegressor]:
    """
    Fit GP on (X,y) and return x_next in [0,1]^n that maximizes the acquisition.
    Assumes objective is to MAXIMIZE f.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("Shapes must be X: (k,n), y: (k,)")

    n = X.shape[1]
    if not (2 <= n <= 8):
        raise ValueError("This helper is intended for 2 <= n <= 8 as you specified.")

    if np.any(X < 0) or np.any(X > 1):
        raise ValueError("All X must lie in the unit cube [0,1]^n.")

    gp = _fit_gp(X, y, cfg)
    y_best = float(np.max(y))

    # 1) sample random points to find good starting locations
    rng = np.random.default_rng(cfg.random_state)
    raw = rng.random((cfg.n_raw_samples, n))
    raw_acq = acquisition(gp, raw, y_best, cfg)
    top_idx = np.argsort(raw_acq)[-cfg.n_restarts:]
    starts = raw[top_idx]

    bounds = [(0.0, 1.0)] * n

    def objective(x1d: np.ndarray) -> float:
        x1d = _clip01(np.asarray(x1d, dtype=float))
        val = acquisition(gp, x1d.reshape(1, -1), y_best, cfg)[0]
        return -float(val)  # minimize negative acquisition

    best_x = None
    best_val = np.inf

    for x0 in starts:
        res = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": cfg.maxiter},
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best_x = _clip01(res.x)

    # Fallback if optimizer fails for some reason:
    if best_x is None:
        best_x = starts[-1]

    return best_x, gp

def bayesopt_loop(
    f: Callable[[np.ndarray], float],
    X_init: np.ndarray,
    y_init: np.ndarray,
    m: int,
    cfg: BOConfig = BOConfig(),
    return_history: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run m sequential BO steps (you call f each time).
    Returns updated (X, y) including the new m points.
    """
    X = np.asarray(X_init, dtype=float).copy()
    y = np.asarray(y_init, dtype=float).ravel().copy()
    points = []

    for t in range(m):
        x_next, _gp = suggest_next_point(X, y, cfg)
        y_next = float(f(x_next))

        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.append(y, y_next)
        points.append([t, x_next, y_next])
        # print(f"Step {t+1}/{m}: x_next={x_next}, y_next={y_next}")

    return X, y, points

def kappaExperiment():
    # Experiment parameters
    PI             = 3.14159265358979323846
    criticalPoints = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], \
                      [0.4, 0.7, 0.4, 0.7, 0.4, 0.7, 0.4, 0.7], \
                      [0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3]]
    kappas         = [2.0, 2.5, 3.0, 3.5, 4.0]
    inputCount     = [10, 15, 20, 20, 30, 35, 40]

    # File to save results
    outputFileName = "..\\Data\\Output.txt"
    outputFile = open(outputFileName, "a")
    
    # Auxiliary functions for the synthetic example
    def dist(x: np.ndarray, x0: np.ndarray) -> float:
        return np.linalg.norm(x - x0)
    
    def auxFunc(x: np.ndarray, x0: np.ndarray, sigma: float) -> float:
        d = dist(x, x0)
        arg = - (d ** 2) / (2 * sigma ** 2)
        return np.exp(arg) / (2 * PI * sigma)

    # Synthetic example: maximize a function on [0,1] ^ numVars
    def f_oracle(x: np.ndarray, numVars) -> float:
        C = [1, 2, 7]
        points = [point[ : numVars] for point in criticalPoints]
        X0 = [np.array(point) for point in points]
        sigma = [0.2, 0.3, 0.1]

        # Sum of exponentials
        x = np.asarray(x)
        return float(sum(C[i] * auxFunc(x, X0[i], sigma[i]) for i in range(3)))
    
    def f_callable(x: np.ndarray) -> float:
        return f_oracle(x, numVars)
        
    for numVars in range(2, 9):
        k = inputCount[numVars - 2]
        rng = np.random.default_rng(0)

        X0 = rng.random((k, numVars))
        y0 = np.array([f_oracle(x, numVars) for x in X0])

        for kappaVal in kappas:
            cfg = BOConfig(acq="ucb", kappa = kappaVal, random_state = 0)
            X_final, y_final, points = bayesopt_loop(f_callable, X0, y0, m = 20, cfg = cfg)
            outputFile.write("Variables:" + str(numVars) + "\n")
            outputFile.write("Kappa: " + str(kappaVal) + "\n")
            outputFile.write("Points: " + str(points) + "\n")

            best_idx = np.argmax(y_final)
            outputFile.write("Initial x values: " + str(X0) + "\n")
            outputFile.write("Initial y values: " + str(y0) + "\n")
            outputFile.write("\nBest found:\n")
            outputFile.write("x* = " + str(X_final[best_idx]) + "\n")
            outputFile.write("f(x*) = " + str(y_final[best_idx]) + "\n")

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    kappaExperiment()