import numpy as np
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import csv

AcqType = Literal["ucb", "ei", "pi"]

@dataclass
class BOConfig:
    acq: AcqType = "ucb"               # UCB: bigger kappa -> more exploration
    kappa: float = 2.0                 # EI/PI: exploration parameter (sometimes called xi)
    xi: float = 1e-3

    # Optimization of acquisition:
    n_restarts: int = 32               # multi-start runs
    n_raw_samples: int = 4096          # random points used to seed restarts
    maxiter: int = 200

    # GP settings:
    alpha: float = 1e-10               # added to diagonal for numerical stability (if no WhiteKernel)
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

def acquisition(gp: GaussianProcessRegressor, Xcand: np.ndarray, y_best: float, cfg: BOConfig, ) -> np.ndarray:
    """
    Returns acquisition values to maximize.
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

def suggest_next_point(X: np.ndarray, y: np.ndarray, cfg: BOConfig = BOConfig(), ) -> Tuple[np.ndarray, GaussianProcessRegressor]:
    """
    Fit GP on (X,y) and return x_next in [0,1]^n that maximizes the acquisition.
    Assumes objective is to maximize f.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("Shapes must be X: (k,n), y: (k,)")

    n = X.shape[1]
    if not (2 <= n <= 8):
        raise ValueError("This helper is intended for 2 <= n <= 8.")

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

def getSuggestedValuesFromData(inputFile):
    variableCount = [2, 2, 3, 4, 4, 5, 6, 8]

    # Reading input data
    with open(inputFile, newline='') as file:
        reader = csv.reader(file)
        inputData = list(map(list, reader))

    # Variable to hold suggested values
    suggestedValues  = []

    # Loop to get suggested values
    for i in range(1, 9):
        # Selecting data for current experiment
        filterData   = [row for row in inputData if row[0] == str(i)]
        dataPoints   = len(filterData)
        numVars      = variableCount[i - 1]
        
        # Creating variables 
        XData, YData = [[] for _ in range(dataPoints)], []
        for rowNum in range(dataPoints):
            row      = filterData[rowNum]
            for j in range(1, numVars + 1):
                XData[rowNum].append(float(row[j]))
            YData.append(float(row[-1]))
        
        # Get suggested point
        rng = np.random.default_rng(0)
        cfg = BOConfig(acq="ucb", kappa=2.5, random_state=0)
        X_final, y_final = suggest_next_point(XData, YData, cfg = cfg)

        # Converting to floating values
        suggestedXFloat = list(map(float, X_final))
        suggestedValues.append(tuple(suggestedXFloat))

    return suggestedValues

# ------------
# Main usage
# ------------
if __name__ == "__main__":
    # This assumes input data is in .csv format
    # Each row contains the experiment number, each of the 8 values for X
    # (blank if not applicable), and the value for the corresponding Y
    inputFile = "C:\\Users\\Domingos\\Documents\\Input.csv"
    suggestedValues = getSuggestedValuesFromData(inputFile)
    print(suggestedValues)