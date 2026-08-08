"""
curate/ate.py
=============
LinearATEModel — analytical OLS-based ATE with no retraining.

ATE = beta_T, the treatment coefficient in:
    Y = b_0 + b_T * T + b_X * X

Every add/remove operation is O(d^2) via:
  - Woodbury matrix identity  (exact)
  - 1st-order Neumann series  (approximate, faster)

Design rule
-----------
_dm_orig and _dm_cand are fixed lookup tables — never mutated after __init__.
_XTX_inv, _XTY, _YTY, and _n evolve as operations are applied (apply()).
"""

import numpy as np
import pandas as pd


def _build_dm(T: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """Design matrix  [1 | T | X]  with a 0-based integer index."""
    intercept = pd.Series(np.ones(len(T)), index=T.index, name='intercept')
    return pd.concat([intercept, T.rename('treatment'), X], axis=1)


class LinearATEModel:
    """
    OLS-based ATE model with fast analytical updates.

    Parameters
    ----------
    X_orig, T_orig, Y_orig : original dataset.
    X_cand, T_cand, Y_cand : candidate pool for mode='add' (e.g. synthetic data).
                             Pass None when only mode='remove' is needed.
    """

    def __init__(
        self,
        X_orig: pd.DataFrame,
        T_orig: pd.Series,
        Y_orig: pd.Series,
        X_cand: pd.DataFrame = None,
        T_cand: pd.Series    = None,
        Y_cand: pd.Series    = None,
    ):
        X_orig = X_orig.reset_index(drop=True)
        T_orig = T_orig.reset_index(drop=True)
        Y_orig = Y_orig.reset_index(drop=True)

        # Fixed lookup tables — never modified.
        self._dm_orig = _build_dm(T_orig, X_orig)
        self._Y_orig  = Y_orig

        if X_cand is not None:
            self._dm_cand = _build_dm(
                T_cand.reset_index(drop=True),
                X_cand.reset_index(drop=True),
            )
            self._Y_cand = Y_cand.reset_index(drop=True)
        else:
            self._dm_cand = None
            self._Y_cand  = None

        # Initial OLS fit.
        X_mat         = self._dm_orig.values
        Y_mat         = Y_orig.values.reshape(-1, 1)
        self._XTX_inv = np.linalg.pinv(X_mat.T @ X_mat)
        self._XTY     = X_mat.T @ Y_mat       # running sum — updated on every apply()
        self._beta    = self._XTX_inv @ self._XTY

        # Tracked for CI computation (updated on every apply()).
        self._n   = len(Y_orig)                                  # current #observations
        self._YTY = float(Y_mat.ravel() @ Y_mat.ravel())        # running sum of y²

    # ── Public ─────────────────────────────────────────────────────────────

    @property
    def ate(self) -> float:
        """Current ATE — treatment coefficient in OLS."""
        return float(self._beta[1])

    @property
    def n_obs(self) -> int:
        """Current number of observations (original ± rows added/removed)."""
        return self._n

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """
        95% CI for ATE using current OLS state — O(1), no refit needed.

        Uses the analytical identity:
            RSS = Y^T Y  −  β^T (X^T Y)
        Both terms are tracked incrementally in _YTY and _XTY.
        """
        from scipy.stats import t as t_dist
        p   = self._dm_orig.shape[1]            # fixed number of features
        df  = max(self._n - p, 1)
        rss = max(self._YTY - float((self._beta.T @ self._XTY).ravel()[0]), 0.0)
        se  = float(np.sqrt(rss / df * self._XTX_inv[1, 1]))
        tc  = float(t_dist.ppf(1 - alpha / 2, df=df))
        return self.ate - tc * se, self.ate + tc * se

    def influence_vec(self, indices: np.ndarray, mode: str, approx: bool = False) -> np.ndarray:
        """
        Vectorized influence for a subset of pool indices.
        Equivalent to [influence([i], mode, approx) for i in indices]
        but computed in a single matrix operation — O(d² + k·d) instead of O(k·d²).

        Returns ndarray of shape (len(indices),).

        Derivation (rank-1 Woodbury, remove mode):
            new_beta = beta + h_i · (ŷ_i − y_i) / (1 − s_i)
        where h_i = XTX_inv @ x_i,  s_i = x_i @ h_i,  ŷ_i = x_i @ beta
        influence_i = h_treat_i · (ŷ_i − y_i) / (1 − s_i)

        Add mode flips signs: (y_i − ŷ_i) / (1 + s_i).
        Neumann approximation drops the denominator correction.
        """
        dm = self._dm_orig if mode == 'remove' else self._dm_cand
        Y  = self._Y_orig  if mode == 'remove' else self._Y_cand

        X_mat = dm.values[indices]          # (k, d)
        Y_arr = Y.values[indices].ravel()   # (k,)

        H       = self._XTX_inv @ X_mat.T          # (d, k)
        h_treat = H[1, :]                           # (k,)
        S       = (X_mat * H.T).sum(axis=1)         # (k,) leverage scores
        Y_hat   = (X_mat @ self._beta).ravel()      # (k,) fitted values

        if approx:
            if mode == 'remove':
                return h_treat * (Y_hat - Y_arr * (1.0 + S))
            else:
                return h_treat * (Y_arr * (1.0 - S) - Y_hat)
        else:
            if mode == 'remove':
                denom = 1.0 - S
                denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
                return h_treat * (Y_hat - Y_arr) / denom
            else:
                denom = 1.0 + S
                denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
                return h_treat * (Y_arr - Y_hat) / denom

    def influence(self, indices: list[int], mode: str, approx: bool = False) -> float:
        """
        ΔATE = new_ate − current_ate for a hypothetical operation.
        Non-mutating — does not change internal state.

        Parameters
        ----------
        indices : indices in the candidate pool (mode='add') or original data (mode='remove').
        mode    : 'add' | 'remove'
        approx  : True → Neumann (fast). False → exact Woodbury.
        """
        if not indices:
            return 0.0
        X_rows, Y_rows       = self._lookup(indices, mode)
        new_inv, new_xty     = self._compute_update(X_rows, Y_rows, mode, approx)
        return float((new_inv @ new_xty)[1]) - self.ate

    def apply(self, indices: list[int], mode: str, approx: bool = False) -> float:
        """
        Permanently apply the operation and update internal state.
        Returns the new ATE.
        """
        if not indices:
            return self.ate
        X_rows, Y_rows           = self._lookup(indices, mode)
        self._XTX_inv, self._XTY = self._compute_update(X_rows, Y_rows, mode, approx)
        self._beta               = self._XTX_inv @ self._XTY
        y_sq = float(Y_rows.ravel() @ Y_rows.ravel())
        if mode == 'add':
            self._YTY += y_sq
            self._n   += len(indices)
        else:
            self._YTY -= y_sq
            self._n   -= len(indices)
        return self.ate

    # ── Internal ───────────────────────────────────────────────────────────

    def _lookup(self, indices: list[int], mode: str) -> tuple[np.ndarray, np.ndarray]:
        if mode == 'remove':
            dm, Y = self._dm_orig, self._Y_orig
        elif mode == 'add':
            if self._dm_cand is None:
                raise ValueError("mode='add' requires X_cand/T_cand/Y_cand at construction.")
            dm, Y = self._dm_cand, self._Y_cand
        else:
            raise ValueError(f"mode must be 'add' or 'remove', got '{mode}'")
        return dm.loc[indices].values, Y.loc[indices].values.reshape(-1, 1)

    def _compute_update(
        self, X_rows: np.ndarray, Y_rows: np.ndarray, mode: str, approx: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return updated (XTX_inv, XTY) without mutating self."""
        if mode == 'add':
            new_inv = self._neumann_add(X_rows)    if approx else self._woodbury_add(X_rows)
            new_xty = self._XTY + X_rows.T @ Y_rows
        else:
            new_inv = self._neumann_remove(X_rows) if approx else self._woodbury_remove(X_rows)
            new_xty = self._XTY - X_rows.T @ Y_rows
        return new_inv, new_xty

    # Woodbury rank-k identities ───────────────────────────────────────────

    def _woodbury_add(self, U: np.ndarray) -> np.ndarray:
        k   = U.shape[0]
        mid = np.linalg.inv(np.eye(k) + U @ self._XTX_inv @ U.T)
        return self._XTX_inv - self._XTX_inv @ U.T @ mid @ U @ self._XTX_inv

    def _woodbury_remove(self, U: np.ndarray) -> np.ndarray:
        k   = U.shape[0]
        mid = np.linalg.inv(np.eye(k) - U @ self._XTX_inv @ U.T)
        return self._XTX_inv + self._XTX_inv @ U.T @ mid @ U @ self._XTX_inv

    def _neumann_add(self, U: np.ndarray) -> np.ndarray:
        return self._XTX_inv - self._XTX_inv @ U.T @ U @ self._XTX_inv

    def _neumann_remove(self, U: np.ndarray) -> np.ndarray:
        return self._XTX_inv + self._XTX_inv @ U.T @ U @ self._XTX_inv


# ── LinearCATEModel ────────────────────────────────────────────────────────────

class LinearCATEModel:
    """
    Dual OLS model for CATE gap equalization.

    Maintains two LinearATEModel instances — one per condition group (0 and 1).
    The CATE gap is defined as:  gap = ATE_1 - ATE_0

    All candidate indices passed to the public API are positions in the FULL
    candidate pool (union of both groups).  Internally, each index is mapped to
    its per-group sub-pool before calling the appropriate sub-model.

    Parameters
    ----------
    X_{0,1}_ori, T_{0,1}_ori, Y_{0,1}_ori : original data split by condition group.
    X_{0,1}_cand, T_{0,1}_cand, Y_{0,1}_cand : synthetic candidates split by group
        (pass None when mode='remove' only).
    cand_group_mask : ndarray of shape (n_cand,) with values 0/1 — group of each
        candidate in the FULL pool.  Required when mode='add'.
    orig_group_mask : ndarray of shape (n_orig,) with values 0/1 — group of each
        original row.  Required when mode='remove' (orig rows are the candidate pool).
    """

    def __init__(
        self,
        X_0_ori: pd.DataFrame, T_0_ori: pd.Series, Y_0_ori: pd.Series,
        X_1_ori: pd.DataFrame, T_1_ori: pd.Series, Y_1_ori: pd.Series,
        X_0_cand: pd.DataFrame = None, T_0_cand: pd.Series = None, Y_0_cand: pd.Series = None,
        X_1_cand: pd.DataFrame = None, T_1_cand: pd.Series = None, Y_1_cand: pd.Series = None,
        cand_group_mask: np.ndarray = None,
        orig_group_mask: np.ndarray = None,
    ):
        self._model_0 = LinearATEModel(X_0_ori, T_0_ori, Y_0_ori, X_0_cand, T_0_cand, Y_0_cand)
        self._model_1 = LinearATEModel(X_1_ori, T_1_ori, Y_1_ori, X_1_cand, T_1_cand, Y_1_cand)

        # full-pool index → per-group local index, for candidate pool (add mode)
        self._cand_to_local_0: dict[int, int] = {}
        self._cand_to_local_1: dict[int, int] = {}
        if cand_group_mask is not None:
            g0 = g1 = 0
            for full, grp in enumerate(cand_group_mask):
                if int(grp) == 0:
                    self._cand_to_local_0[full] = g0; g0 += 1
                else:
                    self._cand_to_local_1[full] = g1; g1 += 1

        # full-pool index → per-group local index, for original data (remove mode)
        self._orig_to_local_0: dict[int, int] = {}
        self._orig_to_local_1: dict[int, int] = {}
        if orig_group_mask is not None:
            g0 = g1 = 0
            for full, grp in enumerate(orig_group_mask):
                if int(grp) == 0:
                    self._orig_to_local_0[full] = g0; g0 += 1
                else:
                    self._orig_to_local_1[full] = g1; g1 += 1

    # ── Public ─────────────────────────────────────────────────────────────

    @property
    def gap(self) -> float:
        """Current CATE gap: ATE_1 - ATE_0."""
        return self._model_1.ate - self._model_0.ate

    @property
    def ate_0(self) -> float:
        return self._model_0.ate

    @property
    def ate_1(self) -> float:
        return self._model_1.ate

    def ci_0(self, alpha: float = 0.05) -> tuple[float, float]:
        """95% CI for ATE of group 0."""
        return self._model_0.ci(alpha)

    def ci_1(self, alpha: float = 0.05) -> tuple[float, float]:
        """95% CI for ATE of group 1."""
        return self._model_1.ci(alpha)

    def gap_influence_vec(self, indices: np.ndarray, mode: str, approx: bool = False) -> np.ndarray:
        """
        Vectorised gap influence for an array of full-pool indices.
        Returns ndarray of shape (len(indices),).

        group-0 records: gap_influence = -delta_ATE_0  (reduces ATE_0 → widens gap)
        group-1 records: gap_influence = +delta_ATE_1  (raises ATE_1 → widens gap)
        """
        if mode == 'add':
            to_local_0, to_local_1 = self._cand_to_local_0, self._cand_to_local_1
        else:
            to_local_0, to_local_1 = self._orig_to_local_0, self._orig_to_local_1

        result = np.zeros(len(indices), dtype=np.float64)

        pos_0 = [i for i, idx in enumerate(indices) if int(idx) in to_local_0]
        pos_1 = [i for i, idx in enumerate(indices) if int(idx) in to_local_1]

        if pos_0:
            local_0 = np.array([to_local_0[int(indices[i])] for i in pos_0], dtype=np.intp)
            result[pos_0] = -self._model_0.influence_vec(local_0, mode, approx)

        if pos_1:
            local_1 = np.array([to_local_1[int(indices[i])] for i in pos_1], dtype=np.intp)
            result[pos_1] = self._model_1.influence_vec(local_1, mode, approx)

        return result

    def gap_influence(self, full_idx: int, mode: str, approx: bool = False) -> float:
        """
        Predicted change in gap (ATE_1 - ATE_0) if this candidate is applied.

        Records in group 0 only affect ATE_0 → gap_influence = -delta_ATE_0.
        Records in group 1 only affect ATE_1 → gap_influence = +delta_ATE_1.
        """
        grp, local = self._resolve(full_idx, mode)
        if grp == 0:
            return -self._model_0.influence([local], mode, approx)
        else:
            return self._model_1.influence([local], mode, approx)

    def apply(self, full_idx: int, mode: str, approx: bool = False) -> float:
        """Permanently apply operation to the correct group model. Returns new gap."""
        grp, local = self._resolve(full_idx, mode)
        if grp == 0:
            self._model_0.apply([local], mode, approx)
        else:
            self._model_1.apply([local], mode, approx)
        return self.gap

    # ── Internal ───────────────────────────────────────────────────────────

    def _resolve(self, full_idx: int, mode: str) -> tuple[int, int]:
        """Return (group, local_index) for a full-pool index."""
        if mode == 'add':
            if full_idx in self._cand_to_local_0:
                return 0, self._cand_to_local_0[full_idx]
            if full_idx in self._cand_to_local_1:
                return 1, self._cand_to_local_1[full_idx]
        else:
            if full_idx in self._orig_to_local_0:
                return 0, self._orig_to_local_0[full_idx]
            if full_idx in self._orig_to_local_1:
                return 1, self._orig_to_local_1[full_idx]
        raise KeyError(f"Index {full_idx} not found in {mode} pool.")