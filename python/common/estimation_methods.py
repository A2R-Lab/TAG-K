import numpy as np

# ============================== RLS ==============================

class RLS:
    """RLS with adaptive forgetting factor (lambda) using 1-D state."""
    def __init__(self, num_params, theta_hat=None, forgetting_factor=0.3, c=1000.0):
        self.n = num_params
        self.P = np.eye(num_params) * float(c)          # (n, n)
        self.theta_hat = (np.zeros(self.n) if theta_hat is None
                          else np.asarray(theta_hat).reshape(-1))       # (n,)
        self.lambda_ = float(forgetting_factor)

    def iterate(self, A, b):
        A = np.asarray(A)                                # (m, n)
        b = np.asarray(b).reshape(-1)                    # (m,)
        m, n = A.shape
        assert n == self.n, "A.shape[1] must match num_params"

        S = A @ self.P @ A.T + self.lambda_ * np.eye(m)  # (m, m)
        K = self.P @ A.T @ np.linalg.inv(S)              # (n, m)

        innov = b - A @ self.theta_hat                   # (m,)
        self.theta_hat = self.theta_hat + K @ innov      # (n,)
        self.P = (self.P - K @ A @ self.P) / self.lambda_
        return self.theta_hat


# ============================== KF ===============================

class KF:
    """
    Basic (flipped-model) Kalman filter treating parameters as the state.
    Uses 1-D theta_hat, 2-D P/Q/R.
    """
    def __init__(self, num_params, process_noise, measurement_noise, theta_hat=None, c=10.0):
        self.n = num_params
        self.theta_hat = (np.zeros(self.n) if theta_hat is None
                          else np.asarray(theta_hat).reshape(-1))       # (n,)
        self.P = np.eye(self.n) * float(c)     
                                 # (n, n)
        self.Q = np.asarray(process_noise, dtype=float)                 # (n, n)
        self.R = np.asarray(measurement_noise, dtype=float)             # (m, m) at runtime

    def iterate(self, A, b):
        A = np.asarray(A)                               # (m, n)
        b = np.asarray(b).reshape(-1)                   # (m,)
        m, n = A.shape
        assert n == self.n, "A.shape[1] must match num_params"
        assert self.R.shape == (m, m), "R must be (m, m) for current A"

        # Predict
        self.P = self.P + self.Q

        # Update
        S = A @ self.P @ A.T + self.R                   # (m, m)
        K = self.P @ A.T @ np.linalg.inv(S)             # (n, m)
        innov = b - A @ self.theta_hat                  # (m,)
        self.theta_hat = self.theta_hat + K @ innov     # (n,)
        self.P = self.P - K @ A @ self.P
        return self.theta_hat


# ============================== RK ===============================

class RK:
    """Classic Randomized Kaczmarz with 1-D state; m*n inner updates per call."""
    def __init__(self, num_params, x0=None):
        self.num_params = num_params
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0).reshape(-1)

    def iterate(self, A, b, eps=1e-12):
        A = np.asarray(A)                    # (m, n)
        b = np.asarray(b).reshape(-1)        # (m,)
        m, n = A.shape
        assert n == self.num_params

        x = self.x0.copy()                   # (n,)
        row_norms_sq = (A * A).sum(axis=1) + eps
        probs = row_norms_sq / row_norms_sq.sum()

        num_iter = m
        
        for _ in range(num_iter):
            i = np.random.choice(m, p=probs)
            ai = A[i]                        # (n,)
            residual = b[i] - ai @ x
            x = x + (residual / row_norms_sq[i]) * ai

        self.x0 = x
        return x

# ============================== RK ===============================

def ruiz_equilibrate(A, iters=5, eps=1e-12):
    m, n = A.shape
    Dr = np.ones(m); Dc = np.ones(n)
    B = A.copy()
    for _ in range(iters):
        r = np.sqrt((B*B).sum(axis=1)) + eps
        Dr *= 1.0/r
        B = (B.T / r).T
        c = np.sqrt((B*B).sum(axis=0)) + eps
        Dc *= 1.0/c
        B = B / c
    return B, np.diag(Dr), np.diag(Dc)

def block_rk_step(A, b, x, p=4):
    m = A.shape[0]
    idx = np.random.choice(m, size=p, replace=False, p=None)  # or by probs
    Ab = A[idx]; bb = b[idx]
    # project onto solution of Ab x = bb (least-squares)
    # x ← x + Ab^T (Ab Ab^T)^+ (bb - Ab x)
    G = Ab @ Ab.T
    x += Ab.T @ np.linalg.pinv(G) @ (bb - Ab @ x)
    return x

def tikhonov_augment(A, b, lam=1e-5):
    """
    Return augmented (A_aug, b_aug) for ridge: min ||A x - b||^2 + lam ||x||^2
      => [A; sqrt(lam) I] x = [b; 0]
    """
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)
    m, n = A.shape
    A_aug = np.vstack([A, np.sqrt(lam) * np.eye(n)])
    b_aug = np.concatenate([b, np.zeros(n)])
    return A_aug, b_aug



class RK_ColScaled:
    """
    Randomized Kaczmarz with simple RIGHT preconditioning (column scaling).
    Scales columns by their std-dev to reduce unit imbalance.
    """
    def __init__(self, num_params, x0=None):
        self.num_params = int(num_params)
        self.x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0).reshape(-1)

    def iterate(self, A, b, eps=1e-12):
        A = np.asarray(A, dtype=float)              # (m, n)
        b = np.asarray(b, dtype=float).reshape(-1)  # (m,)
        m, n = A.shape
        assert n == self.num_params

        # Right preconditioner: A_col = A * D^{-1}, where D = diag(col_scale)
        col_scale = A.std(axis=0) + 1e-12
        A_col = A / col_scale

        # Work in y = D x  (=> x = D^{-1} y)
        y = (self.x0 * col_scale).copy()

        # Row sampling ∝ ||row||^2 of preconditioned matrix
        row_norms_sq = (A_col * A_col).sum(axis=1) + eps
        probs = row_norms_sq / row_norms_sq.sum()

        num_iter = m
        for _ in range(num_iter):
            i = np.random.choice(m, p=probs)
            ai = A_col[i]                         # (n,)
            r  = b[i] - ai @ y
            y += (r / row_norms_sq[i]) * ai

        x = y / col_scale
        self.x0 = x
        return x


class RK_Equi:
    """
    Randomized Kaczmarz with RUIZ equilibration (LEFT + RIGHT preconditioning).
    Alternates row/column scalings to make A nearly isotropic.
    """
    def __init__(self, num_params, x0=None, ruiz_iters=5):
        self.num_params = int(num_params)
        self.x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0).reshape(-1)
        self.ruiz_iters = int(ruiz_iters)

    def _ruiz(self, A, iters=5, eps=1e-12):
        m, n = A.shape
        Dr = np.ones(m)
        Dc = np.ones(n)
        B = A.copy()
        for _ in range(iters):
            r = np.sqrt((B*B).sum(axis=1)) + eps
            Dr *= 1.0 / r
            B = (B.T / r).T
            c = np.sqrt((B*B).sum(axis=0)) + eps
            Dc *= 1.0 / c
            B = B / c
        return B, Dr, Dc

    def iterate(self, A, b, eps=1e-12):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        # Dr A Dc  y = Dr b  with x = Dc y
        Aeq, Dr, Dc = self._ruiz(A, iters=self.ruiz_iters, eps=eps)
        beq = Dr * b

        # Work in y, with warm start y0 = Dc^{-1} x0
        y = (self.x0 / (Dc + 1e-18)).copy()

        row_norms_sq = (Aeq * Aeq).sum(axis=1) + eps
        probs = row_norms_sq / row_norms_sq.sum()

        num_iter = m
        for _ in range(num_iter):
            i = np.random.choice(m, p=probs)
            ai = Aeq[i]
            r  = beq[i] - ai @ y
            y += (r / row_norms_sq[i]) * ai

        x = Dc * y
        self.x0 = x
        return x


class RK_Tikh:
    """
    Randomized Kaczmarz on the Tikhonov-augmented system:
        [ A          ] x ≈ [ b ]
        [ sqrt(l) I ]       [ 0 ]
    which yields the ridge solution argmin ||A x - b||^2 + l ||x||^2.
    """
    def __init__(self, num_params, x0=None, lam=None, lam_scale=1e-5):
        self.num_params = int(num_params)
        self.x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0).reshape(-1)
        self.lam = lam          # if None, choose from lam_scale * ||A||_2^2 each call
        self.lam_scale = float(lam_scale)

    def iterate(self, A, b, eps=1e-12):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        # Choose lambda if not fixed
        lam = self.lam
        if lam is None:
            # use spectral norm estimate via ||A||_F as cheap upper bound if you want:
            # lam = self.lam_scale * (np.linalg.norm(A, 2)**2)
            lam = self.lam_scale * (np.linalg.norm(A, ord='fro')**2 / max(m, 1))

        # Augmented system
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(n)])
        b_aug = np.concatenate([b, np.zeros(n)])

        x = self.x0.copy()
        row_norms_sq = (A_aug * A_aug).sum(axis=1) + eps
        probs = row_norms_sq / row_norms_sq.sum()

        num_iter = A_aug.shape[0]
        for _ in range(num_iter):
            i = np.random.choice(A_aug.shape[0], p=probs)
            ai = A_aug[i]
            r  = b_aug[i] - ai @ x
            x += (r / row_norms_sq[i]) * ai

        self.x0 = x
        return x

# ============================== TARK =============================

class TARK:
    """
    Tail-Averaged Randomized Kaczmarz (m*n updates per call).
    Returns the tail-averaged estimate; keeps last x in self.x0.
    """
    def __init__(self, num_params, x0=None):
        self.num_params = num_params
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0).reshape(-1)

    def iterate(self, A, b, burnin: int = 0, eps: float = 1e-12): # TODO: debug burnin steps
        A = np.asarray(A)                    # (m, n)
        b = np.asarray(b).reshape(-1)        # (m,)
        m, n = A.shape
        assert n == self.num_params

        x = self.x0.copy()
        x_sum = np.zeros_like(x)
        count = 0

        row_norms_sq = (A * A).sum(axis=1) + eps
        probs = row_norms_sq / row_norms_sq.sum()

        num_iter = m 

        for s in range(num_iter):
            i = np.random.choice(m, p=probs)
            ai = A[i]
            residual = b[i] - ai @ x
            x = x + (residual / row_norms_sq[i]) * ai
            if s >= burnin:
                x_sum += x
                count += 1

        x_avg = x_sum / max(count, 1)
        self.x0 = x
        return x_avg
    
class RK_EquiTikh:
    """
    Randomized Kaczmarz with BOTH:
      - Ruiz equilibration (left+right scaling), and
      - Tikhonov augmentation (ridge)

    Solves in y with x = diag(Dc) * y, using the augmented system:
        [ Aeq         ] y ≈ [ beq ]
        [ sqrt(lam)Dc ]     [  0  ]
    where Aeq = Dr * A * Dc, beq = Dr * b.
    """
    def __init__(self, num_params, x0=None, ruiz_iters=5, lam=None, lam_scale=1e-3):
        self.num_params = int(num_params)
        self.x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0).reshape(-1)
        self.ruiz_iters = int(ruiz_iters)
        self.lam = lam
        self.lam_scale = float(lam_scale)

    def _ruiz(self, A, iters=5, eps=1e-12):
        m, n = A.shape
        Dr = np.ones(m)
        Dc = np.ones(n)
        B = A.copy()
        for _ in range(iters):
            r = np.sqrt((B * B).sum(axis=1)) + eps
            Dr *= 1.0 / r
            B = (B.T / r).T
            c = np.sqrt((B * B).sum(axis=0)) + eps
            Dc *= 1.0 / c
            B = B / c
        return B, Dr, Dc

    def iterate(self, A, b, eps=1e-12):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        # 1) Ruiz equilibration
        Aeq, Dr, Dc = self._ruiz(A, iters=self.ruiz_iters, eps=eps)
        beq = Dr * b

        # 2) Ridge strength (auto if lam is None)
        lam = self.lam
        if lam is None:
            # cheap, scale-invariant-ish choice
            lam = self.lam_scale * (np.linalg.norm(A, ord='fro')**2 / max(m, 1))

        # 3) Augmented system in y (x = Dc * y)
        #    Top: Dr * A * Dc, RHS: Dr * b
        #    Bottom: sqrt(lam) * Dc, RHS: 0
        A_bot = np.sqrt(lam) * np.diag(Dc)
        A_aug = np.vstack([Aeq, A_bot])
        b_aug = np.concatenate([beq, np.zeros(n)])

        # 4) RK on augmented, equilibrated system (warm start in y)
        y = (self.x0 / (Dc + 1e-18)).copy()
        row_norms_sq = (A_aug * A_aug).sum(axis=1) + eps
        probs = row_norms_sq / row_norms_sq.sum()

        num_iter = A_aug.shape[0] 
        for _ in range(num_iter):
            i = np.random.choice(A_aug.shape[0], p=probs)
            ai = A_aug[i]
            r  = b_aug[i] - ai @ y
            y += (r / row_norms_sq[i]) * ai

        x = Dc * y
        self.x0 = x
        return x

# ---------- 1) GRK + Tikhonov ----------
class GRK_Tikh:
    """
    Greedy RK on the Tikhonov-augmented system:
        [ A          ] x ≈ [ b ]
        [ sqrt(l) I ]       [ 0 ]
    Equivalent to ridge regression with weight 'lam'.

    Args:
      num_params: problem dimension n
      x0: warm start
      lam: fixed ridge coefficient; if None, chosen from lam_scale * ||A||_F^2 / m
      lam_scale: scale used when lam is None (kept consistent with your RK_Tikh)
      num_passes: how many (m*n) inner iterations
    """
    def __init__(self, num_params, x0=None, lam=None, lam_scale=1e-5):
        self.num_params = int(num_params)
        self.x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0).reshape(-1)
        self.lam = lam          # if None, choose from lam_scale * ||A||_2^2 each call
        self.lam_scale = float(lam_scale)

    def iterate(self,
                A: np.ndarray,
                b: np.ndarray,
                l: int | None = None,
                x0: np.ndarray | None = None,
                tol: float = 0.0,
                rng: np.random.Generator | None = None) -> np.ndarray:
        A = np.asarray(A, float)
        b = np.asarray(b, float).reshape(-1)
        m, n = A.shape

        # Choose lambda if not fixed
        lam = self.lam
        if lam is None:
            # use spectral norm estimate via ||A||_F as cheap upper bound if you want:
            # lam = self.lam_scale * (np.linalg.norm(A, 2)**2)
            lam = self.lam_scale * (np.linalg.norm(A, ord='fro')**2 / max(m, 1))

        # Augmented system
        A = np.vstack([A, np.sqrt(lam) * np.eye(n)])
        b = np.concatenate([b, np.zeros(n)])

        m, n = A.shape
        assert n == self.num_params, "num_params must equal A.shape[1]"
        assert b.shape[0] == m, "b must have length m"

        x = (self.x0 if x0 is None else np.asarray(x0, float).reshape(-1)).copy()

        # exact norms
        row_norms_sq = (A * A).sum(axis=1)         # ||A^(i)||_2^2
        fro_sq = float((A * A).sum())              # ||A||_F^2
        fro_sq_safe = max(fro_sq, 1e-300)

        if l is None:
            l = m

        if rng is None:
            rng = np.random.default_rng()

        for _ in range(int(l)):
            r = b - A @ x
            rnorm_sq = float(r @ r)

            if rnorm_sq <= tol**2:
                break

            # Eq. (2.3): epsilon_k
            with np.errstate(divide='ignore', invalid='ignore'):
                crit = np.where(row_norms_sq > 0.0, (r * r) / row_norms_sq, 0.0)
            rnorm_sq_safe = max(rnorm_sq, 1e-300)
            eps_k = 0.5 * (crit.max() / rnorm_sq_safe + 1.0 / fro_sq_safe)

            # Eq. (2.4): v_k set
            tau_mask = (r * r) >= (eps_k * rnorm_sq * row_norms_sq)

            # Numerical fallback: ensure non-empty candidate set
            if not np.any(tau_mask):
                i = int(np.argmax(np.abs(r)))
                tau_mask = np.zeros(m, dtype=bool)
                tau_mask[i] = True

            # r~ and selection probs (Eq. 2.5)
            r_tilde = np.where(tau_mask, r, 0.0)
            rt_sq = float(r_tilde @ r_tilde)
            if rt_sq <= 0.0:
                # degenerate—choose the largest |r_i| in tau
                candidates = np.flatnonzero(tau_mask)
                i_k = int(candidates[np.argmax(np.abs(r[candidates]))])
            else:
                probs = (r_tilde * r_tilde) / rt_sq
                # choice over all m with zero probs outside v_k
                i_k = int(rng.choice(m, p=probs))

            # Row update (line 6)
            ai = A[i_k, :]
            den = float(ai @ ai)                   # ||A^(i_k)||_2^2
            den_safe = max(den, 1e-300)
            res_i = b[i_k] - float(ai @ x)
            x = x + (res_i / den_safe) * ai

        self.x0 = x
        return x


# ---------- 2) GRK + Tail Averaging ----------
class GRK_TailAvg:
    """
    Greedy RK with Polyak tail-averaging of iterates.
    We take the average of the last 'tail_frac' portion of the total steps.

    Args:
      num_params: dimension n
      x0: warm start
      tail_frac: fraction in (0,1], e.g. 0.5 averages the last 50% of steps
      num_passes: how many (m*n) inner iterations
    """
    def __init__(self, num_params: int, x0: np.ndarray | None = None, burnin: int = 0):
        self.num_params = int(num_params)
        self.x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0, float).reshape(-1)
        self.burnin = burnin

    def iterate(self,
                A: np.ndarray,
                b: np.ndarray,
                l: int | None = None,
                x0: np.ndarray | None = None,
                tol: float = 0.0,
                rng: np.random.Generator | None = None) -> np.ndarray:
        A = np.asarray(A, float)
        b = np.asarray(b, float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params, "num_params must equal A.shape[1]"
        assert b.shape[0] == m, "b must have length m"

        x = (self.x0 if x0 is None else np.asarray(x0, float).reshape(-1)).copy()
        x_sum = np.zeros_like(x)
        count = 0

        # exact norms
        row_norms_sq = (A * A).sum(axis=1)         # ||A^(i)||_2^2
        fro_sq = float((A * A).sum())              # ||A||_F^2
        fro_sq_safe = max(fro_sq, 1e-300)

        if l is None:
            l = m

        if rng is None:
            rng = np.random.default_rng()

        for ss in range(int(l)):
            r = b - A @ x
            rnorm_sq = float(r @ r)

            if rnorm_sq <= tol**2:
                break

            # Eq. (2.3): epsilon_k
            with np.errstate(divide='ignore', invalid='ignore'):
                crit = np.where(row_norms_sq > 0.0, (r * r) / row_norms_sq, 0.0)
            rnorm_sq_safe = max(rnorm_sq, 1e-300)
            eps_k = 0.5 * (crit.max() / rnorm_sq_safe + 1.0 / fro_sq_safe)

            # Eq. (2.4): v_k set
            tau_mask = (r * r) >= (eps_k * rnorm_sq * row_norms_sq)

            # Numerical fallback: ensure non-empty candidate set
            if not np.any(tau_mask):
                i = int(np.argmax(np.abs(r)))
                tau_mask = np.zeros(m, dtype=bool)
                tau_mask[i] = True

            # r~ and selection probs (Eq. 2.5)
            r_tilde = np.where(tau_mask, r, 0.0)
            rt_sq = float(r_tilde @ r_tilde)
            if rt_sq <= 0.0:
                # degenerate—choose the largest |r_i| in tau
                candidates = np.flatnonzero(tau_mask)
                i_k = int(candidates[np.argmax(np.abs(r[candidates]))])
            else:
                probs = (r_tilde * r_tilde) / rt_sq
                # choice over all m with zero probs outside v_k
                i_k = int(rng.choice(m, p=probs))

            # Row update (line 6)
            ai = A[i_k, :]
            den = float(ai @ ai)                   # ||A^(i_k)||_2^2
            den_safe = max(den, 1e-300)
            res_i = b[i_k] - float(ai @ x)
            x = x + (res_i / den_safe) * ai

            if ss >= self.burnin:
                x_sum += x
                count += 1

        x_avg = x_sum / max(count, 1)
        self.x0 = x_avg
        return x_avg


# ---------- 3) GRK + Tail Averaging + Tikhonov ----------
class GRK_TailAvg_Tikh:
    """
    Greedy RK + Tikhonov augmentation, returning a tail-averaged iterate.
    This combines the stability of ridge with the variance reduction of tail averaging.
    """
    def __init__(self, num_params, x0=None, lam=None, lam_scale=1e-3):
        self.num_params = int(num_params)
        self.x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0).reshape(-1)
        self.lam = lam          # if None, choose from lam_scale * ||A||_2^2 each call
        self.lam_scale = float(lam_scale)

    def iterate(self,
                A: np.ndarray,
                b: np.ndarray,
                l: int | None = None,
                x0: np.ndarray | None = None,
                tol: float = 0.0,
                rng: np.random.Generator | None = None,
                burnin: int = 0) -> np.ndarray:
        A = np.asarray(A, float)
        b = np.asarray(b, float).reshape(-1)
        m, n = A.shape

        # Choose lambda if not fixed
        lam = self.lam
        if lam is None:
            # use spectral norm estimate via ||A||_F as cheap upper bound if you want:
            # lam = self.lam_scale * (np.linalg.norm(A, 2)**2)
            lam = self.lam_scale * (np.linalg.norm(A, ord='fro')**2 / max(m, 1))

        # Augmented system
        A = np.vstack([A, np.sqrt(lam) * np.eye(n)])
        b = np.concatenate([b, np.zeros(n)])

        m, n = A.shape
        assert n == self.num_params, "num_params must equal A.shape[1]"
        assert b.shape[0] == m, "b must have length m"

        x = (self.x0 if x0 is None else np.asarray(x0, float).reshape(-1)).copy()
        x_sum = np.zeros_like(x)
        count = 0

        # exact norms
        row_norms_sq = (A * A).sum(axis=1)         # ||A^(i)||_2^2
        fro_sq = float((A * A).sum())              # ||A||_F^2
        fro_sq_safe = max(fro_sq, 1e-300)

        if l is None:
            l = m 

        if rng is None:
            rng = np.random.default_rng()

        for ss in range(int(l)):
            r = b - A @ x
            rnorm_sq = float(r @ r)

            if rnorm_sq <= tol**2:
                break

            # Eq. (2.3): epsilon_k
            with np.errstate(divide='ignore', invalid='ignore'):
                crit = np.where(row_norms_sq > 0.0, (r * r) / row_norms_sq, 0.0)
            rnorm_sq_safe = max(rnorm_sq, 1e-300)
            eps_k = 0.5 * (crit.max() / rnorm_sq_safe + 1.0 / fro_sq_safe)

            # Eq. (2.4): v_k set
            tau_mask = (r * r) >= (eps_k * rnorm_sq * row_norms_sq)

            # Numerical fallback: ensure non-empty candidate set
            if not np.any(tau_mask):
                i = int(np.argmax(np.abs(r)))
                tau_mask = np.zeros(m, dtype=bool)
                tau_mask[i] = True

            # r~ and selection probs (Eq. 2.5)
            r_tilde = np.where(tau_mask, r, 0.0)
            rt_sq = float(r_tilde @ r_tilde)
            if rt_sq <= 0.0:
                # degenerate—choose the largest |r_i| in tau
                candidates = np.flatnonzero(tau_mask)
                i_k = int(candidates[np.argmax(np.abs(r[candidates]))])
            else:
                probs = (r_tilde * r_tilde) / rt_sq
                # choice over all m with zero probs outside v_k
                i_k = int(rng.choice(m, p=probs))

            # Row update (line 6)
            ai = A[i_k, :]
            den = float(ai @ ai)                   # ||A^(i_k)||_2^2
            den_safe = max(den, 1e-300)
            res_i = b[i_k] - float(ai @ x)
            x = x + (res_i / den_safe) * ai

            if ss >= burnin:
                x_sum += x
                count += 1

        x_avg = x_sum / max(count, 1)
        self.x0 = x
        return x_avg


# ============================== IGRK =============================

class IGRK:
    def __init__(self, num_params, x0=None):
        self.num_params = num_params
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0).reshape(-1)

    def iterate(self, A, b, eps=1e-12):
        A = np.asarray(A)
        b = np.asarray(b).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        x = self.x0.copy()
        row_norms_sq = (A * A).sum(axis=1) + eps    # ||a_i||^2
        num_iter = m * n

        for _ in range(num_iter):
            r = A @ x - b                           # residual
            crit = (r * r) / row_norms_sq           # |<a_i,x>-b_i|^2 / ||a_i||^2
            
            # (7) N_k = { i : |<a_i,x>-b_i| != 0 }, and Γ_k = sum_{i∈N_k} ||a_i||^2
            Nk = (np.abs(r) > 0)
            if not Nk.any():                                 
                # print("residual converged")
                break
            Gamma_k = row_norms_sq[Nk].sum()

            # thresh = 1/2 * ( max_i crit_i + ||r||^2 / Γ_k )
            thresh = 0.5 * (crit.max() + (np.dot(r, r) / max(Gamma_k, eps)))

            # Only consider valid rows (nonzero residuals & sensible norms)
            valid = Nk & (row_norms_sq > eps)
            Jk = np.flatnonzero(valid & (crit >= thresh))

            # If empty (can happen numerically), fall back to the best valid row
            if Jk.size == 0:
                print("No valid rows found; using best valid row.")
                break

            # Sample i ∈ J_k with probabilities p_i ∝ crit_i (any prob. works; this is natural)
            weights = crit[Jk]
            if not np.isfinite(weights).any() or weights.sum() <= 0:
                print("Invalid weights")
                break
            else:
                weights = weights / weights.sum()
            i = np.random.choice(Jk, p=weights)

            ai = A[i]
            den = row_norms_sq[i]
            ri = r[i]
            step = (ri / den) * ai

            x = x - step

        self.x0 = x
        return x


# ============================== MGRK =============================

class MGRK:
    """
    Improved Greedy Randomized Kaczmarz with momentum (Algorithm 2).

    Args
    ----
    num_params : int
    x0         : (n,) initial point; used for x^(0) = x^(1)
    alpha      : float > 0  (relaxation/step size)
    beta       : float >= 0 (momentum)
    theta      : float in [0, 1] (mixes 'greedy' and 'global' thresholds)
    """
    def __init__(self, num_params, x0=None, alpha=1.0, beta=0.25, theta=0.5):
        self.num_params = num_params
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        self.x_prev = self.x0.copy()     # x^(0) = x^(1)
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.theta = float(theta)

    def iterate(self, A, b, eps=1e-12):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        x      = self.x0.copy()
        x_prev = self.x_prev.copy()
        row_norms_sq = (A * A).sum(axis=1) + eps

        num_iter = m * n  # passes over the rows

        for _ in range(num_iter):
            r = A @ x - b                                 # residual
            Nk = (np.abs(r) > 0)                         # Step 1: N_k
            if not Nk.any():
                break
            Gamma_k = row_norms_sq[Nk].sum()             # Γ_k

            crit = (r * r) / row_norms_sq                # |<a_i,x>-b_i|^2 / ||a_i||^2

            # ------- Step 2: S_k via Eq. (17) -------
            # thresh = θ * max_i crit_i + (1-θ) * ||r||^2 / Γ_k
            thresh = self.theta * crit.max() + (1.0 - self.theta) * (np.dot(r, r) / max(Gamma_k, eps))
            valid = Nk & (row_norms_sq > eps)
            Sk = np.flatnonzero(valid & (crit >= thresh))

            # Fallback if empty due to numerical issues
            if Sk.size == 0:
                print("No valid rows found")
                break

            # ------- Step 3: choose i_k in S_k (prob ∝ crit) -------
            weights = crit[Sk]
            ws = weights.sum()
            p = (weights / ws) if np.isfinite(ws) and ws > 0 else np.ones(Sk.size) / Sk.size
            i = np.random.choice(Sk, p=p)

            # ------- Step 4: momentum update -------
            ai  = A[i]
            den = row_norms_sq[i]
            ri  = ai @ x - b[i]

            grad = (ri / den) * ai                       # Kaczmarz direction
            x_new = x - self.alpha * grad + self.beta * (x - x_prev)

            x_prev, x = x, x_new

        # keep state for next call (x^(k+1) becomes both x0 and x_prev)
        self.x0 = x.copy()
        self.x_prev = x.copy()
        return x


# ============================== REK ==============================

class REK:
    """
    Randomized Extended Kaczmarz (Algorithm 3).

    x^0 = 0 (or user x0), z^0 = b.
    Each iteration:
      - Column projection:  z ← z - <A_:j, z>/||A_:j||^2 * A_:j
      - Row update:         x ← x + (b_i - z_i - <a_i, x>)/||a_i||^2 * a_i
    Stop every 8 * min(m,n) iters if the two normalized tests are ≤ eps.
    """

    def __init__(self, num_params, x0=None):
        self.num_params = num_params
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0, float).reshape(-1)

    def iterate(self, A, b, eps=1e-6, max_passes=2):
        A = np.asarray(A, float)
        b = np.asarray(b, float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params, "num_params must match A.shape[1]"

        # Precompute norms/probabilities
        row_norms_sq = (A * A).sum(axis=1)
        col_norms_sq = (A * A).sum(axis=0)
        fro2 = row_norms_sq.sum() + 1e-12 # == col_norms_sq.sum()

        row_probs = row_norms_sq / fro2 
        col_probs = col_norms_sq / fro2 
        row_probs = np.where(row_norms_sq > 0, row_probs, 0.0)
        col_probs = np.where(col_norms_sq > 0, col_probs, 0.0)
        
        # Renormalize if needed
        row_probs /= row_probs.sum()
        col_probs /= col_probs.sum()

        # Init
        x = self.x0.copy()
        z = b.copy()            # z^(0) = b
        max_iters = max(1, int(max_passes * m))
        check_every = max(1, 8 * min(m, n))
        fro = np.sqrt(fro2) if fro2 > 0 else 1.0

        for k in range(max_iters):
            # --- pick column and update z (Step 6), but keep old z for x-step ---
            z_prev = z  # keep z^{(k)} for the x-update below
            j = np.random.choice(n, p=col_probs)
            aj = A[:, j]
            den_c = col_norms_sq[j] if col_norms_sq[j] > 0 else 1.0
            z = z - (aj @ z) / den_c * aj  # z^{(k+1)}

            # --- pick row and update x (Step 7) using z^{(k)}, not z^{(k+1)} ---
            i = np.random.choice(m, p=row_probs)
            ai = A[i, :]
            den_r = row_norms_sq[i] if row_norms_sq[i] > 0 else 1.0
            x = x + (b[i] - z_prev[i] - ai @ x) / den_r * ai  # uses z_prev = z^{(k)}

            # ---- termination check (Step 8) ----
            if (k + 1) % check_every == 0:
                Ax = A @ x
                xnorm = np.linalg.norm(x)
                denom_x = max(xnorm, 1e-12)
                primal = np.linalg.norm(Ax - (b - z)) / (fro * denom_x)
                dual   = np.linalg.norm(A.T @ z)       / ((fro * fro) * denom_x)
                if primal <= eps and dual <= eps:
                    break
        # if (np.linalg.norm(A @ x - b)) > 1e-4:
        #     print("residual", np.linalg.norm(A @ x - b))
        #     import pdb; pdb.set_trace()

        self.x0 = x
        return x


# ============================== GRK ==============================

class GRK:
    """
    Greedy Randomized Kaczmarz (GRK), Algorithm 2 in your screenshot.

    Inputs to iterate():
        A : (m,n) array
        b : (m,) array
        l : number of iterations (if None, uses m*n)
        x0: optional initial guess (overrides constructor's x0)

    Iteration k (exactly as in the figure):
      r_k   = b - A x_k
      ε_k   = 1/2 * ( max_i ( |r_i|^2 / ||A^(i)||_2^2 ) / ||r||_2^2  +  1/||A||_F^2 )      (Eq. 2.3)
      v_k   = { i : |r_i|^2 >= ε_k * ||r||_2^2 * ||A^(i)||_2^2 }                           (Eq. 2.4)
      r~    = r masked to v_k  (r~_i = r_i if i∈v_k, else 0)
      pick i_k in v_k with pr(i_k) = r~_i^2 / ||r~||_2^2                                   (Eq. 2.5)
      x_{k+1} = x_k + (b_{i_k} - A^(i_k) x_k)/||A^(i_k)||_2^2 * (A^(i_k))^T                (line 6)

    Stopping:
      - run exactly l iterations if given;
      - or stop early if ||r||^2 == 0 (within tol).

    Notes:
      - No epsilon is added to Frobenius/row norms; we only guard divides.
      - If v_k is empty (rare due to numerics), we fall back to the largest |r_i|.
    """

    def __init__(self, num_params: int, x0: np.ndarray | None = None):
        self.num_params = int(num_params)
        self.x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0, float).reshape(-1)

    def iterate(self,
                A: np.ndarray,
                b: np.ndarray,
                l: int | None = None,
                x0: np.ndarray | None = None,
                tol: float = 0.0,
                rng: np.random.Generator | None = None) -> np.ndarray:
        A = np.asarray(A, float)
        b = np.asarray(b, float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params, "num_params must equal A.shape[1]"
        assert b.shape[0] == m, "b must have length m"

        x = (self.x0 if x0 is None else np.asarray(x0, float).reshape(-1)).copy()

        # exact norms
        row_norms_sq = (A * A).sum(axis=1)         # ||A^(i)||_2^2
        fro_sq = float((A * A).sum())              # ||A||_F^2
        fro_sq_safe = max(fro_sq, 1e-300)

        if l is None:
            l = m 

        if rng is None:
            rng = np.random.default_rng()

        for _ in range(int(l)):
            r = b - A @ x
            rnorm_sq = float(r @ r)

            if rnorm_sq <= tol**2:
                break

            # Eq. (2.3): epsilon_k
            with np.errstate(divide='ignore', invalid='ignore'):
                crit = np.where(row_norms_sq > 0.0, (r * r) / row_norms_sq, 0.0)
            rnorm_sq_safe = max(rnorm_sq, 1e-300)
            eps_k = 0.5 * (crit.max() / rnorm_sq_safe + 1.0 / fro_sq_safe)

            # Eq. (2.4): v_k set
            tau_mask = (r * r) >= (eps_k * rnorm_sq * row_norms_sq)

            # Numerical fallback: ensure non-empty candidate set
            if not np.any(tau_mask):
                i = int(np.argmax(np.abs(r)))
                tau_mask = np.zeros(m, dtype=bool)
                tau_mask[i] = True

            # r~ and selection probs (Eq. 2.5)
            r_tilde = np.where(tau_mask, r, 0.0)
            rt_sq = float(r_tilde @ r_tilde)
            if rt_sq <= 0.0:
                # degenerate—choose the largest |r_i| in tau
                candidates = np.flatnonzero(tau_mask)
                i_k = int(candidates[np.argmax(np.abs(r[candidates]))])
            else:
                probs = (r_tilde * r_tilde) / rt_sq
                # choice over all m with zero probs outside v_k
                i_k = int(rng.choice(m, p=probs))

            # Row update (line 6)
            ai = A[i_k, :]
            den = float(ai @ ai)                   # ||A^(i_k)||_2^2
            den_safe = max(den, 1e-300)
            res_i = b[i_k] - float(ai @ x)
            x = x + (res_i / den_safe) * ai

        self.x0 = x
        return x

# ============================== FDBK ==============================

import numpy as np

class FDBK:
    """
    Fast Deterministic Block Kaczmarz (FDBK).

    Update (per iter k):
      r = b - A x_k
      ε_k = 1/2 * ( max_i |r_i|^2 / ||a_i||^2 / ||r||^2  +  1/||A||_F^2 )
      τ_k = { i : |r_i|^2 >= ε_k * ||r||^2 * ||a_i||^2 }
      η_k = r masked to τ_k  (η_i = r_i if i∈τ_k else 0)
      x_{k+1} = x_k + (η_k^T r / ||A^T η_k||^2) * A^T η_k

    Args
    ----
    num_params : int
    x0         : optional (n,) initial vector
    """

    def __init__(self, num_params, x0=None):
        self.num_params = num_params
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0, float).reshape(-1)

    def iterate(self, A, b, tol=1e-4, eps_row=1e-12):
        A = np.asarray(A, float)
        b = np.asarray(b, float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        # Precompute norms
        row_norms_sq = (A * A).sum(axis=1)          # shape (m,)
        fro_sq       = float((A * A).sum())         # scalar = ||A||_F^2

        x = self.x0.copy()
        max_iters = m 

        for _ in range(max_iters):
            r = b - A @ x
            rnorm_sq = float(r @ r)
            if rnorm_sq <= tol:
                # print("converged")
                break

            # (2.1) epsilon_k
            # Guard only when dividing by a near-zero denominator:
            r_safe = max(float(r @ r), 1e-300)          # ||r||^2
            # For rows with zero norm, define crit_i = 0 (they carry no constraint);
            # equivalently, ignore them in the max:
            with np.errstate(divide='ignore', invalid='ignore'):
                crit = np.where(row_norms_sq > 0, (r * r) / row_norms_sq, 0.0)
            eps_k = 0.5 * (crit.max() / r_safe + 1.0 / max(fro_sq, 1e-300))

            # (2.2) tau_k
            # condition: |r_i|^2 >= eps_k * ||r||^2 * ||A^(i)||^2  (exact form)
            tau_mask = (r * r) >= (eps_k * r_safe * row_norms_sq)

            # ensure non-empty block (very rare numerically)
            if not tau_mask.any():
                print("No valid rows found")
                break

            # ----- (2.3) η_k -----
            eta = np.zeros(m, dtype=float)
            eta[tau_mask] = r[tau_mask]                       # η_i = r_i if i in τ_k

            # (2.4) block step
            At_eta = A.T @ eta
            denom  = float(At_eta @ At_eta)             # ||A^T eta||^2
            alpha  = float(eta @ r) / max(denom, 1e-300)
            x = x + alpha * At_eta

        self.x0 = x
        return x



# ============================== DEKA ==============================

class DEKA:
    """
    Deterministic Kaczmarz with Smoothing & Damping (DeKA).
    Greedy row-set selection each outer step; optional multi-step per call.

    Args (constructor):
        num_params (int): dimension n of the parameter vector.
        x0 (np.ndarray | None): initial (n,). Defaults to zeros.
        eps (float): tiny number to avoid divide-by-zero.
        gamma (float): damping factor (scales update).
        reg (float): Tikhonov regularization for denominator.
        alpha (float): exponential smoothing factor in [0,1]. Higher = smoother.
        steps_per_call (int): number of DeKA outer iterations per .iterate() call.
        tol (float): early-stop on residual^2 if <= tol (0 disables).
    """
    def __init__(
        self,
        num_params,
        x0=None,
        eps=1e-12,
        gamma=1.0,
        reg=1e-12,
        alpha=0.1,
        tol=1e-8,
    ):
        self.num_params = int(num_params)
        self.x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        assert self.x0.shape == (self.num_params,)
        self.eps   = float(eps)
        self.gamma = float(gamma)
        self.reg   = float(reg)
        self.alpha = float(alpha)
        self.tol   = float(tol)

        # smoothed estimate (kept across calls)
        self.x_smooth = self.x0.copy()

    def iterate(self, A, b):
        """
        Run 'steps_per_call' DeKA outer iterations.
        Shapes:
            A: (m, n)
            b: (m,)
        Returns:
            np.ndarray: current smoothed estimate (n,)
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params, f"expected n={self.num_params}, got {n}"
        assert b.shape == (m,), f"b must be (m,), got {b.shape}"

        x = self.x0.copy()          # raw (unsmoothed) estimate carried within this call

        # Precompute norms used in thresholds
        row_norm2 = (A * A).sum(axis=1) + self.eps       # (m,)
        fro2 = (A * A).sum() + self.eps                  # ||A||_F^2

        num_iter = m*n
        for _ in range(num_iter):
            r = b - A @ x                                # residual vector (m,)
            r2 = float(r @ r)                            # ||r||_2^2

            if self.tol > 0.0 and r2 <= self.tol:
                break

            # Per-row “energy”: |b_i - a_i x|^2 / ||a_i||^2
            num = (r ** 2)                               # since r_i = b_i - a_i x
            ratios = num / row_norm2                     # (m,)
            ratios_max = float(ratios.max())

            # Threshold epsilon_k (half of: max-ratio scaled by 1/||r||^2 plus 1/||A||_F^2)
            # If r2 is ~0, fall back to the 1/||A||_F^2 term.
            if r2 > self.eps:
                eps_k = 0.5 * (ratios_max / r2 + 1.0 / fro2)
            else:
                eps_k = 0.5 * (1.0 / fro2)

            # Greedy candidate set τ_k:
            # Keep rows whose normalized error exceeds eps_k * ||r||^2 * ||a_i||^2
            # Equivalent check using already computed terms:
            #   num[i]/row_norm2[i] >= eps_k * r2 * row_norm2[i]
            # -> num[i] >= eps_k * r2 * (row_norm2[i]**2)
            thresh = eps_k * r2 * (row_norm2 ** 2)
            mask = (num >= thresh)

            if not np.any(mask):
                # Fallback: pick the single worst row by ratio
                mask[np.argmax(ratios)] = True

            # Aggregated residual η_k: only entries in τ_k kept
            eta = np.where(mask, r, 0.0)                 # (m,)

            # Update direction with regularization:
            AT_eta = A.T @ eta                           # (n,)
            denom = float(AT_eta @ AT_eta) + self.reg    # ||A^T η||^2 + reg
            # Scale by η^T r (energy alignment)
            scale = float(eta @ r) / denom
            delta = scale * AT_eta                        # (n,)

            # Damping
            delta *= self.gamma

            # Raw update
            x = x + delta

            # Exponential smoothing (stateful across calls)
            self.x_smooth = self.alpha * self.x_smooth + (1.0 - self.alpha) * x

        # Persist for next call
        self.x0 = x.copy()
        return self.x_smooth.copy()
