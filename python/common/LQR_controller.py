# src/lqr_controller.py

import numpy as np

from scipy.linalg import solve_discrete_are

def dlqr(A, B, Q, R):
    # robust DARE solve
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P

# def dlqr(A: np.ndarray,
#          B: np.ndarray,
#          Q: np.ndarray,
#          R: np.ndarray,
#          max_iters: int = 1000,
#          tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Solve the discrete-time Algebraic Riccati equation by iteration,
#     returning the optimal gain K and cost-to-go matrix P.

#     A, B:   system matrices
#     Q, R:   state- and input-cost matrices
#     max_iters, tol: stop criteria for convergence of P
#     """
#     P = Q.copy()
#     for i in range(max_iters):
#         # Compute gain K
#         S = R + B.T @ P @ B
#         K = np.linalg.solve(S, B.T @ P @ A)
#         # Riccati update
#         P_next = Q + A.T @ P @ (A - B @ K)
#         if np.linalg.norm(P_next - P, ord=2) < tol:
#             P = P_next
#             break
#         P = P_next
#     # Final gain
#     K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
#     return K, P


class LQRController:
    """
    Simple infinite-horizon LQR controller:
        u = -K x
    """

    def __init__(self,
                 A: np.ndarray,
                 B: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray):
        """
        Pre-compute the optimal gain K.
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K, self.P = dlqr(A, B, Q, R)

    def update_linearized_dynamics(self, A: np.ndarray, B: np.ndarray):
        """
        Update the linearized dynamics matrices A and B.
        """
        self.A = A
        self.B = B
        self.K, self.P = dlqr(A, B, self.Q, self.R)

    def control(self, x: np.ndarray) -> np.ndarray:
        """
        Compute control action for state x.
        """
        # ensure x is column-vector
        x = x.reshape(-1, 1)  
        u = -self.K @ x
        return u.squeeze()

    def _dbg_report(self):
        # spectral radius of closed-loop
        eig = np.linalg.eigvals(self.A - self.B @ self.K)
        rho = max(abs(eig))
        # conditioning of S = R + B^T P B
        S = self.R + self.B.T @ self.P @ self.B
        try:
            condS = np.linalg.cond(S)
        except Exception:
            condS = np.inf
        print(f"[LQR] ||K||={np.linalg.norm(self.K):.2e}  "
            f"rho(A-BK)={rho:.4f}  cond(S)={condS:.2e}")