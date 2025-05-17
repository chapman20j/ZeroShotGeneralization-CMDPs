# tabular.py
"""
Code for CEBE with the tabular cliff walking environment.

Plots:
1. Original environment reward (CEBE is exact here)
2. Reward with goal reward p^(-2) and penalty p^(-1) for falling. (Figure 1a)
3. Reward with goal reward 1/(1 + p) and penalty 1/(1 + p) for falling. (Figure 1b)
"""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import Expr
from tqdm import tqdm

TICK_SIZE: int = 14
FONT_SIZE: int = 16


# * Code for solving
def solve_exact(pi: np.ndarray, transition, rewards, gamma):

    # pi is (s, a)
    # transition is (s, a, s')
    # rewards is (s, a, s')
    s, a = pi.shape
    # Expected reward
    rhs = np.sum(transition * rewards, axis=-1).flatten()

    # Transition kernel (T * pi)
    tpi = transition.reshape((*transition.shape, -1)) * pi

    # I - gamma * T * pi
    lhs = np.eye(s * a) - gamma * tpi.reshape(s * a, s * a)

    # Solve for q
    q = np.linalg.solve(lhs, rhs)
    return q.reshape(s, a)


# * Code for cliff walker


# Enum for tabular actions
class ACT(Enum):
    LEFT = (0, (-1, 0))
    RIGHT = (1, (1, 0))
    UP = (2, (0, -1))
    DOWN = (3, (0, 1))


# This is the transition function for a grid world
# This will be for cliff walking
def make_transition(m: int, n: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Creates transition tensor for cliffwalking environment.

    Args:
        m: number of rows
        n: number of columns
        p: slip probability

    Returns:
        transition tensor T of shape (m, n, 4, m, n)
        grad tensor dT/dp of shape (m, n, 4, m, n)
    """
    out = np.zeros((m, n, len(ACT), m, n), dtype=np.float64)
    grad = np.zeros((m, n, len(ACT), m, n), dtype=np.float64)

    # Function to check if the index is valid
    def is_valid(i, j):
        return i >= 0 and i <= m - 1 and j >= 0 and j <= n - 1

    # Function to compute transitions for initial state (i, j)
    def trans(i, j):

        # Absorbing states first
        if i == 0 and j > 0:
            out[i, j, :, i, j] = 1
            return

        count = len(ACT) - 1

        for a in ACT:
            next_spot = (i + a.value[1][0], j + a.value[1][1])
            is_val = is_valid(*next_spot)
            if is_val:
                out[i, j, a.value[0], next_spot[0], next_spot[1]] = 1 - p
                grad[i, j, a.value[0], next_spot[0], next_spot[1]] = -1
            else:
                # just don't move, but still have slip probability
                out[i, j, a.value[0], i, j] = 1 - p
                grad[i, j, a.value[0], i, j] = -1
            for b in ACT:
                if a == b:
                    continue
                next_spot2 = (i + b.value[1][0], j + b.value[1][1])
                if is_valid(*next_spot2):
                    out[i, j, a.value[0], next_spot2[0], next_spot2[1]] = p / count
                    grad[i, j, a.value[0], next_spot2[0], next_spot2[1]] = 1 / count
                else:
                    out[i, j, a.value[0], i, j] += p / count
                    grad[i, j, a.value[0], i, j] += 1 / count
        return

    # Compute all transitions
    for i in range(m):
        for j in range(n):
            trans(i, j)

    return out, grad


def make_reward0(
    m: int, n: int, p: float, penalty: float = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Creates a reward tensor independent of p.

    Args:
        m: number of rows
        n: number of columns
        p: slip probability
        penalty: falling penalty. Defaults to 100.

    Returns:
        reward tensor R of shape (m, n, A, m, n)
        grad tensor dR/dp of shape (m, n, A, m, n)
    """
    out = np.zeros((m, n, len(ACT), m, n), dtype=np.float64)
    # Falling off cliff
    out[1, 1 : n - 1, 0, 0, 1 : n - 1] = -penalty
    # Falling from start
    out[0, 0, :, 0, 1] = -penalty

    # Reaching goal
    out[1, n - 1, :, 0, n - 1] = 1

    return out, np.zeros_like(out)


def make_reward1(
    m: int, n: int, p: float, penalty: float = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Creates a reward tensor with goal reward p^(-2) and penalty p^(-1) for falling.

    Args:
        m: number of rows
        n: number of columns
        p: slip probability
        penalty: falling penalty. Defaults to 100.

    Returns:
        reward tensor R of shape (m, n, A, m, n)
        grad tensor dR/dp of shape (m, n, A, m, n)
    """
    out = np.zeros((m, n, 4, m, n), dtype=np.float64)
    # Falling off cliff
    out[1, 1 : n - 1, 0, 0, 1 : n - 1] = -penalty / p
    # Falling from start
    out[0, 0, :, 0, 1] = -penalty / p

    # Reaching goal
    out[1, n - 1, :, 0, n - 1] = 1 / p**2

    grad = np.zeros_like(out)
    grad[1, 1 : n - 1, 0, 0, 1 : n - 1] = penalty / p**2
    grad[0, 0, :, 0, 1] = penalty / p**2
    grad[1, n - 1, :, 0, n - 1] = -2 / p**3

    return out, grad


def make_reward_any(
    m: int, n: int, p: float, cliff_fn: Expr, goal_fn: Expr, penalty: float = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Creates a reward tensor with goal goal_fn and penalty cliff_fn for falling.

    Args:
        m: number of rows
        n: number of columns
        p: slip probability
        cliff_fn: Sympy function for cliff penalty. Should be a function of p.
        goal_fn: Sympy function for goal reward. Should be a function of p.
        penalty: falling penalty. Defaults to 100.

    Returns:
        reward tensor R of shape (m, n, A, m, n)
        grad tensor dR/dp of shape (m, n, A, m, n)
    """
    # this assumes that rew_fn is a function of var(p)
    varp = sympy.symbols("p")
    cliff_deriv = sympy.diff(cliff_fn, varp)
    goal_deriv = sympy.diff(goal_fn, varp)

    out = np.zeros((m, n, 4, m, n), dtype=np.float64)
    # Falling off cliff
    out[1, 1 : n - 1, 0, 0, 1 : n - 1] = -penalty * cliff_fn.subs(varp, p)
    # Falling from start
    out[0, 0, :, 0, 1] = -penalty * cliff_fn.subs(varp, p)
    # Reaching goal
    out[1, n - 1, :, 0, n - 1] = goal_fn.subs(varp, p)

    grad = np.zeros_like(out)
    grad[1, 1 : n - 1, 0, 0, 1 : n - 1] = -penalty * cliff_deriv.subs(varp, p)
    grad[0, 0, :, 0, 1] = -penalty * cliff_deriv.subs(varp, p)
    grad[1, n - 1, :, 0, n - 1] = goal_deriv.subs(varp, p)

    return out, grad


# CEBE
def tabular_cebe_approximation_error(
    context0, context1, m, n, penalty, rew_fn, display: bool = False
):
    # context0 is the base context
    # context1 is the new context
    transition0, dT0 = make_transition(m, n, context0)
    transition1, dT1 = make_transition(m, n, context1)
    rewards0, dr0 = rew_fn(m, n, context0, penalty)
    rewards1, dr1 = rew_fn(m, n, context1, penalty)
    pi = np.ones((m * n, 4)) / 4
    gamma = 0.9

    # * Reshape
    transition0 = transition0.reshape((m * n, 4, m * n))
    rewards0 = rewards0.reshape((m * n, 4, m * n))
    dT0 = dT0.reshape((m * n, 4, m * n))
    dr0 = dr0.reshape((m * n, 4, m * n))

    transition1 = transition1.reshape((m * n, 4, m * n))
    rewards1 = rewards1.reshape((m * n, 4, m * n))
    dT1 = dT1.reshape((m * n, 4, m * n))
    dr1 = dr1.reshape((m * n, 4, m * n))

    # * Make sample enhacement
    se_transition = transition0 + dT0 * (context1 - context0)
    se_transition = se_transition / np.sum(se_transition, axis=-1, keepdims=True)
    se_rewards = rewards0 + dr0 * (context1 - context0)

    # * Solve
    se_soln = solve_exact(pi, se_transition, se_rewards, gamma)
    soln1 = solve_exact(pi, transition1, rewards1, gamma)

    diff = se_soln - soln1
    if display:
        diff = diff.reshape((m, n, 4))
        tmp = np.where(np.abs(diff) > 1)
        for x, y, a in zip(*tmp):
            print(x, y, a)

    return np.abs(se_soln - soln1).max()


if __name__ == "__main__":
    verify = False

    m, n = 5, 6
    gamma = 0.9
    p = 0.1

    # Lets train stuff
    pvar = sympy.symbols("p")
    rwfn0 = lambda m, n, p, penalty=100: make_reward0(m, n, p, penalty=penalty)
    rwfn1 = lambda m, n, p, penalty=100: make_reward1(m, n, p, penalty=penalty)
    cfn = 1 / (1 + pvar)
    gfn = 1 / (1 + pvar) ** 1.5
    rwfn_power = lambda m, n, p, penalty=10: make_reward_any(
        m, n, p, cfn, gfn, penalty=penalty
    )

    # * Verify first order accurate
    rwfn_list = [rwfn0, rwfn1, rwfn_power]
    penalty_list = [100, 100, 10]
    low_list = [-8, -8, -8]
    high_list = [-1, -3, -1]
    for rwfn, pen, low, high in zip(rwfn_list, penalty_list, low_list, high_list):
        eps_arr = np.linspace(low, high, 100)
        eps_arr = np.exp(eps_arr)
        out = []
        for eps in tqdm(eps_arr):
            out.append(
                tabular_cebe_approximation_error(
                    0.1, 0.1 + eps, m, n, penalty=pen, rew_fn=rwfn
                )
            )
        plt.loglog(eps_arr, out, label="True Error")

        # Fit a line to the log-log data
        log_eps = np.log(eps_arr)
        log_out = np.log(out)
        coeffs = np.polyfit(log_eps[:10], log_out[:10], 1)
        best_fit_line = np.poly1d(coeffs)
        print(best_fit_line)

        # Plot the line of best fit
        plt.loglog(
            eps_arr,
            np.exp(best_fit_line(log_eps)),
            label=f"{best_fit_line}",
            linestyle="--",
        )
        plt.legend(fontsize=FONT_SIZE)
        plt.xticks(fontsize=TICK_SIZE)
        plt.yticks(fontsize=TICK_SIZE)

        plt.ylabel("Approximation Error", fontsize=FONT_SIZE)
        plt.xlabel(r"$c - c_0$", fontsize=FONT_SIZE)
        plt.xlim(0, None)
        plt.tight_layout()
        plt.show()
