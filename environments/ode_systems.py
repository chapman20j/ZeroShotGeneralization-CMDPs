# ode_systems.py
"""
File which first creates a differential equation.
Then, it will solve the differential equation and return the numerical solution.

Each function returns:
    transition equations
    observables
    context variables
    control variables
    Reward expression

Need to specify all the variables it depends on (including ones that only influence reward expression)
Need to have observables in the order they would appear for the equations after converting to first order
"""

import numpy as np
import sympy
from sympy import Expr

from .ode_sympy import tvar


def make_pendulum_goal_eqns() -> tuple[list, list, list, list, Expr]:

    # Context
    gvar = sympy.Symbol("g")
    mvar = sympy.Symbol("m")
    lvar = sympy.Symbol("l")
    tau = sympy.Symbol("tau")

    # Action
    uvar = sympy.Symbol("u")

    # Equation
    theta = sympy.Function("theta")(tvar, gvar, mvar, lvar, tau)

    eqn = sympy.Eq(
        sympy.Derivative(theta, (tvar, 2)),
        3 * gvar / (2 * lvar) * sympy.sin(theta) + 3 * uvar / (mvar * lvar**2),
    )
    dtheta_dt = sympy.Derivative(theta, tvar)
    observables = [theta, dtheta_dt]
    # Add the desired torque (goal torque)
    goal = sympy.asin(-tau / (mvar * gvar * lvar))
    reward_expr = -(
        np.pi**2 * sympy.sin((theta - goal) / 2) ** 2
        + 0.1 * dtheta_dt**2
        + 0.001 * uvar**2
    )
    return [eqn], observables, [gvar, mvar, lvar, tau], [uvar], reward_expr


def make_cartpole_goal_eqns() -> tuple[list, list, list, list, Expr]:
    # Context
    gvar = sympy.Symbol("g")
    pmvar = sympy.Symbol("m_p")
    cmvar = sympy.Symbol("m_c")
    lvar = sympy.Symbol("l")
    goalvar = sympy.Symbol("goal")

    # Action
    fvar = sympy.Symbol("F")

    # Equation
    theta = sympy.Function("theta")(tvar, gvar, pmvar, cmvar, lvar, goalvar)
    x = sympy.Function("x")(tvar, gvar, pmvar, cmvar, lvar, goalvar)

    theta_eqn = sympy.Eq(
        sympy.Derivative(theta, (tvar, 2)),
        (
            gvar * sympy.sin(theta)
            + sympy.cos(theta)
            * (
                -fvar
                - pmvar
                * lvar
                * sympy.Derivative(theta, (tvar, 1)) ** 2
                * sympy.sin(theta)
            )
            / (pmvar + cmvar)
        )
        / (lvar * (4 / 3 - pmvar * sympy.cos(theta) ** 2 / (pmvar + cmvar))),
    )
    x_eqn = sympy.Eq(
        sympy.Derivative(x, (tvar, 2)),
        (
            fvar
            + pmvar
            * lvar
            * (
                sympy.Derivative(theta, (tvar, 1)) ** 2 * sympy.sin(theta)
                - sympy.cos(theta) * theta_eqn.rhs
            )
        )
        / (pmvar + cmvar),
    )
    observables = [
        x,
        sympy.Derivative(x, tvar),
        theta,
        sympy.Derivative(theta, tvar),
    ]
    reward_expr = 1 + (1 - sympy.sqrt(1 + (x - goalvar) ** 2))
    return (
        [x_eqn, theta_eqn],
        observables,
        [gvar, pmvar, cmvar, lvar, goalvar],
        [fvar],
        reward_expr,
    )
