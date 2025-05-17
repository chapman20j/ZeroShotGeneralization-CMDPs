# ode_sympy.py
"""
File containing functions for converting from original ode system to a system of first order odes.

The pipeline looks roughly like
0. Plug in equations. Functions should depend on t and context parameters.
1. Create context derivatives
2. Convert to symbols
3. Convert to first order system
"""

from copy import deepcopy
from typing import Optional

import numpy as np
import sympy
from scipy.integrate import solve_ivp
from sympy import Derivative, Equality, Function, Symbol, lambdify

VAR_DELIM = "~"
DERIV_DELIM = "#"
tvar = Symbol("t")


# * Utility functions
def get_order(fn: Derivative, var: Symbol) -> int | None:
    r"""Gets the order of the derivative of a function with respect to a variable

    Args:
        fn: Function like dx/dvar
        var: variable to get the order of

    Returns:
        int representing the order of the derivative
        None if the variable is not in the function
    """
    for tup in fn.variable_count:
        if tup[0] == var:
            return tup[1]
    return None


def fun_to_fun(fn: Function):
    r"""Converts functions of multiple variables to functions of the time variable
    eg. x(t, a, b, c) -> x(t)

    Args:
        fn: sympy function like x(t, a, b, c)

    Returns:
        sympy function like x(t)
    """
    return Function(fn.name)(tvar)


def deriv_to_fun(
    deriv: Derivative, include_tvar: bool = False
) -> Function | Derivative:
    r"""Converts derivatives into a function for later numerical solving
    If include_tvar is true, it will include the tvar in the function name

    eg. include_tvar = False
        d^2 x(t, a)/(dadt) -> d(x~a#1)(t)/dt

    eg. include_tvar = True
        d^2 x(t, a)/(dadt) -> x~a#1t#1(t)

    The part before ~ is the function name
    The part after ~ is the ordered variable names
    The number after # is the order of the derivative
    ~ is specified by VAR_DELIM
    # is specified by DERIV_DELIM

    Args:
        deriv: Derivative to convert
        include_tvar: Whether to include tvar in new name. Defaults to False.

    Returns:
        Resulting function or Derivative
    """

    # get function name and derivatives
    base_fn = deriv.atoms(Function).pop()
    name = base_fn.name
    derivs = deriv.variable_count

    # Sort by the variable name
    derivs = sorted(derivs, key=lambda x: x[0].name)

    # put tvar at the end if in the list
    tvar_stuff = None
    for d in derivs:
        if d[0] == tvar:
            tvar_stuff = d
            derivs.remove(d)
            break
    if tvar_stuff is not None:
        derivs.append(tvar_stuff)

    # If including tvar, put all variables in the name
    if include_tvar:
        fn_name = VAR_DELIM.join(
            [name] + [f"{x[0]}{DERIV_DELIM}{x[1]}" for x in derivs]
        )
        return Function(fn_name)(tvar)

    fn_name = VAR_DELIM.join(
        [name] + [f"{x[0]}{DERIV_DELIM}{x[1]}" for x in derivs if x[0] != tvar]
    )

    tvar_count = 0
    d = derivs[-1]
    if d[0] == tvar:
        tvar_count = d[1]

    modified_no_dt = Function(fn_name)(tvar)
    # This returns a function if tvar_count is 0
    return Derivative(modified_no_dt, (tvar, tvar_count))


# * Context Derivatives
def context_deriv(eqn: Equality, var: Symbol) -> Equality:
    r"""Computes the derivative of the equation with respect to a variable
    eg. dx/dt = ax + bf and var=a
    -> d^2 x/(dtda) = x + a dx/da

    Args:
        eqn: Equation to compute the derivative of
        var: Variable to compute the derivative with respect to

    Returns:
        Differentiated equation
    """

    dlhs = sympy.diff(eqn.lhs, var)
    drhs = sympy.diff(eqn.rhs, var)
    return Equality(dlhs, drhs)


def system_with_context_derivs(
    eqn_list: list[Equality], context_vars: list[Symbol]
) -> list[Equality]:
    r"""Computes the derivatives of the equations with respect to the context variables
    Output is in the following form
    [eqn1, eqn2, ..., eqnM, d_eqn1/d_var1, d_eqn2/d_var1, ..., d_eqnM/d_varN]
    This format makes it efficient to turn derivatives into a jacobian matrix

    Args:
        eqn_list: list of equations to compute the derivatives of
        context_vars: variables to compute the derivatives with respect to

    Raises:
        ValueError: tvar cannot be in context variables. It should be treated separately.

    Returns:
        List of differentiated equations
    """
    if tvar in context_vars:
        raise ValueError(
            "tvar cannot be in context variables. It should be treated separately."
        )

    out = deepcopy(eqn_list)
    for var in context_vars:
        for eq in eqn_list:
            out.append(context_deriv(eq, var))
    return out


# * Convert to symbols


def orig_eqns_to_syms(eqn_list: list[Equality]) -> list[Equality]:
    r"""Converts systems of equations to simplified forms.
    eg. dx/dt = ax + bf -> dx/dt = ax + bf
    eg. d^2 x/(dadt) = x + a dx/da -> d(x~a#1)/dt = x + a x~a#1
    eg. d^3 x/(da dt^2) = x + a dx/da -> d^2(x~a#1)/dt^2 = x + a x~a#1

    Note that input functions look like
    x(t, a, b) and this gets reduced to x(t)

    Args:
        eqn_list: list of equations to convert

    Returns:
        simplified equations
    """

    fn_dict = dict()

    # Convert all functions and derivatives to simplified forms
    # Preserves the derivatives of tvar
    for eq in eqn_list:
        lhs_fns = eq.lhs.atoms(Derivative)
        for fn in lhs_fns:
            if fn not in fn_dict:
                fn_dict[fn] = deriv_to_fun(fn)

        for deriv in eq.rhs.atoms(Derivative):
            if deriv not in fn_dict:
                fn_dict[deriv] = deriv_to_fun(deriv, include_tvar=True)
        for fn in eq.rhs.atoms(Function):
            if fn not in fn_dict:
                if hasattr(fn, "name"):
                    fn_dict[fn] = fun_to_fun(fn)

    # Substitute the functions
    new_eqns = []
    for eq in eqn_list:
        lhs = eq.lhs.subs(fn_dict)
        rhs = eq.rhs.subs(fn_dict)
        new_eqns.append(Equality(lhs, rhs))

    return new_eqns


# * Make System
# This part will convert the previous equations into a system of first order odes


def to_first_order(eqn_list):
    r"""Converts a system of higher order odes into a system of first order odes
    NOTE: This assumes that the sympy functions depend only on t
        ie x(t, a, b) has been converted to x(t)

    If input is [eqn1, eqn2, ..., eqnM]
    The output will be
    [eqn1_t0, eqn1_t1, ..., eqn1_to1, eqn2_t0, eqn2_t1, ..., eqn2_to2, ..., eqnM_t0, eqnM_t1, ..., eqnM_toM]
    where oi denotes the order of the t derivative in the i'th equation

    eg. d^2 x/(dt^2) = ax + bf -> dx/dt = x~t#1, d(x~t#1)/dt = ax + bf

    Args:
        eqn_list: list of equations to convert

    Returns:
        first order system of equations
    """

    # Now we just need to look at the degrees and create a system from that
    first_order_system = []
    for eq in eqn_list:
        lhs = eq.lhs

        # Get the degree of the t derivative
        deg = 0
        for fn in lhs.atoms(Derivative):
            deg = get_order(fn, tvar)

        # Create the system
        base_fn = lhs.atoms(Derivative).pop().atoms(Function).pop()
        variables = [base_fn]
        var_dict = {base_fn: base_fn}
        for i in range(1, deg):
            tmp = Derivative(base_fn, (tvar, i))
            tmp2 = deriv_to_fun(tmp, include_tvar=True)
            variables.append(tmp2)
            var_dict[tmp] = tmp2

        for i in range(deg - 1):
            my_eqn = Equality(Derivative(variables[i], tvar), variables[i + 1])
            first_order_system.append(my_eqn)

        first_order_system.append(
            Equality(Derivative(variables[-1], tvar), eq.rhs).subs(var_dict)
        )

    return first_order_system


# * Full conversion
def full_conversion(
    eqn_list: list[Equality],
    context_vars: list[Symbol],
    display: bool = False,
    latex: bool = False,
    compute_context_derivs: bool = True,
) -> list[Equality]:
    r"""Converts a system of equations into a first order system with context derivatives

    Args:
        eqn_list: list of equations to convert
        context_vars: context variables to compute the derivatives with respect to
        display: whether to display the conversion process . Defaults to False.
        latex: whether to print the equations in latex. Defaults to False.
        compute_context_derivs: whether to compute context derivatives. Defaults to True.

    Returns:
        First order system of equations
    """

    if display:
        print("ORIGINAL SYSTEM")
        for eq in eqn_list:
            sympy.pprint(eq)

    if compute_context_derivs:
        sys_with_context = system_with_context_derivs(eqn_list, context_vars)
    else:
        sys_with_context = deepcopy(eqn_list)

    if display:
        print("\nSYSTEM WITH CONTEXT DERIVATIVES:")
        for eq in sys_with_context:
            sympy.pprint(eq)

    if latex:
        print("\nLATEX")
        for eq in sys_with_context:
            print(sympy.latex(eq))

    first_conversion = orig_eqns_to_syms(sys_with_context)
    if display:
        print("\nNEW EQUATIONS:")
        for eq in first_conversion:
            sympy.pprint(eq)

    final_conversion = to_first_order(first_conversion)
    if display:
        print("\nFIRST ORDER SYSTEM:")
        for eq in final_conversion:
            sympy.pprint(eq)
        print()

    return final_conversion


class ODEContextSystem:

    def __init__(
        self,
        original_eqns: list[sympy.Eq],
        observables: list[sympy.Symbol],
        context_vars: list[sympy.Symbol],
        control_variables: list[sympy.Symbol],
        reward_expr: Optional[sympy.Eq] = None,
        compute_context_derivatives: bool = True,
        solver: str = "euler",
    ):
        if not compute_context_derivatives:
            raise NotImplementedError(
                "Must compute derivatives of context variables for now"
            )
        self.original_eqns = original_eqns
        self.observables = []
        for obs in observables:
            if isinstance(obs, sympy.Derivative):
                self.observables.append(deriv_to_fun(obs, include_tvar=True))
            else:
                self.observables.append(fun_to_fun(obs))
        self.observables = tuple(self.observables)
        self.context_vars = tuple(context_vars)
        self.control_variables = tuple(control_variables)
        self.context_control_vars = context_vars + control_variables
        self.reward_expr = reward_expr
        if solver not in ["euler", "solve_ivp", "rk45"]:
            # Pick euler for fastest.
            raise NotImplementedError(
                f"Solver {solver} not implemented. Use euler, solve_ivp, or rk45"
            )
        self.solver = solver

        # * Set up first order system
        self.first_order_system = full_conversion(
            original_eqns,
            context_vars,
            display=False,
            latex=False,
            compute_context_derivs=compute_context_derivatives,
        )

        self.first_order_state_variables = [
            eq.lhs.atoms(sympy.Function).pop() for eq in self.first_order_system
        ]

        # * Scipy stuff
        # Assumes state variables in y
        # Then context variables in args
        # Then control variables at end of args

        # * Optimized solve_fn
        self.solve_fn = lambdify(
            (
                tvar,
                self.first_order_state_variables,
                self.context_vars,
                self.control_variables,
            ),
            [eqn.rhs for eqn in self.first_order_system],
        )

        if reward_expr is not None:
            # Convert to symbols
            dummy_var = sympy.Symbol("dummy")
            reward_expr = orig_eqns_to_syms([sympy.Eq(dummy_var, reward_expr)])[0].rhs
            # * Optimized reward
            self._reward_lambda = lambdify(
                (
                    tvar,
                    self.first_order_state_variables,
                    self.context_vars,
                    self.control_variables,
                ),
                reward_expr,
            )

            # dr_dcontext
            self.dr_dcontext = [
                sympy.diff(reward_expr, var) for var in self.context_vars
            ]

            # * Optimized reward derivatives
            self._dr_context_lambda = lambdify(
                (
                    tvar,
                    self.first_order_state_variables,
                    self.context_vars,
                    self.control_variables,
                ),
                self.dr_dcontext,
            )
            self.dr_ds = [
                sympy.diff(reward_expr, var) for var in self.first_order_state_variables
            ]
            self._dr_ds_lambda = lambdify(
                (
                    tvar,
                    self.first_order_state_variables,
                    self.context_vars,
                    self.control_variables,
                ),
                self.dr_ds,
            )

        self.state = np.zeros(len(self.first_order_state_variables), dtype=np.float64)

    def step(self, context_values, action, dt):
        self._step_state(context_values, action, dt)
        obs, dobs_dcontext = self._unpack_state(self.state)

        return obs, dobs_dcontext

    def reward(self, context_values, action):
        reward = self._reward_lambda(0, self.state, context_values, action)
        dr_dcontext = self._dr_context_lambda(0, self.state, context_values, action)
        dr_ds = self._dr_ds_lambda(0, self.state, context_values, action)
        dr_dobs = self._unpack_dr_ds(dr_ds)

        return reward, dr_dcontext, dr_dobs

    def _step_state(self, context_values, action, dt):
        assert len(context_values) == len(self.context_vars)
        assert action.shape[-1] == len(self.control_variables)

        self._set_derivs_to_zero()

        if self.solver == "euler":
            sol = self.solve_fn(0, self.state, context_values, action)

            self.state += dt * np.array(sol)
        else:
            sol = solve_ivp(
                self.solve_fn,
                (0, dt),
                self.state,
                args=(context_values, action),
                first_step=None if self.solver == "solve_ivp" else dt,
            )
            self.state = sol.y[:, -1]
        return

    def _unpack_state(self, state):
        o = len(self.observables)
        p = len(self.context_vars)
        dobs_dcontext = state[o:].reshape(p, -1).T
        return state[:o], dobs_dcontext

    def _unpack_dr_ds(self, dr_ds):
        return dr_ds[: len(self.observables)]

    def _set_derivs_to_zero(self):
        o = len(self.observables)
        self.state[o:] = 0
        return

    def summary(self):
        return f"""ODEContextSystem:
        # base eqns  = {len(self.original_eqns)} 
        observables  = {self.observables}
        context vars = {self.context_vars}
        control vars = {self.control_variables}"""

    def __repr__(self):
        # Detailed summary
        header = self.summary()
        all_eqns = "\n".join([sympy.pretty(eq) for eq in self.first_order_system])
        if self.reward_expr is None:
            rew = ""
        else:
            rew = "\nREWARD:\n"
            rew += (
                sympy.pretty(
                    sympy.Eq(sympy.Symbol("R"), self.reward_expr, evaluate=False)
                )
                + "\n"
            )
            rew += "\n" + "\n".join(
                [
                    sympy.pretty(
                        sympy.Eq(sympy.Symbol(f"dR/d{var}"), deriv, evaluate=False)
                    )
                    for var, deriv in zip(self.context_vars, self.dr_dcontext)
                ]
            )
            rew += "\n" + "\n".join(
                [
                    sympy.pretty(
                        sympy.Eq(sympy.Symbol(f"dR/d{var}"), deriv, evaluate=False)
                    )
                    for var, deriv in zip(self.first_order_state_variables, self.dr_ds)
                ]
            )

        return header + "\nALL EQUATIONS:\n" + all_eqns + rew

    def __str__(self):
        header = self.summary()
        orig_eqns = "\n".join([sympy.pretty(eq) for eq in self.original_eqns])
        if self.reward_expr is None:
            rew = ""
        else:
            rew = "\nREWARD:\n"
            rew += sympy.pretty(self.reward_expr)

        return header + "\nORIGINAL EQUATIONS:\n" + orig_eqns + rew
