# Specifications newton_rootfinder

## Solver

Implement Newton-Raphson method and other related methods

## Solver - Model Integration

- Avoid any unnecessary evaluation of the model
- When the solver has found a solution, the user should be able to access any latent variable of the model without reevaluating the model
- The solver should be able to work with models implementing a jacobian evaluation or switch to finite-difference


## Solver parametrization

- Ability to choose a mode in which the parametrization can be changed without having to recompile the code launching the resolution (runtime parametrization)
- Ability to choose a mode in which all the parameters are set at compile-time for performance.

## Error handling

The solver should be able to recover from differents kinds of error from the model.

There are 2 kinds of errors expected from the model:
- Inaccurate values
- Other errors

For other errors it is recoverable if it does not happen at the first or last iteration.
For inaccurate values it is always recoverable unless it happens at the last iteration.

The statregy for recovering is the following one:
- if the attempted step was using a quasi-newton method, redo that step with the classical Newton-Raphson method
- if the attempted step was already a Newton-Raphson step, rollback one iteration and redo it with damping

