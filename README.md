newton_rootfinder: Newton based methods for rootfinding
========================================================


This crate allows you to use [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) for rootfinding.

It aims to implement several Newton based methods (Broyden, ...), whether the jacobian function is provided or not.

It also aims to work on a complex model, limiting the number of model calls to a minimum.

A minimal solver is also provided for basic usages and benchmarking purposes.

# Minimal solver

A minimal solver is provided for basic usages in the `solver_minimal` module.
Don't hesitate to check in this module documentation for examples.

This minimal solver works only on basic 1D functions.

The speed of the advanced solver will be benchmarked against this one to estimate the overhead.


# Advanced solver

An advanced solver is available for n-dimension problems.

To get improved interactions with the user problem (usually a function),
the user is required to implement the `Model` trait in order to use the solver.
This ensures a reduced number of calls to the function and a better debugging experience if needed.

It is defined in the `solver` module.
Don't hesitate to check in this module documentation for examples.

The focus of this crate is the development of this solver.

## Key features
  1. Works whether the jacobian is provided or not (evaluating it with finite-differentiation).
  2. In-detail parametrization of iterative variables, residuals and stopping criteria.
  3. Several Newton-based methods will be made available (not yet)
  4. The advanced solver is designed to interact with a complex model computing other outputs and having memory effects.
      The requirements of this model are defined by the `Model` trait.
      The struct `UserModelWithFunc` is provided to easily adapt a given function to the required trait.
  5. Real world use cases and an extensive function database are included in the crate for integration testing and benchmarking. (work in progress)

 ## Current limitations

 1. The inputs and outputs of the model are assumed to be `nalgebra` vectors.
 2. Only the finite-difference version is currently available.
 3. Benchmarking vs the minimal-solver is not yet in place.



 # Comparison with other rust crates

 Note: Crates may have evolved since this comparison was established.

 | crate                 | version | 1-dimension  | n-dimension | Jacobian not required | Other algorithms¹ |
 |-----------------------|--------:|:------------:|:-----------:|----------------------:|------------------:|
 | **newton_rootfinder** |   0.1.0 |       ✔️     |      ✔️     |  ✔️                  | ❌ (not yet)      |
 | newton-raphson        |   0.1.0 |       ✔️     |      ❌     |  ❌                  | ❌                |
 | nrfind                |   1.0.3 |       ✔️     |      ❌     |  ❌                  | ❌                |
 | rootfind              |   0.7.0 |       ✔️     |      ❌     |  ❌                  |  ✔️               |
 | roots                 |   0.6.0 |       ✔️     |      ❌     |  ❌                  |  ✔️               |
 | peroxide              |  0.21.7 |       ✔️     |      ✔️     |  ❌                  | ❌                |

 1. Other algorithms than the Newton-Raphson method.
