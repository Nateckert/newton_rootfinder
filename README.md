Newton based methods for rootfinding
========================================================

![crates.io](https://img.shields.io/crates/v/newton_rootfinder.svg)
[![Build Status](https://travis-ci.com/Nateckert/newton_rootfinder.svg?branch=master)](https://travis-ci.com/Nateckert/newton_rootfinder)

This crate allows you to use [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) for rootfinding.

It aims to implement several Newton based methods (Broyden, ...), whether the jacobian function is provided or not.

It also aims to work on a complex model, limiting the number of model calls to a minimum.

A minimal solver is also provided for basic usages and benchmarking purposes.

# Minimal solver

A minimal solver is provided for basic usages in the `solver_minimal` module.

This minimal solver works only on basic 1D functions.

The speed of the advanced solver will be benchmarked against this one to estimate the overhead.

# Examples
```rust
extern crate newton_rootfinder as nrf;
use nrf::solver_minimal::{solver1d, solver1d_fd};

fn square2(x: f64) -> f64 {
    x.powi(2)-2.0
}
fn dsquare(x: f64) -> f64 {
    2.0*x
}

fn main() {
    let max_iter = 50;
    let tolerance = 1e-6;
    let finite_diff_dx = 1e-8;

    let x1 = solver1d(1.0, square2, dsquare, max_iter, tolerance);
    let x2 = solver1d_fd(1.0, square2, max_iter, tolerance, finite_diff_dx);

    println!("{}", x1);                         // 1.4142135623746899
    println!("{}", x2);                         // 1.4142135623746772
    println!("{}", std::f64::consts::SQRT_2);   // 1.4142135623730951
}
```

# Advanced solver

An advanced solver is available for n-dimension problems.

To get improved interactions with the user problem (usually a function),
the user is required to implement the `Model` trait in order to use the solver.
This ensures a reduced number of calls to the function and a better debugging experience if needed.

It is defined in the `solver_advanced` module.
Don't hesitate to check in this module documentation for examples.

The focus of this crate is the development of this solver.

## Key features
 1. Works whether the jacobian is provided or not (evaluating it with finite-differentiation).
 2. In-detail parametrization of iterative variables, residuals and stopping criteria.
 3. Debugging informations available through a .txt log file.
 4. The advanced solver is designed to interact with a complex model computing other outputs and having memory effects. The requirements of this model are defined by the `Model` trait. The struct `UserModelWithFunc` is provided to easily adapt a given function to the required trait.
 5. Real world use cases and an extensive function database are included in the crate for integration testing and benchmarking. (work in progress)

## Current limitations

 1. The inputs and outputs of the model are assumed to be `nalgebra` vectors.
 2. The test base is still in construction



# Comparison with other rust crates

Note: Crates may have evolved since this comparison was established.

N-dimensional :

| crate                 | version | Advanced <br> Parametrization | Simulation <br> Log | Other iterative<br> algorithms |
|-----------------------|--------:|:-----------------------------:|:-------------------:|-------------------------------:|
| **newton_rootfinder** |   0.5.0 |       ✔️                      |      ✔️             |  ✔️                  |
| peroxide              |  0.21.7 |       ❌                      |      ❌             |   ❌                          |



If you are looking for one dimensional crates, several options are available.
As a reminder, the focus of newton_rootfinder is **NOT** the development of the 1D solver.

One dimension :

| crate                 | version | Newton-Raphson | Other Iterative methods | Analytical methods  |
|-----------------------|--------:|---------------:|------------------------:|--------------------:|
| **newton_rootfinder** |   0.5.0 |  ✔️            | ❌                     | ❌                  |
| newton-raphson        |   0.1.0 |  ✔️            | ❌                     | ❌                  |
| nrfind                |   1.0.3 |  ✔️            | ❌                     | ❌                  |
| rootfind              |   0.7.0 |  ✔️            | ✔️                     | ❌                  |
| roots                 |   0.6.0 |  ✔️            | ✔️                     | ✔️                  |
| peroxide              |  0.21.7 |  ✔️            | ❌                     | ❌                  |
