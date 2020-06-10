//! Advanced solver
//!
//! ## Definitions
//!
//! ### Model
//! For a given function:
//!```ignore
//! f : (1,n) -> (1,n)
//!      f(X) -> Y
//!```
//!
//! A n-dimensional rootfinding alogrithm attempts,
//! to find X a vector such as norm(Y) < tolerance, i.e Y = 0
//!
//! Most rootfinding algorithm returns the X values.
//!
//! However, in common use case, the end-user is not interested by the values of these values,
//! but by other values computed by its model once the root is found.
//! For exemple, a model can compute other quantities that are the main focus of the end-user.
//! To compute these ones, the user would have to perform an extra call to the model with the solution.
//!
//! To spare this extra call, the solver is designed to work on a model defined through the `Model` trait.
//! The solver mutate the model and returns it once it is converged,
//! allowing the user to access the results if getter to other variables are implemented.
//!
//! ### Iteratives variables
//! The inputs of the functions X values are called the iteratives variables,
//! as the resolution method iterates on them and changes their values.
//! Iteratives variables can be parametrized in order to defines their behavior during the iterations.
//!
//! For a more complete description, check the `iteratives` module
//!
//! ### Residuals variables
//! The output of the model are called the residuals.
//! Residuals values can be computed in different ways according to the residuals definition.
//! The residuals are actually defined as left and right parts of an equation,
//! in order to be able to change its calculation without changing the model outputs
//! (check below example)
//!
//! For a more complete description, check the `residuals` module
//!
//!
//! ### Examples
//! Let's consider the following thermodynamical problem
//!
//! We have a pipe with a gas flow of mass flow W, a total pressure Pt and a total enthalpy ht.
//!
//! The pipe has at the input the section A
//!
//! We have the following equations :
//!
//! - W = rho_s.v.A : Conservation of mass flow
//! - Pt = Ps + rho_s.(v^2)/2 : Conservation of momentum
//! - ht = f(Ts) + (v^2)/2 : Conservation of energy
//! - Ps = rho_s.R.Ts : Perfect gas law
//!
//!
//! Are known :
//! - The geometry of the pipe is known: section A
//! - The inputs characteristics : W, Pt, ht
//! - The gas constant R
//! - The function f(T) used in the conservation of energy (it is the enthalpy function)
//!
//! Are unknown:
//! - The speed v
//! - The static temperature Ts
//! - The static fluid density rho_s
//! - The static pressure Ps
//!
//! We have 4 unknowns and 4 equations,
//! it is a problem that can be solved thanks to a rootfinding algorithm
//!
//! The 4 iteratives variables will be:
//! - v, Ts, rho_s, Ps
//!
//! A first way to write the residuals would be to have the model outputs the following quantities:
//!```ignore
//! model(v, Ts, rho_s, Ps) -> | W - rho_s*v*A
//!                            | Pt - Ps + rho_s*(v**2)/2
//!                            | ht - f(Ts) + (v**2)/2
//!                            | Ps - rho_s*R*Ts
//!```
//!
//! That would require the user to change its model to have the desired residuals.
//! However, the solver works with `ResidualsValues` that are pairs of outputs:
//!```ignore
//! model(v, Ts, rho_s, Ps) -> | (W, rho_s*v*A)
//!                            | (Pt, Ps + rho_s*(v**2)/2)
//!                            | (ht, f(Ts) + (v**2)/2)
//!                            | (Ps, rho_s*R*Ts)
//!```
//!
//! For the given problem, the quantities don't have the same order of magnitude:
//! - Pressure in 1e5
//! - flow in the range 1e-1 - 1e3 (according to the system considered)
//!
//! Hence it is numerically a bad practice to impose the same tolerance on all outputs.
//! The residual configuration allows you to chose a normalization formula well suited to your problem.
//!
//! # Features
//! 1. Iteratives parametrization: check the `iteratives` module
//! 2. Residuals  parametrization: check the `residuals` module
//! 3. Solver interaction with through the `Model` trait, check the `model` module
//! 4. Simulation log: writing of a log to follow the resolution parameters to ease debugging
//! 5. Configuration parsing: it is possible to define the solver paramters in an external configuration
//! file, allowing a more ergonomic use (and a parametrization only known at run-time): check the `util` module
//! 6. Many solver options : check the `solver_advanced` module
//! 7. This solver is available through Python: check https://github.com/Nateckert/py_nrf
//!
//!
//!
//! # Upcoming Features
//! 1. Implement other algorithms (Broyden, Martinez, Huang, Tomas, Greenstadt: see https://doi.org/10.1007/BF02684472)
//! 2. Implement new test cases
//! 4. Implement tests cases with Automatic Differentation for benchmarking (https://crates.io/crates/fwd_ad)

pub mod iteratives;
pub mod model;
pub mod residuals;
pub mod solver;
pub mod test_cases;
pub mod util;
