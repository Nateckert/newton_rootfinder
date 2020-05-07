//! newton_rootfinder: Newton based methods for rootfinding
//! ========================================================
//!
//! This crate allows you to use [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) for rootfinding.
//! It aims to implement several Newton based method (Broyden, ...), whereas the jacobian function is provided or not.
//! It also aims to work on a complex model, limiting the number of model calls at a minimum.
//! A minimal solver is also provided for basic usages and benchmarking purposes.
//!
//! ## Key selling-points
//!
//!  1. **Minimal solver** available for basic 1D function. The speed of the industrial solver will be benchmarked vs this one to estimate the overhead cost.
//!  2. **Advanced solver** for n-dimension including advanced stopping and update criteria, step limitation and more. It is designed to work with the jacobian provided or not, evaluating it thanks to finite-differentiation.
//!  3. **Model interaction**: the advanced solver is designed to interact with a complex model computing others outputs and having memory effects. A definition of such a model is given through the `Model` trait. The struct `UserModelWithFunc` is provided to easily adapt model defined with a function to the required trait.
//!  4. **Extended tests examples**: the crate aims at having an import functions bases for integration testing and benchmarks, which is not available in plug and play form.
//!
//! ## Current limitations
//!
//! 1. The inputs and outputs of the model are assuming to be vector from the `nalgebra` crate.
//! 2. Only the finite-difference version is currently available.
//! 3. Benchmarking vs the minimal-solver is not in place.
//!
//! ## Examples
//!
//! ```rust
//! extern crate newton_rootfinder as nrf;
//! use nrf::model::Model;
//!
//! extern crate nalgebra;
//!
//! /// Equation : x**2 - 2 = 0
//! fn square2(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
//!     let mut y = x * x;
//!     y[0] -= 2.0;
//!     y
//! }
//!
//! fn main() {
//!   let problem_size = 1;
//!   let init_guess = nalgebra::DVector::from_vec(vec![1.0]);
//!   let rf = nrf::solver::RootFinderFD::default_with_guess(init_guess);
//!   let mut user_model =
//!       nrf::model_with_func::UserModelWithFunc::new(problem_size, square2);
//!
//!   rf.solve(&mut user_model);
//!
//!   println!("{}", user_model.get_iteratives()[0]);
//!   // print 1.4142135623747443
//! }
//! ```
//!
//!
//! # Comparision with other rust libraries
//!
//! Crates may have evolved since this comparision was established
//!
//! | crate                 | version | 1-dimension  | n-dimension | Jacobian not required | Other algorithms¹ |
//! |-----------------------|--------:|:------------:|:-----------:|----------------------:|------------------:|
//! | **newton_rootfinder** |   0.1.0 |       ✔️     |      ✔️     |  ✔️                  | ❌ (not yet)      |
//! | newton-raphson        |   0.1.0 |       ✔️     |      ❌     |  ❌                  | ❌                |
//! | nrfind                |   1.0.3 |       ✔️     |      ❌     |  ❌                  | ❌                |
//! | rootfind              |   0.7.0 |       ✔️     |      ❌     |  ❌                  |  ✔️               |
//! | roots                 |   0.6.0 |       ✔️     |      ❌     |  ❌                  |  ✔️               |
//! | peroxide              |  0.21.7 |       ✔️     |      ✔️     |  ❌                  | ❌                |
//!
//! 1. Other algorithms than the Newton-Raphson method.


pub mod util;

pub mod solver;

pub mod model;

pub mod model_with_func;
