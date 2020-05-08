//! Newton based methods for rootfinding
//! ========================================================
//!
//! This crate allows you to use [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) for rootfinding.
//!
//! It aims to implement several Newton based methods (Broyden, ...), whether the jacobian function is provided or not.
//! It also aims to work on a complex model, limiting the number of model calls to a minimum.
//! A minimal solver is also provided for basic usages and benchmarking purposes.
//!
//! ## Key features
//!
//!  1. **Minimal solver** available for basic 1D functions. The speed of the advanced solver will be benchmarked against this one to estimate the overhead.
//!  2. **Advanced solver** for n-dimension problems including advanced stopping, update criteria, step limitation and more.
//!      It is designed to work with the jacobian provided or not, evaluating it with finite-differentiation.
//!  3. **Model interaction**: The advanced solver is designed to interact with a complex model computing other outputs and having memory effects.
//!      A definition of such a model is given through the `Model` trait.
//!      The struct `UserModelWithFunc` is provided to easily adapt a given function to the required trait.
//!  4. **Extended test examples**: Real world use cases and an extensive function database are included in the crate for integration testing and benchmarking.
//!
//! ## Current limitations
//!
//! 1. The inputs and outputs of the model are assumed to be `nalgebra` vectors.
//! 2. Only the finite-difference version is currently available.
//! 3. Benchmarking vs the minimal-solver is not yet in place.
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
//! # Comparison with other rust crates
//!
//! Note: Crates may have evolved since this comparison was established.
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
pub mod solver_minimal;

pub mod model;

pub mod model_with_func;
