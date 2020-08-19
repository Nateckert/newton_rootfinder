//! Newton based methods for rootfinding
//! ========================================================
//!
//! This crate allows you to use [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) for rootfinding.
//!
//! It aims to implement several Newton based methods (Broyden, ...), whether the jacobian function is provided or not.
//!
//! It also aims to work on a complex model, limiting the number of model calls to a minimum.
//!
//! A minimal solver is also provided for basic usages and benchmarking purposes.
//!
//! # Minimal solver
//!
//! A minimal solver is provided for basic usages in the `solver_minimal` module.
//!
//! This minimal solver works only on basic 1D functions.
//!
//! The speed of the advanced solver will be benchmarked against this one to estimate the overhead.
//!
//!
//! # Advanced solver
//!
//! An advanced solver is available for n-dimension problems.
//!
//! To get improved interactions with the user problem (usually a function),
//! the user is required to implement the `Model` trait in order to use the solver.
//! This ensures a reduced number of calls to the function and a better debugging experience if needed.
//!
//! It is defined in the `solver_advanced` module.
//! Don't hesitate to check in this module documentation for examples.
//!
//! The focus of this crate is the development of this solver.
//!
//! ## Key features
//!  1. Works whether the jacobian is provided or not (evaluating it with finite-differentiation).
//!  2. In-detail parametrization of iterative variables, residuals and stopping criteria.
//!  3. Debugging informations available through a .txt log file.
//!  4. The advanced solver is designed to interact with a complex model computing other outputs and having memory effects.
//!      The requirements of this model are defined by the `Model` trait.
//!      The struct `UserModelWithFunc` is provided to easily adapt a given function to the required trait.
//!  5. Real world use cases and an extensive function database are included in the crate for integration testing and benchmarking. (work in progress)
//!
//! ## Current limitations
//!
//! 1. The inputs and outputs of the model are assumed to be `nalgebra` vectors.
//! 2. The test base is still in construction
//!
//! ## Examples
//! ```
//! extern crate newton_rootfinder;
//! use newton_rootfinder::solver_advanced as nrf;
//! use nrf::model::Model; // trait import
//!
//! extern crate nalgebra;
//!
//! // Function to optimize: x**2 = 2
//! pub fn square2(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
//!     let mut y = x * x;
//!     y[0] -= 2.0;
//!    y
//! }
//!
//! fn main() {
//!
//!     let problem_size = 1;
//!
//!     // Parametrization of the iteratives variables
//!     let vec_iter_params = nrf::iteratives::default_vec_iteratives_fd(problem_size);
//!     let iter_params = nrf::iteratives::Iteratives::new(&vec_iter_params);
//!
//!     // Parametrization of the residuals
//!     let stopping_residuals = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
//!     let update_methods = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
//!     let res_config = nrf::residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
//!
//!     // Parametrization of the solver
//!     let init = nalgebra::DVector::from_vec(vec![1.0]);
//!     let resolution_method = nrf::solver::ResolutionMethod::NewtonRaphson;
//!     let damping = false;
//!     let mut rf = nrf::solver::default_with_guess(
//!         init,
//!         &iter_params,
//!         &res_config,
//!         resolution_method,
//!         damping,
//!     );
//!
//!     // Adpatation of the function to solve to the Model trait.
//!     let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, square2);
//!
//!     rf.solve(&mut user_model);
//!
//!     println!("{}", user_model.get_iteratives()[0]); // 1.4142135623747443
//!     println!("{}", std::f64::consts::SQRT_2);       // 1.4142135623730951
//! }
//! ```
//!
//! # Comparison with other rust crates
//!
//! Note: Crates may have evolved since this comparison was established.
//!
//! N-dimensional :
//!
//! | crate                 | version | Advanced <br> Parametrization | Simulation <br> Log | Other iterative<br> algorithms |
//! |-----------------------|--------:|:-----------------------------:|:-------------------:|-------------------------------:|
//! | **newton_rootfinder** |   0.5.0 |       ✔️                      |      ✔️             |  ✔️                           |
//! | peroxide              |  0.21.7 |       ❌                      |      ❌             |   ❌                          |
//!
//!
//!
//! If you are looking for one dimensional crates, several options are available.
//!
//! One dimension :
//!
//! | crate                 | version | Newton-Raphson | Other Iterative methods | Analytical methods  | Error handling |
//! |-----------------------|--------:|---------------:|------------------------:|--------------------:|---------------:|
//! | newton-raphson        |   0.1.0 |  ✔️            | ❌                     | ❌                  | ❌             |
//! | nrfind                |   1.0.3 |  ✔️            | ❌                     | ❌                  | ✔️             |
//! | rootfind              |   0.7.0 |  ✔️            | ✔️                     | ❌                  | ✔️             |
//! | roots                 |   0.6.0 |  ✔️            | ✔️                     | ✔️                  | ✔️             |
//!

pub mod solver_advanced;
pub mod solver_minimal;
