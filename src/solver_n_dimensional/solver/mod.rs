//! Solver configuration
//!
//! The solver is defined in the `RootFinder` struct.
//!
//! To create a new solver, it is required to give the 4 following:
//! - The parameters through the `SolverParameters` struct
//! - The iteratives configuration through a reference to a slice of `Iteratives`
//! - The residuals configuration through a reference to a slice of `ResidualsConfig`
//! - The initial guess to use by the solver
//!
//! # Features
//! 1. Simulation log available for debugging, check the `set_debug()` method
//! 2. Damping, check the `set_damping()` method
//!
//!
//! ## Examples
//!
//! ```
//! extern crate newton_rootfinder;
//! use newton_rootfinder as nrf;
//! use nrf::model::Model;
//! use nrf::iteratives;
//! use nrf::residuals;
//! use nrf::solver::ResolutionMethod;
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
//!   let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
//!   let iter_params = iteratives::Iteratives::new(&vec_iter_params);
//!   let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
//!   let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
//!   let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
//!   let damping = false;
//!   let mut rf = nrf::solver::default_with_guess(init_guess, &iter_params, &res_config, ResolutionMethod::NewtonRaphson, damping);
//!   let mut user_model =
//!       nrf::model::UserModelWithFunc::new(problem_size, square2);
//!
//!   rf.solve(&mut user_model);
//!
//!   println!("{}", user_model.get_iteratives()[0]);
//!   // print 1.4142135623747443
//! }
//! ```

mod default;
mod jacobian;
mod log;
mod parameters;
mod resolution_method;
mod rootfinder;

pub use default::default_with_guess;
pub use jacobian::jacobian_evaluation;
pub use jacobian::JacobianMatrix;
pub use parameters::SolverParameters;
pub use resolution_method::greenstadt_second_method_udpate_jac;
pub use resolution_method::{
    broyden_first_method_udpate_inv_jac, broyden_second_method_udpate_inv_jac,
};
pub use resolution_method::{broyden_first_method_udpate_jac, broyden_second_method_udpate_jac};
pub use resolution_method::{quasi_method_update_inv_jac, quasi_method_update_jac};
pub use resolution_method::{QuasiNewtonMethod, ResolutionMethod, UpdateQuasiNewtonMethod};
pub use rootfinder::RootFinder;
