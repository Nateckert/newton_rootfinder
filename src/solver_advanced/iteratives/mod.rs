//! This module defines the iteratives variables
//!
//! Iteratives variables are defined through the `Iterative` trait
//!
//! Two kind of iterative variables are provided :
//! - `IterativeParams`
//! - `IterativeParamsFD` that extends the previous one to work with finite-difference
//!
//! The struct `Iteratives` (plural) is holding the array or vector of the previous parameters
//! and is the one that will be used by the solver
//!
//! It is possible to create default iteratives variables with the following lines:
//!
//! # Examples
//! ```
//! extern crate newton_rootfinder;
//! use newton_rootfinder::solver_advanced as nrf;
//! use nrf::iteratives::*;
//!
//! let size = 5;
//! let my_iters_fd = Iteratives::new(&vec![IterativeParamsFD::default(); size]);
//! let my_iters = Iteratives::new(&vec![IterativeParams::default(); size]);
//! ```
//!

mod iterative_var;
mod iterative_var_fd;
mod iteratives_base;

pub use iterative_var::IterativeParams; // struct re-export
pub use iterative_var_fd::IterativeParamsFD; // struct re-export
pub use iterative_var_fd::PerturbationMethod; // enum re-export
pub use iteratives_base::Iterative; // trait re-export
pub use iteratives_base::Iteratives; // struct re-export

/// Constructor with default values for iteratives parameters
///
/// # Examples
///```
/// extern crate newton_rootfinder;
/// use newton_rootfinder::solver_advanced as nrf;
///
/// let size = 2;
/// let iteratives_vec = nrf::iteratives::default_vec_iteratives(size);
/// assert_eq!(iteratives_vec.len(), size);
/// for i in 0..size {
///     assert_eq!(iteratives_vec[i].get_min_value(), f64::NEG_INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_value(), f64::INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_step_abs(), f64::INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_step_abs(), f64::INFINITY);
/// }
///```
pub fn default_vec_iteratives(size: usize) -> Vec<IterativeParams> {
    vec![IterativeParams::default(); size]
}

/// Constructor with default values for iteratives parameters with finite-differences
///
/// # Examples
///```
/// extern crate newton_rootfinder;
/// use newton_rootfinder::solver_advanced as nrf;
///
/// let size = 2;
/// let iteratives_vec = nrf::iteratives::default_vec_iteratives_fd(size);
/// assert_eq!(iteratives_vec.len(), size);
/// for i in 0..size {
///     assert_eq!(iteratives_vec[i].get_min_value(), f64::NEG_INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_value(), f64::INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_step_abs(), f64::INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_step_abs(), f64::INFINITY);
///     assert_eq!(iteratives_vec[i].get_dx_abs(), 5e-8);
///     assert_eq!(iteratives_vec[i].get_dx_rel(), 5e-8);
///     assert_eq!(iteratives_vec[i].get_perturbation_method(), nrf::iteratives::PerturbationMethod::Max);
/// }
///```
pub fn default_vec_iteratives_fd(size: usize) -> Vec<IterativeParamsFD> {
    vec![IterativeParamsFD::default(); size]
}
