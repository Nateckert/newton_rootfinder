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
//! use newton_rootfinder as nrf;
//! use nrf::iteratives::{Iteratives, IterativeParams, IterativeParamsFD};
//!
//! let size = 5;
//! let my_iters_fd = Iteratives::new(&vec![IterativeParamsFD::default(); size]);
//! let my_iters = Iteratives::new(&vec![IterativeParams::default(); size]);
//! ```
//!

mod default;
mod iterative_var;
mod iterative_var_fd;
mod iteratives_base;

pub use iterative_var::IterativeParams; // struct re-export
pub use iterative_var_fd::IterativeParamsFD; // struct re-export
pub use iterative_var_fd::PerturbationMethod; // enum re-export
pub use iteratives_base::Iterative; // trait re-export
pub use iteratives_base::Iteratives; // struct re-export

pub use default::{default_vec_iteratives, default_vec_iteratives_fd};
