//! Model definition
//!
//! This module contains the model behavior definition through the `Model` trait
//!
//! The advanced solver is working only with model that implements this trait
//!
//! To ease the adaptation of a function to the required trait,
//! the following structs are provided :
//! - `UserModelWithFunc`: to work with a function defining the problem, finite-difference will be used
//! - `UserModelWithFuncJac`: to work with two functions, one for the model and one for the jacobian

mod model_definition;
mod model_from_func;

pub use model_definition::Model;
pub use model_from_func::{UserModelWithFunc, UserModelWithFuncJac};
