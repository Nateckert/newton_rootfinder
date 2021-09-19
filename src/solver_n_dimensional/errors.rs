//! Solver errors
use std::error::Error;

/// Errors for solver control flow
///
/// These error are not exposed directly to the API
#[derive(Debug)]
pub enum SolverInternalError {
    InvalidJacobianError(Box<dyn Error>),
}

pub enum SolverError {
    NonConvergence,
}
