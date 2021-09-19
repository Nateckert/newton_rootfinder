//! Solver errors
use std::fmt;

/// Errors for solver control flow
///
/// These error are not exposed directly to the API
pub enum SolverInternalError<M, D>
where
    M: crate::model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
    InvalidJacobianError(crate::model::ModelError<M, D>),
    InvalidJacobianInverseError,
}

pub struct NonInvertibleJacobian;

#[derive(Debug)]
pub enum SolverError {
    NonConvergence(String),
    ModelInitialEvaluationError(String),
}

impl<M, D> fmt::Display for SolverInternalError<M, D>
where
    M: crate::model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidJacobianError(error) => write!(f, "Invalid jacobian: {}", error),
            Self::InvalidJacobianInverseError => write!(f, "Non invertible jacobian"),
        }
    }
}

impl<M, D> fmt::Debug for SolverInternalError<M, D>
where
    M: crate::model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
