//! Solver errors
use std::error::Error;
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

pub struct NonInvertibleJacobian;

pub enum SolverError<M, D>
where
    M: crate::model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
    NonConvergenceError,
    ModelInitialEvaluationError(String),
    ModelEvaluationError(crate::model::ModelError<M, D>),
    JacobianError(SolverInternalError<M, D>),
    FinalEvaluationError,
}

impl<M, D> fmt::Display for SolverError<M, D>
where
    M: crate::model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::NonConvergenceError => write!(f, "Convergence not reached"),
            Self::ModelInitialEvaluationError(error) => {
                write!(f, "Initial model evaluation failed: {}", error)
            }
            Self::ModelEvaluationError(error) => {
                write!(f, "Model evaluation failed: {}", error)
            }
            Self::JacobianError(error) => {
                write!(f, "Jacobian error: {}", error)
            }
            Self::FinalEvaluationError => {
                write!(f, "Final model evaluation failed")
            }
        }
    }
}

impl<M, D> fmt::Debug for SolverError<M, D>
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

impl<M, D> Error for SolverError<M, D>
where
    M: crate::model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
}
