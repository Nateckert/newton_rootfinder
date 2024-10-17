//! Solver errors
//!
//! The error API exposed to the end user is represented by the enum [SolverError]
//!
//! However, to have optimal integration between solver and model,
//! it is expected to define the potential errors raised by the model
//! through the associated types:
//! - [crate::model::Model::InaccurateValuesError]
//! - [crate::model::Model::UnusableValuesError]
//!
//! If such error cannot occur, you can default the values to [std::convert::Infallible]
//!
//! The use of such associated types allows the user model
//! to classify its error into defined categories that the solver can react to.
//! It also allows the model to define subcategories that the solver don't need to know about,
//! in order to improve the quality of the error message to ease the debugging experience.
//!
//! An explanation behind the rational for the error categories can be found
//! in the documentation of [crate::model::ModelError]
//!
//! Here is a working example implementing the associated types in the model:
//!
//! ```
//! use std::error::Error;
//! use std::fmt;
//!
//! use newton_rootfinder as nrf;
//! use nrf::iteratives;
//! use nrf::model::Model;
//! use nrf::residuals;
//!
//! struct MyDummyModel {
//!     iteratives: nalgebra::DVector<f64>,
//!     residuals: nalgebra::DVector<f64>,
//! }
//!
//! impl MyDummyModel {
//!     pub fn new() -> Self {
//!         let iteratives = nalgebra::DVector::zeros(1);
//!         let residuals = nalgebra::DVector::zeros(1);
//!         MyDummyModel {
//!             iteratives,
//!             residuals,
//!         }
//!     }
//! }
//!
//! #[derive(Debug)]
//! pub enum MyCustomErrors {
//!     NotAGoodValue,
//! }
//!
//! impl fmt::Display for MyCustomErrors {
//!     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//!         match self {
//!             _ => write!(f, "{}", "Not a good value"),
//!         }
//!     }
//! }
//!
//! impl Error for MyCustomErrors {}
//!
//! impl Model<nalgebra::Dyn> for MyDummyModel {
//!     type InaccurateValuesError = MyCustomErrors;
//!     type UnusableValuesError = MyCustomErrors;
//!
//!     fn len_problem(&self) -> usize {
//!         1
//!     }
//!
//!     fn get_iteratives(&self) -> nalgebra::DVector<f64> {
//!         return self.iteratives.clone();
//!     }
//!
//!     fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>) {
//!         self.iteratives = iteratives.clone();
//!     }
//!
//!     fn get_residuals(&self) -> nrf::residuals::ResidualsValues<nalgebra::Dyn> {
//!         return nrf::residuals::ResidualsValues::new(
//!             self.residuals.clone(),
//!             nalgebra::DVector::zeros(1),
//!         );
//!     }
//!
//!     fn evaluate(&mut self) -> Result<(), nrf::model::ModelError<Self, nalgebra::Dyn>> {
//!         self.residuals[0] = self.iteratives[0].powi(2) - 2.0;
//!         Err(nrf::model::ModelError::InaccurateValuesError(
//!             MyCustomErrors::NotAGoodValue,
//!         ))
//!     }
//! }
//!
//! fn main() {
//!     let problem_size = 1;
//!     let mut init = nalgebra::DVector::zeros(problem_size);
//!     init[0] = 1.0;
//!
//!     let damping = false;
//!
//!     let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
//!     let iter_params = iteratives::Iteratives::new(&vec_iter_params);
//!     let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
//!     let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
//!     let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
//!     let mut rf = nrf::solver::default_with_guess(
//!         init,
//!         &iter_params,
//!         &res_config,
//!         nrf::solver::ResolutionMethod::NewtonRaphson,
//!         damping,
//!     );
//!
//!     let mut my_model = MyDummyModel::new();
//!
//!     let result = rf.solve(&mut my_model).unwrap_err();
//!     let expected: nrf::errors::SolverError<nrf::model::UserModelFromFunction, nalgebra::Dyn> =
//!         nrf::errors::SolverError::FinalEvaluationError;
//!     assert_eq!(expected.to_string(), result.to_string());
//!     assert!(float_cmp::approx_eq!(
//!         f64,
//!         my_model.get_iteratives()[0],
//!         std::f64::consts::SQRT_2,
//!         epsilon = 1e-6
//!     ));
//! }
//! ```

use std::error::Error;
use std::fmt;

/// Errors for solver control flow
///
/// These error are not exposed directly to the API
pub enum SolverInternalError<M, D>
where
    M: crate::model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    InvalidJacobianError(crate::model::ModelError<M, D>),
    InvalidJacobianInverseError,
}

impl<M, D> fmt::Display for SolverInternalError<M, D>
where
    M: crate::model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
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
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

pub struct NonInvertibleJacobian;

/// Error returned by the [crate::solver::RootFinder::solve] method
///
/// Exit status:
/// - [SolverError::NonConvergenceError] : finished all the iterations but didn't find a root
/// - [SolverError::ModelInitialEvaluationError] : the algorithm must be able to evaluate the model correctly at the begin of the resolution process, it failed in that case
/// - [SolverError::ModelEvaluationError] : during the iterative process, while performing an update, a model error occured
/// - [SolverError::JacobianError] : during the jacobian evaluation, an error occured
/// - [SolverError::FinalEvaluationError] : the algorithm managed to converged but the model returned an error at convergence
pub enum SolverError<M, D>
where
    M: crate::model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
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
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
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
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<M, D> Error for SolverError<M, D>
where
    M: crate::model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
}
