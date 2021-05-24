//! The solver has been designed to operate on a user defined model.
//! To define such a model, the user must adapt the problem through the implementation of the [Model] trait.
//!
//! # Implementation
//!
//! Don't hesitate to check the [Model] trait documentation !
//!
//! # Motivations
//!
//! ## Access to additional computed values
//!
//! An usual rootfinding algorithm operates on a mathematical function `f(iteratives) -> residuals`.
//! The solver attempts to find `iteratives` such that `norm(f(iteratives)) < tolerance`.
//!
//! However, in real life cases, the model computes other quantities that are the main focus of the end-user of the solver.
//! In fact, the `residuals` parameters have often few to no significations,
//! the user being mostly interested by other quantities.
//!
//! With most available solvers, computing the other quantities requires another function call to extract them.
//! This extra function call being made with the `iteratives` values found by the solver.
//!
//! ## Solver state differentiation
//!
//! A rootfinding algorithm will call several times a given model at different steps of the resolution process.
//!
//! For example, it calls a model to evaluate the residuals and will also call the model to evaluate the jacobian matrix.
//! During this different steps, one can want to have different behaviors in :
//! - the error handling
//! - the computation process
//!
//! *Error handling differentiation*:
//!
//! If a computation outside of the validity domain is performed during the convergence process,
//! the solver could recover from it and the final solution could still be in the validity domain.
//! If an error is raised during this computation, the program would abort.
//! Hence the potential need for the user to want to handle errors differently at different steps of the resolution process.
//!
//! *Computation process differentiation*:
//!
//! To speed-up computation, one can cache results that can be reused during the evaluation of the jacobian.
//! For rootfinding algorithm working with finite differantiation, the interesting cached results would be the one from the reference point.
//!
//! # Quick & easy adaptation from user defined function
//!
//! In practice, in most of the case, the user's problem is defined through a function or a closure.
//! A mecanism has been provided to implement the [Model] trait automatically given a user defined function.
//!
//! To ease the adaptation of a function to the required trait,
//! the following structs are provided :
//! - [UserModelFromFunction]: to work with a function defining the problem, finite-difference will be used
//! - [UserModelFromFunctionAndJacobian]: to work with two functions, one for the model and one for the jacobian
//! - [UserModelFromClosure]: to work with a closure defining the problem, finite-difference will be used
//! - [UserModelFromClosureAndJacobian]: to work with two closures, one for the model and one for the jacobian

mod model_definition;
mod model_from_closure;
mod model_from_func;

pub use model_definition::Model;
pub use model_from_closure::{UserModelFromClosure, UserModelFromClosureAndJacobian};
pub use model_from_func::{UserModelFromFunction, UserModelFromFunctionAndJacobian};
