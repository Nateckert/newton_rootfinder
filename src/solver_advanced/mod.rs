//! Advanced solver
//!
//! Solver designed to interact with a `Model`
//!
//! Vocabulary :
//! If f(X) = Y, with X and Y vectors of the same size
//! - Rootfinding methods try to find X such as Y = 0
//! - X are called the iteratives variables, as the resolution method iterates on them and changes their values.
//! - Y are called residuals.
//!
//! - Iteratives variables can be parametrized in order to defines their behavior during the iterations.
//! - Residuals values can be computed in different ways according to the residuals definition.
//!
//! # Features
//! 1. Iteratives parametrization: check the `iteratives` module
//! 2. Residuals  parametrization: check the `util/residuals` module
//! 3. Solver interaction with through the `Model` trait, check the `model` module
//! 4. Many solver options : check the `solver` module
//!
//!
//! # Upcoming features
//!

pub mod iteratives;
pub mod model;
pub mod residuals;
pub mod solver;
pub mod util;
