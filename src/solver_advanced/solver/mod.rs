//! Solver configuration
//!
//! # Features
//! 1. Simulation log available for debugging, check the `set_debug()` method
//! 2. Damping, check the `set_damping()` method

mod default;
mod log;
mod solver_advanced;

pub use default::default_with_guess;
pub use solver_advanced::RootFinder;
pub use solver_advanced::SolverParameters;
