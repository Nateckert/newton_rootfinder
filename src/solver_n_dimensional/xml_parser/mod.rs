//! Useful functions
//!
//! # Parser
//! A parser to an xml configuration file is provided to ease the construction of the parameters:
//! - [from_xml_finite_diff()]
//! - [from_xml_jacobian()]
//!
//! For the meaning of each parameters, please refer to the documentation of the related module:
//! - solver: [crate::solver::SolverParameters]
//! - iteratives: [crate::iteratives]
//! - residuals: [crate::residuals]
//!
//! Parse a xml configuration file to create the elements that are required by a [crate::solver::RootFinder]:
//!
//! ```xml
//! <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
//! <nrf>
//!     <solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="NR"/>
//!     <iteratives min_value="-inf"  max_value="inf" max_step_abs="inf" max_step_rel="inf">
//!         <iterative id="0" min_value="-inf"  max_value="inf" max_step_abs="100" max_step_rel="0.5"/>
//!         <iterative id="1" min_value="0"     max_value="inf" max_step_abs="inf" max_step_rel="0.5"/>
//!         <iterative id="2" min_value="-inf"  max_value="12"  max_step_abs="100" max_step_rel="inf"/>
//!     </iteratives>
//!     <residuals stopping_criteria="Abs" update_method="Abs">
//!         <residual id="0" stopping_criteria="Adapt"     update_method="Abs"/>
//!         <residual id="1" stopping_criteria="Rel"       update_method="Abs"/>
//!         <residual id="2" stopping_criteria="Adapt"     update_method="Rel"/>
//!     </residuals>
//! </nrf>
//! ```
//!
//! The values provided in the iteratives and residuals nodes will act as default values.
//!
//! These values are taken into account only if none are provided for a given iterative or residual
//!

mod node_iterative;
mod node_iterative_fd;
mod node_iterative_jac;
mod node_residual;
mod node_solver;
mod util;
mod xml_file_fd;
mod xml_file_jac;

pub use xml_file_fd::from_xml_finite_diff;
pub use xml_file_jac::from_xml_jacobian;
