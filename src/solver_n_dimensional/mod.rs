//! Advanced solver

pub mod iteratives;
pub mod model;
pub mod residuals;
pub mod solver;
mod util_nalgebra;
pub mod xml_parser;

pub use util_nalgebra::ovector_zeros_like;
