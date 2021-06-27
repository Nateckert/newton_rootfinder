//! Advanced solver

pub mod iteratives;
pub mod model;
pub mod residuals;
pub mod solver;
mod util_nalgebra;
pub mod xml_parser;

pub use util_nalgebra::{
    omatrix_zeros_from_shape, omatrix_zeros_like, ovector_zeros_from_shape, ovector_zeros_like,
};
