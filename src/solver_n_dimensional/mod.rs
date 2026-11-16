//! Advanced solver

pub mod errors;
pub mod iteratives;
pub mod log;
pub mod model;
pub mod residuals;
pub mod solver;
#[cfg(test)]
mod test_util;
mod util_nalgebra;

#[cfg(feature = "xml_config_file")]
pub mod xml_parser;

pub(crate) use util_nalgebra::{
    omatrix_zeros_from_shape, omatrix_zeros_like_ovector, ovector_zeros_from_shape,
    ovector_zeros_like,
};
