//! Definition of residuals
//!
//! It is splitted in-between the left and right terms of the equation

mod config;
mod values;
pub use config::ResidualConfig;
pub use config::ResidualsConfig;
pub use values::JacobianValues;
pub use values::ResidualsValues;

use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum NormalizationMethod {
    Abs,
    Rel,
    Adapt,
}

pub enum StoppingCriteria {
    OutputTol,
    InputTol,
    SumInputOutputTol,
}

impl fmt::Display for NormalizationMethod {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let result = match self {
            NormalizationMethod::Abs => &"Absolute Normalization",
            NormalizationMethod::Rel => &"Relative Normalization",
            NormalizationMethod::Adapt => &"Adaptative Normalization",
        };

        write!(f, "{}", result)
    }
}

/// Compute the residue according to the normalization method
/// Abs (absolute) is the plain difference evaluation
/// Rel (relative) is the relative value evaluation
/// Adapt (adaptative) is designed to behave like Abs for near zero values and like Rel for big values
/// # Examples
/// ```
/// extern crate newton_rootfinder;
/// use newton_rootfinder::solver_advanced as nrf;
/// extern crate float_cmp;
/// use float_cmp::*;
/// use nrf::residuals::*;
///
/// let small_values_abs = normalization(0.1, -0.15, NormalizationMethod::Abs);
/// assert!(approx_eq!(f64, small_values_abs, 0.25, ulps = 2));
///
/// let small_values_rel = normalization(0.1, -0.15, NormalizationMethod::Rel);
/// assert!(approx_eq!(f64, small_values_rel, 10.0, ulps = 2));
///
/// let small_values_adapt = normalization(0.1, -0.15, NormalizationMethod::Adapt);
/// assert!(approx_eq!(f64, small_values_adapt, 0.24390243902439027, ulps = 2));
///
/// let big_values_abs = normalization(101.1, 101.25, NormalizationMethod::Abs);
/// assert!(approx_eq!(f64, big_values_abs, -0.15000000000000568, ulps = 2));
///
/// let big_values_rel = normalization(101.1, 101.25, NormalizationMethod::Rel);
/// assert!(approx_eq!(f64, big_values_rel, -0.0014825796886583217, ulps = 2));
///
/// let big_values_adapt = normalization(101.1, 101.25, NormalizationMethod::Adapt);
/// assert!(approx_eq!(f64, big_values_adapt, -0.0014680694886225172, ulps = 2));
/// ```

pub fn normalization(x: f64, y: f64, normalization_method: NormalizationMethod) -> f64 {
    match normalization_method {
        NormalizationMethod::Abs => x - y,
        NormalizationMethod::Rel => (x - y) / ((x + y).abs() / 2.0),
        NormalizationMethod::Adapt => (x - y) / (1.0 + (x + y).abs() / 2.0),
    }
}

pub fn deriv_normalization(
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
    normalization_method: NormalizationMethod,
) -> f64 {
    match normalization_method {
        NormalizationMethod::Abs => dx - dy,
        NormalizationMethod::Rel => {
            let diff = x - y;
            let deriv_diff = dx - dy;
            let sum = x + y;
            let deriv_sum = dx + dy;

            2.0 * ((deriv_diff) * sum.abs() - deriv_sum * diff * sum.signum()) / (sum.powi(2))
        }
        NormalizationMethod::Adapt => {
            let diff = x - y;
            let deriv_diff = dx - dy;
            let avg = (x + y) / 2.0;
            let deriv_avg = (dx + dy) / 2.0;
            let denominator = 1.0 + avg.abs();
            let deriv_denominator = deriv_avg * avg.signum();

            (deriv_diff * denominator - deriv_denominator * diff) / (denominator.powi(2))
        }
    }
}

pub fn default_vec_residuals(size: usize) -> Vec<ResidualConfig> {
    vec![ResidualConfig::default(); size]
}
