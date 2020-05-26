//! Definition of residuals
//!
//! The residuals are splitted between the solver parametrization and the model output:
//! - `ResidualsConfig` for the solver
//! - `ResidualsValues` for the model output
//!
//! In addition to this two base struct, the following one are introduced:
//! - `ResidualConfig` to make easier to create the `ResidualsConfig`from a slice of the ladder
//! - `JacobianValues` to manipulate the jacobian outputs of a model when it is provided (non applicable for finite-differences)

mod config;
mod values;
pub use config::ResidualConfig;
pub use config::ResidualsConfig;
pub use values::JacobianValues;
pub use values::ResidualsValues;

use std::fmt;

/// Normalization method used by the `normalization` function.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum NormalizationMethod {
    Abs,
    Rel,
    Adapt,
}

/// Not used yet
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
///
/// - Abs (absolute) is the plain difference evaluation
/// - Rel (relative) is the relative value evaluation
/// - Adapt (adaptative) is designed to behave like Abs for near zero values and like Rel for big values
///
/// The formula are:
/// - Abs: left - right
/// - Rel: (left - right)/(abs(left+right)/2)
/// - Adapt: (left - right)/(1+abs(left+right)/2)
///
/// Default of each formula:
/// - Abs: does not take into account the order of magnitude of the residuals
/// - Rel: behave poorly if the residual is close to zero
/// - Adapt: behave poorly if one member of the residual is close to zero and the other one is big, as the value will be close to either -2 or 2.
///
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

/// Derivation of the normalization method
///
/// This method is used when the jacobian is provided by the model and not calculated through finite-difference
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

/// Default method to construct a residual with Abs values
///
/// # Examples
/// ```
/// extern crate newton_rootfinder;
/// use newton_rootfinder::solver_advanced as nrf;
///
/// let problem_size = 3;
/// let stopping_criterias_ref = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
/// let update_methods_ref = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
///
/// let residuals_default = nrf::residuals::default_vec_residuals(problem_size);
///
/// for i in 0..problem_size {
///     assert_eq!(residuals_default[i].get_stopping_criteria(), stopping_criterias_ref[i]);
///     assert_eq!(residuals_default[i].get_update_method(), update_methods_ref[i]);
/// }
///```
pub fn default_vec_residuals(size: usize) -> Vec<ResidualConfig> {
    vec![ResidualConfig::default(); size]
}
