use std::f64;
use std::fmt;

#[derive(Debug, Copy, Clone)]
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
        //        let mut result = String::new();
        let result = match self {
            NormalizationMethod::Abs => &"Absolute Normalization Method",
            NormalizationMethod::Rel => &"Relative Normalization Method",
            NormalizationMethod::Adapt => &"Adaptative Normalization Method",
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
/// extern crate newton_rootfinder as nrf;
/// extern crate float_cmp;
/// use float_cmp::*;
/// use nrf::util::residuals::*;
///
/// let small_values_abs = normalization(0.1, -0.15, &NormalizationMethod::Abs);
/// assert!(approx_eq!(f64, small_values_abs, 0.25, ulps = 2));
///
/// let small_values_rel = normalization(0.1, -0.15, &NormalizationMethod::Rel);
/// assert!(approx_eq!(f64, small_values_rel, 10.0, ulps = 2));
///
/// let small_values_adapt = normalization(0.1, -0.15, &NormalizationMethod::Adapt);
/// assert!(approx_eq!(f64, small_values_adapt, 0.24390243902439027, ulps = 2));
///
/// let big_values_abs = normalization(101.1, 101.25, &NormalizationMethod::Abs);
/// assert!(approx_eq!(f64, big_values_abs, -0.15000000000000568, ulps = 2));
///
/// let big_values_rel = normalization(101.1, 101.25, &NormalizationMethod::Rel);
/// assert!(approx_eq!(f64, big_values_rel, -0.0014825796886583217, ulps = 2));
///
/// let big_values_adapt = normalization(101.1, 101.25, &NormalizationMethod::Adapt);
/// assert!(approx_eq!(f64, big_values_adapt, -0.0014680694886225172, ulps = 2));
/// ```

pub fn normalization(x: f64, y: f64, normalization_method: NormalizationMethod) -> f64 {
    match normalization_method {
        NormalizationMethod::Abs => x - y,
        NormalizationMethod::Rel => (x - y) / ((x + y).abs() / 2.0),
        NormalizationMethod::Adapt => (x - y) / (1.0 + (x + y).abs() / 2.0),
    }
}

#[derive(Debug)]
pub struct ResidualsValues {
    left: nalgebra::DVector<f64>,
    right: nalgebra::DVector<f64>,
}
impl ResidualsValues {
    pub fn new(left: nalgebra::DVector<f64>, right: nalgebra::DVector<f64>) -> Self {
        ResidualsValues { left, right }
    }

    pub fn get_values(&self, index: usize) -> (f64, f64) {
        (self.left[index], self.right[index])
    }
}

#[derive(Debug)]
pub struct ResidualsConfig {
    stopping_critera: Vec<NormalizationMethod>,
    iteration_update_method: Vec<NormalizationMethod>,
    problem_size: usize,
}

impl ResidualsConfig {
    pub fn new(
        stopping_critera: Vec<NormalizationMethod>,
        iteration_update_method: Vec<NormalizationMethod>,
    ) -> Self {
        if stopping_critera.len() != iteration_update_method.len() {
            panic!("Dimension mismatch : stopping_critera.len() = {}, iteration_update_method.len() = {}",
                    stopping_critera.len(), iteration_update_method.len());
        }
        let problem_size = stopping_critera.len();

        ResidualsConfig {
            stopping_critera,
            iteration_update_method,
            problem_size,
        }
    }

    pub fn default_with_size(problem_size: usize) -> Self {
        let stopping_critera = vec![NormalizationMethod::Abs; problem_size];
        let iteration_update_method = vec![NormalizationMethod::Abs; problem_size];
        ResidualsConfig {
            stopping_critera,
            iteration_update_method,
            problem_size,
        }
    }

    pub fn evaluate_update_residuals(&self, values: &ResidualsValues) -> nalgebra::DVector<f64> {
        let mut update_residuals: nalgebra::DVector<f64> =
            nalgebra::DVector::zeros(self.problem_size);

        for i in 0..self.problem_size {
            let (left, right) = values.get_values(i);
            update_residuals[i] = normalization(left, right, self.iteration_update_method[i]);
        }
        update_residuals
    }

    pub fn evaluate_stopping_residuals(&self, values: &ResidualsValues) -> nalgebra::DVector<f64> {
        let mut stopping_residuals: nalgebra::DVector<f64> =
            nalgebra::DVector::zeros(self.problem_size);

        for i in 0..self.problem_size {
            let (left, right) = values.get_values(i);
            stopping_residuals[i] = normalization(left, right, self.stopping_critera[i]).abs();
        }
        stopping_residuals
    }
}
