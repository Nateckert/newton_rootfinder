//! Iteratives definition
//!
//! The iteratives variables are the inputs variables X in f(X) = 0.
//!
//! It is not only a float value, that changes during the iterative resolution process.
//!
//! One might want to limit the update steps, by either:
//! - limiting the range of values to avoid non-sense values
//! - limiting the size of an update step

use std::fmt;

pub trait Iterative {
    fn set_max_steps(&mut self, max_step_abs: f64, max_step_rel: f64);
    fn set_max_values(&mut self, min_value: f64, max_value: f64);
    fn step_limitation(&self, value_current: f64, raw_step: f64) -> f64;
    fn compute_perturbation(&self, #[allow(unused_variables)] x: f64)  -> f64 {
        unimplemented!();
    }
}

pub struct Iteratives<'a, T: Iterative> {
    iteratives_params: &'a [T]
}

impl<'a, T> Iteratives<'a, T>
where
    T: Iterative,
{
    pub fn new(iteratives_params: &'a [T]) -> Self {
        Iteratives {iteratives_params}
    }

    pub fn len(&self) -> usize {
        self.iteratives_params.len()
    }
    /// Compute a limited step
    /// Return the new value after the application of the step limitation (and not the step)
    /// This is required as it can be limited by an interval for the iteratives.
    pub fn step_limitations(&self,
        values: &nalgebra::DVector<f64>,
        raw_step: &nalgebra::DVector<f64>,
        problem_size: usize,
    ) -> nalgebra::DVector<f64> {
        let mut step_lim: nalgebra::DVector<f64> = nalgebra::DVector::zeros(problem_size);

        for (i, iterative_params) in (self.iteratives_params).iter().enumerate() {
            step_lim[i] = iterative_params.step_limitation(values[i], raw_step[i]);
        }
        step_lim
    }

    pub fn compute_perturbations(&self,
        iterative_values: &nalgebra::DVector<f64>,
        problem_size: usize,
    ) -> nalgebra::DVector<f64> {
        let mut perturbations: nalgebra::DVector<f64> = nalgebra::DVector::zeros(problem_size);

        for (i, iterative_var) in (self.iteratives_params).iter().enumerate() {
            perturbations[i] = iterative_var.compute_perturbation(iterative_values[i]);
        }
        perturbations
    }
}

impl<'a, T> fmt::Display for Iteratives<'a, T>
where
    T: Iterative + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut content = String::from("Iteratives parameters:\n");
        content.push_str("==============\n");
        for (i, elt) in self.iteratives_params.iter().enumerate() {
            content.push_str("   - Iterative var ");
            content.push_str(&i.to_string());
            content.push_str(&elt.to_string());
            content.push_str("\n");
        }

        content.push_str("\n");
        write!(f, "{}", content)
    }
}
