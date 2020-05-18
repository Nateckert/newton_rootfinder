use std::fmt;
use crate::iteratives::Iterative;

#[derive(Debug, Clone)]
pub struct IterativeParams {
    max_step_abs: f64,
    max_step_rel: f64,
    min_value: f64,
    max_value: f64,
}

impl IterativeParams {
    pub fn new() -> Self {
        IterativeParams::default()
    }
}

impl Default for IterativeParams {
    fn default() -> IterativeParams {
        IterativeParams {
            max_step_abs: f64::INFINITY,
            max_step_rel: f64::INFINITY,
            min_value: f64::NEG_INFINITY,
            max_value: f64::INFINITY,
        }
    }
}

impl Iterative for IterativeParams {
    fn set_max_steps(&mut self, max_step_abs: f64, max_step_rel: f64) {
        if max_step_abs <= 0.0 {
            panic!(
                "maxStepAbs must be strictly positive, provided value was {}",
                max_step_abs
            );
        }
        if max_step_rel <= 0.0 {
            panic!(
                "maxStepRel must be strictly positive, provided value was {}",
                max_step_rel
            );
        }

        self.max_step_abs = max_step_abs;
        self.max_step_rel = max_step_rel;
    }

    fn set_max_values(&mut self, min_value: f64, max_value: f64) {
        if min_value >= max_value {
            panic!(
                "minValue must be strictly inferior to maxValue, provided values are {} > {}",
                min_value, max_value
            );
        }

        self.min_value = min_value;
        self.max_value = max_value;
    }

    /// Compute a limited update step
    ///
    /// # Examples
    /// ```
    /// extern crate newton_rootfinder as nrf;
    /// use nrf::iteratives::*;
    ///
    /// let mut iterative_var = IterativeParams::new();
    ///
    /// iterative_var.set_max_steps(1.0, 1.0);
    /// assert_eq!(iterative_var.step_limitation(1.0, 1.0), 2.0);
    /// assert_eq!(iterative_var.step_limitation(1.0, 3.0), 2.0);
    ///
    /// iterative_var.set_max_steps(0.1, 0.5);
    /// assert_eq!(iterative_var.step_limitation(1.5, 0.5), 1.6);
    ///
    /// iterative_var.set_max_steps(0.1, 0.5);
    /// assert_eq!(iterative_var.step_limitation(0.1, 3.0), 0.15000000000000002);
    /// ```

    fn step_limitation(&self, value_current: f64, raw_step: f64) -> f64 {
        let max_step = self
            .max_step_abs
            .min(self.max_step_rel * value_current.abs());

        let abs_step = raw_step.abs();
        let sign_step = raw_step.signum();

        let step_lim = (max_step.min(abs_step)) * sign_step;
        // limitation by max_step_abs and max_step_rel
        let value_next_lim = value_current + step_lim;

        // limitation by min_value and max_value
        (value_next_lim.max(self.min_value)).min(self.max_value)
    }
}

impl fmt::Display for IterativeParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut content = String::new();
        content.push_str(&format!(" {:width$}|", &self.max_step_abs.to_string(), width = 13));
        content.push_str(&format!(" {:width$}|", &self.max_step_rel.to_string(), width = 13));
        content.push_str(&format!(" {:width$}|", &self.min_value.to_string(), width = 13));
        content.push_str(&format!(" {:width$}|", &self.max_value.to_string(), width = 13));
        write!(f, "{}", content)
    }
}
