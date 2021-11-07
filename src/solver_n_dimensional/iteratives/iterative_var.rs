use super::Iterative;
use std::fmt;

/// The parameters of an iterative variable
///
/// This parameters are used by the `step_limitation()` method from the `Iterative` trait to reduce the size of a step
#[derive(Debug, Clone, PartialEq)]
pub struct IterativeParams {
    max_step_abs: f64,
    max_step_rel: f64,
    min_value: f64,
    max_value: f64,
}

impl IterativeParams {
    /// The parameters are used by the `step_limitation()` method from the `Iterative` trait to reduce the size of a step
    ///
    /// Both value `max_step_abs` and `max_step_rel` must be positive
    ///
    /// The `min_value` must be lower than the `max_value`
    pub fn new(max_step_abs: f64, max_step_rel: f64, min_value: f64, max_value: f64) -> Self {
        if max_step_abs <= 0.0 {
            panic!(
                "max_step_abs must be strictly positive, provided value was {}",
                max_step_abs
            );
        }
        if max_step_rel <= 0.0 {
            panic!(
                "max_step_rel must be strictly positive, provided value was {}",
                max_step_rel
            );
        }

        if min_value >= max_value {
            panic!(
                "min_value must be strictly inferior to max_value, provided values are {} > {}",
                min_value, max_value
            );
        }

        IterativeParams {
            max_step_abs,
            max_step_rel,
            min_value,
            max_value,
        }
    }

    pub fn get_min_value(&self) -> f64 {
        self.min_value
    }

    pub fn get_max_value(&self) -> f64 {
        self.max_value
    }

    pub fn get_max_step_abs(&self) -> f64 {
        self.max_step_abs
    }

    pub fn get_max_step_rel(&self) -> f64 {
        self.max_step_rel
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
    /// Compute a limited update step
    ///
    /// The step size is reduced according to the following criteria :
    ///```block
    /// abs(step_size) < max_step_abs
    /// abs(step_size) < max_step_rel*abs(iterative_value)
    ///```
    /// Also, the step must not violated the constraints on the `min_value` and `max_value` of the iterative variable.
    ///
    /// **Warning**:
    /// setting the parameters max_step_rel to a value different from infinity
    /// might lead to very reduced step size if the iterative value is near zero.
    ///
    /// # Examples
    /// ```
    /// use newton_rootfinder as nrf;
    /// use nrf::iteratives::*;
    ///
    /// let (max_step_abs, max_step_rel, min_value, max_value) = (1.0, 1.0, f64::NEG_INFINITY, f64::INFINITY);
    /// let mut iterative_var = IterativeParams::new(max_step_abs, max_step_rel, min_value, max_value);
    /// assert_eq!(iterative_var.step_limitation(1.0, 1.0), 2.0);
    /// assert_eq!(iterative_var.step_limitation(1.0, 3.0), 2.0);
    ///
    /// let (max_step_abs, max_step_rel, min_value, max_value) = (0.1, 0.5, f64::NEG_INFINITY, f64::INFINITY);
    /// let mut iterative_var = IterativeParams::new(max_step_abs, max_step_rel, min_value, max_value);
    /// assert_eq!(iterative_var.step_limitation(1.5, 0.5), 1.6);
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
        content.push_str(&format!(
            " {:width$}|",
            &self.max_step_abs.to_string(),
            width = 13
        ));
        content.push_str(&format!(
            " {:width$}|",
            &self.max_step_rel.to_string(),
            width = 13
        ));
        content.push_str(&format!(
            " {:width$}|",
            &self.min_value.to_string(),
            width = 13
        ));
        content.push_str(&format!(
            " {:width$}|",
            &self.max_value.to_string(),
            width = 13
        ));
        write!(f, "{}", content)
    }
}
