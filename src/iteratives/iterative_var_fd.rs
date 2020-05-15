//! Extension of iteratives. FD stands for Iterative Finite-Difference
//!
//! In addition to the already defined parameters of an iteratives,
//! The fact that the jacobian is going to be evaluated with finite-differences
//! leads to the necessity to parametrize the way the perturbation on the iteratives are made.
//!
//! Hence if x is an iterative variable, (x + dx) is used for the jacobian evaluation.
//!
//! The parametrization here defines dx with regards to :
//! - dx_abs: the absolute perturbation step
//! - dx_rel: the relative perturbation step
//!
//! For each case we would have :
//! - dx = dx_abs
//! - dx = dx_rel*abs(x)
//!
//! The implementation here allows you to choose and combine the formulas:
//! - dx = max(dx_abs, dx_rel*abs(x))
//! - dx = dx_abs + dx_rel*abs(x)
//!
//! It is also possible to get one of the two basic cases by setting the other to 0:
//! - dx_abs = 0 implies dx = dx_rel*abs(x)
//! - dx_rel = 0 implies dx = dx_abs

use std::fmt;
use crate::iteratives::Iterative;
use crate::iteratives::iterative_var;

#[derive(Debug, Clone)]
pub enum PerturbationMethod {
    Max,
    Sum,
}

impl fmt::Display for PerturbationMethod {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let result = match self {
            PerturbationMethod::Max => &"Max",
            PerturbationMethod::Sum => &"Sum",
        };

        write!(f, "{}", result)
    }
}

#[derive(Debug, Clone)]
pub struct IterativeParamsFD {
    iterative_params: iterative_var::IterativeParams,
    max_step_method: PerturbationMethod,
    dx_abs: f64,
    dx_rel: f64,
}

impl Default for IterativeParamsFD {
    fn default() -> IterativeParamsFD {
        IterativeParamsFD {
            iterative_params: iterative_var::IterativeParams::default(),
            max_step_method: PerturbationMethod::Max,
            dx_abs: 5.0e-8,
            dx_rel: 5.0e-8,
        }
    }
}

impl IterativeParamsFD {
    pub fn set_max_step_method(&mut self, max_step_method: PerturbationMethod) {
        self.max_step_method = max_step_method;
    }

    pub fn set_dx_values(&mut self, dx_abs: f64, dx_rel: f64) {
        if dx_abs <= 0.0 {
            panic!(
                "dx_abs must be strictly positive, provided value was {}",
                dx_abs
            );
        }
        if dx_rel <= 0.0 {
            panic!(
                "dx_rel must be strictly positive, provided value was {}",
                dx_rel
            );
        }

        self.dx_abs = dx_abs;
        self.dx_rel = dx_rel;
    }
}

impl Iterative for IterativeParamsFD {
    fn set_max_steps(&mut self, max_step_abs: f64, max_step_rel: f64) {
        self.iterative_params
            .set_max_steps(max_step_abs, max_step_rel)
    }

    fn set_max_values(&mut self, min_value: f64, max_value: f64) {
        self.iterative_params.set_max_values(min_value, max_value)
    }

    fn step_limitation(&self, value_current: f64, value_next: f64) -> f64 {
        self.iterative_params
            .step_limitation(value_current, value_next)
    }

    fn compute_perturbation(&self, x: f64) -> f64 {
        match self.max_step_method {
            PerturbationMethod::Max => (self.dx_abs).max(x.abs() * self.dx_rel),
            PerturbationMethod::Sum => self.dx_abs + x.abs() * self.dx_rel,
        }
    }

    fn with_finite_diff(&self) -> bool {
        true
    }
}

impl fmt::Display for IterativeParamsFD {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let column_float = String::from("--------------+");
        let width = column_float.len()-2;
        let mut content = self.iterative_params.to_string();
        content.push_str(&format!(" {:width$}|", &self.max_step_method.to_string(), width = "-----------------+".len()-2));
        content.push_str(&format!(" {:width$}|", &self.dx_abs.to_string(), width = width));
        content.push_str(&format!(" {:width$}|", &self.dx_rel.to_string(), width = width));

        write!(f, "{}", content)
    }
}
