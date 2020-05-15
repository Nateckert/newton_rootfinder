//! Advanced solver
//!
//! ## Examples
//!
//! ```
//! extern crate newton_rootfinder as nrf;
//! use nrf::model::Model;
//! use nrf::iteratives;
//!
//! extern crate nalgebra;
//!
//! /// Equation : x**2 - 2 = 0
//! fn square2(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
//!     let mut y = x * x;
//!     y[0] -= 2.0;
//!     y
//! }
//!
//! fn main() {
//!   let problem_size = 1;
//!   let init_guess = nalgebra::DVector::from_vec(vec![1.0]);
//!   let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
//!   let iter_params = iteratives::Iteratives::new(&vec_iter_params);
//!   let rf = nrf::solver::RootFinder::default_with_guess_fd(init_guess, iter_params);
//!   let mut user_model =
//!       nrf::model_with_func::UserModelWithFunc::new(problem_size, square2);
//!
//!   rf.solve(&mut user_model);
//!
//!   println!("{}", user_model.get_iteratives()[0]);
//!   // print 1.4142135623747443
//! }
//! ```

use std::fmt;

extern crate nalgebra;

use crate::model;
use crate::iteratives;
use crate::iteratives::Iterative;

use crate::util::jacobian;
use crate::util::residuals;
use crate::log;

pub struct RootFinder<'a, T>
where
    T: Iterative + fmt::Display
{
    initial_guess: nalgebra::DVector<f64>,
    iters_params: iteratives::Iteratives<'a, T>,
    residuals_config: residuals::ResidualsConfig,
    problem_size: usize,
    tolerance: f64,
    max_iter: usize,
    damping: bool,
    debug: bool,
    solver_log: log::SolverLog,
}

impl<'a, T> RootFinder<'a, T>
where
    T: Iterative + fmt::Display
{
    pub fn new(
        initial_guess: nalgebra::DVector<f64>,
        iters_params: iteratives::Iteratives<'a, T>,
        residuals_config: residuals::ResidualsConfig,
        problem_size: usize,
        tolerance: f64,
        max_iter: usize,
    ) -> Self {
        let damping = false;
        let debug = false;
        let solver_log = log::SolverLog::new();

        RootFinder {
            initial_guess,
            iters_params,
            residuals_config,
            problem_size,
            tolerance,
            max_iter,
            damping,
            debug,
            solver_log
        }
    }

    pub fn write_log(&self) {
        self.solver_log.write();
    }

    pub fn parameters_to_log(&mut self) {
        let mut str_parameters = String::from("Solver parameters\n");
        str_parameters.push_str("=================\n\n");
        str_parameters.push_str("Max iteration: ");
        str_parameters.push_str(&self.max_iter.to_string());
        str_parameters.push_str("\n----------------------------\n\n");
        str_parameters.push_str("Tolerance: ");
        str_parameters.push_str(&self.tolerance.to_string());
        str_parameters.push_str("\n----------------------------\n\n");
        str_parameters.push_str(&self.residuals_config.to_string());
        str_parameters.push_str("\n----------------------------\n\n");
        str_parameters.push_str(&self.iters_params.to_string());
        str_parameters.push_str("\n----------------------------\n\n");
        str_parameters.push_str("Init guess:\n");
        str_parameters.push_str(&self.initial_guess.to_string());
        str_parameters.push_str("\n----------------------------\n\n");
        self.solver_log.add_content(&str_parameters);
    }

    pub fn set_damping(&mut self, damping: bool) {
        self.damping = damping;
    }

    fn evaluate_max_error<M: model::Model>(&self, model: &M) -> f64 {
        let residuals_values = model.get_residuals();
        let stopping_residuals = self
            .residuals_config
            .evaluate_stopping_residuals(&residuals_values);
        stopping_residuals.amax()
    }

    fn compute_jac_func<M: model::Model>(&self, model: &mut M) -> nalgebra::DMatrix<f64> {
        let residuals_values = model.get_residuals();

        let jacobians = model.get_jacobian();
        let normalization_method = self.residuals_config.get_update_method();
        jacobians.normalize(&residuals_values, &normalization_method)
    }

    fn compute_jac_fd<M: model::Model>(&self, model: &mut M) -> nalgebra::DMatrix<f64> {
        let iters_values = model.get_iteratives();

        let perturbations = self.iters_params.compute_perturbations(
            &iters_values,
            self.problem_size,
        );

        jacobian::jacobian_evaluation(model, &perturbations, &(self.residuals_config))
    }

    fn compute_next<M: model::Model>(&self, model: &mut M) -> nalgebra::DVector<f64> {
        let jac = if model.jacobian_provided() {
            self.compute_jac_func(model)
        } else {
            self.compute_jac_fd(model)
        };

        let residuals = self
            .residuals_config
            .evaluate_update_residuals(&model.get_residuals());

        let raw_step = jacobian::newton_raw_step_size(&residuals, &jac);

        let iter_values = model.get_iteratives();

        self.iters_params.step_limitations(
            &iter_values,
            &raw_step,
            self.problem_size,
        )
    }

    fn update_model<M: model::Model>(
        &self,
        model: &mut M,
        proposed_guess: &nalgebra::DVector<f64>,
    ) -> f64 {
        let max_error = self.evaluate_max_error(model);
        let current_guess = model.get_iteratives();

        model.set_iteratives(proposed_guess);
        model.evaluate();
        let mut max_error_next = self.evaluate_max_error(model);

        if self.damping {
            if max_error_next > max_error {
                let damped_guess = (proposed_guess + &current_guess) / 2.0;
                model.set_iteratives(&damped_guess);
                model.evaluate();
                max_error_next = self.evaluate_max_error(model);
            }
        }

        max_error_next
    }

    pub fn solve<M>(&self, model: &mut M)
    where
        M: model::Model,
    {
        model.init();
        model.set_iteratives(&self.initial_guess);
        model.evaluate();

        let mut max_error = self.evaluate_max_error(model);

        let mut iter = 0;

        while max_error > self.tolerance && iter < self.max_iter {
            iter += 1;
            let proposed_guess = self.compute_next(model);
            max_error = self.update_model(model, &proposed_guess);
        }
    }

    pub fn default_with_guess_fd(initial_guess: nalgebra::DVector<f64>,
                                    iters_params: iteratives::Iteratives<'a, T>) -> Self {
        let problem_size = initial_guess.len();
        let residuals_config = residuals::ResidualsConfig::default_with_size(problem_size);
        let tolerance: f64 = 1e-6;
        let max_iter: usize = 50;
        let damping: bool = false;
        let debug: bool = false;
        let solver_log = log::SolverLog::new();

        RootFinder {
            initial_guess,
            iters_params,
            residuals_config,
            problem_size,
            tolerance,
            max_iter,
            damping,
            debug,
            solver_log
        }
    }
}
