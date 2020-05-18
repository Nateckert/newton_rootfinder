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
//!   let mut rf = nrf::solver::RootFinder::default_with_guess(init_guess, iter_params);
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
    iter: usize,
    damping: bool,
    debug: bool,
    solver_log: super::log::SolverLog,
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
        let solver_log = super::log::SolverLog::new();
        let iter = 0;

        RootFinder {
            initial_guess,
            iters_params,
            residuals_config,
            problem_size,
            tolerance,
            max_iter,
            iter,
            damping,
            debug,
            solver_log
        }
    }

    pub fn set_damping(&mut self, damping: bool) {
        self.damping = damping;
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    fn evaluate_errors<M: model::Model>(&self, model: &M) -> nalgebra::DVector<f64> {
        let residuals_values = model.get_residuals();
        self.residuals_config
            .evaluate_stopping_residuals(&residuals_values)
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

    fn compute_next<M: model::Model>(&mut self, model: &mut M) -> nalgebra::DVector<f64> {
        let jac = if model.jacobian_provided() {
            self.compute_jac_func(model)
        } else {
            self.compute_jac_fd(model)
        };

        if self.debug {
            self.jac_to_log(&jac);
        }

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
        &mut self,
        model: &mut M,
        proposed_guess: &nalgebra::DVector<f64>,
    ) -> nalgebra::DVector<f64> {
        let errors = self.evaluate_errors(model);
        let max_error = errors.amax();
        let current_guess = model.get_iteratives();

        model.set_iteratives(proposed_guess);
        model.evaluate();
        let mut errors_next = self.evaluate_errors(model);
        let mut max_error_next = errors_next.amax();

        if self.debug {
            self.iteration_to_log(model, &errors_next);
        }

        if self.damping {
            if max_error_next > max_error {
                // update formula : X = X - damping_factor*res/jac
                // proposed_guess is with damping_factor = 1
                let damping_factor = 1.0/2.0;
                let damped_guess = &current_guess*(1.0-damping_factor) + proposed_guess*damping_factor;
                model.set_iteratives(&damped_guess);
                model.evaluate();
                errors_next = self.evaluate_errors(model);
                max_error_next = errors_next.amax();

                if self.debug {
                    self.damping_to_log(model, &errors_next);
                }
            }
        }

        errors_next
    }

    pub fn solve<M>(&mut self, model: &mut M)
    where
        M: model::Model,
    {
        model.init();
        model.set_iteratives(&self.initial_guess);
        model.evaluate();

        let mut errors = self.evaluate_errors(model);
        let mut max_error = errors.amax();

        if self.debug {
            self.parameters_to_log();
            self.iteration_to_log(model, &errors);
        }

        while max_error > self.tolerance && self.iter < self.max_iter {
            self.iter += 1;
            let proposed_guess = self.compute_next(model);
            errors = self.update_model(model, &proposed_guess);
            max_error = errors.amax();

        }
    }

    pub fn default_with_guess(initial_guess: nalgebra::DVector<f64>,
                                iters_params: iteratives::Iteratives<'a, T>) -> Self {
        let problem_size = initial_guess.len();
        let residuals_config = residuals::ResidualsConfig::default_with_size(problem_size);
        let tolerance: f64 = 1e-6;
        let max_iter: usize = 50;
        let iter: usize = 0;
        let damping: bool = false;
        let debug: bool = false;
        let solver_log = super::log::SolverLog::new();

        RootFinder {
            initial_guess,
            iters_params,
            residuals_config,
            problem_size,
            tolerance,
            max_iter,
            iter,
            damping,
            debug,
            solver_log
        }
    }

    // Writing of simulation log
    fn parameters_to_log(&mut self) {

        let parameters = super::log::Parameters::new(&self.max_iter.to_string(),
                                            &self.tolerance.to_string(),
                                            &self.residuals_config.to_string(),
                                            &self.iters_params.to_string(),
                                            &self.initial_guess.to_string());

        self.solver_log.add_parameters(parameters);
    }

    // Writing of simulation log
    fn iteration_to_log<M>(&mut self, model: &M, errors: &nalgebra::DVector<f64>)
    where
        M: model::Model,
    {
        let iteratives = model.get_iteratives();
        let residuals = model.get_residuals();
        self.solver_log.add_new_iteration(&iteratives, &residuals, errors, self.iter);
    }

    fn damping_to_log<M>(&mut self, model: &M, errors: &nalgebra::DVector<f64>)
    where
        M: model::Model,
    {
        let iteratives = model.get_iteratives();
        let residuals = model.get_residuals();
        self.solver_log.add_damping(&iteratives, &residuals, errors);
    }

    fn jac_to_log(&mut self, jac: &nalgebra::DMatrix<f64>) {
        self.solver_log.add_jac(jac);
    }

    /// Writing of simulation log
    /// The debug field of the solver must be activated
    /// during the simulation in order to be able to write it.
    /// This can be achived thanks to the `set_debug()` method
    pub fn write_log(&self, path: &str) {
        if !self.debug {
            panic!("The debug field was not activated during the simulation");
        }
        self.solver_log.write(path);
    }



}
