//! Advanced solver
//!
//! ## Examples
//!
//! ```
//! extern crate newton_rootfinder;
//! use newton_rootfinder::solver_advanced as nrf;
//! use nrf::model::Model;
//! use nrf::iteratives;
//! use nrf::residuals;
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
//!   let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
//!   let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
//!   let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
//!   let mut rf = nrf::solver::default_with_guess(init_guess, iter_params, res_config);
//!   let mut user_model =
//!       nrf::model::UserModelWithFunc::new(problem_size, square2);
//!
//!   rf.solve(&mut user_model);
//!
//!   println!("{}", user_model.get_iteratives()[0]);
//!   // print 1.4142135623747443
//! }
//! ```

use std::fmt;

extern crate nalgebra;

use crate::solver_advanced::iteratives;
use crate::solver_advanced::iteratives::Iterative;
use crate::solver_advanced::model;

use crate::solver_advanced::residuals;
use crate::solver_advanced::util::jacobian;

/// Solver for rootfinding
///
/// The solver operates on the model and mutate it
///
/// The only fields of the solver that can change during the resolution are:
/// - iter : it holds the current iteration number
/// - solver_log : it holds the simulation log information for debugging
pub struct RootFinder<'a, T>
where
    T: Iterative + fmt::Display,
{
    initial_guess: nalgebra::DVector<f64>,
    iters_params: iteratives::Iteratives<'a, T>,
    residuals_config: residuals::ResidualsConfig<'a>,
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
    T: Iterative + fmt::Display,
{
    pub fn new(
        initial_guess: nalgebra::DVector<f64>,
        iters_params: iteratives::Iteratives<'a, T>,
        residuals_config: residuals::ResidualsConfig<'a>,
        problem_size: usize,
        tolerance: f64,
        max_iter: usize,
    ) -> Self {
        let damping = false;
        let debug = false;
        let solver_log = super::log::SolverLog::new();
        let iter = 0;

        if residuals_config.len() != problem_size {
            panic!(
                "Dimension mismatch :\n residuals_config.len() = {} and problem_size = {}",
                residuals_config.len(),
                problem_size
            );
        }
        if initial_guess.len() != problem_size {
            panic!(
                "Dimension mismatch :\n initial_guess.len() = {} and problem_size = {}",
                initial_guess.len(),
                problem_size
            );
        }
        if iters_params.len() != problem_size {
            panic!(
                "Dimension mismatch :\n iters_params.len() = {} and problem_size = {}",
                iters_params.len(),
                problem_size
            );
        }

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
            solver_log,
        }
    }

    /// Activate the damping to improve convergence
    ///
    /// Plain resolution according to Newton is made through the formula
    /// X = X - J^-1*F(X)
    ///
    /// However, if the proposed update is not performing (deterioriating the solution)
    /// it is likely it is due to a much to step-size too important.
    /// Reducing the step-size might be the solution
    ///
    /// The damping formula is then :
    /// X = X - damping_factor*J^-1*F(X)
    /// with 0 < damping_factor <= 1
    ///
    /// As long as the error is reduced damping_factor = 1.
    /// If it is not the case, a factor is applied
    /// (the value might change according to the versions).
    pub fn set_damping(&mut self, damping: bool) {
        self.damping = damping;
    }

    /// Activate the gathering of the log informations
    ///
    /// It is required to write the log after the resolution.
    /// The path must also be provided (as .txt file)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// extern crate newton_rootfinder;
    /// use newton_rootfinder::solver_advanced as nrf;
    /// # use nrf::iteratives;
    /// # use nrf::residuals;
    /// # pub fn square2(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    /// #   let mut y = x * x;
    /// #   y[0] -= 2.0;
    /// #   y
    /// # }
    /// # let problem_size = 1;
    /// # let init_guess = nalgebra::DVector::from_vec(vec![1.0]);
    /// # let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    /// # let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    /// # let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    /// # let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    /// # let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    /// # let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, square2);
    /// let mut rf = nrf::solver::default_with_guess(init_guess, iter_params, res_config);
    /// rf.set_debug(true);
    /// rf.solve(&mut user_model);
    /// rf.write_log(&"solver_log.txt");
    /// ```
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
        let normalization_method = self.residuals_config.get_update_methods();
        jacobians.normalize(&residuals_values, &normalization_method)
    }

    fn compute_jac_fd<M: model::Model>(&self, model: &mut M) -> nalgebra::DMatrix<f64> {
        let iters_values = model.get_iteratives();

        let perturbations = self
            .iters_params
            .compute_perturbations(&iters_values, self.problem_size);

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

        self.iters_params
            .step_limitations(&iter_values, &raw_step, self.problem_size)
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

        if self.debug {
            self.iteration_to_log(model, &errors_next);
        }

        if self.damping {
            let max_error_next = errors_next.amax();
            if max_error_next > max_error {
                // update formula : X = X - damping_factor*res/jac
                // proposed_guess is with damping_factor = 1
                let damping_factor = 1.0 / 2.0;
                let damped_guess =
                    &current_guess * (1.0 - damping_factor) + proposed_guess * damping_factor;
                model.set_iteratives(&damped_guess);
                model.evaluate();
                errors_next = self.evaluate_errors(model);

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

    // Writing of simulation log
    fn parameters_to_log(&mut self) {
        let parameters = super::log::Parameters::new(
            &self.max_iter.to_string(),
            &self.tolerance.to_string(),
            &self.residuals_config.to_string(),
            &self.iters_params.to_string(),
            &self.initial_guess.to_string(),
        );

        self.solver_log.add_parameters(parameters);
    }

    // Writing of simulation log
    fn iteration_to_log<M>(&mut self, model: &M, errors: &nalgebra::DVector<f64>)
    where
        M: model::Model,
    {
        let iteratives = model.get_iteratives();
        let residuals = model.get_residuals();
        self.solver_log
            .add_new_iteration(&iteratives, &residuals, errors, self.iter);
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
