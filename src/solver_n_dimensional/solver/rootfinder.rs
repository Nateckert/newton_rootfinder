use std::fmt;

extern crate nalgebra;

use crate::iteratives;
use crate::iteratives::Iterative;
use crate::model;
use crate::residuals;

use super::greenstadt_second_method_udpate_jac;
use super::{broyden_first_method_udpate_inv_jac, broyden_second_method_udpate_inv_jac};
use super::{broyden_first_method_udpate_jac, broyden_second_method_udpate_jac};
use super::{jacobian_evaluation, JacobianMatrix, SolverParameters};
use super::{quasi_method_update_inv_jac, quasi_method_update_jac};
use super::{QuasiNewtonMethod, ResolutionMethod, UpdateQuasiNewtonMethod};

/// Solver for rootfinding
///
/// The solver operates on the model and mutate it
///
/// The core functionnality is the `solve()` method
///
/// The user can activate the debugging before the resolution thanks to the `set_debug()` method
pub struct RootFinder<'a, T>
where
    T: Iterative + fmt::Display,
{
    // user inputs
    parameters: SolverParameters,
    initial_guess: nalgebra::DVector<f64>,
    iters_params: &'a iteratives::Iteratives<'a, T>,
    residuals_config: &'a residuals::ResidualsConfig<'a>,
    debug: bool,

    // solver placeholder
    iter: usize,
    solver_log: Option<super::log::SolverLog>,
    jacobian: JacobianMatrix,
    compute_jac_next_iter: bool,
    last_iter_with_computed_jacobian: usize,
    iteratives_step_size: Option<nalgebra::DVector<f64>>,
    residuals_step_size: Option<nalgebra::DVector<f64>>,
    residuals_values_current: Option<nalgebra::DVector<f64>>,
}

impl<'a, T> RootFinder<'a, T>
where
    T: Iterative + fmt::Display,
{
    pub fn new(
        parameters: SolverParameters,
        initial_guess: nalgebra::DVector<f64>,
        iters_params: &'a iteratives::Iteratives<'a, T>,
        residuals_config: &'a residuals::ResidualsConfig<'a>,
    ) -> Self {
        let debug = false;
        let solver_log = None;
        let iter = 0;

        if residuals_config.len() != parameters.get_problem_size() {
            panic!(
                "Dimension mismatch :\n residuals_config.len() = {} and problem_size = {}",
                residuals_config.len(),
                parameters.get_problem_size()
            );
        }
        if initial_guess.len() != parameters.get_problem_size() {
            panic!(
                "Dimension mismatch :\n initial_guess.len() = {} and problem_size = {}",
                initial_guess.len(),
                parameters.get_problem_size()
            );
        }
        if iters_params.len() != parameters.get_problem_size() {
            panic!(
                "Dimension mismatch :\n iters_params.len() = {} and problem_size = {}",
                iters_params.len(),
                parameters.get_problem_size()
            );
        }

        let jacobian = JacobianMatrix::new();
        let compute_jac_next_iter = true;
        let last_iter_with_computed_jacobian = 0;
        let iteratives_step_size = None;
        let residuals_step_size = None;
        let residuals_values_current = None;

        RootFinder {
            parameters,
            initial_guess,
            iters_params,
            residuals_config,
            debug,
            iter,
            solver_log,
            jacobian,
            compute_jac_next_iter,
            last_iter_with_computed_jacobian,
            iteratives_step_size,
            residuals_step_size,
            residuals_values_current,
        }
    }

    /// Activate the gathering of the log
    ///
    /// The path must be provided (as .txt file)
    /// This generate a .txt file at the given path with simulation informations.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// extern crate newton_rootfinder;
    /// use newton_rootfinder as nrf;
    /// # use nrf::iteratives;
    /// # use nrf::residuals;
    /// # use nrf::solver::ResolutionMethod;
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
    /// # let mut user_model = nrf::model::UserModelFromFunction::new(problem_size, square2);
    /// # let damping = false;
    /// let mut rf = nrf::solver::default_with_guess(init_guess, &iter_params, &res_config, ResolutionMethod::NewtonRaphson, damping);
    ///
    /// rf.activate_debug(&"solver_log.txt");
    /// rf.solve(&mut user_model);
    /// ```
    pub fn activate_debug(&mut self, path: &str) {
        self.debug = true;
        self.solver_log = Some(super::log::SolverLog::new(path));
    }

    fn evaluate_errors<M: model::Model>(&self, model: &M) -> nalgebra::DVector<f64> {
        let residuals_values = model.get_residuals();
        self.residuals_config
            .evaluate_stopping_residuals(&residuals_values)
    }

    fn compute_jac_func<M: model::Model>(&mut self, model: &mut M) {
        let residuals_values = model.get_residuals();

        let jacobians = model.get_jacobian();
        let normalization_method = self.residuals_config.get_update_methods();
        self.jacobian
            .update_jacobian(jacobians.normalize(&residuals_values, &normalization_method));
    }

    fn compute_jac_fd<M: model::Model>(&mut self, model: &mut M) {
        let iters_values = model.get_iteratives();

        let perturbations = self
            .iters_params
            .compute_perturbations(&iters_values, self.parameters.get_problem_size());

        self.jacobian.update_jacobian(jacobian_evaluation(
            model,
            &perturbations,
            &(self.residuals_config),
        ));
    }

    fn compute_jac<M: model::Model>(&mut self, model: &mut M) {
        if model.jacobian_provided() {
            self.compute_jac_func(model);
        } else {
            self.compute_jac_fd(model);
        }

        self.compute_jac_next_iter = false;
        self.last_iter_with_computed_jacobian = self.iter;
    }

    fn compute_newton_raphson_step<M: model::Model>(
        &mut self,
        model: &mut M,
    ) -> nalgebra::DVector<f64> {
        self.compute_jac(model);

        if self.debug {
            self.jac_to_log();
        }

        self.compute_next_from_inv_jac(model)
    }

    fn approximate_jacobian(&mut self, method: UpdateQuasiNewtonMethod) {
        let jac_next = match method {
            UpdateQuasiNewtonMethod::BroydenFirstMethod => broyden_first_method_udpate_jac(
                self.jacobian.get_jacobian().as_ref().unwrap(),
                self.iteratives_step_size.as_ref().unwrap(),
                self.residuals_step_size.as_ref().unwrap(),
            ),
            UpdateQuasiNewtonMethod::BroydenSecondMethod => broyden_second_method_udpate_jac(
                self.jacobian.get_jacobian().as_ref().unwrap(),
                self.iteratives_step_size.as_ref().unwrap(),
                self.residuals_step_size.as_ref().unwrap(),
            ),
            UpdateQuasiNewtonMethod::GreenstadtFirstMethod => quasi_method_update_jac(
                self.jacobian.get_jacobian().as_ref().unwrap(),
                self.iteratives_step_size.as_ref().unwrap(),
                self.residuals_step_size.as_ref().unwrap(),
                self.residuals_values_current.as_ref().unwrap(),
            ),
            UpdateQuasiNewtonMethod::GreenstadtSecondMethod => {
                let c = self.jacobian.get_inverse().as_ref().unwrap()
                    * self.residuals_step_size.as_ref().unwrap();
                greenstadt_second_method_udpate_jac(
                    self.jacobian.get_jacobian().as_ref().unwrap(),
                    self.iteratives_step_size.as_ref().unwrap(),
                    self.residuals_step_size.as_ref().unwrap(),
                    &c,
                )
            }
        };

        self.jacobian.update_jacobian(jac_next);
    }

    fn approximate_inv_jacobian(&mut self, method: UpdateQuasiNewtonMethod) {
        let inv_jac_next = match method {
            UpdateQuasiNewtonMethod::BroydenFirstMethod => broyden_first_method_udpate_inv_jac(
                self.jacobian.get_inverse().as_ref().unwrap(),
                self.iteratives_step_size.as_ref().unwrap(),
                self.residuals_step_size.as_ref().unwrap(),
            ),
            UpdateQuasiNewtonMethod::BroydenSecondMethod => broyden_second_method_udpate_inv_jac(
                self.jacobian.get_inverse().as_ref().unwrap(),
                self.iteratives_step_size.as_ref().unwrap(),
                self.residuals_step_size.as_ref().unwrap(),
            ),
            UpdateQuasiNewtonMethod::GreenstadtFirstMethod => quasi_method_update_inv_jac(
                self.jacobian.get_inverse().as_ref().unwrap(),
                self.iteratives_step_size.as_ref().unwrap(),
                self.residuals_step_size.as_ref().unwrap(),
                self.residuals_values_current.as_ref().unwrap(),
            ),
            UpdateQuasiNewtonMethod::GreenstadtSecondMethod => {
                let c = self.jacobian.get_inverse().as_ref().unwrap().transpose()
                    * self.jacobian.get_inverse().as_ref().unwrap()
                    * self.residuals_step_size.as_ref().unwrap();
                quasi_method_update_inv_jac(
                    self.jacobian.get_inverse().as_ref().unwrap(),
                    self.iteratives_step_size.as_ref().unwrap(),
                    self.residuals_step_size.as_ref().unwrap(),
                    &c,
                )
            }
        };

        self.jacobian.update_inverse(inv_jac_next);
    }

    fn compute_quasi_newton_step<M: model::Model>(
        &mut self,
        model: &mut M,
        resolution_method: QuasiNewtonMethod,
    ) -> nalgebra::DVector<f64> {
        if self.compute_jac_next_iter {
            self.compute_jac(model);
        } else {
            match resolution_method {
                QuasiNewtonMethod::StationaryNewton => (),
                QuasiNewtonMethod::JacobianUpdate(method) => {
                    self.approximate_jacobian(method);
                }
                QuasiNewtonMethod::InverseJacobianUpdate(method) => {
                    self.approximate_inv_jacobian(method);
                }
            };
        }

        if self.debug {
            self.jac_to_log();
        }

        self.compute_next_from_inv_jac(model)
    }

    fn compute_next_from_inv_jac<M: model::Model>(&self, model: &M) -> nalgebra::DVector<f64> {
        let residuals = self
            .residuals_config
            .evaluate_update_residuals(&model.get_residuals());

        let raw_step = -self.jacobian.get_inverse().as_ref().unwrap() * residuals;

        let iter_values = model.get_iteratives();

        self.iters_params.step_limitations(
            &iter_values,
            &raw_step,
            self.parameters.get_problem_size(),
        )
    }

    fn damping<M: model::Model>(
        &mut self,
        model: &mut M,
        max_error: f64,
        current_guess: &nalgebra::DVector<f64>,
        proposed_guess: &nalgebra::DVector<f64>,
        errors_next: &mut nalgebra::DVector<f64>,
    ) {
        let max_error_next = errors_next.amax();
        if max_error_next > max_error {
            // see documentation of the `SolverParameters` struct
            if self.parameters.get_resolution_method() != ResolutionMethod::NewtonRaphson
                && self.last_iter_with_computed_jacobian != self.iter
            {
                self.compute_jac_next_iter = true;
                if self.debug {
                    self.recompute_jacobian_to_log();
                }
            } else {
                let damping_factor = 1.0 / 2.0;
                let damped_guess =
                    current_guess * (1.0 - damping_factor) + proposed_guess * damping_factor;
                model.set_iteratives(&damped_guess);
                model.evaluate();
                *errors_next = self.evaluate_errors(model);

                if self.debug {
                    self.damping_to_log(model, &errors_next);
                }
            }
        }
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

        if self.parameters.get_damping() {
            self.damping(
                model,
                max_error,
                &current_guess,
                &proposed_guess,
                &mut errors_next,
            );
        }

        match self.parameters.get_resolution_method() {
            ResolutionMethod::NewtonRaphson => (),
            ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton) => (),
            _ => {
                self.iteratives_step_size = Some(model.get_iteratives() - current_guess);
                self.residuals_step_size = Some(errors_next.clone() - errors);
                self.residuals_values_current = Some(errors_next.clone())
            }
        };

        errors_next
    }

    /// The core function performing the resolution on a given `Model`
    pub fn solve<M>(&mut self, model: &mut M)
    where
        M: model::Model,
    {
        model.set_iteratives(&self.initial_guess);
        model.evaluate();

        let mut errors = self.evaluate_errors(model);
        let mut max_error = errors.amax();

        if self.debug {
            self.parameters_to_log();
            self.iteration_to_log(model, &errors);
        }

        // Warning: unrolling by hand the first iteration as the following code does
        //          is actually slowing down the code (run benchmarks to see it)
        //
        // first iteration: always a Newton-Raphson step
        //    if max_error > self.parameters.get_tolerance() {
        //        self.iter += 1;
        //        let proposed_guess = self.compute_newton_raphson_step(model);
        //        errors = self.update_model(model, &proposed_guess);
        //        max_error = errors.amax();
        //    }

        // other iterations
        while max_error > self.parameters.get_tolerance()
            && self.iter < self.parameters.get_max_iter()
        {
            self.iter += 1;

            let proposed_guess = match self.parameters.get_resolution_method() {
                ResolutionMethod::NewtonRaphson => self.compute_newton_raphson_step(model),
                ResolutionMethod::QuasiNewton(quasi_newton_method) => {
                    self.compute_quasi_newton_step(model, quasi_newton_method)
                }
            };

            errors = self.update_model(model, &proposed_guess);
            max_error = errors.amax();
        }
    }

    fn parameters_to_log(&self) {
        self.solver_log.as_ref().unwrap().add_parameters(
            &self.parameters.to_string(),
            &self.iters_params.to_string(),
            &self.residuals_config.to_string(),
        );
    }

    fn iteration_to_log<M>(&self, model: &M, errors: &nalgebra::DVector<f64>)
    where
        M: model::Model,
    {
        let iteratives = model.get_iteratives();
        let residuals = model.get_residuals();
        self.solver_log.as_ref().unwrap().add_new_iteration(
            &iteratives,
            &residuals,
            errors,
            self.iter,
        );
    }

    fn recompute_jacobian_to_log(&self) {
        self.solver_log.as_ref().unwrap().add_content(
            &"Iteration refused, the jacobian will be recomputed at the next iteration\n\n",
        );
    }

    fn damping_to_log<M>(&self, model: &M, errors: &nalgebra::DVector<f64>)
    where
        M: model::Model,
    {
        let iteratives = model.get_iteratives();
        let residuals = model.get_residuals();
        self.solver_log
            .as_ref()
            .unwrap()
            .add_damping(&iteratives, &residuals, errors);
    }

    fn jac_to_log(&self) {
        self.solver_log
            .as_ref()
            .unwrap()
            .add_content(&self.jacobian.to_string());
    }
}
