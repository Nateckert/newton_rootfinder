use std::fmt;

extern crate nalgebra;

use crate::errors;
use crate::iteratives;
use crate::iteratives::Iterative;
use crate::model;
use crate::model::ModelError;
use crate::residuals;

use super::{
    approximate_inv_jacobian, approximate_jacobian, evaluate_jacobian_from_analytical_function,
    evaluate_jacobian_from_finite_difference, JacobianMatrix, SolverParameters,
};

use super::{QuasiNewtonMethod, ResolutionMethod};

/// Solver for rootfinding
///
/// The solver operates on the model and mutate it
///
/// The core functionnality is the `solve()` method
///
/// The user can activate the debugging before the resolution thanks to the `set_debug()` method
pub struct RootFinder<'a, T, D>
where
    T: Iterative + fmt::Display,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, nalgebra::U1, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<(usize, usize), D>,
{
    // user inputs
    parameters: SolverParameters,
    initial_guess: nalgebra::OVector<f64, D>,
    iters_params: &'a iteratives::Iteratives<'a, T>,
    residuals_config: &'a residuals::ResidualsConfig<'a>,
    debug: bool,

    // solver placeholder
    iter: usize,
    solver_log: Option<super::log::SolverLog>,
    jacobian: JacobianMatrix<D>,
    iteratives_step_size: Option<nalgebra::OVector<f64, D>>,
    residuals_step_size: Option<nalgebra::OVector<f64, D>>,
    residuals_values_current: Option<nalgebra::OVector<f64, D>>,
    valid_last_model_evaluation: bool,
}

impl<'a, T, D> RootFinder<'a, T, D>
where
    T: Iterative + fmt::Display,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, nalgebra::U1, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<(usize, usize), D>,
{
    pub fn new(
        parameters: SolverParameters,
        initial_guess: nalgebra::OVector<f64, D>,
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
        let iteratives_step_size = None;
        let residuals_step_size = None;
        let residuals_values_current = None;
        let valid_last_model_evaluation = true;

        RootFinder {
            parameters,
            initial_guess,
            iters_params,
            residuals_config,
            debug,
            iter,
            solver_log,
            jacobian,
            iteratives_step_size,
            residuals_step_size,
            residuals_values_current,
            valid_last_model_evaluation,
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

    fn evaluate_errors<M>(&self, model: &M) -> nalgebra::OVector<f64, D>
    where
        M: model::Model<D>,
    {
        let residuals_values = model.get_residuals();
        self.residuals_config
            .evaluate_stopping_residuals(&residuals_values)
    }

    fn compute_jac<M>(&mut self, model: &mut M) -> Result<(), errors::SolverInternalError<M, D>>
    where
        M: model::Model<D>,
    {
        let successful_jac_computation = if model.jacobian_provided() {
            evaluate_jacobian_from_analytical_function(
                &mut self.jacobian,
                model,
                self.residuals_config,
            )
        } else {
            evaluate_jacobian_from_finite_difference(
                &mut self.jacobian,
                model,
                self.iters_params,
                self.residuals_config,
            )
        };

        match successful_jac_computation {
            Ok(())
            | Err(errors::SolverInternalError::InvalidJacobianError(
                ModelError::InaccurateValuesError(_),
            )) => Ok(()),
            Err(model_error) => Err(model_error),
        }
    }

    fn compute_newton_raphson_step<M>(
        &mut self,
        model: &mut M,
    ) -> Result<nalgebra::OVector<f64, D>, crate::errors::SolverInternalError<M, D>>
    where
        M: model::Model<D>,
    {
        let successful_jac_computation = self.compute_jac(model);

        if self.debug {
            self.jac_to_log();
        }
        match successful_jac_computation {
            Ok(()) => Ok(self.compute_next_from_inv_jac(model)),
            Err(error) => Err(error),
        }
    }

    /// Perform the jacobian evaluation
    ///
    /// Based on the resolution method:
    /// - the jacobian can be computed and inverted
    /// - the jacobian can be approximated and inverted
    /// - the inverse of the jacobian can be approximated
    fn evaluate_jacobian_quasi_newton_step<M>(
        &mut self,
        model: &mut M,
        resolution_method: QuasiNewtonMethod,
    ) -> Result<(), crate::errors::SolverInternalError<M, D>>
    where
        M: model::Model<D>,
    {
        if self.jacobian.compute_jacobian() {
            let successful_jac_computation = self.compute_jac(model);

            match successful_jac_computation {
                Ok(()) => (),
                Err(error) => {
                    if self.debug {
                        self.jac_to_log();
                    }
                    return Err(error);
                }
            }
        } else {
            match resolution_method {
                QuasiNewtonMethod::StationaryNewton => (),
                QuasiNewtonMethod::JacobianUpdate(method) => {
                    match approximate_jacobian(
                        &mut self.jacobian,
                        method,
                        self.iteratives_step_size.as_ref().unwrap(),
                        self.residuals_step_size.as_ref().unwrap(),
                        self.residuals_values_current.as_ref().unwrap(),
                    ) {
                        Ok(()) => (),
                        Err(_) => {
                            return Err(errors::SolverInternalError::InvalidJacobianInverseError)
                        }
                    }
                }
                QuasiNewtonMethod::InverseJacobianUpdate(method) => {
                    approximate_inv_jacobian(
                        &mut self.jacobian,
                        method,
                        self.iteratives_step_size.as_ref().unwrap(),
                        self.residuals_step_size.as_ref().unwrap(),
                        self.residuals_values_current.as_ref().unwrap(),
                    );
                }
            };
        }

        if self.debug {
            self.jac_to_log();
        }

        Ok(())
    }

    fn compute_quasi_newton_step<M>(
        &mut self,
        model: &mut M,
        resolution_method: QuasiNewtonMethod,
    ) -> Result<nalgebra::OVector<f64, D>, crate::errors::SolverInternalError<M, D>>
    where
        M: model::Model<D>,
    {
        match self.evaluate_jacobian_quasi_newton_step(model, resolution_method) {
            Ok(()) => Ok(self.compute_next_from_inv_jac(model)),
            Err(crate::errors::SolverInternalError::InvalidJacobianError(error)) => Err(
                crate::errors::SolverInternalError::InvalidJacobianError(error),
            ),
            Err(crate::errors::SolverInternalError::InvalidJacobianInverseError) => {
                Err(crate::errors::SolverInternalError::InvalidJacobianInverseError)
            }
        }
    }

    fn compute_next_from_inv_jac<M>(&self, model: &M) -> nalgebra::OVector<f64, D>
    where
        M: model::Model<D>,
    {
        let residuals = self
            .residuals_config
            .evaluate_update_residuals(&model.get_residuals());

        let raw_step = -self.jacobian.get_inverse().as_ref().unwrap() * residuals;

        let iter_values = model.get_iteratives();

        self.iters_params.step_limitations(&iter_values, &raw_step)
    }

    fn damping<M>(
        &mut self,
        model: &mut M,
        max_error: f64,
        current_guess: &nalgebra::OVector<f64, D>,
        proposed_guess: &nalgebra::OVector<f64, D>,
        errors_next: &mut nalgebra::OVector<f64, D>,
    ) where
        M: model::Model<D>,
    {
        let max_error_next = errors_next.amax();
        if max_error_next > max_error {
            // see documentation of the `SolverParameters` struct
            if self.parameters.get_resolution_method() != ResolutionMethod::NewtonRaphson
                && self.jacobian.is_jacobian_approximated()
            {
                self.jacobian.force_jacobian_computation();
                if self.debug {
                    self.recompute_jacobian_to_log();
                }
            } else {
                let damping_factor = 1.0 / 2.0;
                let damped_guess =
                    current_guess * (1.0 - damping_factor) + proposed_guess * damping_factor;
                model.set_iteratives(&damped_guess);
                model.evaluate().unwrap();
                *errors_next = self.evaluate_errors(model);

                if self.debug {
                    self.damping_to_log(model, errors_next);
                }
            }
        }
    }

    fn update_model<M>(
        &mut self,
        model: &mut M,
        proposed_guess: &nalgebra::OVector<f64, D>,
    ) -> Result<nalgebra::OVector<f64, D>, errors::SolverError<M, D>>
    where
        M: model::Model<D>,
    {
        let errors = self.evaluate_errors(model);
        let max_error = errors.amax();
        let current_guess = model.get_iteratives();

        model.set_iteratives(proposed_guess);
        match model.evaluate() {
            Ok(()) => {
                self.valid_last_model_evaluation = true;
            }
            Err(ModelError::InaccurateValuesError(_)) => {
                self.valid_last_model_evaluation = false;
            }
            Err(e) => {
                self.valid_last_model_evaluation = false;
                return Err(errors::SolverError::ModelEvaluationError(e));
            }
        }
        let mut errors_next = self.evaluate_errors(model);

        if self.debug {
            self.iteration_to_log(model, &errors_next);
        }

        if self.parameters.get_damping() {
            self.damping(
                model,
                max_error,
                &current_guess,
                proposed_guess,
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

        Ok(errors_next)
    }

    /// The core function performing the resolution on a given `Model`
    pub fn solve<M>(&mut self, model: &mut M) -> Result<(), crate::errors::SolverError<M, D>>
    where
        M: model::Model<D>,
    {
        model.set_iteratives(&self.initial_guess);

        // The first evaluation must yield usuable values
        // However, then don't need to be accurate
        match model.evaluate() {
            Ok(()) => (),
            Err(ModelError::InaccurateValuesError(_)) => (),
            Err(ModelError::UnusableValuesError(error)) => {
                return Err(crate::errors::SolverError::ModelInitialEvaluationError(
                    error.to_string(),
                ))
            }
        }

        let mut errors = self.evaluate_errors(model);
        let mut max_error = errors.amax();

        if self.debug {
            self.parameters_to_log();
            self.iteration_to_log(model, &errors);
        }

        // Warning: unrolling by hand the first iteration (which is always a Newton-Raphson step)
        //          is actually slowing down the code (run benchmarks to see it)
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

            match proposed_guess {
                Ok(value) => match self.update_model(model, &value) {
                    Ok(value) => errors = value,
                    Err(e) => return Err(e),
                },
                Err(error) => {
                    return Err(errors::SolverError::JacobianError(error));
                }
            }

            max_error = errors.amax();
        }

        if max_error > self.parameters.get_tolerance() {
            Err(crate::errors::SolverError::NonConvergenceError)
        } else if self.valid_last_model_evaluation {
            Ok(())
        } else {
            Err(crate::errors::SolverError::FinalEvaluationError)
        }
    }

    fn parameters_to_log(&self) {
        self.solver_log.as_ref().unwrap().add_parameters(
            &self.parameters.to_string(),
            &self.iters_params.to_string(),
            &self.residuals_config.to_string(),
        );
    }

    fn iteration_to_log<M>(&self, model: &M, errors: &nalgebra::OVector<f64, D>)
    where
        M: model::Model<D>,
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
            "Iteration refused, the jacobian will be recomputed at the next iteration\n\n",
        );
    }

    fn damping_to_log<M>(&self, model: &M, errors: &nalgebra::OVector<f64, D>)
    where
        M: model::Model<D>,
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
