use std::fmt;

use crate::errors;
use crate::iteratives;
use crate::iteratives::Iterative;
use crate::log::{Detail, Evaluation, JacobianSource, Log, ResolutionLog};
use crate::model;
use crate::model::ModelError;
use crate::residuals;

use super::{
    JacobianMatrix, Point, SolverParameters, approximate_inv_jacobian, approximate_jacobian,
    evaluate_jacobian_from_analytical_function, evaluate_jacobian_from_finite_difference_at,
};

use super::{QuasiNewtonMethod, ResolutionMethod};

/// Solver for rootfinding
///
/// The solver operates on the model and mutate it
///
/// The core functionnality is the `solve()` method.
/// Every call is a resolution of its own, started afresh from the iteratives the model holds:
/// a solver can be reused for several resolutions, on the same model or on other models of the same dimension.
///
/// The resolutions are written in a [Log] once it is given with [RootFinder::log_to]
pub struct RootFinder<'a, T, D>
where
    T: Iterative + fmt::Debug,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::U1, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    // user inputs
    parameters: SolverParameters,
    iters_params: &'a iteratives::Iteratives<'a, T>,
    residuals_config: &'a residuals::ResidualsConfig<'a>,
    log: Option<ResolutionLog>,

    // solver placeholder
    iter: usize,
    jacobian: JacobianMatrix<D>,
    iteratives_step_size: Option<nalgebra::OVector<f64, D>>,
    residuals_step_size: Option<nalgebra::OVector<f64, D>>,
    residuals_values_current: Option<nalgebra::OVector<f64, D>>,
    valid_last_model_evaluation: bool,
}

impl<'a, T, D> RootFinder<'a, T, D>
where
    T: Iterative + fmt::Debug,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::U1, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// A solver from its parameters and the configurations of the iteratives and of the residuals,
    /// which must have the same length, the dimension of the problem.
    /// The dimension type `D` is the one of the models the solver is called on, a static or a dynamic one.
    ///
    /// # Panics
    ///
    /// When the two configurations differ in length.
    pub fn new(
        parameters: SolverParameters,
        iters_params: &'a iteratives::Iteratives<'a, T>,
        residuals_config: &'a residuals::ResidualsConfig<'a>,
    ) -> Self {
        let log = None;
        let iter = 0;

        if iters_params.len() != residuals_config.len() {
            panic!(
                "Dimension mismatch between the iteratives and the residuals configurations: {} iteratives, {} residuals",
                iters_params.len(),
                residuals_config.len()
            );
        }

        let jacobian = JacobianMatrix::new();
        let iteratives_step_size = None;
        let residuals_step_size = None;
        let residuals_values_current = None;
        let valid_last_model_evaluation = true;

        RootFinder {
            parameters,
            iters_params,
            residuals_config,
            log,
            iter,
            jacobian,
            iteratives_step_size,
            residuals_step_size,
            residuals_values_current,
            valid_last_model_evaluation,
        }
    }

    /// Write the resolutions of this solver in `log`, under `label`
    ///
    /// Each call to [RootFinder::solve] is a resolution of the log,
    /// with the detail the log has when it starts: see the [log module](crate::log).
    ///
    /// # Examples
    ///
    /// ```
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
    /// # nrf::model::Model::set_iteratives(&mut user_model, &init_guess);
    /// let log = nrf::log::Log::new(std::io::sink());
    /// let mut rf = nrf::solver::default(&iter_params, &res_config, ResolutionMethod::NewtonRaphson, damping);
    ///
    /// rf.log_to(&log, "square root of 2");
    /// rf.solve(&mut user_model).unwrap();
    /// ```
    pub fn log_to(&mut self, log: &Log, label: &str) {
        self.log = Some(ResolutionLog::new(log, label));
    }

    /// Start a resolution afresh: the iteration count, the jacobian,
    /// the steps of the quasi-Newton updates and the validity of the last evaluation
    /// are the ones of a new solver, nothing being kept from a previous call to `solve`
    fn reset(&mut self) {
        self.iter = 0;
        self.jacobian = if self.reads_jacobian_matrix() {
            JacobianMatrix::new()
        } else {
            JacobianMatrix::factorization_only()
        };
        self.iteratives_step_size = None;
        self.residuals_step_size = None;
        self.residuals_values_current = None;
        self.valid_last_model_evaluation = true;
    }

    /// The matrix of the jacobian is read back by the methods updating it and by a log writing it:
    /// the other resolutions keep its factorization only, sparing a copy at each jacobian
    fn reads_jacobian_matrix(&self) -> bool {
        self.log
            .as_ref()
            .is_some_and(|log| log.writes(Detail::Jacobians))
            || matches!(
                self.parameters.get_resolution_method(),
                ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(_))
            )
    }

    fn evaluate_errors<M>(&self, model: &M) -> nalgebra::OVector<f64, D>
    where
        M: model::Model<D>,
    {
        let residuals_values = model.get_residuals();
        self.residuals_config
            .evaluate_stopping_residuals(&residuals_values)
    }

    /// Indices of the equations whose residual is not finite
    ///
    /// The maximum error and the comparisons with it ignore NaN,
    /// so the solver must look for non-finite residuals itself:
    /// without it a NaN residual could be reported converged, or escape the damping.
    /// A non-finite residual always gives a non-finite error,
    /// so the residuals are read only when an error is not finite.
    /// The converse does not hold: the relative normalization of 0 = 0 is NaN.
    fn non_finite_residuals<M>(&self, model: &M, errors: &nalgebra::OVector<f64, D>) -> Vec<usize>
    where
        M: model::Model<D>,
    {
        if errors.iter().all(|error| error.is_finite()) {
            return Vec::new();
        }
        model.get_residuals().non_finite_indices()
    }

    /// Fails when a residual is not finite
    fn check_finite_residuals<M>(
        &self,
        model: &M,
        errors: &nalgebra::OVector<f64, D>,
    ) -> Result<(), errors::SolverError<M, D>>
    where
        M: model::Model<D>,
    {
        let indices = self.non_finite_residuals(model, errors);
        if indices.is_empty() {
            Ok(())
        } else {
            Err(errors::SolverError::NonFiniteResiduals(indices))
        }
    }

    fn compute_jac<M>(
        &mut self,
        model: &mut M,
        current: &Point<M, D>,
    ) -> Result<(), errors::SolverInternalError<M, D>>
    where
        M: model::Model<D>,
    {
        let (successful_jac_computation, source) = if model.jacobian_provided() {
            (
                evaluate_jacobian_from_analytical_function(
                    &mut self.jacobian,
                    model,
                    self.residuals_config,
                ),
                JacobianSource::Analytic,
            )
        } else {
            (
                evaluate_jacobian_from_finite_difference_at(
                    &mut self.jacobian,
                    model,
                    current,
                    self.iters_params,
                    self.residuals_config,
                ),
                JacobianSource::FiniteDifferences,
            )
        };

        if let Some(log) = &self.log {
            log.jacobian(self.iter, source, &self.jacobian);
        }

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
        current: &Point<M, D>,
    ) -> Result<nalgebra::OVector<f64, D>, crate::errors::SolverInternalError<M, D>>
    where
        M: model::Model<D>,
    {
        let successful_jac_computation = self.compute_jac(model, current);

        match successful_jac_computation {
            Ok(()) => Ok(self.compute_next_from_jacobian(current)),
            Err(error) => Err(error),
        }
    }

    /// Perform the jacobian evaluation
    ///
    /// Based on the resolution method:
    /// - the jacobian can be computed and factorized
    /// - the jacobian can be approximated and factorized
    /// - the inverse of the jacobian can be approximated
    fn evaluate_jacobian_quasi_newton_step<M>(
        &mut self,
        model: &mut M,
        current: &Point<M, D>,
        resolution_method: QuasiNewtonMethod,
    ) -> Result<(), crate::errors::SolverInternalError<M, D>>
    where
        M: model::Model<D>,
    {
        if self.jacobian.compute_jacobian() {
            let successful_jac_computation = self.compute_jac(model, current);

            successful_jac_computation?;
        } else {
            match resolution_method {
                QuasiNewtonMethod::StationaryNewton => (),
                QuasiNewtonMethod::JacobianUpdate(method) => {
                    let approximation = approximate_jacobian(
                        &mut self.jacobian,
                        method,
                        self.iteratives_step_size.as_ref().unwrap(),
                        self.residuals_step_size.as_ref().unwrap(),
                        self.residuals_values_current.as_ref().unwrap(),
                    );
                    if let Some(log) = &self.log {
                        log.jacobian(self.iter, JacobianSource::Update(method), &self.jacobian);
                    }
                    if approximation.is_err() {
                        return Err(errors::SolverInternalError::InvalidJacobianInverseError);
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
                    if let Some(log) = &self.log {
                        log.jacobian(self.iter, JacobianSource::Update(method), &self.jacobian);
                    }
                }
            };
        }

        Ok(())
    }

    fn compute_quasi_newton_step<M>(
        &mut self,
        model: &mut M,
        current: &Point<M, D>,
        resolution_method: QuasiNewtonMethod,
    ) -> Result<nalgebra::OVector<f64, D>, crate::errors::SolverInternalError<M, D>>
    where
        M: model::Model<D>,
    {
        match self.evaluate_jacobian_quasi_newton_step(model, current, resolution_method) {
            Ok(()) => Ok(self.compute_next_from_jacobian(current)),
            Err(crate::errors::SolverInternalError::InvalidJacobianError(error)) => Err(
                crate::errors::SolverInternalError::InvalidJacobianError(error),
            ),
            Err(crate::errors::SolverInternalError::InvalidJacobianInverseError) => {
                Err(crate::errors::SolverInternalError::InvalidJacobianInverseError)
            }
        }
    }

    fn compute_next_from_jacobian<M>(&self, current: &Point<M, D>) -> nalgebra::OVector<f64, D>
    where
        M: model::Model<D>,
    {
        let residuals = self
            .residuals_config
            .evaluate_update_residuals(&current.residuals);

        // The step solves `jacobian * step = -residuals`
        let raw_step = -self.jacobian.solve(&residuals).unwrap();
        if let Some(log) = &self.log {
            log.step(self.iter, &raw_step);
        }

        self.iters_params
            .step_limitations(&current.iteratives, &raw_step)
    }

    fn damping<M>(
        &mut self,
        model: &mut M,
        max_error: f64,
        current: &Point<M, D>,
        proposed_guess: &nalgebra::OVector<f64, D>,
        errors_next: &mut nalgebra::OVector<f64, D>,
    ) -> Result<(), errors::SolverError<M, D>>
    where
        M: model::Model<D>,
    {
        // A step to a non-finite residual deteriorates the solution,
        // whatever the maximum of the errors, which ignores NaN
        let max_error_next = errors_next.amax();
        if max_error_next > max_error || !self.non_finite_residuals(model, errors_next).is_empty() {
            // see documentation of the `SolverParameters` struct
            if self.parameters.get_resolution_method() != ResolutionMethod::NewtonRaphson
                && self.jacobian.is_jacobian_approximated()
            {
                self.jacobian.force_jacobian_computation();
                if let Some(log) = &self.log {
                    log.refused(self.iter);
                }
            } else {
                let damping_factor = 1.0 / 2.0;
                let damped_guess =
                    &current.iteratives * (1.0 - damping_factor) + proposed_guess * damping_factor;
                // The damped guess starts from the memory of the current point,
                // not from the one the rejected proposed guess left
                model.set_memory(&current.memory);
                model.set_iteratives(&damped_guess);
                self.evaluate_model(model, Evaluation::Damping)?;
                *errors_next = self.evaluate_errors(model);
                self.log_evaluation(model, Evaluation::Damping, errors_next);
            }
        }
        Ok(())
    }

    fn evaluate_model<M>(
        &mut self,
        model: &mut M,
        of: Evaluation,
    ) -> Result<(), errors::SolverError<M, D>>
    where
        M: model::Model<D>,
    {
        match model.evaluate() {
            Ok(()) => {
                self.valid_last_model_evaluation = true;
            }
            Err(ModelError::InaccurateValuesError(_)) => {
                self.valid_last_model_evaluation = false;
            }
            Err(ModelError::UnusableValuesError(error)) => {
                self.valid_last_model_evaluation = false;
                if let Some(log) = &self.log {
                    log.failed_evaluation(
                        self.iter,
                        of,
                        &model.get_iteratives(),
                        &error.to_string(),
                    );
                }
                return Err(errors::SolverError::ModelEvaluationError(
                    ModelError::UnusableValuesError(error),
                ));
            }
        }
        Ok(())
    }

    /// Evaluate the proposed guess, damped if needed,
    /// and return the point the model ends on with its errors
    fn update_model<M>(
        &mut self,
        model: &mut M,
        current: &Point<M, D>,
        proposed_guess: &nalgebra::OVector<f64, D>,
    ) -> Result<(Point<M, D>, nalgebra::OVector<f64, D>), errors::SolverError<M, D>>
    where
        M: model::Model<D>,
    {
        let errors = self
            .residuals_config
            .evaluate_stopping_residuals(&current.residuals);
        let max_error = errors.amax();

        model.set_iteratives(proposed_guess);
        self.evaluate_model(model, Evaluation::Step)?;
        let mut errors_next = self.evaluate_errors(model);
        self.log_evaluation(model, Evaluation::Step, &errors_next);

        if self.parameters.get_damping() {
            self.damping(model, max_error, current, proposed_guess, &mut errors_next)?;
        }

        let next = Point::of(model);

        match self.parameters.get_resolution_method() {
            ResolutionMethod::NewtonRaphson => (),
            ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton) => (),
            _ => {
                let residuals_current = match self.residuals_values_current.take() {
                    Some(residuals) => residuals,
                    None => self
                        .residuals_config
                        .evaluate_update_residuals(&current.residuals),
                };
                let residuals_next = self
                    .residuals_config
                    .evaluate_update_residuals(&next.residuals);
                self.iteratives_step_size = Some(&next.iteratives - &current.iteratives);
                self.residuals_step_size = Some(&residuals_next - residuals_current);
                self.residuals_values_current = Some(residuals_next)
            }
        };

        Ok((next, errors_next))
    }

    /// The core function performing the resolution on a given `Model`
    ///
    /// When the resolution fails because the model or the jacobian could not be evaluated,
    /// or because a residual is not finite, the model gets back the iteratives and the memory
    /// it held when `solve` was called, so that a new resolution can start from them;
    /// its residuals are the ones of the failed evaluation.
    /// A [errors::SolverError::NonConvergenceError] or a [errors::SolverError::FinalEvaluationError]
    /// leaves the model on the last iterate.
    ///
    /// The resolution starts afresh: from the iteratives the model holds, with the iteration count at zero
    /// and a jacobian computed at the first iteration, nothing being kept from a previous call.
    /// A model with another number of iteratives than the configurations of the solver is refused
    /// with [errors::SolverError::DimensionMismatch], before any evaluation.
    /// With a [Log] given by [RootFinder::log_to], each call is a resolution of the log.
    pub fn solve<M>(&mut self, model: &mut M) -> Result<(), crate::errors::SolverError<M, D>>
    where
        M: model::Model<D>,
    {
        if let Some(log) = &mut self.log {
            log.start(&self.parameters, self.iters_params, self.residuals_config);
        }
        self.reset();
        let result = self.solve_from_the_model(model);
        if let Some(log) = &mut self.log {
            log.end(&result, self.iter);
        }
        result
    }

    /// The resolution, restoring the point the model held on the failures leaving it unusable
    fn solve_from_the_model<M>(
        &mut self,
        model: &mut M,
    ) -> Result<(), crate::errors::SolverError<M, D>>
    where
        M: model::Model<D>,
    {
        let entry_iteratives = model.get_iteratives();
        if entry_iteratives.len() != self.iters_params.len() {
            return Err(errors::SolverError::DimensionMismatch {
                model: entry_iteratives.len(),
                solver: self.iters_params.len(),
            });
        }
        let entry_memory = model.get_memory();

        let result = self.solve_from_entry(model);

        match result {
            Err(errors::SolverError::ModelInitialEvaluationError(_))
            | Err(errors::SolverError::ModelEvaluationError(_))
            | Err(errors::SolverError::JacobianEvaluationError(_))
            | Err(errors::SolverError::NonInvertibleJacobian)
            | Err(errors::SolverError::NonFiniteResiduals(_)) => {
                model.set_memory(&entry_memory);
                model.set_iteratives(&entry_iteratives);
            }
            Ok(())
            | Err(errors::SolverError::NonConvergenceError)
            | Err(errors::SolverError::FinalEvaluationError)
            | Err(errors::SolverError::DimensionMismatch { .. }) => (),
        }

        result
    }

    fn solve_from_entry<M>(&mut self, model: &mut M) -> Result<(), crate::errors::SolverError<M, D>>
    where
        M: model::Model<D>,
    {
        // The first evaluation must yield usuable values
        // However, then don't need to be accurate
        match model.evaluate() {
            Ok(()) => (),
            Err(ModelError::InaccurateValuesError(_)) => (),
            Err(ModelError::UnusableValuesError(error)) => {
                if let Some(log) = &self.log {
                    log.failed_evaluation(
                        self.iter,
                        Evaluation::Guess,
                        &model.get_iteratives(),
                        &error.to_string(),
                    );
                }
                return Err(crate::errors::SolverError::ModelInitialEvaluationError(
                    error,
                ));
            }
        }

        // The point the solver stands on: the model may hold a perturbed one,
        // after a jacobian evaluated by finite-difference
        let mut current = Point::of(model);
        let mut errors = self.evaluate_errors(model);
        let mut max_error = errors.amax();
        self.log_evaluation(model, Evaluation::Guess, &errors);

        self.check_finite_residuals(model, &errors)?;

        // Warning: unrolling by hand the first iteration (which is always a Newton-Raphson step)
        //          is actually slowing down the code (run benchmarks to see it)
        while max_error > self.parameters.get_tolerance()
            && self.iter < self.parameters.get_max_iter()
        {
            self.iter += 1;

            let proposed_guess = match self.parameters.get_resolution_method() {
                ResolutionMethod::NewtonRaphson => {
                    self.compute_newton_raphson_step(model, &current)
                }
                ResolutionMethod::QuasiNewton(quasi_newton_method) => {
                    self.compute_quasi_newton_step(model, &current, quasi_newton_method)
                }
            };

            match proposed_guess {
                Ok(value) => match self.update_model(model, &current, &value) {
                    Ok((next, value)) => {
                        current = next;
                        errors = value;
                    }
                    Err(e) => return Err(e),
                },
                Err(error) => return Err(error.into()),
            }

            self.check_finite_residuals(model, &errors)?;
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

    /// Write an evaluation the model made in the log, if any
    #[inline]
    fn log_evaluation<M>(&mut self, model: &M, of: Evaluation, errors: &nalgebra::OVector<f64, D>)
    where
        M: model::Model<D>,
    {
        if self.log.is_some() {
            self.write_evaluation(model, of, errors);
        }
    }

    #[cold]
    #[inline(never)]
    fn write_evaluation<M>(&mut self, model: &M, of: Evaluation, errors: &nalgebra::OVector<f64, D>)
    where
        M: model::Model<D>,
    {
        if let Some(log) = &mut self.log {
            if log.writes(Detail::Iterations) {
                log.evaluation(
                    self.iter,
                    of,
                    &model.get_iteratives(),
                    &model.get_residuals(),
                    errors,
                );
            } else {
                log.max_error(errors);
            }
        }
    }
}

impl<'a, T, D> fmt::Debug for RootFinder<'a, T, D>
where
    T: Iterative + fmt::Debug,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::U1, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Rootfinder")
            .field("Solver parameters", &self.parameters)
            .field("Iterative configuration", &self.iters_params)
            .field("Residual configuration", &self.residuals_config)
            .field("Log", &self.log)
            .field("Current iteration", &self.iter)
            .field(
                "Status last model evaluation",
                &self.valid_last_model_evaluation,
            )
            .field("Placeholder jacobian", &self.jacobian)
            .field("Placeholder iterative step", &self.iteratives_step_size)
            .field("Placeholder residual step", &self.residuals_step_size)
            .field(
                "Placeholder current residual",
                &self.residuals_values_current,
            )
            .finish()
    }
}
