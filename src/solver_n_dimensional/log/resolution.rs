//! The lines a resolution writes

use std::fmt;

use super::{Detail, Fields, Log, json};
use crate::errors::SolverError;
use crate::iteratives::{Iterative, Iteratives};
use crate::model::Model;
use crate::residuals::{NormalizationMethod, ResidualsConfig, ResidualsValues};
use crate::solver::{
    JacobianMatrix, QuasiNewtonMethod, ResolutionMethod, SolverParameters, UpdateQuasiNewtonMethod,
};

/// The point a model is evaluated at
#[derive(Debug, Clone, Copy)]
pub(crate) enum Evaluation {
    Guess,
    Step,
    Damping,
}

impl Evaluation {
    fn name(self) -> &'static str {
        match self {
            Evaluation::Guess => "guess",
            Evaluation::Step => "step",
            Evaluation::Damping => "damping",
        }
    }
}

/// How a jacobian was obtained
#[derive(Debug, Clone, Copy)]
pub(crate) enum JacobianSource {
    FiniteDifferences,
    Analytic,
    Update(UpdateQuasiNewtonMethod),
}

impl JacobianSource {
    fn name(self) -> &'static str {
        match self {
            JacobianSource::FiniteDifferences => "finite_differences",
            JacobianSource::Analytic => "analytic",
            JacobianSource::Update(method) => update_method_name(method),
        }
    }
}

/// The log of the resolutions of one solver, under its label
///
/// Its methods are cold and never inlined: a resolution without a log
/// keeps a test of the option of its solver at each event, and nothing else of them in its code
pub(crate) struct ResolutionLog {
    log: Log,
    label: String,
    /// The detail of the resolution in progress, read from the log at its start
    detail: Detail,
    /// The resolution in progress
    id: Option<u64>,
    /// The largest error of the last evaluation, for the `end` line
    max_error: Option<f64>,
}

impl ResolutionLog {
    pub(crate) fn new(log: &Log, label: &str) -> Self {
        ResolutionLog {
            log: log.clone(),
            label: label.to_owned(),
            detail: log.get_detail(),
            id: None,
            max_error: None,
        }
    }

    /// Whether the resolution in progress writes the events of this detail
    pub(crate) fn writes(&self, detail: Detail) -> bool {
        self.detail >= detail
    }

    #[cold]
    #[inline(never)]
    pub(crate) fn start<T: Iterative>(
        &mut self,
        parameters: &SolverParameters,
        iteratives: &Iteratives<T>,
        residuals: &ResidualsConfig,
    ) {
        self.detail = self.log.get_detail();
        self.max_error = None;
        let label = &self.label;
        let id = self.log.open(|line, id, parent, time| {
            line.push_str("{\"event\":\"start\",\"id\":");
            json::unsigned(line, id);
            json::key(line, "parent");
            json::optional_unsigned(line, parent);
            json::key(line, "time");
            json::number(line, time);
            json::key(line, "label");
            json::string(line, label);
            json::key(line, "method");
            json::string(line, method_name(parameters.get_resolution_method()));
            json::key(line, "tolerance");
            json::number(line, parameters.get_tolerance());
            json::key(line, "max_iterations");
            json::unsigned(line, parameters.get_max_iter() as u64);
            json::key(line, "damping");
            json::boolean(line, parameters.get_damping());

            json::key(line, "iteratives");
            line.push('[');
            for (i, iterative) in iteratives.params().iter().enumerate() {
                if i > 0 {
                    line.push(',');
                }
                line.push_str("{\"settings\":");
                let mut settings = Fields::open(line);
                iterative.log_settings(&mut settings);
                settings.close();
                line.push('}');
            }
            line.push(']');

            json::key(line, "residuals");
            line.push('[');
            let stopping = residuals.get_stopping_criterias();
            let update = residuals.get_update_methods();
            for i in 0..residuals.len() {
                if i > 0 {
                    line.push(',');
                }
                line.push_str("{\"settings\":");
                let mut settings = Fields::open(line);
                settings
                    .text("stopping", normalization_name(stopping[i]))
                    .text("update", normalization_name(update[i]));
                settings.close();
                line.push('}');
            }
            line.push(']');
            line.push('}');
        });
        self.id = Some(id);
    }

    /// The line of an event of the resolution in progress, its object left open
    fn line(&self, event: &str, id: u64) -> String {
        let mut line = String::with_capacity(256);
        line.push_str("{\"event\":");
        json::string(&mut line, event);
        json::key(&mut line, "id");
        json::unsigned(&mut line, id);
        json::key(&mut line, "time");
        json::number(&mut line, self.log.elapsed());
        line
    }

    /// The id of the resolution in progress, when it writes the events of this detail
    fn written(&self, detail: Detail) -> Option<u64> {
        if self.writes(detail) { self.id } else { None }
    }

    /// The largest error of an evaluation, kept for the `end` line,
    /// when the detail does not write the evaluations
    #[cold]
    #[inline(never)]
    pub(crate) fn max_error<D>(&mut self, errors: &nalgebra::OVector<f64, D>)
    where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    {
        self.max_error = Some(errors.amax());
    }

    #[cold]
    #[inline(never)]
    pub(crate) fn evaluation<D>(
        &mut self,
        iteration: usize,
        of: Evaluation,
        iteratives: &nalgebra::OVector<f64, D>,
        residuals: &ResidualsValues<D>,
        errors: &nalgebra::OVector<f64, D>,
    ) where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    {
        let max_error = errors.amax();
        self.max_error = Some(max_error);
        let Some(id) = self.written(Detail::Iterations) else {
            return;
        };
        let mut line = self.line("evaluation", id);
        json::key(&mut line, "iteration");
        json::unsigned(&mut line, iteration as u64);
        json::key(&mut line, "of");
        json::string(&mut line, of.name());
        json::key(&mut line, "iteratives");
        json::numbers(&mut line, iteratives.iter().copied());
        json::key(&mut line, "left");
        json::numbers(&mut line, residuals.left().iter().copied());
        json::key(&mut line, "right");
        json::numbers(&mut line, residuals.right().iter().copied());
        json::key(&mut line, "errors");
        json::numbers(&mut line, errors.iter().copied());
        json::key(&mut line, "max_error");
        json::number(&mut line, max_error);
        line.push('}');
        self.log.write(&line);
    }

    /// An evaluation the model could not make: the iteratives, and the message of the model
    #[cold]
    #[inline(never)]
    pub(crate) fn failed_evaluation<D>(
        &self,
        iteration: usize,
        of: Evaluation,
        iteratives: &nalgebra::OVector<f64, D>,
        message: &str,
    ) where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    {
        let Some(id) = self.written(Detail::Iterations) else {
            return;
        };
        let mut line = self.line("evaluation", id);
        json::key(&mut line, "iteration");
        json::unsigned(&mut line, iteration as u64);
        json::key(&mut line, "of");
        json::string(&mut line, of.name());
        json::key(&mut line, "iteratives");
        json::numbers(&mut line, iteratives.iter().copied());
        json::key(&mut line, "error");
        json::string(&mut line, message);
        line.push('}');
        self.log.write(&line);
    }

    /// The step solving the linear system, before the limits of the iteratives
    #[cold]
    #[inline(never)]
    pub(crate) fn step<D>(&self, iteration: usize, step: &nalgebra::OVector<f64, D>)
    where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    {
        let Some(id) = self.written(Detail::Iterations) else {
            return;
        };
        let mut line = self.line("step", id);
        json::key(&mut line, "iteration");
        json::unsigned(&mut line, iteration as u64);
        json::key(&mut line, "step");
        json::numbers(&mut line, step.iter().copied());
        line.push('}');
        self.log.write(&line);
    }

    /// What the resolution holds of the jacobian it has just obtained:
    /// the matrix, singular when it could not be factorized, or the inverse.
    /// Nothing when it holds neither, after an evaluation the model could not make
    #[cold]
    #[inline(never)]
    pub(crate) fn jacobian<D>(
        &self,
        iteration: usize,
        source: JacobianSource,
        jacobian: &JacobianMatrix<D>,
    ) where
        D: nalgebra::DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
    {
        let Some(id) = self.written(Detail::Jacobians) else {
            return;
        };
        let mut line = self.line("jacobian", id);
        json::key(&mut line, "iteration");
        json::unsigned(&mut line, iteration as u64);
        json::key(&mut line, "how");
        json::string(&mut line, source.name());
        match (jacobian.get_jacobian(), jacobian.get_inverse()) {
            (Some(matrix), _) => {
                json::key(&mut line, "matrix");
                json::matrix(&mut line, matrix);
                if !jacobian.is_factorized() {
                    json::key(&mut line, "singular");
                    json::boolean(&mut line, true);
                }
            }
            (None, Some(inverse)) => {
                json::key(&mut line, "inverse");
                json::matrix(&mut line, inverse);
            }
            (None, None) => return,
        }
        line.push('}');
        self.log.write(&line);
    }

    /// A step of a quasi-Newton method refused: the jacobian is computed at the next iteration
    #[cold]
    #[inline(never)]
    pub(crate) fn refused(&self, iteration: usize) {
        let Some(id) = self.written(Detail::Iterations) else {
            return;
        };
        let mut line = self.line("refused", id);
        json::key(&mut line, "iteration");
        json::unsigned(&mut line, iteration as u64);
        line.push('}');
        self.log.write(&line);
    }

    #[cold]
    #[inline(never)]
    pub(crate) fn end<M, D>(&mut self, result: &Result<(), SolverError<M, D>>, iterations: usize)
    where
        M: Model<D>,
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    {
        let Some(id) = self.id.take() else {
            return;
        };
        let mut line = self.line("end", id);
        json::key(&mut line, "outcome");
        match result {
            Ok(()) => json::string(&mut line, "converged"),
            Err(error) => {
                json::string(&mut line, outcome_name(error));
                json::key(&mut line, "message");
                json::string(&mut line, &error.to_string());
                if let SolverError::NonFiniteResiduals(indices) = error {
                    json::key(&mut line, "residuals");
                    json::unsigneds(&mut line, indices.iter().map(|&i| i as u64));
                }
            }
        }
        json::key(&mut line, "iterations");
        json::unsigned(&mut line, iterations as u64);
        json::key(&mut line, "max_error");
        match self.max_error {
            Some(max_error) => json::number(&mut line, max_error),
            None => line.push_str("null"),
        }
        line.push('}');
        self.log.close_node(id, &line, result.is_err());
    }
}

/// A resolution interrupted by a panic in the model ends when its solver is dropped, unwinding
impl Drop for ResolutionLog {
    #[cold]
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            let mut line = self.line("end", id);
            json::key(&mut line, "outcome");
            json::string(&mut line, "panic");
            line.push('}');
            self.log.close_node(id, &line, true);
        }
    }
}

impl fmt::Debug for ResolutionLog {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ResolutionLog")
            .field("label", &self.label)
            .field("log", &self.log)
            .finish()
    }
}

fn method_name(method: ResolutionMethod) -> &'static str {
    match method {
        ResolutionMethod::NewtonRaphson => "newton_raphson",
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton) => "stationary_newton",
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(method)) => {
            update_method_name(method)
        }
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::InverseJacobianUpdate(method)) => {
            match method {
                UpdateQuasiNewtonMethod::BroydenFirstMethod => "broyden_first_method_inverse",
                UpdateQuasiNewtonMethod::BroydenSecondMethod => "broyden_second_method_inverse",
                UpdateQuasiNewtonMethod::GreenstadtFirstMethod => "greenstadt_first_method_inverse",
                UpdateQuasiNewtonMethod::GreenstadtSecondMethod => {
                    "greenstadt_second_method_inverse"
                }
            }
        }
    }
}

fn update_method_name(method: UpdateQuasiNewtonMethod) -> &'static str {
    match method {
        UpdateQuasiNewtonMethod::BroydenFirstMethod => "broyden_first_method",
        UpdateQuasiNewtonMethod::BroydenSecondMethod => "broyden_second_method",
        UpdateQuasiNewtonMethod::GreenstadtFirstMethod => "greenstadt_first_method",
        UpdateQuasiNewtonMethod::GreenstadtSecondMethod => "greenstadt_second_method",
    }
}

fn normalization_name(method: NormalizationMethod) -> &'static str {
    match method {
        NormalizationMethod::Abs => "abs",
        NormalizationMethod::Rel => "rel",
        NormalizationMethod::Adapt => "adapt",
    }
}

fn outcome_name<M, D>(error: &SolverError<M, D>) -> &'static str
where
    M: Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    match error {
        SolverError::NonConvergenceError => "non_convergence_error",
        SolverError::ModelInitialEvaluationError(_) => "model_initial_evaluation_error",
        SolverError::ModelEvaluationError(_) => "model_evaluation_error",
        SolverError::JacobianEvaluationError(_) => "jacobian_evaluation_error",
        SolverError::NonInvertibleJacobian => "non_invertible_jacobian",
        SolverError::FinalEvaluationError => "final_evaluation_error",
        SolverError::NonFiniteResiduals(_) => "non_finite_residuals",
        SolverError::DimensionMismatch { .. } => "dimension_mismatch",
    }
}
