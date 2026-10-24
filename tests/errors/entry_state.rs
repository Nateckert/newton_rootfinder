use std::error::Error;
use std::fmt;

use newton_rootfinder as nrf;
use nrf::errors::SolverError;
use nrf::iteratives;
use nrf::model::{Model, ModelError};
use nrf::residuals;
use nrf::solver::compute_jacobian_from_finite_difference;

/// The residual `atan(x)` from `x = 2`: the jacobian perturbs `x` to
/// `2.0000001`, then the first Newton-Raphson step proposes `x ≈ -3.54`.
///
/// The memory holds the iterative of the last evaluation. The evaluation fails
/// on `unusable`, and gives a NaN residual on `not_a_number`.
struct Arctangent {
    x: f64,
    residual: f64,
    last_x: f64,
    unusable: Option<(f64, f64)>,
    not_a_number: Option<(f64, f64)>,
}

impl Arctangent {
    /// A model left by a previous resolution on `x = 7`, its memory `5`
    fn new(unusable: Option<(f64, f64)>, not_a_number: Option<(f64, f64)>) -> Self {
        Arctangent {
            x: 7.0,
            residual: 7.0_f64.atan(),
            last_x: 5.0,
            unusable,
            not_a_number,
        }
    }
}

fn within(x: f64, range: Option<(f64, f64)>) -> bool {
    matches!(range, Some((low, high)) if low < x && x < high)
}

#[derive(Debug)]
struct Unusable;

impl fmt::Display for Unusable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unusable value")
    }
}

impl Error for Unusable {}

impl Model<nalgebra::Const<1>> for Arctangent {
    type InaccurateValuesError = std::convert::Infallible;
    type UnusableValuesError = Unusable;
    type Memory = f64;

    fn len_problem(&self) -> usize {
        1
    }

    fn set_iteratives(&mut self, iteratives: &nalgebra::SVector<f64, 1>) {
        self.x = iteratives[0];
    }

    fn get_iteratives(&self) -> nalgebra::SVector<f64, 1> {
        nalgebra::SVector::from([self.x])
    }

    fn evaluate(&mut self) -> Result<(), ModelError<Self, nalgebra::Const<1>>> {
        self.last_x = self.x;
        if within(self.x, self.unusable) {
            return Err(ModelError::UnusableValuesError(Unusable));
        }
        self.residual = if within(self.x, self.not_a_number) {
            f64::NAN
        } else {
            self.x.atan()
        };
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Const<1>> {
        residuals::ResidualsValues::new(
            nalgebra::SVector::from([self.residual]),
            nalgebra::SVector::zeros(),
        )
    }

    fn get_memory(&self) -> f64 {
        self.last_x
    }

    fn set_memory(&mut self, memory: &f64) {
        self.last_x = *memory;
    }
}

fn solve(
    model: &mut Arctangent,
    max_iter: usize,
) -> Result<(), SolverError<Arctangent, nalgebra::Const<1>>> {
    let vec_iter_params = iteratives::default_vec_iteratives_fd(1);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; 1];
    let update_methods = vec![residuals::NormalizationMethod::Abs; 1];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let damping = false;
    let parameters = nrf::solver::SolverParameters::new(
        1,
        1e-6,
        max_iter,
        nrf::solver::ResolutionMethod::NewtonRaphson,
        damping,
    );
    let mut rf = nrf::solver::RootFinder::new(
        parameters,
        nalgebra::SVector::from([2.0]),
        &iter_params,
        &res_config,
    );
    rf.solve(model)
}

fn assert_entry_state(model: &Arctangent) {
    assert_eq!(model.x, 7.0);
    assert_eq!(model.last_x, 5.0);
}

#[test]
fn a_failed_jacobian_column_restores_the_memory_of_the_reference_point() {
    let mut model = Arctangent::new(Some((2.00000005, 2.0000002)), None);
    model.set_iteratives(&nalgebra::SVector::from([2.0]));
    model.evaluate().unwrap();
    let stopping = vec![residuals::NormalizationMethod::Abs; 1];
    let config = residuals::ResidualsConfig::new(&stopping, &stopping);
    let perturbations = nalgebra::SVector::from([0.0000001]);
    compute_jacobian_from_finite_difference(&mut model, &perturbations, &config).unwrap_err();
    assert_eq!(model.last_x, 2.0);
}

#[test]
fn an_unusable_initial_guess_restores_the_entry_state() {
    let mut model = Arctangent::new(Some((1.9, 2.1)), None);
    let error = solve(&mut model, 50).unwrap_err();
    assert!(matches!(
        error,
        SolverError::ModelInitialEvaluationError(Unusable)
    ));
    assert_eq!(
        error.to_string(),
        "Initial model evaluation failed: Unusable value"
    );
    assert_entry_state(&model);
}

#[test]
fn an_unusable_proposed_guess_restores_the_entry_state() {
    let mut model = Arctangent::new(Some((-4.0, -3.0)), None);
    let error = solve(&mut model, 50).unwrap_err();
    assert!(matches!(error, SolverError::ModelEvaluationError(_)));
    assert_entry_state(&model);
}

#[test]
fn a_failed_jacobian_restores_the_entry_state() {
    let mut model = Arctangent::new(Some((2.00000005, 2.0000002)), None);
    let error = solve(&mut model, 50).unwrap_err();
    assert!(matches!(error, SolverError::JacobianError(_)));
    assert_entry_state(&model);
}

#[test]
fn a_non_finite_residual_restores_the_entry_state() {
    let mut model = Arctangent::new(None, Some((-4.0, -3.0)));
    let error = solve(&mut model, 50).unwrap_err();
    assert!(matches!(error, SolverError::NonFiniteResiduals(_)));
    assert_entry_state(&model);
}

#[test]
fn a_non_convergence_leaves_the_last_iterate() {
    let mut model = Arctangent::new(None, None);
    let error = solve(&mut model, 1).unwrap_err();
    assert!(matches!(error, SolverError::NonConvergenceError));
    assert!(model.x < -3.5);
    assert_eq!(model.last_x, model.x);
}
