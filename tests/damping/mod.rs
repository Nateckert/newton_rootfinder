mod non_finite;

use std::error::Error;
use std::fmt;

use newton_rootfinder as nrf;
use nrf::iteratives;
use nrf::model::{Model, ModelError};
use nrf::residuals;

/// The residual `atan(x)` from `x = 2`: the first Newton-Raphson step
/// overshoots to `x ≈ -3.54`, where the residual is larger, so the damping
/// evaluates the damped guess `x ≈ -0.77`, halfway between them.
///
/// The memory holds the iterative of the last evaluation, a state the
/// evaluation keeps between calls; `evaluations` records, for each one,
/// the iterative and the memory it started from.
struct Arctangent {
    x: f64,
    residual: f64,
    last_x: f64,
    evaluations: Vec<(f64, f64)>,
    unusable_between: Option<(f64, f64)>,
}

impl Arctangent {
    fn new(unusable_between: Option<(f64, f64)>) -> Self {
        Arctangent {
            x: 0.0,
            residual: 0.0,
            last_x: f64::NAN,
            evaluations: Vec::new(),
            unusable_between,
        }
    }
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
        self.evaluations.push((self.x, self.last_x));
        self.last_x = self.x;
        if let Some((low, high)) = self.unusable_between {
            if low < self.x && self.x < high {
                return Err(ModelError::UnusableValuesError(Unusable));
            }
        }
        self.residual = self.x.atan();
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Const<1>> {
        residuals::ResidualsValues::new(
            nalgebra::SVector::from([self.residual]),
            nalgebra::SVector::zeros(),
        )
    }

    fn get_memory(&self) -> nalgebra::DVector<f64> {
        nalgebra::DVector::from_vec(vec![self.last_x])
    }

    fn set_memory(&mut self, memory: &nalgebra::DVector<f64>) {
        self.last_x = memory[0];
    }
}

fn solve(
    model: &mut Arctangent,
) -> Result<(), nrf::errors::SolverError<Arctangent, nalgebra::Const<1>>> {
    let vec_iter_params = iteratives::default_vec_iteratives_fd(1);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; 1];
    let update_methods = vec![residuals::NormalizationMethod::Abs; 1];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let damping = true;
    let mut rf = nrf::solver::default_with_guess(
        nalgebra::SVector::from([2.0]),
        &iter_params,
        &res_config,
        nrf::solver::ResolutionMethod::NewtonRaphson,
        damping,
    );
    rf.solve(model)
}

#[test]
fn the_damped_guess_is_evaluated_from_the_memory_of_the_current_point() {
    let mut model = Arctangent::new(None);
    solve(&mut model).unwrap();
    assert!(model.x.abs() < 1e-6);

    // The current point, the column of the jacobian, the proposed guess,
    // then the damped guess.
    let (current, _) = model.evaluations[0];
    let (proposed, from) = model.evaluations[2];
    let (damped, damped_from) = model.evaluations[3];
    assert_eq!(current, 2.0);
    assert!(proposed < -3.5);
    assert_eq!(from, current);
    assert!(-1.0 < damped && damped < 0.0);
    // Not from the memory the rejected proposed guess left.
    assert_eq!(damped_from, current);
}

#[test]
fn the_damped_guess_is_halfway_from_the_current_point() {
    let mut model = Arctangent::new(None);
    solve(&mut model).unwrap();

    // The current point, the column of the jacobian (the current point
    // perturbed), the proposed guess, then the damped guess.
    let (current, _) = model.evaluations[0];
    let (perturbed, _) = model.evaluations[1];
    let (proposed, _) = model.evaluations[2];
    let (damped, _) = model.evaluations[3];
    assert!(perturbed > current);
    // From the current point, not from the one the jacobian perturbed last.
    assert_eq!(damped, (current + proposed) / 2.0);
}

#[test]
fn an_unusable_damped_guess_ends_the_resolution_with_an_error() {
    let mut model = Arctangent::new(Some((-1.0, 0.0)));
    let error = solve(&mut model).unwrap_err();
    assert!(matches!(
        error,
        nrf::errors::SolverError::ModelEvaluationError(ModelError::UnusableValuesError(_))
    ));
    let (damped, _) = *model.evaluations.last().unwrap();
    assert!(-1.0 < damped && damped < 0.0);
}
