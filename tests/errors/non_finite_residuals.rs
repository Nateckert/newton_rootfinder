use newton_rootfinder as nrf;

use nrf::iteratives;
use nrf::model::Model;
use nrf::residuals;

fn solve(
    problem_size: usize,
    init: nalgebra::DVector<f64>,
    normalization: residuals::NormalizationMethod,
    function: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
) -> Result<(), nrf::errors::SolverError<nrf::model::UserModelFromFunction, nalgebra::Dyn>> {
    let damping = false;

    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![normalization; problem_size];
    let update_methods = vec![normalization; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf = nrf::solver::default(
        &iter_params,
        &res_config,
        nrf::solver::ResolutionMethod::NewtonRaphson,
        damping,
    );
    let mut user_model = nrf::model::UserModelFromFunction::new(problem_size, function);

    user_model.set_iteratives(&init);
    rf.solve(&mut user_model)
}

fn assert_non_finite(
    result: Result<(), nrf::errors::SolverError<nrf::model::UserModelFromFunction, nalgebra::Dyn>>,
    indices: Vec<usize>,
) {
    let expected: nrf::errors::SolverError<nrf::model::UserModelFromFunction, nalgebra::Dyn> =
        nrf::errors::SolverError::NonFiniteResiduals(indices);
    assert_eq!(expected.to_string(), result.unwrap_err().to_string());
}

fn nan(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut outputs = nalgebra::DVector::zeros(x.len());
    outputs[0] = f64::NAN;
    outputs
}

fn nan_and_finite(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut outputs = nalgebra::DVector::zeros(x.len());
    outputs[0] = f64::NAN;
    outputs[1] = x[1] - 5.0;
    outputs
}

fn infinite(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut outputs = nalgebra::DVector::zeros(x.len());
    outputs[0] = x[0] - 1.0;
    outputs[1] = f64::INFINITY;
    outputs
}

// No root: the first Newton step from 1.0 goes to -3.0, where the square root is NaN
fn nan_after_a_step(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut outputs = nalgebra::DVector::zeros(x.len());
    outputs[0] = x[0].sqrt() + 1.0;
    outputs
}

fn identity(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    x.clone()
}

#[test]
fn nan_residual_at_the_initial_evaluation() {
    let init = nalgebra::DVector::from_element(1, 1.0);
    let result = solve(1, init, residuals::NormalizationMethod::Abs, nan);
    assert_non_finite(result, vec![0]);
}

#[test]
fn nan_residual_next_to_a_finite_one() {
    let init = nalgebra::DVector::from_element(2, 1.0);
    let result = solve(2, init, residuals::NormalizationMethod::Abs, nan_and_finite);
    assert_non_finite(result, vec![0]);
}

#[test]
fn infinite_residual() {
    let init = nalgebra::DVector::from_element(2, 1.0);
    let result = solve(2, init, residuals::NormalizationMethod::Adapt, infinite);
    assert_non_finite(result, vec![1]);
}

#[test]
fn nan_residual_after_an_update() {
    let init = nalgebra::DVector::from_element(1, 1.0);
    let result = solve(
        1,
        init,
        residuals::NormalizationMethod::Abs,
        nan_after_a_step,
    );
    assert_non_finite(result, vec![0]);
}

// The relative normalization of 0 = 0 is NaN, but the residuals are finite: converged
#[test]
fn relative_normalization_of_zero_residuals() {
    let init = nalgebra::DVector::zeros(1);
    let result = solve(1, init, residuals::NormalizationMethod::Rel, identity);
    assert!(result.is_ok());
}
