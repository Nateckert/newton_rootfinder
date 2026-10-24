use newton_rootfinder as nrf;

use nrf::iteratives;
use nrf::residuals;

fn solve(
    problem_size: usize,
    function: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    damping: bool,
) -> (
    Result<(), nrf::errors::SolverError<nrf::model::UserModelFromFunction, nalgebra::Dyn>>,
    nalgebra::DVector<f64>,
) {
    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf = nrf::solver::default(
        &iter_params,
        &res_config,
        nrf::solver::ResolutionMethod::NewtonRaphson,
        damping,
    );
    let mut user_model = nrf::model::UserModelFromFunction::new(problem_size, function);
    nrf::model::Model::set_iteratives(
        &mut user_model,
        &nalgebra::DVector::from_element(problem_size, 9.0),
    );
    let result = rf.solve(&mut user_model);
    (result, nrf::model::Model::get_iteratives(&user_model))
}

// From 9, the first Newton step goes to -3, where the square root is NaN;
// the damped guess, halfway, is 3
fn square_root(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut outputs = nalgebra::DVector::zeros(x.len());
    outputs[0] = x[0].sqrt() - 1.0;
    outputs
}

// The same, next to a linear residual the first step solves:
// the maximum of the errors after it, NaN ignored, is 0
fn square_root_and_linear(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut outputs = nalgebra::DVector::zeros(x.len());
    outputs[0] = x[0].sqrt() - 1.0;
    outputs[1] = x[1] - 5.0;
    outputs
}

#[test]
fn a_step_to_a_nan_residual_is_damped() {
    let (result, iteratives) = solve(1, square_root, true);
    result.unwrap();
    assert!((iteratives[0] - 1.0).abs() < 1e-5);
}

#[test]
fn a_step_to_a_nan_residual_next_to_a_solved_one_is_damped() {
    let (result, iteratives) = solve(2, square_root_and_linear, true);
    result.unwrap();
    assert!((iteratives[0] - 1.0).abs() < 1e-5);
    assert!((iteratives[1] - 5.0).abs() < 1e-6);
}

#[test]
fn without_damping_a_step_to_a_nan_residual_fails() {
    let (result, _) = solve(1, square_root, false);
    assert!(matches!(
        result,
        Err(nrf::errors::SolverError::NonFiniteResiduals(indices)) if indices == vec![0]
    ));
}
