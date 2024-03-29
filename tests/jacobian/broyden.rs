use newton_rootfinder as nrf;
use nrf::model::Model;
use nrf::residuals;
use nrf::solver::compute_jacobian_from_finite_difference;

use crate::common::float_matrix_comparison;
use util::test_cases::broyden1965;

#[test]
fn jacobian_evaluation_broyden1965_case5() {
    let inputs = broyden1965::init_broyden1965_case5();
    let problem_size = inputs.len();
    let mut user_model =
        nrf::model::UserModelFromFunction::new(problem_size, broyden1965::broyden1965_case5);
    user_model.set_iteratives(&inputs);
    user_model.evaluate().unwrap();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = compute_jacobian_from_finite_difference(&mut user_model, &perturbations, &res_config)
        .unwrap();
    let jac_ref = broyden1965::broyden1965_case5_jac(&inputs);

    float_matrix_comparison(&jac, &jac_ref, 1e-6);
}

#[test]
fn jacobian_evaluation_broyden1965_case6() {
    let inputs = broyden1965::init_broyden1965_case6();
    let problem_size = inputs.len();
    let mut user_model =
        nrf::model::UserModelFromFunction::new(problem_size, broyden1965::broyden1965_case6);
    user_model.set_iteratives(&inputs);
    user_model.evaluate().unwrap();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = compute_jacobian_from_finite_difference(&mut user_model, &perturbations, &res_config)
        .unwrap();
    let jac_ref = broyden1965::broyden1965_case6_jac(&inputs);

    float_matrix_comparison(&jac, &jac_ref, 1e-6);

    println!("{}", jac);
    println!("{}", jac_ref);
}

#[test]
fn jacobian_evaluation_broyden1965_case7() {
    let inputs = broyden1965::init_broyden1965_case7();
    let problem_size = inputs.len();
    let mut user_model =
        nrf::model::UserModelFromFunction::new(problem_size, broyden1965::broyden1965_case7);

    user_model.set_iteratives(&inputs);
    user_model.evaluate().unwrap();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = compute_jacobian_from_finite_difference(&mut user_model, &perturbations, &res_config)
        .unwrap();
    let jac_ref = broyden1965::broyden1965_case7_jac(&inputs);

    float_matrix_comparison(&jac, &jac_ref, 1e-6);
}

#[test]
fn jacobian_evaluation_broyden1965_case8() {
    let inputs = broyden1965::init_broyden1965_case8();
    let problem_size = inputs.len();
    let mut user_model =
        nrf::model::UserModelFromFunction::new(problem_size, broyden1965::broyden1965_case8);

    user_model.set_iteratives(&inputs);
    user_model.evaluate().unwrap();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = compute_jacobian_from_finite_difference(&mut user_model, &perturbations, &res_config)
        .unwrap();
    let jac_ref = broyden1965::broyden1965_case8_jac(&inputs);

    float_matrix_comparison(&jac, &jac_ref, 1e-6);
}

#[test]
fn jacobian_evaluation_broyden1965_case9() {
    let inputs = broyden1965::init_broyden1965_case9();
    let problem_size = inputs.len();
    let mut user_model =
        nrf::model::UserModelFromFunction::new(problem_size, broyden1965::broyden1965_case9);

    user_model.set_iteratives(&inputs);
    user_model.evaluate().unwrap();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = compute_jacobian_from_finite_difference(&mut user_model, &perturbations, &res_config)
        .unwrap();
    let jac_ref = broyden1965::broyden1965_case9_jac(&inputs);

    float_matrix_comparison(&jac, &jac_ref, 1e-6);
}

#[test]
fn jacobian_evaluation_broyden1965_case10() {
    let inputs = broyden1965::init_broyden1965_case10();
    let problem_size = inputs.len();
    let mut user_model =
        nrf::model::UserModelFromFunction::new(problem_size, broyden1965::broyden1965_case10);

    user_model.set_iteratives(&inputs);
    user_model.evaluate().unwrap();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = compute_jacobian_from_finite_difference(&mut user_model, &perturbations, &res_config)
        .unwrap();
    let jac_ref = broyden1965::broyden1965_case10_jac(&inputs);

    float_matrix_comparison(&jac, &jac_ref, 1e-6);
}
