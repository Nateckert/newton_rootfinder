extern crate newton_rootfinder;
use newton_rootfinder::solver_advanced as nrf;
use nrf::model::Model;
use nrf::residuals;
use nrf::util::jacobian;

use crate::common::broyden1965;
use crate::common::float_matrix_comparison;

extern crate float_cmp;
extern crate nalgebra;

#[test]
fn jacobian_evaluation_broyden1965_case5() {
    let inputs = broyden1965::init_broyden1965_case5();
    let problem_size = inputs.len();
    let mut user_model =
        nrf::model::UserModelWithFunc::new(problem_size, broyden1965::broyden1965_case5);
    user_model.set_iteratives(&inputs);
    user_model.evaluate();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = jacobian::jacobian_evaluation(&mut user_model, &perturbations, &res_config);
    let jac_ref = broyden1965::broyden1965_case5_jac(&inputs);

    float_matrix_comparison(&jac, &jac_ref, 1e-6);
}

#[test]
fn jacobian_evaluation_broyden1965_case6() {
    let inputs = broyden1965::init_broyden1965_case6();
    let problem_size = inputs.len();
    let mut user_model =
        nrf::model::UserModelWithFunc::new(problem_size, broyden1965::broyden1965_case6);
    user_model.set_iteratives(&inputs);
    user_model.evaluate();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = jacobian::jacobian_evaluation(&mut user_model, &perturbations, &res_config);
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
        nrf::model::UserModelWithFunc::new(problem_size, broyden1965::broyden1965_case7);

    user_model.set_iteratives(&inputs);
    user_model.evaluate();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = jacobian::jacobian_evaluation(&mut user_model, &perturbations, &res_config);
    let jac_ref = broyden1965::broyden1965_case7_jac(&inputs);

    float_matrix_comparison(&jac, &jac_ref, 1e-6);
}

#[test]
fn jacobian_evaluation_broyden1965_case8() {
    let inputs = broyden1965::init_broyden1965_case8();
    let problem_size = inputs.len();
    let mut user_model =
        nrf::model::UserModelWithFunc::new(problem_size, broyden1965::broyden1965_case8);

    user_model.set_iteratives(&inputs);
    user_model.evaluate();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = jacobian::jacobian_evaluation(&mut user_model, &perturbations, &res_config);
    let jac_ref = broyden1965::broyden1965_case8_jac(&inputs);

    float_matrix_comparison(&jac, &jac_ref, 1e-6);
}

#[test]
fn jacobian_evaluation_broyden1965_case9() {
    let inputs = broyden1965::init_broyden1965_case9();
    let problem_size = inputs.len();
    let mut user_model =
        nrf::model::UserModelWithFunc::new(problem_size, broyden1965::broyden1965_case9);

    user_model.set_iteratives(&inputs);
    user_model.evaluate();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = jacobian::jacobian_evaluation(&mut user_model, &perturbations, &res_config);
    let jac_ref = broyden1965::broyden1965_case9_jac(&inputs);

    float_matrix_comparison(&jac, &jac_ref, 1e-6);
}

#[test]
fn jacobian_evaluation_broyden1965_case10() {
    let inputs = broyden1965::init_broyden1965_case10();
    let problem_size = inputs.len();
    let mut user_model =
        nrf::model::UserModelWithFunc::new(problem_size, broyden1965::broyden1965_case10);

    user_model.set_iteratives(&inputs);
    user_model.evaluate();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8; problem_size]);
    let jac = jacobian::jacobian_evaluation(&mut user_model, &perturbations, &res_config);
    let jac_ref = broyden1965::broyden1965_case10_jac(&inputs);

    float_matrix_comparison(&jac, &jac_ref, 1e-6);
}
