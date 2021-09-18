extern crate newton_rootfinder;
use newton_rootfinder as nrf;

use util::test_cases::broyden1965::*;

use crate::common::float_matrix_comparison;

extern crate nalgebra;
use nrf::model::Model;

#[test]
fn broyden_case5_jac() {
    let init_guess = init_broyden1965_case5();
    let problem_size = init_guess.len();

    let mut user_model = nrf::model::UserModelFromFunctionAndJacobian::new(
        problem_size,
        broyden1965_case5,
        broyden1965_case5_jac,
    );

    user_model.set_iteratives(&init_guess);
    user_model.evaluate();
    let jacobians = user_model.get_jacobian().unwrap();
    let (left_jac, right_jac) = jacobians.get_jacobians();

    let jac_ref = broyden1965_case5_jac(&init_guess);
    let zeros = nalgebra::DMatrix::zeros(problem_size, problem_size);

    float_matrix_comparison(&left_jac, &jac_ref, 1e-6);
    float_matrix_comparison(&right_jac, &zeros, 1e-6);
}

#[test]
fn broyden_case6_jac() {
    let init_guess = init_broyden1965_case6();
    let problem_size = init_guess.len();

    let mut user_model = nrf::model::UserModelFromFunctionAndJacobian::new(
        problem_size,
        broyden1965_case6,
        broyden1965_case6_jac,
    );

    user_model.set_iteratives(&init_guess);
    user_model.evaluate();
    let jacobians = user_model.get_jacobian().unwrap();
    let (left_jac, right_jac) = jacobians.get_jacobians();

    let jac_ref = broyden1965_case6_jac(&init_guess);
    let zeros = nalgebra::DMatrix::zeros(problem_size, problem_size);

    float_matrix_comparison(&left_jac, &jac_ref, 1e-6);
    float_matrix_comparison(&right_jac, &zeros, 1e-6);
}

#[test]
fn broyden_case7_jac() {
    let init_guess = init_broyden1965_case7();
    let problem_size = init_guess.len();

    let mut user_model = nrf::model::UserModelFromFunctionAndJacobian::new(
        problem_size,
        broyden1965_case7,
        broyden1965_case7_jac,
    );

    user_model.set_iteratives(&init_guess);
    user_model.evaluate();
    let jacobians = user_model.get_jacobian().unwrap();
    let (left_jac, right_jac) = jacobians.get_jacobians();

    let jac_ref = broyden1965_case7_jac(&init_guess);
    let zeros = nalgebra::DMatrix::zeros(problem_size, problem_size);

    float_matrix_comparison(&left_jac, &jac_ref, 1e-6);
    float_matrix_comparison(&right_jac, &zeros, 1e-6);
}

#[test]
fn broyden_case8_jac() {
    let init_guess = init_broyden1965_case8();
    let problem_size = init_guess.len();

    let mut user_model = nrf::model::UserModelFromFunctionAndJacobian::new(
        problem_size,
        broyden1965_case8,
        broyden1965_case8_jac,
    );

    user_model.set_iteratives(&init_guess);
    user_model.evaluate();
    let jacobians = user_model.get_jacobian().unwrap();
    let (left_jac, right_jac) = jacobians.get_jacobians();

    let jac_ref = broyden1965_case8_jac(&init_guess);
    let zeros = nalgebra::DMatrix::zeros(problem_size, problem_size);

    float_matrix_comparison(&left_jac, &jac_ref, 1e-6);
    float_matrix_comparison(&right_jac, &zeros, 1e-6);
}

#[test]
fn broyden_case9_jac() {
    let init_guess = init_broyden1965_case9();
    let problem_size = init_guess.len();

    let mut user_model = nrf::model::UserModelFromFunctionAndJacobian::new(
        problem_size,
        broyden1965_case9,
        broyden1965_case9_jac,
    );

    user_model.set_iteratives(&init_guess);
    user_model.evaluate();
    let jacobians = user_model.get_jacobian().unwrap();
    let (left_jac, right_jac) = jacobians.get_jacobians();

    let jac_ref = broyden1965_case9_jac(&init_guess);
    let zeros = nalgebra::DMatrix::zeros(problem_size, problem_size);

    float_matrix_comparison(&left_jac, &jac_ref, 1e-6);
    float_matrix_comparison(&right_jac, &zeros, 1e-6);
}

#[test]
fn broyden_case10_jac() {
    let init_guess = init_broyden1965_case10();
    let problem_size = init_guess.len();

    let mut user_model = nrf::model::UserModelFromFunctionAndJacobian::new(
        problem_size,
        broyden1965_case10,
        broyden1965_case10_jac,
    );

    user_model.set_iteratives(&init_guess);
    user_model.evaluate();
    let jacobians = user_model.get_jacobian().unwrap();
    let (left_jac, right_jac) = jacobians.get_jacobians();

    let jac_ref = broyden1965_case10_jac(&init_guess);
    let zeros = nalgebra::DMatrix::zeros(problem_size, problem_size);

    float_matrix_comparison(&left_jac, &jac_ref, 1e-6);
    float_matrix_comparison(&right_jac, &zeros, 1e-6);
}
