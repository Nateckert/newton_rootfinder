use crate::common::broyden1965::*;
extern crate newton_rootfinder;
use newton_rootfinder::solver_advanced as nrf;

extern crate nalgebra;
use nrf::iteratives;
use nrf::model::Model;
use nrf::residuals;

#[test]
fn broyden_case5_fd() {
    let problem_size = 5;
    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case5(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case5);

    rf.solve(&mut user_model);

    let solution = solution_broyden1965_case5();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

#[test]
fn broyden_case5_jac() {
    let problem_size = 5;
    let vec_iter_params = iteratives::default_vec_iteratives(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case5(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFuncJac::new(
        problem_size,
        broyden1965_case5,
        broyden1965_case5_jac,
    );

    rf.solve(&mut user_model);

    let solution = solution_broyden1965_case5();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

#[test]
fn broyden_case6_fd() {
    let problem_size = 5;
    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case6(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case6);

    rf.set_debug(true);
    rf.solve(&mut user_model);
    rf.write_log(&"log_fd.txt");

    let solution = solution_broyden1965_case6();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

#[test]
fn broyden_case6_jac() {
    let problem_size = 5;
    let vec_iter_params = iteratives::default_vec_iteratives(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case6(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFuncJac::new(
        problem_size,
        broyden1965_case6,
        broyden1965_case6_jac,
    );

    rf.set_debug(true);
    rf.solve(&mut user_model);
    rf.write_log(&"log_jac.txt");

    let solution = solution_broyden1965_case6();

    for i in 0..problem_size {
        println!("{} = {}", user_model.get_iteratives()[i], solution[i]);
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

#[test]
fn broyden_case7_fd() {
    let problem_size = 10;
    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case7(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case7);

    rf.solve(&mut user_model);

    let solution = solution_broyden1965_case7();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

#[test]
fn broyden_case7_jac() {
    let problem_size = 10;
    let vec_iter_params = iteratives::default_vec_iteratives(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case7(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFuncJac::new(
        problem_size,
        broyden1965_case7,
        broyden1965_case7_jac,
    );

    rf.solve(&mut user_model);

    let solution = solution_broyden1965_case7();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

#[test]
fn broyden_case8_fd() {
    let problem_size = 20;
    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case8(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case8);

    rf.solve(&mut user_model);

    let solution = solution_broyden1965_case8();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

#[test]
fn broyden_case8_jac() {
    let problem_size = 20;
    let vec_iter_params = iteratives::default_vec_iteratives(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case8(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFuncJac::new(
        problem_size,
        broyden1965_case8,
        broyden1965_case8_jac,
    );

    rf.solve(&mut user_model);

    let solution = solution_broyden1965_case8();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

#[test]
fn broyden_case9_fd() {
    let problem_size = 2;
    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case9(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case9);

    rf.solve(&mut user_model);

    let solution = solution_broyden1965_case9();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

#[test]
fn broyden_case9_jac() {
    let problem_size = 2;
    let vec_iter_params = iteratives::default_vec_iteratives(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case9(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFuncJac::new(
        problem_size,
        broyden1965_case9,
        broyden1965_case9_jac,
    );

    rf.solve(&mut user_model);

    let solution = solution_broyden1965_case9();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

#[test]
#[should_panic] // This panic is unexpected, see file common/broyden1965
fn broyden_case10_fd() {
    let problem_size = 2;
    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case10(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case10);

    rf.solve(&mut user_model);

    let solution = solution_broyden1965_case10();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

#[test]
fn broyden_case10_jac() {
    let problem_size = 2;
    let vec_iter_params = iteratives::default_vec_iteratives(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf =
        nrf::solver::default_with_guess(init_broyden1965_case10(), &iter_params, &res_config);
    let mut user_model = nrf::model::UserModelWithFuncJac::new(
        problem_size,
        broyden1965_case10,
        broyden1965_case10_jac,
    );

    rf.solve(&mut user_model);

    let solution = solution_broyden1965_case10();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}
