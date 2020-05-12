use crate::common::broyden1965::*;
extern crate newton_rootfinder as nrf;

extern crate nalgebra;
use nrf::model::Model;

#[test]
fn broyden_case5() {
    let problem_size = 5;
    let rf = nrf::solver::RootFinder::default_with_guess(init_broyden1965_case5());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case5);

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
fn broyden_case6() {
    let problem_size = 5;
    let rf = nrf::solver::RootFinder::default_with_guess(init_broyden1965_case6());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case6);

    rf.solve(&mut user_model);

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
fn broyden_case7() {
    let problem_size = 10;
    let rf = nrf::solver::RootFinder::default_with_guess(init_broyden1965_case7());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case7);

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
fn broyden_case8() {
    let problem_size = 20;
    let rf = nrf::solver::RootFinder::default_with_guess(init_broyden1965_case8());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case8);

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
fn broyden_case9() {
    let problem_size = 2;
    let rf = nrf::solver::RootFinder::default_with_guess(init_broyden1965_case9());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case9);

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
fn broyden_case10() {
    let problem_size = 2;
    let rf = nrf::solver::RootFinder::default_with_guess(init_broyden1965_case10());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case10);

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
