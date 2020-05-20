extern crate nalgebra;
extern crate newton_rootfinder;
use newton_rootfinder::solver_advanced as nrf;

use nrf::iteratives;
use nrf::model::Model;
use nrf::residuals; // trait import

use crate::common::polynom;

extern crate float_cmp;

#[test]
fn square() {
    let problem_size = 1;
    let init_guess = nalgebra::DVector::from_vec(vec![1.0]);
    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf = nrf::solver::default_with_guess(init_guess, iter_params, res_config);

    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, polynom::square2);

    rf.solve(&mut user_model);

    assert!(float_cmp::approx_eq!(
        f64,
        user_model.get_iteratives()[0],
        2_f64.sqrt(),
        epsilon = 1e-6
    ));
}

#[test]
fn root_with_high_derivative() {
    let problem_size = 1;
    let init_guess = nalgebra::DVector::from_vec(vec![0.15]);
    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf = nrf::solver::default_with_guess(init_guess, iter_params, res_config);

    let mut user_model =
        nrf::model::UserModelWithFunc::new(problem_size, polynom::root_with_high_derivative);

    rf.solve(&mut user_model);
    assert!(float_cmp::approx_eq!(
        f64,
        user_model.get_iteratives()[0],
        0.1,
        epsilon = 1e-4
    ));
}
