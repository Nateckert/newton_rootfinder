extern crate nalgebra;
extern crate newton_rootfinder as nrf;

use nrf::model::Model; // trait import

use crate::common::polynom;

extern crate float_cmp;

#[test]
fn square() {
    let problem_size = 1;
    let init_guess = nalgebra::DVector::from_vec(vec![1.0]);
    let rf = nrf::solver::RootFinderFD::default_with_guess(init_guess);
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, polynom::square2);

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
    let rf = nrf::solver::RootFinderFD::default_with_guess(init_guess);
    let mut user_model = nrf::model_with_func::UserModelWithFunc::new(
        problem_size,
        polynom::root_with_high_derivative,
    );

    rf.solve(&mut user_model);
    assert!(float_cmp::approx_eq!(
        f64,
        user_model.get_iteratives()[0],
        0.1,
        epsilon = 1e-4
    ));
}
