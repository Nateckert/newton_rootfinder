extern crate newton_rootfinder as nrf;
use nrf::model::Model;
use nrf::util::residuals;
use nrf::util::jacobian;

use crate::common::broyden1965;

extern crate float_cmp;

#[test]
fn jacobian_evaluation_broyden1965_case10() {
    let problem_size = 2;
    let mut user_model = nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965::broyden1965_case10);
    let inputs = broyden1965::init_broyden1965_case10();
    user_model.set_iteratives(&inputs);
    user_model.evaluate();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; 2];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(stopping_residuals, update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![5e-8, 5e-8]);
    let jac = jacobian::jacobian_evaluation(&mut user_model, &perturbations, &res_config);
    let jac_ref = broyden1965::broyden1965_case10_jac(&inputs);

    assert!(float_cmp::approx_eq!(
        f64,
        jac[(0, 0)],
        jac_ref[(0, 0)],
        epsilon = 1e-6
    ));

    assert!(float_cmp::approx_eq!(
        f64,
        jac[(0, 1)],
        jac_ref[(0, 1)],
        epsilon = 1e-6
    ));

    assert!(float_cmp::approx_eq!(
        f64,
        jac[(1, 0)],
        jac_ref[(1, 0)],
        epsilon = 1e-6
    ));

    assert!(float_cmp::approx_eq!(
        f64,
        jac[(1, 1)],
        jac_ref[(1, 1)],
        epsilon = 1e-6
    ));

}
