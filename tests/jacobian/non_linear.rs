extern crate nalgebra;
extern crate newton_rootfinder;
use newton_rootfinder as nrf;
use nrf::model::Model;
use nrf::residuals;
use nrf::solver::jacobian_evaluation;

pub fn non_linear(inputs: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut outputs = nalgebra::DVector::zeros(2);
    outputs[0] = inputs[0] + 4.0 * inputs[1];
    outputs[1] = inputs[0] * inputs[1];
    outputs
}

#[test]
fn jacobian_evaluation_non_linear() {
    let problem_size = 2;
    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, non_linear);
    let inputs = nalgebra::DVector::from_vec(vec![1.0, 2.0]);
    user_model.set_iteratives(&inputs);
    user_model.evaluate();

    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_residuals = stopping_residuals.clone();
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_residuals);
    let perturbations = nalgebra::DVector::from_vec(vec![0.0001; problem_size]);
    let jac = jacobian_evaluation(&mut user_model, &perturbations, &res_config);

    assert_eq!(jac[(0, 0)], 0.9999999999976694);
    assert_eq!(jac[(0, 1)], 4.000000000008441);
    assert_eq!(jac[(1, 0)], 1.9999999999997797);
    assert_eq!(jac[(1, 1)], 1.0000000000021103);
}
