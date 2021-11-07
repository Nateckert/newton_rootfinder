use newton_rootfinder as nrf;

use nrf::iteratives;
use nrf::model::Model;
use nrf::residuals;

pub fn run_function_case_fd(
    problem_size: usize,
    func: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    init: nalgebra::DVector<f64>,
    solution: nalgebra::DVector<f64>,
    resolution_method: nrf::solver::ResolutionMethod,
    damping: bool,
) {
    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf = nrf::solver::default_with_guess(
        init,
        &iter_params,
        &res_config,
        resolution_method,
        damping,
    );
    let mut user_model = nrf::model::UserModelFromFunction::new(problem_size, func);

    rf.solve(&mut user_model).unwrap();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}

pub fn run_function_case_jac(
    problem_size: usize,
    func: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    jac: fn(&nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64>,
    init: nalgebra::DVector<f64>,
    solution: nalgebra::DVector<f64>,
    resolution_method: nrf::solver::ResolutionMethod,
    damping: bool,
) {
    let vec_iter_params = iteratives::default_vec_iteratives(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf = nrf::solver::default_with_guess(
        init,
        &iter_params,
        &res_config,
        resolution_method,
        damping,
    );
    let mut user_model = nrf::model::UserModelFromFunctionAndJacobian::new(problem_size, func, jac);

    rf.solve(&mut user_model).unwrap();

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }
}
