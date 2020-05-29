extern crate newton_rootfinder;
use newton_rootfinder::solver_advanced as nrf;
use nrf::test_cases::broyden1965::*;

use nrf::model::Model;

#[test]
fn broyden_case10_fd() {
    const FILEPATH: &'static str = "./tests/advanced_parametrization/broyden_case10.xml";

    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::util::from_xml_finite_diff(&FILEPATH);

    let iteratives = nrf::iteratives::Iteratives::new(&iteratives_vec);
    let residuals_config =
        nrf::residuals::ResidualsConfig::new(&stopping_criterias, &update_methods);
    let problem_size = solver_parameters.get_problem_size();

    let mut rf = nrf::solver::RootFinder::new(
        solver_parameters,
        init_broyden1965_case10(),
        &iteratives,
        &residuals_config,
    );

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
