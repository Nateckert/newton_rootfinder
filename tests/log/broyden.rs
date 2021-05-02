extern crate newton_rootfinder;
use newton_rootfinder as nrf;
use nrf::test_cases::broyden1965::*;

use std::fs::File;
use std::io::{BufRead, BufReader};

#[test]
fn broyden_case10_fd() {
    const FILEPATH: &'static str = "./tests/log/broyden_case10.xml";
    const LOG_PATH: &'static str = "./tests/log/log.txt";
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(&FILEPATH);

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
    rf.activate_debug(&LOG_PATH);

    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case10);

    rf.solve(&mut user_model);

    let log_ref = File::open(&"./tests/log/log_ref.txt").unwrap();
    let log_new = File::open(&LOG_PATH).unwrap();

    let log_new_reader = BufReader::new(log_new);
    let log_ref_reader = BufReader::new(log_ref);

    let mut lines_new = log_new_reader.lines();
    let mut lines_ref = log_ref_reader.lines();

    // Parse the runner informations
    for _i in 0..3 {
        let line_new = lines_new.next().unwrap();
        let line_ref = lines_ref.next().unwrap();
        assert_eq!(line_new.unwrap(), line_ref.unwrap());
    }

    // ignore the OS line
    lines_new.next();
    lines_ref.next();
    // ignore the host line
    lines_new.next();
    lines_ref.next();
    // ignore the username line
    lines_new.next();
    lines_ref.next();
    // ignore the rustc version line
    lines_new.next();
    lines_ref.next();
    // ignore the crate version line
    lines_new.next();
    lines_ref.next();
    // ignore the time line
    lines_new.next();
    lines_ref.next();
    // ignore the UTC time line
    lines_new.next();
    lines_ref.next();
    // ignore the Local time line
    lines_new.next();
    lines_ref.next();

    for (elt_new, elt_ref) in lines_new.zip(lines_ref) {
        assert_eq!(elt_new.unwrap(), elt_ref.unwrap());
    }
}
