extern crate newton_rootfinder;
use newton_rootfinder::solver_advanced as nrf;
use nrf::test_cases::broyden1965::*;

use std::fs::File;
use std::io::{BufRead, BufReader};

#[test]
fn broyden_case10_fd() {
    const FILEPATH: &'static str = "./tests/log/broyden_case10.xml";
    const LOG_PATH: &'static str = "./tests/log/log.txt";
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
    rf.set_debug(true);

    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case10);

    rf.solve(&mut user_model);

    rf.write_log(&LOG_PATH);

    let log_ref = File::open(&"./tests/log/log_ref.txt").unwrap();
    let log_new = File::open(&LOG_PATH).unwrap();

    let log_new_reader = BufReader::new(log_new);
    let log_ref_reader = BufReader::new(log_ref);

    let mut lines_new = log_new_reader.lines();
    let mut lines_ref = log_ref_reader.lines();

    // Parse the runner informations except the time
    for _i in 0..9 {
        let line_new = lines_new.next().unwrap();
        let line_ref = lines_ref.next().unwrap();
        assert_eq!(line_new.unwrap(), line_ref.unwrap());
    }

    // ignore the time lines
    lines_new.next();
    lines_new.next();
    lines_ref.next();
    lines_ref.next();

    for (elt_new, elt_ref) in lines_new.zip(lines_ref) {
        assert_eq!(elt_new.unwrap(), elt_ref.unwrap());
    }

    //let mut log_contents = String::new();
    //let mut log_ref_contents = String::new();
    //log_reader.read_to_string(&mut log_contents).unwrap();
    //log_ref_reader
    //    .read_to_string(&mut log_ref_contents)
    //    .unwrap();
    //assert_eq!(log_contents, log_ref_contents);
}
