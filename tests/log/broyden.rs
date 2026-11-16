use newton_rootfinder as nrf;
use nrf::log::{Detail, Log};
use nrf::model::Model;
use util::test_cases::broyden1965::*;

use super::Buffer;

/// The log of a resolution, compared line by line with the reference.
/// `NRF_UPDATE_LOG_REFERENCE=1 cargo test --features xml_config_file` rewrites the reference, the diff to review
#[test]
fn broyden_case10_fd() {
    const FILEPATH: &str = "./tests/log/broyden_case10.xml";
    const REFERENCE: &str = "./tests/log/broyden_case10.jsonl";
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(FILEPATH);

    let iteratives = nrf::iteratives::Iteratives::new(&iteratives_vec);
    let residuals_config =
        nrf::residuals::ResidualsConfig::new(&stopping_criterias, &update_methods);
    let problem_size = iteratives_vec.len();

    let buffer = Buffer::default();
    let log = Log::new(buffer.clone()).detail(Detail::Jacobians);
    let mut rf = nrf::solver::RootFinder::new(solver_parameters, &iteratives, &residuals_config);
    rf.log_to(&log, "Broyden 1965, case 10");

    let mut user_model = nrf::model::UserModelFromFunction::new(problem_size, broyden1965_case10);
    user_model.set_iteratives(&init_broyden1965_case10());

    rf.solve(&mut user_model).unwrap();
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    if std::env::var_os("NRF_UPDATE_LOG_REFERENCE").is_some() {
        std::fs::write(REFERENCE, lines.join("\n") + "\n").unwrap();
    }
    let reference = std::fs::read_to_string(REFERENCE).unwrap();
    assert_eq!(lines, reference.lines().collect::<Vec<_>>());
}
