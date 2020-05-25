extern crate newton_rootfinder;
use newton_rootfinder::solver_advanced as nrf;

use nrf::iteratives;
use nrf::residuals;
use nrf::util;

#[test]
fn parse_file() {
    //let path = env::current_dir().unwrap();
    //println!("The current directory is {}", path.display());
    const FILEPATH: &'static str = "./tests/parser/data_fd.xml";
    let (solver_parameters, iteratives_parsed, stopping_criterias, update_methods) =
        util::from_xml_finite_diff(&FILEPATH);

    assert_eq!(solver_parameters.get_problem_size(), 3);
    assert_eq!(solver_parameters.get_max_iter(), 60);
    assert_eq!(solver_parameters.get_tolerance(), 1e-6);
    assert_eq!(solver_parameters.get_damping(), true);

    let iterative1_ref = iteratives::IterativeParamsFD::new(
        100.0,
        0.5,
        f64::NEG_INFINITY,
        f64::INFINITY,
        5e-8,
        5e-9,
        iteratives::PerturbationMethod::Sum,
    );

    let iterative2_ref = iteratives::IterativeParamsFD::new(
        f64::INFINITY,
        0.5,
        0.0,
        f64::INFINITY,
        5e-8,
        5e-9,
        iteratives::PerturbationMethod::Sum,
    );

    let iterative3_ref = iteratives::IterativeParamsFD::new(
        100.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        12.0,
        5e-8,
        5e-9,
        iteratives::PerturbationMethod::Sum,
    );

    let iteratives_ref = vec![iterative1_ref, iterative2_ref, iterative3_ref];
    assert_eq!(iteratives_parsed, iteratives_ref);

    let stopping_criterias_ref = vec![
        residuals::NormalizationMethod::Adapt,
        residuals::NormalizationMethod::Rel,
        residuals::NormalizationMethod::Adapt,
    ];

    let update_methods_ref = vec![
        residuals::NormalizationMethod::Abs,
        residuals::NormalizationMethod::Abs,
        residuals::NormalizationMethod::Rel,
    ];

    assert_eq!(stopping_criterias, stopping_criterias_ref);
    assert_eq!(update_methods, update_methods_ref);
}
