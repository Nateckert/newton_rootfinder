//! Parse a xml configuration file to create a RootFinder
//! ```xml
//! <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
//! <nrf>
//!     <solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true"/>
//!     <iteratives dx_abs="5e-8" dx_rel="5e-8" min_value="-inf"  max_value="inf" max_step_abs="inf" max_step_rel="inf">
//!         <iterative id="0" min_value="-inf"  max_value="inf" max_step_abs="100" max_step_rel="0.5"/>
//!         <iterative id="1" min_value="0"     max_value="inf" max_step_abs="inf" max_step_rel="0.5"/>
//!         <iterative id="2" min_value="-inf"  max_value="12"  max_step_abs="100" max_step_rel="inf"/>
//!     </iteratives>
//!     <residuals stopping_criteria="Abs" update_method="Abs">
//!         <residual id="0" stopping_criteria="Adapt"     update_method="Abs"/>
//!         <residual id="1" stopping_criteria="Rel"       update_method="Abs"/>
//!         <residual id="2" stopping_criteria="Adapt"     update_method="Rel"/>
//!     </residuals>
//! </nrf>
//! ```
//! The values provided in the iteratives and residuals nodes will act as default values
//! These values are taken into account only if non are provided for a given iterative or residual

use std::fs;
extern crate minidom;

use minidom::Element;

use crate::solver_advanced::iteratives;
use crate::solver_advanced::residuals;
use crate::solver_advanced::solver::SolverParameters;

/// Parser for a solver operating with a model with the jacobian provided
pub fn from_xml_jacobian(
    filepath: &str,
) -> (
    SolverParameters,
    Vec<iteratives::IterativeParams>,
    Vec<residuals::NormalizationMethod>,
    Vec<residuals::NormalizationMethod>,
) {
    let content = fs::read_to_string(filepath).unwrap();
    parse_root_node(&content)
}

/// Parser for a solver operating with a model with the jacobian not provided
///
/// The use of finite difference requires additional parameters for the iteratives variables
pub fn from_xml_finite_diff(
    filepath: &str,
) -> (
    SolverParameters,
    Vec<iteratives::IterativeParamsFD>,
    Vec<residuals::NormalizationMethod>,
    Vec<residuals::NormalizationMethod>,
) {
    let content = fs::read_to_string(filepath).unwrap();
    parse_root_node_fd(&content)
}

fn parse_root_node(
    content: &str,
) -> (
    SolverParameters,
    Vec<iteratives::IterativeParams>,
    Vec<residuals::NormalizationMethod>,
    Vec<residuals::NormalizationMethod>,
) {
    let root: Element = content.parse().unwrap();
    if root.name() != "nrf" {
        panic!("Expected the first node to be \"nrf\", got {}", root.name());
    }

    let mut tree = root.children();

    let solver_node = tree.next().unwrap();
    check_node_name_and_panic(solver_node, &"solver");
    let parameters = parse_solver_node(solver_node);

    let iteratives_node = tree.next().unwrap();
    check_node_name_and_panic(iteratives_node, &"iteratives");
    let iteratives = parse_iteratives_node(iteratives_node);

    let residuals_node = tree.next().unwrap();
    check_node_name_and_panic(residuals_node, &"residuals");
    let (stopping_criterias, update_methods) = parse_residuals_node(residuals_node);

    if parameters.get_problem_size() != iteratives.len() {
        panic!("Dimension mismatch, got problem_size = {} and the number of iteratives variables is {}", parameters.get_problem_size(), iteratives.len());
    }

    if parameters.get_problem_size() != stopping_criterias.len() {
        panic!(
            "Dimension mismatch, got problem_size = {} and the number of residuals variables is {}",
            parameters.get_problem_size(),
            stopping_criterias.len()
        );
    }

    (parameters, iteratives, stopping_criterias, update_methods)
}

fn parse_root_node_fd(
    content: &str,
) -> (
    SolverParameters,
    Vec<iteratives::IterativeParamsFD>,
    Vec<residuals::NormalizationMethod>,
    Vec<residuals::NormalizationMethod>,
) {
    let root: Element = content.parse().unwrap();
    if root.name() != "nrf" {
        panic!("Expected the first node to be \"nrf\", got {}", root.name());
    }

    let mut tree = root.children();

    let solver_node = tree.next().unwrap();
    check_node_name_and_panic(solver_node, &"solver");
    let parameters = parse_solver_node(solver_node);

    let iteratives_node = tree.next().unwrap();
    check_node_name_and_panic(iteratives_node, &"iteratives");
    let iteratives = parse_iteratives_fd_node(iteratives_node);

    let residuals_node = tree.next().unwrap();
    check_node_name_and_panic(residuals_node, &"residuals");
    let (stopping_criterias, update_methods) = parse_residuals_node(residuals_node);

    if parameters.get_problem_size() != iteratives.len() {
        panic!("Dimension mismatch, got problem_size = {} and the number of iteratives variables is {}", parameters.get_problem_size(), iteratives.len());
    }

    if parameters.get_problem_size() != stopping_criterias.len() {
        panic!(
            "Dimension mismatch, got problem_size = {} and the number of residuals variables is {}",
            parameters.get_problem_size(),
            stopping_criterias.len()
        );
    }

    (parameters, iteratives, stopping_criterias, update_methods)
}

/// Parse a solver node
fn parse_solver_node(solver_node: &Element) -> SolverParameters {
    let node_info = "solver node";
    let problem_size = parse_int_attribute(solver_node, &"problem_size", &node_info);
    let max_iter = parse_int_attribute(solver_node, &"max_iter", &node_info);
    let tolerance = parse_float_attribute(solver_node, &"tolerance", &node_info);

    let damping: bool = match solver_node.attr(&"damping") {
        Some(value) => value.parse().expect("The attribute \"damping\" is not a valid boolean, valid values are \"true\" and \"false\" (case sensitive)"),
        None => false,
    };

    SolverParameters::new(problem_size, tolerance, max_iter, damping)
}

fn parse_iteratives_node(iteratives_node: &Element) -> Vec<iteratives::IterativeParams> {
    let mut iteratives = Vec::new();

    let iterative_default = parse_iterative_node(&iteratives_node, &"iteratives node");

    for (expected_id, iterative_node) in iteratives_node.children().enumerate() {
        if iterative_node.name() != "iterative" {
            panic!(
                "Node below iteratives are expected to be \"iterative\", got {}",
                iterative_node.name()
            );
        }
        let id = parse_id(iterative_node, expected_id, &"iterative node");
        let node_info = format!("iterative node id = {}", id);
        let iterative =
            parse_iterative_node_with_default(&iterative_node, &iterative_default, &node_info);

        iteratives.push(iterative);
    }

    iteratives
}

fn parse_iteratives_fd_node(iteratives_node: &Element) -> Vec<iteratives::IterativeParamsFD> {
    let mut iteratives = Vec::new();

    let iterative_fd_default = parse_iterative_fd_node(&iteratives_node, &"iteratives node");

    for (expected_id, iterative_node) in iteratives_node.children().enumerate() {
        if iterative_node.name() != "iterative" {
            panic!(
                "Node below iteratives are expected to be \"iterative\", got {}",
                iterative_node.name()
            );
        }
        let id = parse_id(iterative_node, expected_id, &"iterative node");
        let node_info = format!("iterative node id = {}", id);
        let iterative = parse_iterative_fd_node_with_default(
            &iterative_node,
            &iterative_fd_default,
            &node_info,
        );

        iteratives.push(iterative);
    }

    iteratives
}

fn parse_iterative_node(iterative_node: &Element, node_info: &str) -> iteratives::IterativeParams {
    let min_value = parse_float_attribute(iterative_node, &"min_value", &node_info);
    let max_value = parse_float_attribute(iterative_node, &"max_value", &node_info);
    let max_step_abs = parse_float_attribute(iterative_node, &"max_step_abs", &node_info);
    let max_step_rel = parse_float_attribute(iterative_node, &"max_step_rel", &node_info);

    iteratives::IterativeParams::new(max_step_abs, max_step_rel, min_value, max_value)
}

fn parse_iterative_node_with_default(
    iterative_node: &Element,
    iterative_default: &iteratives::IterativeParams,
    node_info: &str,
) -> iteratives::IterativeParams {
    let min_value = parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_min_value(),
        &"min_value",
        &node_info,
    );
    let max_value = parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_max_value(),
        &"max_value",
        &node_info,
    );
    let max_step_abs = parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_max_step_abs(),
        &"max_step_abs",
        &node_info,
    );
    let max_step_rel = parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_max_step_rel(),
        &"max_step_rel",
        &node_info,
    );

    iteratives::IterativeParams::new(max_step_abs, max_step_rel, min_value, max_value)
}

fn parse_iterative_fd_node(
    iterative_node: &Element,
    node_info: &str,
) -> iteratives::IterativeParamsFD {
    let iterative = parse_iterative_node(iterative_node, node_info);

    let dx_abs = parse_float_attribute(iterative_node, &"dx_abs", &node_info);
    let dx_rel = parse_float_attribute(iterative_node, &"dx_rel", &node_info);

    let perturbation_method = parse_perturbation_method(iterative_node, &node_info);

    iteratives::IterativeParamsFD::extend(iterative, dx_abs, dx_rel, perturbation_method)
}

fn parse_iterative_fd_node_with_default(
    iterative_node: &Element,
    iterative_default: &iteratives::IterativeParamsFD,
    node_info: &str,
) -> iteratives::IterativeParamsFD {
    let iterative = parse_iterative_node_with_default(
        iterative_node,
        &iterative_default.get_iterative_params(),
        node_info,
    );

    let dx_abs = parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_dx_abs(),
        &"dx_abs",
        &node_info,
    );
    let dx_rel = parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_dx_rel(),
        &"dx_rel",
        &node_info,
    );

    let perturbation_method = parse_perturbation_method_with_default(
        iterative_node,
        iterative_default.get_perturbation_method(),
        &node_info,
    );

    iteratives::IterativeParamsFD::extend(iterative, dx_abs, dx_rel, perturbation_method)
}

fn parse_residuals_node(
    residuals_node: &Element,
) -> (
    Vec<residuals::NormalizationMethod>,
    Vec<residuals::NormalizationMethod>,
) {
    //Parsing of default values
    let residuals_config_default = parse_residual_node(&residuals_node, &"residuals node");

    let mut residuals = Vec::new();

    for (expected_id, residual_node) in residuals_node.children().enumerate() {
        if residual_node.name() != "residual" {
            panic!(
                "Node below residuals are expected to be \"residuals\", got {}",
                residual_node.name()
            );
        }

        let id = parse_id(&residual_node, expected_id, &"residual_node");
        let node_info = format!("residual node id = {}", id);
        let residual =
            parse_residual_node_with_default(&residual_node, residuals_config_default, &node_info);

        residuals.push(residual);
    }

    let (stopping_criterias, update_methods) =
        residuals::ResidualsConfig::convert_into_vecs(residuals);
    (stopping_criterias, update_methods)
}

fn parse_residual_node(residual_node: &Element, node_info: &str) -> residuals::ResidualConfig {
    let stopping_critera =
        parse_normalization_method_attribute(residual_node, &"stopping_criteria", &node_info);
    let update_method =
        parse_normalization_method_attribute(residual_node, &"update_method", &node_info);

    residuals::ResidualConfig::new(stopping_critera, update_method)
}

fn parse_residual_node_with_default(
    residual_node: &Element,
    residuals_config_default: residuals::ResidualConfig,
    node_info: &str,
) -> residuals::ResidualConfig {
    let stopping_critera = parse_normalization_method_attribute_with_default(
        residual_node,
        residuals_config_default.get_stopping_critera(),
        &"stopping_criteria",
        &node_info,
    );
    let update_method = parse_normalization_method_attribute_with_default(
        residual_node,
        residuals_config_default.get_update_method(),
        &"update_method",
        &node_info,
    );

    residuals::ResidualConfig::new(stopping_critera, update_method)
}

fn parse_perturbation_method(node: &Element, node_info: &str) -> iteratives::PerturbationMethod {
    match node
            .attr(&"perturbation_method")
            .unwrap_or_else(|| panic!("The attribute \"perturbation_method\" is missing in {}", node_info)) {
                "Max" => iteratives::PerturbationMethod::Max,
                "Sum" => iteratives::PerturbationMethod::Sum,
                _     => panic!("The attribute \"perturbation_method\" at the {} has an improper values, valid values are \"Sum\" and \"Max\"", node_info),
            }
}

fn parse_perturbation_method_with_default(
    node: &Element,
    default: iteratives::PerturbationMethod,
    node_info: &str,
) -> iteratives::PerturbationMethod {
    match node
            .attr(&"perturbation_method") {
                None => default,
                Some(value) => match value {
                    "Max" => iteratives::PerturbationMethod::Max,
                    "Sum" => iteratives::PerturbationMethod::Sum,
                    _     => panic!("The attribute \"perturbation_method\" at the {} has an improper values, valid values are \"Sum\" and \"Max\"", node_info),
                },
            }
}

fn parse_normalization_method_attribute(
    node: &Element,
    attribute: &str,
    node_info: &str,
) -> residuals::NormalizationMethod {
    match node
            .attr(attribute)
            .unwrap_or_else(|| panic!("The attribute \"{}\" is missing in {}", attribute, node_info)) {
                "Abs"   => residuals::NormalizationMethod::Abs,
                "Rel"   => residuals::NormalizationMethod::Rel,
                "Adapt" => residuals::NormalizationMethod::Adapt,
                _       => panic!("The attribute \"{}\" at {} has an improper values, valid values are \"Abs\", \"Rel\" and \"Adapt\"", attribute, node_info),
            }
}

fn parse_normalization_method_attribute_with_default(
    node: &Element,
    default: residuals::NormalizationMethod,
    attribute: &str,
    node_info: &str,
) -> residuals::NormalizationMethod {
    match node
            .attr(attribute) {
                None => default,
                Some(value) => match value {
                                    "Abs"   => residuals::NormalizationMethod::Abs,
                                    "Rel"   => residuals::NormalizationMethod::Rel,
                                    "Adapt" => residuals::NormalizationMethod::Adapt,
                                    _       => panic!("The attribute \"{}\" at {} has an improper values, valid values are \"Abs\", \"Rel\" and \"Adapt\"", attribute, node_info),
                }
            }
}

fn parse_id(node: &Element, expected_id: usize, node_info: &str) -> usize {
    let id = parse_int_attribute(node, &"id", node_info);
    if expected_id != id {
        panic!(
            "The ids must be in order starting from 0, got id {} when the expected one was {}",
            id, expected_id
        );
    }

    id
}

fn parse_int_attribute(node: &Element, attribute: &str, node_info: &str) -> usize {
    node.attr(attribute)
        .unwrap_or_else(|| panic!(
            "The attribute \"{}\" is missing in the {}",
            attribute, node_info
        ))
        .parse::<usize>()
        .unwrap_or_else(|_| panic!(
            "The attribute \"{}\" is not a valid positive integer",
            attribute
        ))
}

fn parse_float_attribute(node: &Element, attribute: &str, node_info: &str) -> f64 {
    node
        .attr(attribute)
        .unwrap_or_else(|| panic!("The attribute \"{}\" is missing in the {}", attribute, node_info))
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("The attribute \"{}\" is not a valid float, for infinity, the valid values are \"-inf\" and \"inf\" ", attribute))
}

fn parse_float_attribute_with_default(
    node: &Element,
    default: f64,
    attribute: &str,
    node_info: &str,
) -> f64 {
    match node
            .attr(attribute) {
                None => default,
                Some(value) => value
                            .parse::<f64>()
                            .unwrap_or_else(|_| panic!("The attribute \"{}\" on node {} is not a valid float, for infinity, the valid values are \"-inf\" and \"inf\" ", attribute, node_info))

            }
}

fn check_node_name_and_panic(node: &Element, expected_name: &str) {
    if node.name() != expected_name {
        panic!(
            "The node is expected to be \"{}\", got {}",
            expected_name,
            node.name()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate minidom;
    use crate::solver_advanced::iteratives;
    use minidom::Element;

    #[test]
    fn parsing_solver_node_1() {
        const DATA: &'static str =
            r#"<solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let solver_parameters = parse_solver_node(&solver_node);
        assert_eq!(solver_parameters.get_problem_size(), 3);
        assert_eq!(solver_parameters.get_max_iter(), 60);
        assert_eq!(solver_parameters.get_tolerance(), 1e-6);
        assert_eq!(solver_parameters.get_damping(), true);
    }
    #[test]
    #[should_panic(expected = "The attribute \"problem_size\" is missing in the solver node")]
    fn parsing_solver_node_2() {
        const DATA: &'static str =
            r#"<solver problem_Size="3" max_iter="60" tolerance="1e-6" damping="true"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let _solver_parameters = parse_solver_node(&solver_node);
    }
    #[test]
    #[should_panic(expected = "The attribute \"problem_size\" is not a valid positive integer")]
    fn parsing_solver_node_3() {
        const DATA: &'static str =
            r#"<solver problem_size="3.0" max_iter="60" tolerance="1e-6" damping="true"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let _solver_parameters = parse_solver_node(&solver_node);
    }
    #[test]
    #[should_panic(expected = "The attribute \"problem_size\" is not a valid positive integer")]
    fn parsing_solver_node_4() {
        const DATA: &'static str =
            r#"<solver problem_size="-3" max_iter="60" tolerance="1e-6" damping="true"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let _solver_parameters = parse_solver_node(&solver_node);
    }
    #[test]
    fn parsing_solver_node_5() {
        const DATA: &'static str = r#"<solver problem_size="3" max_iter="60" tolerance="1e-6"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let solver_parameters = parse_solver_node(&solver_node);
        assert_eq!(solver_parameters.get_problem_size(), 3);
        assert_eq!(solver_parameters.get_max_iter(), 60);
        assert_eq!(solver_parameters.get_tolerance(), 1e-6);
        assert_eq!(solver_parameters.get_damping(), false);
    }
    #[test]
    fn parsing_iterative_node_1() {
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" min_value="-inf", max_value="inf"/>"#;
        let iterative_node: Element = DATA.parse().unwrap();
        let node_info = "iterative node id = 0";
        let iterative = parse_iterative_node(&iterative_node, node_info);

        let iterative_ref =
            iteratives::IterativeParams::new(10.0, 0.4, f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(iterative, iterative_ref);
    }

    #[test]
    fn parsing_iterative_node_2() {
        let iterative_default = iteratives::IterativeParams::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" min_value="-inf", max_value="inf"/>"#;
        let iterative_node: Element = DATA.parse().unwrap();
        let node_info = "iterative node id = 0";
        let iterative =
            parse_iterative_node_with_default(&iterative_node, &iterative_default, &node_info);

        let iterative_ref =
            iteratives::IterativeParams::new(10.0, 0.4, f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(iterative, iterative_ref);
    }

    #[test]
    #[should_panic(
        expected = "The attribute \"min_value\" is missing in the iterative node id = 0"
    )]
    fn parsing_iterative_node_3() {
        const DATA: &'static str = r#"<iterative id="0"/>"#;
        let node_info = "iterative node id = 0";
        let iterative_node: Element = DATA.parse().unwrap();
        let iterative = parse_iterative_node(&iterative_node, &node_info);

        let iterative_ref =
            iteratives::IterativeParams::new(10.0, 0.5, f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(iterative, iterative_ref);
    }

    #[test]
    fn parsing_iterative_node_4() {
        let iterative_default =
            iteratives::IterativeParams::new(10.0, 0.5, f64::NEG_INFINITY, f64::INFINITY);
        const DATA: &'static str = r#"<iterative id="0"/>"#;
        let node_info = "iterative node id = 0";
        let iterative_node: Element = DATA.parse().unwrap();
        let iterative =
            parse_iterative_node_with_default(&iterative_node, &iterative_default, &node_info);

        let iterative_ref =
            iteratives::IterativeParams::new(10.0, 0.5, f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(iterative, iterative_ref);
    }

    #[test]
    #[should_panic(expected = "max_step_rel must be strictly positive, provided value was -0.4")]
    fn parsing_iterative_node_5() {
        let iterative_default = iteratives::IterativeParams::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="-0.4" min_value="-inf", max_value="inf"/>"#;
        let iterative_node: Element = DATA.parse().unwrap();
        let node_info = "iterative node id = 0";
        let _iterative =
            parse_iterative_node_with_default(&iterative_node, &iterative_default, &node_info);
    }

    #[test]
    fn parsing_iterative_fd_node_1() {
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" min_value="-inf" max_value="inf" dx_abs="0.1" dx_rel="0.2" perturbation_method="Max"/>"#;
        let node_info = "iterative node id = 0";
        let iterative_node: Element = DATA.parse().unwrap();
        let iterative = parse_iterative_fd_node(&iterative_node, &node_info);

        let iterative_ref = iteratives::IterativeParamsFD::new(
            10.0,
            0.4,
            f64::NEG_INFINITY,
            f64::INFINITY,
            0.1,
            0.2,
            iteratives::PerturbationMethod::Max,
        );
        assert_eq!(iterative, iterative_ref);
    }

    #[test]
    fn parsing_iterative_fd_node_2() {
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" min_value="-inf" max_value="inf" dx_abs="0.1" dx_rel="0.2" perturbation_method="Max"/>"#;
        let iterative_default = iteratives::IterativeParamsFD::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            5.0e-8,
            5.0e-8,
            iteratives::PerturbationMethod::Max,
        );
        let node_info = "iterative node id = 0";
        let iterative_node: Element = DATA.parse().unwrap();
        let iterative =
            parse_iterative_fd_node_with_default(&iterative_node, &iterative_default, &node_info);

        let iterative_ref = iteratives::IterativeParamsFD::new(
            10.0,
            0.4,
            f64::NEG_INFINITY,
            f64::INFINITY,
            0.1,
            0.2,
            iteratives::PerturbationMethod::Max,
        );
        assert_eq!(iterative, iterative_ref);
    }

    #[test]
    fn parsing_iterative_fd_node_3() {
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" min_value="-inf" max_value="inf" dx_abs="0.1" dx_rel="0.2" perturbation_method="Sum"/>"#;
        let node_info = "iterative node id = 0";
        let iterative_node: Element = DATA.parse().unwrap();
        let iterative = parse_iterative_fd_node(&iterative_node, &node_info);

        let iterative_ref = iteratives::IterativeParamsFD::new(
            10.0,
            0.4,
            f64::NEG_INFINITY,
            f64::INFINITY,
            0.1,
            0.2,
            iteratives::PerturbationMethod::Sum,
        );
        assert_eq!(iterative, iterative_ref);
    }

    #[test]
    fn parsing_iterative_fd_node_4() {
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" min_value="-inf" max_value="inf" dx_abs="0.1" dx_rel="0.2" perturbation_method="Sum"/>"#;
        let iterative_default = iteratives::IterativeParamsFD::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            5.0e-8,
            5.0e-8,
            iteratives::PerturbationMethod::Max,
        );
        let node_info = "iterative node id = 0";
        let iterative_node: Element = DATA.parse().unwrap();
        let iterative =
            parse_iterative_fd_node_with_default(&iterative_node, &iterative_default, &node_info);

        let iterative_ref = iteratives::IterativeParamsFD::new(
            10.0,
            0.4,
            f64::NEG_INFINITY,
            f64::INFINITY,
            0.1,
            0.2,
            iteratives::PerturbationMethod::Sum,
        );
        assert_eq!(iterative, iterative_ref);
    }

    #[test]
    fn parsing_iterative_fd_node_5() {
        const DATA: &'static str = r#"<iterative id="0"/>"#;
        let iterative_default = iteratives::IterativeParamsFD::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            5.0e-8,
            5.0e-8,
            iteratives::PerturbationMethod::Max,
        );
        let node_info = "iterative node id = 0";
        let iterative_node: Element = DATA.parse().unwrap();
        let iterative =
            parse_iterative_fd_node_with_default(&iterative_node, &iterative_default, &node_info);

        assert_eq!(iterative, iterative_default);
    }
    #[test]
    fn parsing_iterative_fd_node_6() {
        const DATA: &'static str =
            r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" perturbation_method="Sum"/>"#;
        let iterative_default = iteratives::IterativeParamsFD::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            5.0e-8,
            5.0e-8,
            iteratives::PerturbationMethod::Max,
        );
        let node_info = "iterative node id = 0";
        let iterative_node: Element = DATA.parse().unwrap();
        let iterative =
            parse_iterative_fd_node_with_default(&iterative_node, &iterative_default, &node_info);

        let iterative_ref = iteratives::IterativeParamsFD::new(
            10.0,
            0.4,
            f64::NEG_INFINITY,
            f64::INFINITY,
            5.0e-8,
            5.0e-8,
            iteratives::PerturbationMethod::Sum,
        );
        assert_eq!(iterative, iterative_ref);
    }

    #[test]
    #[should_panic(
        expected = "The attribute \"perturbation_method\" at the iterative node id = 0 has an improper values, valid values are \"Sum\" and \"Max\""
    )]
    fn parsing_iterative_fd_node_7() {
        let node_info = "iterative node id = 0";
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" min_value="-inf" max_value="inf" dx_abs="0.1" dx_rel="0.2" perturbation_method="max"/>"#;
        let iterative_node: Element = DATA.parse().unwrap();
        let _iterative = parse_iterative_fd_node(&iterative_node, &node_info);
    }

    #[test]
    #[should_panic(
        expected = "The attribute \"perturbation_method\" at the iterative node id = 0 has an improper values, valid values are \"Sum\" and \"Max\""
    )]
    fn parsing_iterative_fd_node_8() {
        let iterative_default = iteratives::IterativeParamsFD::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            5.0e-8,
            5.0e-8,
            iteratives::PerturbationMethod::Max,
        );
        let node_info = "iterative node id = 0";
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" min_value="-inf" max_value="inf" dx_abs="0.1" dx_rel="0.2" perturbation_method="max"/>"#;
        let iterative_node: Element = DATA.parse().unwrap();
        let _iterative =
            parse_iterative_fd_node_with_default(&iterative_node, &iterative_default, &node_info);
    }

    #[test]
    fn parsing_residual_node_1() {
        let node_info = "residual node id = 0";
        const DATA: &'static str =
            r#"<residual id="0" stopping_criteria="Adapt" update_method="Abs"/>"#;
        let residual_node: Element = DATA.parse().unwrap();
        let residual = parse_residual_node(&residual_node, &node_info);

        let residual_ref = residuals::ResidualConfig::new(
            residuals::NormalizationMethod::Adapt,
            residuals::NormalizationMethod::Abs,
        );
        assert_eq!(residual, residual_ref);
    }

    #[test]
    fn parsing_residual_node_2() {
        let residual_config_default = residuals::ResidualConfig::new(
            residuals::NormalizationMethod::Rel,
            residuals::NormalizationMethod::Rel,
        );
        let node_info = "residual node id = 0";
        const DATA: &'static str =
            r#"<residual id="0" stopping_criteria="Adapt" update_method="Abs"/>"#;
        let residual_node: Element = DATA.parse().unwrap();
        let residual =
            parse_residual_node_with_default(&residual_node, residual_config_default, &node_info);

        let residual_ref = residuals::ResidualConfig::new(
            residuals::NormalizationMethod::Adapt,
            residuals::NormalizationMethod::Abs,
        );
        assert_eq!(residual, residual_ref);
    }

    #[test]
    #[should_panic(
        expected = "The attribute \"stopping_criteria\" is missing in residual node id = 0"
    )]
    fn parsing_residual_node_3() {
        let node_info = "residual node id = 0";
        const DATA: &'static str = r#"<residual id="0"/>"#;
        let residual_node: Element = DATA.parse().unwrap();
        let residual = parse_residual_node(&residual_node, &node_info);

        let residual_ref = residuals::ResidualConfig::new(
            residuals::NormalizationMethod::Rel,
            residuals::NormalizationMethod::Rel,
        );
        assert_eq!(residual, residual_ref);
    }

    #[test]
    fn parsing_residual_node_4() {
        let residual_config_default = residuals::ResidualConfig::new(
            residuals::NormalizationMethod::Rel,
            residuals::NormalizationMethod::Rel,
        );
        let node_info = "residual node id = 0";
        const DATA: &'static str = r#"<residual id="0"/>"#;
        let residual_node: Element = DATA.parse().unwrap();
        let residual =
            parse_residual_node_with_default(&residual_node, residual_config_default, &node_info);

        let residual_ref = residuals::ResidualConfig::new(
            residuals::NormalizationMethod::Rel,
            residuals::NormalizationMethod::Rel,
        );
        assert_eq!(residual, residual_ref);
    }

    #[test]
    #[should_panic(
        expected = "The attribute \"stopping_criteria\" at residual node id = 0 has an improper values, valid values are \"Abs\", \"Rel\" and \"Adapt\""
    )]
    fn parsing_residual_node_5() {
        let node_info = "residual node id = 0";
        const DATA: &'static str =
            r#"<residual id="0" stopping_criteria="adapt" update_method="Abs"/>"#;
        let residual_node: Element = DATA.parse().unwrap();
        let _residual = parse_residual_node(&residual_node, &node_info);
    }

    #[test]
    #[should_panic(
        expected = "The attribute \"stopping_criteria\" at residual node id = 0 has an improper values, valid values are \"Abs\", \"Rel\" and \"Adapt\""
    )]
    fn parsing_residual_node_6() {
        let residual_config_default = residuals::ResidualConfig::new(
            residuals::NormalizationMethod::Rel,
            residuals::NormalizationMethod::Rel,
        );
        let node_info = "residual node id = 0";
        const DATA: &'static str =
            r#"<residual id="0" stopping_criteria="adapt" update_method="Abs"/>"#;
        let residual_node: Element = DATA.parse().unwrap();
        let _residual =
            parse_residual_node_with_default(&residual_node, residual_config_default, &node_info);
    }

    #[test]
    fn parsing_residuals_node_1() {
        const DATA: &'static str = r#"
            <residuals stopping_criteria="Adapt" update_method="Abs">
                <residual id="0"/>
                <residual id="1"/>
                <residual id="2"/>
            </residuals>"#;
        let residuals_node: Element = DATA.parse().unwrap();
        let (stopping_criterias, update_methods) = parse_residuals_node(&residuals_node);

        let stopping_ref = vec![residuals::NormalizationMethod::Adapt; 3];
        let update_ref = vec![residuals::NormalizationMethod::Abs; 3];

        assert_eq!(stopping_criterias, stopping_ref);
        assert_eq!(update_methods, update_ref);
    }

    #[test]
    fn parsing_residuals_node_2() {
        const DATA: &'static str = r#"
            <residuals stopping_criteria="Adapt" update_method="Abs">
                <residual id="0" stopping_criteria="Rel"/>
                <residual id="1"/>
                <residual id="2"/>
            </residuals>"#;
        let residuals_node: Element = DATA.parse().unwrap();
        let (stopping_criterias, update_methods) = parse_residuals_node(&residuals_node);

        let mut stopping_ref = vec![residuals::NormalizationMethod::Adapt; 3];
        stopping_ref[0] = residuals::NormalizationMethod::Rel;
        let update_ref = vec![residuals::NormalizationMethod::Abs; 3];

        assert_eq!(stopping_criterias, stopping_ref);
        assert_eq!(update_methods, update_ref);
    }

    #[test]
    #[should_panic(
        expected = "The ids must be in order starting from 0, got id 2 when the expected one was 1"
    )]
    fn parsing_residuals_node_3() {
        const DATA: &'static str = r#"
            <residuals stopping_criteria="Adapt" update_method="Abs">
                <residual id="0"/>
                <residual id="2"/>
                <residual id="1"/>
            </residuals>"#;
        let residuals_node: Element = DATA.parse().unwrap();
        let (_stopping_criterias, _update_methods) = parse_residuals_node(&residuals_node);
    }

    #[test]
    #[should_panic(
        expected = "The ids must be in order starting from 0, got id 1 when the expected one was 2"
    )]
    fn parsing_residuals_node_4() {
        const DATA: &'static str = r#"
            <residuals stopping_criteria="Adapt" update_method="Abs">
                <residual id="0"/>
                <residual id="1"/>
                <residual id="1"/>
            </residuals>"#;
        let residuals_node: Element = DATA.parse().unwrap();
        let (_stopping_criterias, _update_methods) = parse_residuals_node(&residuals_node);
    }

    #[test]
    #[should_panic(
        expected = "The ids must be in order starting from 0, got id 3 when the expected one was 2"
    )]
    fn parsing_residuals_node_5() {
        const DATA: &'static str = r#"
            <residuals stopping_criteria="Adapt" update_method="Abs">
                <residual id="0"/>
                <residual id="1"/>
                <residual id="3"/>
            </residuals>"#;
        let residuals_node: Element = DATA.parse().unwrap();
        let (_stopping_criterias, _update_methods) = parse_residuals_node(&residuals_node);
    }

    #[test]
    #[should_panic(expected = "The attribute \"id\" is not a valid positive integer")]
    fn parsing_residuals_node_6() {
        const DATA: &'static str = r#"
            <residuals stopping_criteria="Adapt" update_method="Abs">
                <residual id="-1"/>
                <residual id="0"/>
                <residual id="1"/>
            </residuals>"#;
        let residuals_node: Element = DATA.parse().unwrap();
        let (_stopping_criterias, _update_methods) = parse_residuals_node(&residuals_node);
    }

    #[test]
    fn parsing_iteratives_node_1() {
        const DATA: &'static str = r#"
            <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf">
                <iterative id="0"/>
                <iterative id="1"/>
                <iterative id="2"/>
            </iteratives>"#;
        let iteratives_node: Element = DATA.parse().unwrap();
        let iteratives = parse_iteratives_node(&iteratives_node);

        let iterative_ref = iteratives::IterativeParams::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );
        let iteratives_ref = vec![iterative_ref; 3];

        assert_eq!(iteratives, iteratives_ref);
    }

    #[test]
    fn parsing_iteratives_node_2() {
        const DATA: &'static str = r#"
            <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf">
                <iterative id="0"/>
                <iterative id="1" max_step_rel="0.5"/>
                <iterative id="2"/>
            </iteratives>"#;
        let iteratives_node: Element = DATA.parse().unwrap();
        let iteratives = parse_iteratives_node(&iteratives_node);

        let iterative_ref = iteratives::IterativeParams::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );
        let mut iteratives_ref = vec![iterative_ref; 3];
        iteratives_ref[1] =
            iteratives::IterativeParams::new(f64::INFINITY, 0.5, f64::NEG_INFINITY, f64::INFINITY);

        assert_eq!(iteratives, iteratives_ref);
    }

    #[test]
    #[should_panic(
        expected = "The ids must be in order starting from 0, got id 4 when the expected one was 2"
    )]
    fn parsing_iteratives_node_3() {
        const DATA: &'static str = r#"
            <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf">
                <iterative id="0"/>
                <iterative id="1" max_step_rel="0.5"/>
                <iterative id="4"/>
            </iteratives>"#;
        let iteratives_node: Element = DATA.parse().unwrap();
        let _iteratives = parse_iteratives_node(&iteratives_node);
    }

    #[test]
    fn parsing_iteratives_fd_node_1() {
        const DATA: &'static str = r#"
            <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf" dx_abs="5e-8" dx_rel="5e-8" perturbation_method="Max">
                <iterative id="0"/>
                <iterative id="1"/>
                <iterative id="2"/>
            </iteratives>"#;
        let iteratives_node: Element = DATA.parse().unwrap();
        let iteratives = parse_iteratives_fd_node(&iteratives_node);

        let iterative_ref = iteratives::IterativeParamsFD::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            5e-8,
            5e-8,
            iteratives::PerturbationMethod::Max,
        );
        let iteratives_ref = vec![iterative_ref; 3];

        assert_eq!(iteratives, iteratives_ref);
    }

    #[test]
    fn parsing_iteratives_fd_node_2() {
        const DATA: &'static str = r#"
            <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf" dx_abs="5e-8" dx_rel="5e-8" perturbation_method="Max">
                <iterative id="0" max_step_abs="10" max_step_rel="0.5" min_value="10" max_value="100" dx_abs="3e-8" dx_rel="8e-8" perturbation_method="Max"/>
                <iterative id="1" max_value="0" dx_abs="1.5e-8" dx_rel="2e-8" perturbation_method="Sum"/>
                <iterative id="2" max_value="inf" dx_abs="1.5e-8" dx_rel="2e-8" perturbation_method="Sum"/>
            </iteratives>"#;
        let iteratives_node: Element = DATA.parse().unwrap();
        let iteratives = parse_iteratives_fd_node(&iteratives_node);

        let iterative1_ref = iteratives::IterativeParamsFD::new(
            10.0,
            0.5,
            10.0,
            100.0,
            3e-8,
            8e-8,
            iteratives::PerturbationMethod::Max,
        );

        let iterative2_ref = iteratives::IterativeParamsFD::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            1.5e-8,
            2e-8,
            iteratives::PerturbationMethod::Sum,
        );

        let iterative3_ref = iteratives::IterativeParamsFD::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            1.5e-8,
            2e-8,
            iteratives::PerturbationMethod::Sum,
        );

        let iteratives_ref = vec![iterative1_ref, iterative2_ref, iterative3_ref];

        assert_eq!(iteratives, iteratives_ref);
    }

    #[test]
    #[should_panic(expected = "The attribute \"dx_abs\" is missing in the iteratives node")]
    fn parsing_iteratives_with_iteratives_fd() {
        const DATA: &'static str = r#"
            <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf">
                <iterative id="0"/>
                <iterative id="1"/>
                <iterative id="2"/>
            </iteratives>"#;
        let iteratives_node: Element = DATA.parse().unwrap();
        let _iteratives = parse_iteratives_fd_node(&iteratives_node);
    }

    #[test]
    fn parsing_iteratives_fd_with_iteratives() {
        const DATA: &'static str = r#"
            <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf" dx_abs="5e-8" dx_rel="5e-8" perturbation_method="Max">
                <iterative id="0"/>
                <iterative id="1"/>
                <iterative id="2"/>
            </iteratives>"#;
        let iteratives_node: Element = DATA.parse().unwrap();
        let _iteratives = parse_iteratives_node(&iteratives_node);
    }

    #[test]
    fn parsing_root_fd_1() {
        const DATA: &'static str = r#"
            <nrf>
                <solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true"/>
                <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf" dx_abs="5e-8" dx_rel="5e-9" perturbation_method="Max">
                    <iterative id="0"/>
                    <iterative id="1"/>
                    <iterative id="2"/>
                </iteratives>
                <residuals stopping_criteria="Abs" update_method="Abs">
                    <residual id="0" stopping_criteria="Adapt" update_method="Abs"/>
                    <residual id="1" stopping_criteria="Rel"   update_method="Abs"/>
                    <residual id="2" stopping_criteria="Adapt" update_method="Rel"/>
                </residuals>
            </nrf>"#;
        let (solver_parameters, iteratives_parsed, stopping_criterias, update_methods) =
            parse_root_node_fd(&DATA);

        assert_eq!(solver_parameters.get_problem_size(), 3);
        assert_eq!(solver_parameters.get_max_iter(), 60);
        assert_eq!(solver_parameters.get_tolerance(), 1e-6);
        assert_eq!(solver_parameters.get_damping(), true);

        let iterative_ref = iteratives::IterativeParamsFD::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            5e-8,
            5e-9,
            iteratives::PerturbationMethod::Max,
        );

        let iteratives_ref = vec![iterative_ref; 3];
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

    #[test]
    #[should_panic(
        expected = "Dimension mismatch, got problem_size = 4 and the number of iteratives variables is 3"
    )]
    fn parsing_root_fd_2() {
        const DATA: &'static str = r#"
            <nrf>
                <solver problem_size="4" max_iter="60" tolerance="1e-6" damping="true"/>
                <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf" dx_abs="5e-8" dx_rel="5e-9" perturbation_method="Max">
                    <iterative id="0"/>
                    <iterative id="1"/>
                    <iterative id="2"/>
                </iteratives>
                <residuals stopping_criteria="Abs" update_method="Abs">
                    <residual id="0" stopping_criteria="Adapt" update_method="Abs"/>
                    <residual id="1" stopping_criteria="Rel"   update_method="Abs"/>
                    <residual id="2" stopping_criteria="Adapt" update_method="Rel"/>
                </residuals>
            </nrf>"#;
        let (_solver_parameters, _iteratives_parsed, _stopping_criterias, _update_methods) =
            parse_root_node_fd(&DATA);
    }

    #[test]
    fn parsing_root_1() {
        const DATA: &'static str = r#"
            <nrf>
                <solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true"/>
                <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf">
                    <iterative id="0"/>
                    <iterative id="1"/>
                    <iterative id="2"/>
                </iteratives>
                <residuals stopping_criteria="Abs" update_method="Abs">
                    <residual id="0" stopping_criteria="Adapt" update_method="Abs"/>
                    <residual id="1" stopping_criteria="Rel"   update_method="Abs"/>
                    <residual id="2" stopping_criteria="Adapt" update_method="Rel"/>
                </residuals>
            </nrf>"#;
        let (solver_parameters, iteratives_parsed, stopping_criterias, update_methods) =
            parse_root_node(&DATA);

        assert_eq!(solver_parameters.get_problem_size(), 3);
        assert_eq!(solver_parameters.get_max_iter(), 60);
        assert_eq!(solver_parameters.get_tolerance(), 1e-6);
        assert_eq!(solver_parameters.get_damping(), true);

        let iterative_ref = iteratives::IterativeParams::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );

        let iteratives_ref = vec![iterative_ref; 3];
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

    #[test]
    #[should_panic(
        expected = "Dimension mismatch, got problem_size = 4 and the number of iteratives variables is 3"
    )]
    fn parsing_root_2() {
        const DATA: &'static str = r#"
            <nrf>
                <solver problem_size="4" max_iter="60" tolerance="1e-6" damping="true"/>
                <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf">
                    <iterative id="0"/>
                    <iterative id="1"/>
                    <iterative id="2"/>
                </iteratives>
                <residuals stopping_criteria="Abs" update_method="Abs">
                    <residual id="0" stopping_criteria="Adapt" update_method="Abs"/>
                    <residual id="1" stopping_criteria="Rel"   update_method="Abs"/>
                    <residual id="2" stopping_criteria="Adapt" update_method="Rel"/>
                </residuals>
            </nrf>"#;
        let (_solver_parameters, _iteratives_parsed, _stopping_criterias, _update_methods) =
            parse_root_node(&DATA);
    }

    #[test]
    #[should_panic(expected = "The attribute \"dx_abs\" is missing in the iteratives node")]
    fn parsing_root_with_root_fd() {
        const DATA: &'static str = r#"
            <nrf>
                <solver problem_size="4" max_iter="60" tolerance="1e-6" damping="true"/>
                <iteratives max_step_abs="inf" max_step_rel="inf" min_value="-inf" max_value="inf">
                    <iterative id="0"/>
                    <iterative id="1"/>
                    <iterative id="2"/>
                </iteratives>
                <residuals stopping_criteria="Abs" update_method="Abs">
                    <residual id="0" stopping_criteria="Adapt" update_method="Abs"/>
                    <residual id="1" stopping_criteria="Rel"   update_method="Abs"/>
                    <residual id="2" stopping_criteria="Adapt" update_method="Rel"/>
                </residuals>
            </nrf>"#;
        let (_solver_parameters, _iteratives_parsed, _stopping_criterias, _update_methods) =
            parse_root_node_fd(&DATA);
    }
}
