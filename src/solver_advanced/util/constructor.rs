use crate::solver_advanced::iteratives;
use crate::solver_advanced::residuals;
use crate::solver_advanced::solver::RootFinder;

use std::fs;
extern crate minidom;

use minidom::Element;

/// Parse a xml configuration file to create a RootFinder
/// <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
/// <nrf>
///     <solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true" jacobian_provided="false"/>
///     <iteratives dx_abs="5e-8" dx_rel="5e-8" min_value="-inf"  max_value="inf" max_step_abs="inf" max_step_rel="inf">
///         <iterative id="0" min_value="-inf"  max_value="inf" max_step_abs="100" max_step_rel="0.5"/>
///         <iterative id="1" min_value="0"     max_value="inf" max_step_abs="Inf" max_step_rel="0.5"/>
///         <iterative id="2" min_value="-inf"  max_value="12"  max_step_abs="100" max_step_rel="inf"/>
///     </iteratives>
///     <residuals stopping_criteria="Abs" update_method="Abs">
///         <residual id="0" stopping_criteria="Adapt"     update_method="Abs"/>
///         <residual id="1" stopping_criteria="Rel"       update_method="Abs"/>
///         <residual id="2" stopping_criteria="Adapt"     update_method="Rel"/>
///     </residuals>
/// </nrf>
///
/// If values are provided in the iteratives and residuals nodes, it will act as default values
/// These values are taken into account only if non are provided for a given iterative or residual
///
pub fn from_xml(filepath: &str) {
    // -> RootFinder {
    let content = fs::read_to_string(filepath).unwrap();
    parse_root_node(&content)
}

fn parse_root_node(content: &str) {
    //-> RootFinder {
    let root: Element = content.parse().unwrap();
    let mut tree = root.children();

    let solver_node = tree.next().unwrap();
    if solver_node.name() != "solver" {
        panic!(
            "The first node is expected to be \"solver\", got {}",
            solver_node.name()
        );
    }
}

/// Parse a solver node
fn parse_solver_node(solver_node: &Element) -> (usize, usize, f64, bool) {
    let node_info = "solver node";
    let problem_size = parse_int_attribute(solver_node, &"problem_size", &node_info);
    let max_iter = parse_int_attribute(solver_node, &"max_iter", &node_info);
    let tolerance = parse_float_attribute(solver_node, &"tolerance", &node_info);

    let damping: bool = match solver_node.attr(&"damping") {
        Some(value) => value.parse().expect("The attribute \"damping\" is not a valid boolean, valid values are \"true\" and \"false\" (case sensitive)"),
        None => false,
    };

    (problem_size, max_iter, tolerance, damping)
}

fn parse_iteratives_node(iteratives_node: &Element, jacobian_provided: bool) {

    if jacobian_provided {
        let iterative_default = parse_iterative_node(&iteratives_node, &"iteratives node");
    } else {
        let iterative_fd_default = parse_iterative_fd_node(&iteratives_node, &"iteratives node");
    }


    for (expected_id, iterative_node) in iteratives_node.children().enumerate() {
        if iterative_node.name() != "iterative" {
            panic!(
                "Node below iteratives are expected to be \"iterative\", got {}",
                iterative_node.name()
            );
        }
        let id = parse_id(iterative_node, expected_id, &"iterative node");
        let node_info = format!("iterative node id = {}", id);
        let iterative = if jacobian_provided {
            parse_iterative_node_with_default(&iterative_node, iterative_default, &node_info)
        } else {
            parse_iterative_fd_node_with_default(&iterative_node, iterative_fd_default, &node_info)
        };
    }
}

fn parse_iterative_node(
    iterative_node: &Element,
    node_info: &str,
) -> iteratives::IterativeParams {

    let min_value = parse_float_attribute(
        iterative_node,
        &"min_value",
        &node_info,
    );
    let max_value = parse_float_attribute(
        iterative_node,
        &"max_value",
        &node_info,
    );
    let max_step_abs = parse_float_attribute(
        iterative_node,
        &"max_step_abs",
        &node_info,
    );
    let max_step_rel = parse_float_attribute(
        iterative_node,
        &"max_step_rel",
        &node_info,
    );

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
    let iterative =
        parse_iterative_node(iterative_node, node_info);

    let dx_abs = parse_float_attribute(
        iterative_node,
        &"dx_abs",
        &node_info,
    );
    let dx_rel = parse_float_attribute(
        iterative_node,
        &"dx_rel",
        &node_info,
    );

    let perturbation_method = parse_perturbation_method(
        iterative_node,
        &node_info,
    );


    iteratives::IterativeParamsFD::extend(iterative, dx_abs, dx_rel, perturbation_method)
}


fn parse_iterative_fd_node_with_default(
    iterative_node: &Element,
    iterative_default: &iteratives::IterativeParamsFD,
    node_info: &str,
) -> iteratives::IterativeParamsFD {
    let iterative =
        parse_iterative_node_with_default(iterative_node, &iterative_default.get_iterative_params(), node_info);

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
        let residual = parse_residual_node_with_default(&residual_node, residuals_config_default, &node_info);

        residuals.push(residual);

    }

    let (stopping_criterias, update_methods) = residuals::ResidualsConfig::convert_into_vecs(residuals);
    (stopping_criterias, update_methods)
}

fn parse_residual_node(
    residual_node: &Element,
    node_info: &str,
) -> residuals::ResidualConfig {

    let stopping_critera = parse_normalization_method_attribute(
        residual_node,
        &"stopping_criteria",
        &node_info,
    );
    let update_method = parse_normalization_method_attribute(
        residual_node,
        &"update_method",
        &node_info,
    );

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
            .expect(&format!("The attribute \"perturbation_method\" is missing in {}", node_info)) {
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
            .expect(&format!("The attribute \"{}\" is missing in {}", attribute, node_info)) {
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
        .expect(&format!(
            "The attribute \"{}\" is missing in the {}",
            attribute, node_info
        ))
        .parse::<usize>()
        .expect(&format!(
            "The attribute \"{}\" is not a valid positive integer",
            attribute
        ))
}



fn parse_float_attribute(node: &Element, attribute: &str, node_info: &str) -> f64 {
    node
        .attr(attribute)
        .expect(&format!("The attribute \"{}\" is missing in the {}", attribute, node_info))
        .parse::<f64>()
        .expect(&format!("The attribute \"{}\" is not a valid float, for infinity, the valid values are \"-inf\" and \"inf\" ", attribute))
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
                            .expect(&format!("The attribute \"{}\" on node {} is not a valid float, for infinity, the valid values are \"-inf\" and \"inf\" ", attribute, node_info))

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
        let (problem_size, max_iter, tolerance, damping) = parse_solver_node(&solver_node);
        assert_eq!(problem_size, 3);
        assert_eq!(max_iter, 60);
        assert_eq!(tolerance, 1e-6);
        assert_eq!(damping, true);
    }
    #[test]
    #[should_panic(expected = "The attribute \"problem_size\" is missing in the solver node")]
    fn parsing_solver_node_2() {
        const DATA: &'static str =
            r#"<solver problem_Size="3" max_iter="60" tolerance="1e-6" damping="true"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let (problem_size, max_iter, tolerance, damping) = parse_solver_node(&solver_node);
    }
    #[test]
    #[should_panic(expected = "The attribute \"problem_size\" is not a valid positive integer")]
    fn parsing_solver_node_3() {
        const DATA: &'static str =
            r#"<solver problem_size="3.0" max_iter="60" tolerance="1e-6" damping="true"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let (problem_size, max_iter, tolerance, damping) = parse_solver_node(&solver_node);
    }
    #[test]
    #[should_panic(expected = "The attribute \"problem_size\" is not a valid positive integer")]
    fn parsing_solver_node_4() {
        const DATA: &'static str =
            r#"<solver problem_size="-3" max_iter="60" tolerance="1e-6" damping="true"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let (problem_size, max_iter, tolerance, damping) = parse_solver_node(&solver_node);
    }
    #[test]
    fn parsing_solver_node_5() {
        const DATA: &'static str = r#"<solver problem_size="3" max_iter="60" tolerance="1e-6"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let (problem_size, max_iter, tolerance, damping) = parse_solver_node(&solver_node);
        assert_eq!(problem_size, 3);
        assert_eq!(max_iter, 60);
        assert_eq!(tolerance, 1e-6);
        assert_eq!(damping, false);
    }
    #[test]
    fn parsing_iterative_node_1() {
        let iterative_default = iteratives::IterativeParams::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" min_value="-inf", max_value="inf"/>"#;
        let iterative_node: Element = DATA.parse().unwrap();
        let (id, iterative) = parse_iterative_node(&iterative_node, &iterative_default);
        assert_eq!(id, 0);

        let iterative_ref =
            iteratives::IterativeParams::new(10.0, 0.4, f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(iterative, iterative_ref);
    }

    #[test]
    fn parsing_iterative_node_2() {
        let iterative_default =
            iteratives::IterativeParams::new(10.0, 0.5, f64::NEG_INFINITY, f64::INFINITY);
        const DATA: &'static str = r#"<iterative id="0"/>"#;
        let iterative_node: Element = DATA.parse().unwrap();
        let (id, iterative) = parse_iterative_node(&iterative_node, &iterative_default);
        assert_eq!(id, 0);

        let iterative_ref =
            iteratives::IterativeParams::new(10.0, 0.5, f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(iterative, iterative_ref);
    }

    #[test]
    #[should_panic(expected = "max_step_rel must be strictly positive, provided value was -0.4")]
    fn parsing_iterative_node_3() {
        let iterative_default = iteratives::IterativeParams::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="-0.4" min_value="-inf", max_value="inf"/>"#;
        let iterative_node: Element = DATA.parse().unwrap();
        let (id, iterative) = parse_iterative_node(&iterative_node, &iterative_default);
    }

    #[test]
    fn parsing_iterative_fd_node_1() {
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
        let iterative_node: Element = DATA.parse().unwrap();
        let (id, iterative) = parse_iterative_fd_node(&iterative_node, &iterative_default);
        assert_eq!(id, 0);

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
        let iterative_node: Element = DATA.parse().unwrap();
        let (id, iterative) = parse_iterative_fd_node(&iterative_node, &iterative_default);
        assert_eq!(id, 0);

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
    fn parsing_iterative_fd_node_3() {
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
        let iterative_node: Element = DATA.parse().unwrap();
        let (id, iterative) = parse_iterative_fd_node(&iterative_node, &iterative_default);
        assert_eq!(id, 0);

        assert_eq!(iterative, iterative_default);
    }
    #[test]
    fn parsing_iterative_fd_node_4() {
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
        let iterative_node: Element = DATA.parse().unwrap();
        let (id, iterative) = parse_iterative_fd_node(&iterative_node, &iterative_default);
        assert_eq!(id, 0);

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
    fn parsing_iterative_fd_node_5() {
        let iterative_default = iteratives::IterativeParamsFD::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            5.0e-8,
            5.0e-8,
            iteratives::PerturbationMethod::Max,
        );
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" min_value="-inf" max_value="inf" dx_abs="0.1" dx_rel="0.2" perturbation_method="max"/>"#;
        let iterative_node: Element = DATA.parse().unwrap();
        let (id, iterative) = parse_iterative_fd_node(&iterative_node, &iterative_default);
    }

    #[test]
    fn parsing_residual_node_1() {
        let residual_config_default = residuals::ResidualConfig::new(
            residuals::NormalizationMethod::Rel,
            residuals::NormalizationMethod::Rel,
        );
        const DATA: &'static str =
            r#"<residual id="0" stopping_criteria="Adapt" update_method="Abs"/>"#;
        let residual_node: Element = DATA.parse().unwrap();
        let (id, residual) = parse_residual_node(&residual_node, residual_config_default);
        assert_eq!(id, 0);

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
        const DATA: &'static str = r#"<residual id="0"/>"#;
        let residual_node: Element = DATA.parse().unwrap();
        let (id, residual) = parse_residual_node(&residual_node, residual_config_default);
        assert_eq!(id, 0);

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
    fn parsing_residual_node_3() {
        let residual_config_default = residuals::ResidualConfig::new(
            residuals::NormalizationMethod::Rel,
            residuals::NormalizationMethod::Rel,
        );
        const DATA: &'static str =
            r#"<residual id="0" stopping_criteria="adapt" update_method="Abs"/>"#;
        let residual_node: Element = DATA.parse().unwrap();
        let (id, residual) = parse_residual_node(&residual_node, residual_config_default);
    }

    #[test]
    fn parsing_residuals_node_1() {
        const DATA: &'static str = r#"
            <residuals id="0" stopping_criteria="Adapt" update_method="Abs">
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
            <residuals id="0" stopping_criteria="Adapt" update_method="Abs">
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
            <residuals id="0" stopping_criteria="Adapt" update_method="Abs">
                <residual id="0"/>
                <residual id="2"/>
                <residual id="1"/>
            </residuals>"#;
        let residuals_node: Element = DATA.parse().unwrap();
        let (stopping_criterias, update_methods) = parse_residuals_node(&residuals_node);
    }

    #[test]
    #[should_panic(
        expected = "The ids must be in order starting from 0, got id 1 when the expected one was 2"
    )]
    fn parsing_residuals_node_4() {
        const DATA: &'static str = r#"
            <residuals id="0" stopping_criteria="Adapt" update_method="Abs">
                <residual id="0"/>
                <residual id="1"/>
                <residual id="1"/>
            </residuals>"#;
        let residuals_node: Element = DATA.parse().unwrap();
        let (stopping_criterias, update_methods) = parse_residuals_node(&residuals_node);
    }

    #[test]
    #[should_panic(
        expected = "The ids must be in order starting from 0, got id 3 when the expected one was 2"
    )]
    fn parsing_residuals_node_5() {
        const DATA: &'static str = r#"
            <residuals id="0" stopping_criteria="Adapt" update_method="Abs">
                <residual id="0"/>
                <residual id="1"/>
                <residual id="3"/>
            </residuals>"#;
        let residuals_node: Element = DATA.parse().unwrap();
        let (stopping_criterias, update_methods) = parse_residuals_node(&residuals_node);
    }

    #[test]
    #[should_panic(expected = "The attribute \"id\" is not a valid positive integer")]
    fn parsing_residuals_node_6() {
        const DATA: &'static str = r#"
            <residuals id="0" stopping_criteria="Adapt" update_method="Abs">
                <residual id="-1"/>
                <residual id="0"/>
                <residual id="1"/>
            </residuals>"#;
        let residuals_node: Element = DATA.parse().unwrap();
        let (stopping_criterias, update_methods) = parse_residuals_node(&residuals_node);
    }
}
