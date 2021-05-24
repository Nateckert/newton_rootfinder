use minidom::Element;
use std::fs;

use crate::iteratives;
use crate::residuals;
use crate::solver::SolverParameters;

use super::node_iterative_fd::parse_iteratives_fd_node;
use super::node_residual::parse_residuals_node;
use super::node_solver::parse_solver_node;
use super::util::check_node_name_and_panic;

/// Parser for a solver operating with a model with the jacobian not provided
///
/// The use of finite difference requires additional parameters for the iteratives variables.
///
/// The three additional parameters are:
/// - `dx_abs`
/// - `dx_rel`
/// - `perturbation_method`
///
/// Otherwise, it works in exactly the same way as the `from_xml_jacobian` parser.
/// Refers to this doc for the general explanation. The differences are highlighted here
///
/// The \<iteratives\> node takes the 3 extra arguments as default values. This values can be overwritten in the same way
///
///```xml
/// <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
/// <nrf>
///     <solver>...</solver>
///     <iteratives min_value="-inf" max_value="inf" max_step_abs="inf" max_step_rel="inf" dx_abs="1.5e-6" dx_rel="5e-5" perturbation_method="Max">
///         <iterative id="0" perturbation_method="Sum">
///         <iterative id="1" max_step_abs="100">
///     </iteratives>
///     <residuals>...</residuals>
/// </nrf>
///

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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::solver::ResolutionMethod;

    #[test]
    fn parsing_root_fd_1() {
        const DATA: &'static str = r#"
            <nrf>
                <solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="NR"/>
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
        assert_eq!(
            solver_parameters.get_resolution_method(),
            ResolutionMethod::NewtonRaphson
        );
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
                <solver problem_size="4" max_iter="60" tolerance="1e-6" damping="true" resolution_method="SN"/>
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
    #[should_panic(expected = "The attribute \"resolution_method\" is missing in solver node")]
    fn parsing_root_fd_3() {
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
        let (_solver_parameters, _iteratives_parsed, _stopping_criterias, _update_methods) =
            parse_root_node_fd(&DATA);
    }

    #[test]
    #[should_panic(
        expected = "The attribute \"resolution_method\" at the solver node has an improper values, valid values are \"NR\", \"SN\", \"BROY1\", \"BROY1_INV\", \"BROY2\", \"BROY2_INV\", \"GRST1\", \"GRST1_INV\", \"GRST2\", \"GRST2_INV\""
    )]
    fn parsing_root_fd_4() {
        const DATA: &'static str = r#"
            <nrf>
                <solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="SR"/>
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
}
