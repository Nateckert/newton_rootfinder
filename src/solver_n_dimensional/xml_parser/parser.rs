use std::fs;

use minidom::Element;

use crate::iteratives;
use crate::residuals;
use crate::solver::SolverParameters;
use crate::solver::{QuasiNewtonMethod, ResolutionMethod, UpdateQuasiNewtonMethod};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver_advanced::iteratives;
    use crate::solver_advanced::solver::ResolutionMethod;
    use minidom::Element;

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
    fn parsing_root_1() {
        const DATA: &'static str = r#"
            <nrf>
                <solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="NR"/>
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
        assert_eq!(
            solver_parameters.get_resolution_method(),
            ResolutionMethod::NewtonRaphson
        );
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
                <solver problem_size="4" max_iter="60" tolerance="1e-6" damping="true" resolution_method="NR"/>
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
                <solver problem_size="4" max_iter="60" tolerance="1e-6" damping="true" resolution_method="SN"/>
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
