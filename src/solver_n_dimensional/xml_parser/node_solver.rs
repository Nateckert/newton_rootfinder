use minidom::Element;

use super::util;
use crate::solver::SolverParameters;
use crate::solver::{QuasiNewtonMethod, ResolutionMethod, UpdateQuasiNewtonMethod};

/// Parse a solver node
pub fn parse_solver_node(solver_node: &Element) -> SolverParameters {
    let node_info = "solver node";
    let problem_size = util::parse_int_attribute(solver_node, &"problem_size", &node_info);
    let max_iter = util::parse_int_attribute(solver_node, &"max_iter", &node_info);
    let tolerance = util::parse_float_attribute(solver_node, &"tolerance", &node_info);
    let resolution_method = parse_resolution_method(solver_node, &node_info);

    let damping: bool = match solver_node.attr(&"damping") {
        Some(value) => value.parse().expect("The attribute \"damping\" is not a valid boolean, valid values are \"true\" and \"false\" (case sensitive)"),
        None => false,
    };

    SolverParameters::new(
        problem_size,
        tolerance,
        max_iter,
        resolution_method,
        damping,
    )
}

fn parse_resolution_method(node: &Element, node_info: &str) -> ResolutionMethod {
    match node
            .attr(&"resolution_method")
            .unwrap_or_else(|| panic!("The attribute \"resolution_method\" is missing in {}", node_info)) {
                "NR" => ResolutionMethod::NewtonRaphson,
                "SN" => ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton),
                "BROY1" => ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(UpdateQuasiNewtonMethod::BroydenFirstMethod)),
                "BROY1_INV" => ResolutionMethod::QuasiNewton(QuasiNewtonMethod::InverseJacobianUpdate(UpdateQuasiNewtonMethod::BroydenFirstMethod)),
                "BROY2" => ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(UpdateQuasiNewtonMethod::BroydenSecondMethod)),
                "BROY2_INV" => ResolutionMethod::QuasiNewton(QuasiNewtonMethod::InverseJacobianUpdate(UpdateQuasiNewtonMethod::BroydenSecondMethod)),
                "GRST1" => ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(UpdateQuasiNewtonMethod::GreenstadtFirstMethod)),
                "GRST1_INV" => ResolutionMethod::QuasiNewton(QuasiNewtonMethod::InverseJacobianUpdate(UpdateQuasiNewtonMethod::GreenstadtFirstMethod)),
                "GRST2" => ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(UpdateQuasiNewtonMethod::GreenstadtSecondMethod)),
                "GRST2_INV" => ResolutionMethod::QuasiNewton(QuasiNewtonMethod::InverseJacobianUpdate(UpdateQuasiNewtonMethod::GreenstadtSecondMethod)),
                _     => panic!("The attribute \"resolution_method\" at the {} has an improper values, valid values are \"NR\", \"SN\", \"BROY1\", \"BROY1_INV\", \"BROY2\", \"BROY2_INV\", \"GRST1\", \"GRST1_INV\", \"GRST2\", \"GRST2_INV\"", node_info),
            }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ResolutionMethod;
    use minidom::Element;

    #[test]
    fn parsing_solver_node_1() {
        const DATA: &'static str = r#"<solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="NR"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let solver_parameters = parse_solver_node(&solver_node);
        assert_eq!(solver_parameters.get_problem_size(), 3);
        assert_eq!(solver_parameters.get_max_iter(), 60);
        assert_eq!(
            solver_parameters.get_resolution_method(),
            ResolutionMethod::NewtonRaphson
        );
        assert_eq!(solver_parameters.get_tolerance(), 1e-6);
        assert_eq!(solver_parameters.get_damping(), true);
    }

    #[test]
    fn parsing_solver_node_resolution_method_1() {
        const DATA: &'static str = r#"<solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="SN"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let solver_parameters = parse_solver_node(&solver_node);
        assert_eq!(solver_parameters.get_problem_size(), 3);
        assert_eq!(solver_parameters.get_max_iter(), 60);
        assert_eq!(
            solver_parameters.get_resolution_method(),
            ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton)
        );
        assert_eq!(solver_parameters.get_tolerance(), 1e-6);
        assert_eq!(solver_parameters.get_damping(), true);
    }

    #[test]
    fn parsing_solver_node_resolution_method_2() {
        const DATA: &'static str = r#"<solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="BROY1"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let solver_parameters = parse_solver_node(&solver_node);
        assert_eq!(solver_parameters.get_problem_size(), 3);
        assert_eq!(solver_parameters.get_max_iter(), 60);
        assert_eq!(
            solver_parameters.get_resolution_method(),
            ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
                UpdateQuasiNewtonMethod::BroydenFirstMethod
            ))
        );
        assert_eq!(solver_parameters.get_tolerance(), 1e-6);
        assert_eq!(solver_parameters.get_damping(), true);
    }

    #[test]
    fn parsing_solver_node_resolution_method_3() {
        const DATA: &'static str = r#"<solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="BROY2"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let solver_parameters = parse_solver_node(&solver_node);
        assert_eq!(solver_parameters.get_problem_size(), 3);
        assert_eq!(solver_parameters.get_max_iter(), 60);
        assert_eq!(
            solver_parameters.get_resolution_method(),
            ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
                UpdateQuasiNewtonMethod::BroydenSecondMethod
            ))
        );
        assert_eq!(solver_parameters.get_tolerance(), 1e-6);
        assert_eq!(solver_parameters.get_damping(), true);
    }

    #[test]
    fn parsing_solver_node_resolution_method_4() {
        const DATA: &'static str = r#"<solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="BROY1_INV"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let solver_parameters = parse_solver_node(&solver_node);
        assert_eq!(solver_parameters.get_problem_size(), 3);
        assert_eq!(solver_parameters.get_max_iter(), 60);
        assert_eq!(
            solver_parameters.get_resolution_method(),
            ResolutionMethod::QuasiNewton(QuasiNewtonMethod::InverseJacobianUpdate(
                UpdateQuasiNewtonMethod::BroydenFirstMethod
            ))
        );
        assert_eq!(solver_parameters.get_tolerance(), 1e-6);
        assert_eq!(solver_parameters.get_damping(), true);
    }

    #[test]
    fn parsing_solver_node_resolution_method_5() {
        const DATA: &'static str = r#"<solver problem_size="3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="BROY2_INV"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let solver_parameters = parse_solver_node(&solver_node);
        assert_eq!(solver_parameters.get_problem_size(), 3);
        assert_eq!(solver_parameters.get_max_iter(), 60);
        assert_eq!(
            solver_parameters.get_resolution_method(),
            ResolutionMethod::QuasiNewton(QuasiNewtonMethod::InverseJacobianUpdate(
                UpdateQuasiNewtonMethod::BroydenSecondMethod
            ))
        );
        assert_eq!(solver_parameters.get_tolerance(), 1e-6);
        assert_eq!(solver_parameters.get_damping(), true);
    }

    #[test]
    #[should_panic(expected = "The attribute \"problem_size\" is missing in the solver node")]
    fn parsing_solver_node_2() {
        const DATA: &'static str = r#"<solver problem_Size="3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="NR"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let _solver_parameters = parse_solver_node(&solver_node);
    }
    #[test]
    #[should_panic(expected = "The attribute \"problem_size\" is not a valid positive integer")]
    fn parsing_solver_node_3() {
        const DATA: &'static str = r#"<solver problem_size="3.0" max_iter="60" tolerance="1e-6" damping="true" resolution_method="NR"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let _solver_parameters = parse_solver_node(&solver_node);
    }
    #[test]
    #[should_panic(expected = "The attribute \"problem_size\" is not a valid positive integer")]
    fn parsing_solver_node_4() {
        const DATA: &'static str = r#"<solver problem_size="-3" max_iter="60" tolerance="1e-6" damping="true" resolution_method="NR"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let _solver_parameters = parse_solver_node(&solver_node);
    }
    #[test]
    fn parsing_solver_node_5() {
        const DATA: &'static str =
            r#"<solver problem_size="3" max_iter="60" tolerance="1e-6" resolution_method="SN"/>"#;
        let solver_node: Element = DATA.parse().unwrap();
        let solver_parameters = parse_solver_node(&solver_node);
        assert_eq!(solver_parameters.get_problem_size(), 3);
        assert_eq!(solver_parameters.get_max_iter(), 60);
        assert_eq!(solver_parameters.get_tolerance(), 1e-6);
        assert_eq!(
            solver_parameters.get_resolution_method(),
            ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton)
        );
        assert_eq!(solver_parameters.get_damping(), false);
    }
}
