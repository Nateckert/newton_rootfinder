use minidom::Element;
use std::fs;

use crate::iteratives;
use crate::residuals;
use crate::solver::SolverParameters;

use super::node_iterative_jac::parse_iteratives_jac_node;
use super::node_residual::parse_residuals_node;
use super::node_solver::parse_solver_node;
use super::util::check_node_name_and_panic;

/// Parser for a solver operating with a model with the jacobian provided
///
///
/// ## XML structure
///
/// ### Root
/// It is expected to be an `.xml` document with a root node called `nrf` (newton root finder)
///
/// Three child nodes are expected:
///```xml
/// <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
/// <nrf>
///     <solver>...</solver>
///     <iteratives>...</iteratives>
///     <residuals>...</residuals>
/// </nrf>
///```
///
/// ### Solver node
/// The \<solver\> node contains must contains the parameters of the [SolverParameters](crate::solver::SolverParameters) struct,
/// i.e :
/// - max_iter
/// - damping (true or false)
/// - tolerance
/// - problem_size
/// - resolution_method: (see [crate::solver::ResolutionMethod])
///     - "NR" for [Newton-Raphson](crate::solver::ResolutionMethod::NewtonRaphson)
///     - "SN" for [Stationary Newton](crate::solver::QuasiNewtonMethod::StationaryNewton)
///     - "BROY1" for [Broyden First Method](crate::solver::UpdateQuasiNewtonMethod::BroydenFirstMethod) approximating the [jacobian](crate::solver::QuasiNewtonMethod::JacobianUpdate)
///     - "BROY2" for [Broyden Second Method](crate::solver::UpdateQuasiNewtonMethod::BroydenSecondMethod) approximating the [jacobian](crate::solver::QuasiNewtonMethod::JacobianUpdate)
///     - "GRST1" for [Greenstadt First Method](crate::solver::UpdateQuasiNewtonMethod::GreenstadtFirstMethod) approximating the [jacobian](crate::solver::QuasiNewtonMethod::JacobianUpdate)
///     - "GRST2" for [Greenstadt Second Method](crate::solver::UpdateQuasiNewtonMethod::GreenstadtSecondMethod) approximating the [jacobian](crate::solver::QuasiNewtonMethod::JacobianUpdate)
///     - "BROY1_INV" for [Broyden First Method](crate::solver::UpdateQuasiNewtonMethod::BroydenFirstMethod) approximating the [inverse jacobian](crate::solver::QuasiNewtonMethod::InverseJacobianUpdate)
///     - "BROY2_INV" for [Broyden Second Method](crate::solver::UpdateQuasiNewtonMethod::BroydenSecondMethod) approximating the [inverse jacobian](crate::solver::QuasiNewtonMethod::InverseJacobianUpdate)
///     - "GRST1_INV" for [Greenstadt First Method](crate::solver::UpdateQuasiNewtonMethod::GreenstadtFirstMethod) approximating the [inverse jacobian](crate::solver::QuasiNewtonMethod::InverseJacobianUpdate)
///     - "GRST2_INV" for [Greenstadt Second Method](crate::solver::UpdateQuasiNewtonMethod::GreenstadtSecondMethod) approximating the [inverse jacobian](crate::solver::QuasiNewtonMethod::InverseJacobianUpdate)
///
///```xml
/// <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
/// <nrf>
///     <solver problem_size="2" max_iter="60" tolerance="1e-6" damping="true" resolution_method="NewtonRaphson"/>
///     <iteratives>...</iteratives>
///     <residuals>...</residuals>
/// </nrf>
///```
///
/// ### Iteratives node
/// The \<iteratives\> node contains all the default values for the parameters of the `IterativeParams` constructor:
/// - min_value
/// - max_value
/// - max_step_abs
/// - max_step_rel
///
/// Its childen will be the <iterative> node, each of them having an id starting at zero.
/// Each children will either take the default values if none are provided, or take any that are redefined for the given id.
///
///
///```xml
/// <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
/// <nrf>
///     <solver>...</solver>
///     <iteratives min_value="-inf" max_value="inf" max_step_abs="inf" max_step_rel="inf">
///         <iterative id="0">
///         <iterative id="1" max_step_abs="100">
///     </iteratives>
///     <residuals>...</residuals>
/// </nrf>
///```
///
/// The first one will take the default values, the second also except for max_step_abs that will be equal to 100.
///
/// ### Residuals node
///
/// The \<residuals\> node contains all the default values for the parameters of the `ResidualConfig` constructor:
/// - stopping_criteria
/// - update_method
///
/// Its childen will be the <residual> node, each of them having an id starting at zero.
/// Each children will either take the default values if none are provided, or take any that are redefined for the given id.
///
///
///```xml
/// <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
/// <nrf>
///     <solver>...</solver>
///     <iteratives>...</iteratives>
///     <residuals stopping_criteria="Adapt"  update_method="Abs">
///         <residual id="0">
///         <residual id="1" stopping_criteria="Abs">
///     </residuals>
/// </nrf>
///```
///
/// The first one will take the default values, the second also except for stopping_criteria that will be equal to Abs.
///
///
/// ## Trick
/// You can add any attribute that is not used by the parser,
/// for example if you want to name variables to recognize them:
///```xml
/// <iterative id="0" var_name="myVarName">
///```
///
/// ## Examples
///```no_run
/// use newton_rootfinder as nrf;
///
/// const FILEPATH: &'static str = "./my_path/my_configuration_file.xml";
/// let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
///    nrf::xml_parser::from_xml_finite_diff(&FILEPATH);
///
/// let iteratives = nrf::iteratives::Iteratives::new(&iteratives_vec);
/// let residuals_config =
///    nrf::residuals::ResidualsConfig::new(&stopping_criterias, &update_methods);
/// let problem_size = solver_parameters.get_problem_size();
///
/// let init = nalgebra::DVector::zeros(5);
///
/// let mut rf = nrf::solver::RootFinder::new(
///    solver_parameters,
///    init,
///    &iteratives,
///    &residuals_config,
/// );
///```
pub fn from_xml_jacobian(
    filepath: &str,
) -> (
    SolverParameters,
    Vec<iteratives::IterativeParams>,
    Vec<residuals::NormalizationMethod>,
    Vec<residuals::NormalizationMethod>,
) {
    let content = fs::read_to_string(filepath).unwrap();
    parse_root_node_jac(&content)
}

fn parse_root_node_jac(
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
    check_node_name_and_panic(solver_node, "solver");
    let parameters = parse_solver_node(solver_node);

    let iteratives_node = tree.next().unwrap();
    check_node_name_and_panic(iteratives_node, "iteratives");
    let iteratives = parse_iteratives_jac_node(iteratives_node);

    let residuals_node = tree.next().unwrap();
    check_node_name_and_panic(residuals_node, "residuals");
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

    use crate::iteratives;
    use crate::solver::ResolutionMethod;

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
            parse_root_node_jac(&DATA);

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
            parse_root_node_jac(&DATA);
    }
}
