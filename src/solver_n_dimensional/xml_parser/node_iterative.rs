use crate::iteratives;
use minidom::Element;

pub fn parse_perturbation_method(
    node: &Element,
    node_info: &str,
) -> iteratives::PerturbationMethod {
    match node
            .attr("perturbation_method")
            .unwrap_or_else(|| panic!("The attribute \"perturbation_method\" is missing in {}", node_info)) {
                "Max" => iteratives::PerturbationMethod::Max,
                "Sum" => iteratives::PerturbationMethod::Sum,
                _     => panic!("The attribute \"perturbation_method\" at the {} has an improper values, valid values are \"Sum\" and \"Max\"", node_info),
            }
}

pub fn parse_perturbation_method_with_default(
    node: &Element,
    default: iteratives::PerturbationMethod,
    node_info: &str,
) -> iteratives::PerturbationMethod {
    match node
            .attr("perturbation_method") {
                None => default,
                Some(value) => match value {
                    "Max" => iteratives::PerturbationMethod::Max,
                    "Sum" => iteratives::PerturbationMethod::Sum,
                    _     => panic!("The attribute \"perturbation_method\" at the {} has an improper values, valid values are \"Sum\" and \"Max\"", node_info),
                },
            }
}
