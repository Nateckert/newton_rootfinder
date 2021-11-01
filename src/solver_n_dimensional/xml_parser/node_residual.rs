use super::util;
use crate::residuals;
use minidom::Element;

pub fn parse_residuals_node(
    residuals_node: &Element,
) -> (
    Vec<residuals::NormalizationMethod>,
    Vec<residuals::NormalizationMethod>,
) {
    //Parsing of default values
    let residuals_config_default = parse_residual_node(
        residuals_node,
        "residuals node"
    );

    let mut residuals = Vec::new();

    for (expected_id, residual_node) in residuals_node.children().enumerate() {
        if residual_node.name() != "residual" {
            panic!(
                "Node below residuals are expected to be \"residuals\", got {}",
                residual_node.name()
            );
        }

        let id = util::parse_id(residual_node, expected_id, "residual_node");
        let node_info = format!("residual node id = {}", id);
        let residual =
            parse_residual_node_with_default(
                residual_node,
                residuals_config_default,
                &node_info
            );

        residuals.push(residual);
    }

    let (stopping_criterias, update_methods) =
        residuals::ResidualsConfig::convert_into_vecs(residuals);
    (stopping_criterias, update_methods)
}

fn parse_residual_node(residual_node: &Element, node_info: &str) -> residuals::ResidualConfig {
    let stopping_critera =
        parse_normalization_method_attribute(residual_node, "stopping_criteria", node_info);
    let update_method =
        parse_normalization_method_attribute(residual_node, "update_method", node_info);

    residuals::ResidualConfig::new(stopping_critera, update_method)
}

fn parse_residual_node_with_default(
    residual_node: &Element,
    residuals_config_default: residuals::ResidualConfig,
    node_info: &str,
) -> residuals::ResidualConfig {
    let stopping_critera = parse_normalization_method_attribute_with_default(
        residual_node,
        residuals_config_default.get_stopping_criteria(),
        "stopping_criteria",
        node_info,
    );
    let update_method = parse_normalization_method_attribute_with_default(
        residual_node,
        residuals_config_default.get_update_method(),
        "update_method",
        node_info,
    );

    residuals::ResidualConfig::new(stopping_critera, update_method)
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

#[cfg(test)]
mod tests {
    use super::*;

    use minidom::Element;

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
}
