use crate::iteratives;
use minidom::Element;

pub fn parse_iteratives_jac_node(iteratives_node: &Element) -> Vec<iteratives::IterativeParams> {
    let mut iteratives = Vec::new();

    let iterative_default = parse_iterative_jac_node(iteratives_node, "iteratives node");

    for (expected_id, iterative_node) in iteratives_node.children().enumerate() {
        if iterative_node.name() != "iterative" {
            panic!(
                "Node below iteratives are expected to be \"iterative\", got {}",
                iterative_node.name()
            );
        }
        let id = super::util::parse_id(iterative_node, expected_id, "iterative node");
        let node_info = format!("iterative node id = {}", id);
        let iterative =
            parse_iterative_jac_node_with_default(iterative_node, &iterative_default, &node_info);

        iteratives.push(iterative);
    }

    iteratives
}

pub fn parse_iterative_jac_node(
    iterative_node: &Element,
    node_info: &str,
) -> iteratives::IterativeParams {
    let min_value = super::util::parse_float_attribute(iterative_node, "min_value", node_info);
    let max_value = super::util::parse_float_attribute(iterative_node, "max_value", node_info);
    let max_step_abs =
        super::util::parse_float_attribute(iterative_node, "max_step_abs", node_info);
    let max_step_rel =
        super::util::parse_float_attribute(iterative_node, "max_step_rel", node_info);

    iteratives::IterativeParams::new(max_step_abs, max_step_rel, min_value, max_value)
}

pub fn parse_iterative_jac_node_with_default(
    iterative_node: &Element,
    iterative_default: &iteratives::IterativeParams,
    node_info: &str,
) -> iteratives::IterativeParams {
    let min_value = super::util::parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_min_value(),
        "min_value",
        node_info,
    );
    let max_value = super::util::parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_max_value(),
        "max_value",
        node_info,
    );
    let max_step_abs = super::util::parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_max_step_abs(),
        "max_step_abs",
        node_info,
    );
    let max_step_rel = super::util::parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_max_step_rel(),
        "max_step_rel",
        node_info,
    );

    iteratives::IterativeParams::new(max_step_abs, max_step_rel, min_value, max_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::iteratives;
    use minidom::Element;

    #[test]
    fn parsing_iterative_node_1() {
        const DATA: &'static str = r#"<iterative id="0" max_step_abs="10" max_step_rel="0.4" min_value="-inf", max_value="inf"/>"#;
        let iterative_node: Element = DATA.parse().unwrap();
        let node_info = "iterative node id = 0";
        let iterative = parse_iterative_jac_node(&iterative_node, node_info);

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
            parse_iterative_jac_node_with_default(&iterative_node, &iterative_default, &node_info);

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
        let iterative = parse_iterative_jac_node(&iterative_node, &node_info);

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
            parse_iterative_jac_node_with_default(&iterative_node, &iterative_default, &node_info);

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
            parse_iterative_jac_node_with_default(&iterative_node, &iterative_default, &node_info);
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
        let iteratives = parse_iteratives_jac_node(&iteratives_node);

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
        let iteratives = parse_iteratives_jac_node(&iteratives_node);

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
        let _iteratives = parse_iteratives_jac_node(&iteratives_node);
    }
}
