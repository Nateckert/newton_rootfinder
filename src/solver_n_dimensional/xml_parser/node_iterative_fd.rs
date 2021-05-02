use crate::iteratives;
use minidom::Element;

pub fn parse_iteratives_fd_node(iteratives_node: &Element) -> Vec<iteratives::IterativeParamsFD> {
    let mut iteratives = Vec::new();

    let iterative_fd_default = parse_iterative_fd_node(&iteratives_node, &"iteratives node");

    for (expected_id, iterative_node) in iteratives_node.children().enumerate() {
        if iterative_node.name() != "iterative" {
            panic!(
                "Node below iteratives are expected to be \"iterative\", got {}",
                iterative_node.name()
            );
        }
        let id = super::util::parse_id(iterative_node, expected_id, &"iterative node");
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

fn parse_iterative_fd_node(
    iterative_node: &Element,
    node_info: &str,
) -> iteratives::IterativeParamsFD {
    let iterative = super::node_iterative_jac::parse_iterative_jac_node(iterative_node, node_info);

    let dx_abs = super::util::parse_float_attribute(iterative_node, &"dx_abs", &node_info);
    let dx_rel = super::util::parse_float_attribute(iterative_node, &"dx_rel", &node_info);

    let perturbation_method =
        super::node_iterative::parse_perturbation_method(iterative_node, &node_info);

    iteratives::IterativeParamsFD::extend(iterative, dx_abs, dx_rel, perturbation_method)
}

fn parse_iterative_fd_node_with_default(
    iterative_node: &Element,
    iterative_default: &iteratives::IterativeParamsFD,
    node_info: &str,
) -> iteratives::IterativeParamsFD {
    let iterative = super::node_iterative_jac::parse_iterative_jac_node_with_default(
        iterative_node,
        &iterative_default.get_iterative_params(),
        node_info,
    );

    let dx_abs = super::util::parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_dx_abs(),
        &"dx_abs",
        &node_info,
    );
    let dx_rel = super::util::parse_float_attribute_with_default(
        iterative_node,
        iterative_default.get_dx_rel(),
        &"dx_rel",
        &node_info,
    );

    let perturbation_method = super::node_iterative::parse_perturbation_method_with_default(
        iterative_node,
        iterative_default.get_perturbation_method(),
        &node_info,
    );

    iteratives::IterativeParamsFD::extend(iterative, dx_abs, dx_rel, perturbation_method)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteratives;

    use minidom::Element;

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
