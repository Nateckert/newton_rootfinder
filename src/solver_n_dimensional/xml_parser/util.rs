use minidom::Element;

pub fn parse_int_attribute(node: &Element, attribute: &str, node_info: &str) -> usize {
    node.attr(attribute)
        .unwrap_or_else(|| {
            panic!(
                "The attribute \"{}\" is missing in the {}",
                attribute, node_info
            )
        })
        .parse::<usize>()
        .unwrap_or_else(|_| {
            panic!(
                "The attribute \"{}\" is not a valid positive integer",
                attribute
            )
        })
}

pub fn parse_float_attribute(node: &Element, attribute: &str, node_info: &str) -> f64 {
    node
        .attr(attribute)
        .unwrap_or_else(|| panic!("The attribute \"{}\" is missing in the {}", attribute, node_info))
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("The attribute \"{}\" is not a valid float, for infinity, the valid values are \"-inf\" and \"inf\" ", attribute))
}

pub fn parse_float_attribute_with_default(
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

pub fn check_node_name_and_panic(node: &Element, expected_name: &str) {
    if node.name() != expected_name {
        panic!(
            "The node is expected to be \"{}\", got {}",
            expected_name,
            node.name()
        );
    }
}

pub fn parse_id(node: &Element, expected_id: usize, node_info: &str) -> usize {
    let id = parse_int_attribute(node, "id", node_info);
    if expected_id != id {
        panic!(
            "The ids must be in order starting from 0, got id {} when the expected one was {}",
            id, expected_id
        );
    }

    id
}
