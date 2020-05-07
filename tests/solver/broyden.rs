use crate::common::broyden1965::*;

#[test]
fn disable_never_used_warning() {
    let init_case5 = init_broyden1965_case5();
    broyden1965_case5(&init_case5);

    let init_case6 = init_broyden1965_case6();
    broyden1965_case6(&init_case6);

    let init_case7 = init_broyden1965_case7();
    broyden1965_case7(&init_case7);

    let init_case8 = init_broyden1965_case8();
    broyden1965_case8(&init_case8);

    let init_case9 = init_broyden1965_case9();
    broyden1965_case9(&init_case9);

    let init_case10 = init_broyden1965_case10();
    broyden1965_case10(&init_case10);
}
