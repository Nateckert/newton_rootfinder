pub fn float_matrix_comparison(
    jac: &nalgebra::DMatrix<f64>,
    jac_ref: &nalgebra::DMatrix<f64>,
    epsilon: f64,
) {
    let (n, m) = jac.shape();
    if (n, m) != jac_ref.shape() {
        panic!("Size mismatch");
    }
    if n != m {
        panic!("Matrix are not squared");
    }

    for i in 0..n {
        for j in 0..n {
            assert!(float_cmp::approx_eq!(
                f64,
                jac[(i, j)],
                jac_ref[(i, j)],
                epsilon = epsilon
            ));
        }
    }
}
