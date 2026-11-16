//! The step is the solution of `jacobian * step = -residuals`, computed with
//! the LU factorization of the jacobian: its inverse is formed only by the
//! methods updating it.

use super::super::JacobianMatrix;

fn matrix() -> nalgebra::DMatrix<f64> {
    nalgebra::DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 3.0])
}

#[test]
fn a_jacobian_is_solved_without_its_inverse() {
    let mut jacobian = JacobianMatrix::<nalgebra::Dyn>::new();
    jacobian
        .update_jacobian_with_exact_value(matrix())
        .ok()
        .unwrap();

    assert!(jacobian.get_inverse().is_none());

    let rhs = nalgebra::DVector::from_vec(vec![5.0, 10.0]);
    let solution = jacobian.solve(&rhs).unwrap();
    assert_eq!(solution, nalgebra::DVector::from_vec(vec![1.0, 3.0]));
}

#[test]
fn the_inverse_is_formed_on_request_and_dropped_with_its_jacobian() {
    let mut jacobian = JacobianMatrix::<nalgebra::Dyn>::new();
    jacobian
        .update_jacobian_with_exact_value(matrix())
        .ok()
        .unwrap();

    jacobian.form_inverse();
    let inverse = jacobian.get_inverse().clone().unwrap();
    let identity = &inverse * matrix();
    assert!(float_cmp::approx_eq!(
        f64,
        identity[(0, 0)],
        1.0,
        epsilon = 1e-15
    ));
    assert!(float_cmp::approx_eq!(
        f64,
        identity[(0, 1)],
        0.0,
        epsilon = 1e-15
    ));
    assert!(float_cmp::approx_eq!(
        f64,
        identity[(1, 0)],
        0.0,
        epsilon = 1e-15
    ));
    assert!(float_cmp::approx_eq!(
        f64,
        identity[(1, 1)],
        1.0,
        epsilon = 1e-15
    ));

    jacobian
        .update_jacobian_with_approximated_value(2.0 * matrix())
        .ok()
        .unwrap();
    assert!(jacobian.get_inverse().is_none());
}

#[test]
fn an_updated_inverse_gives_the_solution_by_itself() {
    let mut jacobian = JacobianMatrix::<nalgebra::Dyn>::new();
    jacobian
        .update_jacobian_with_exact_value(matrix())
        .ok()
        .unwrap();

    // The inverse of the diagonal matrix (2, 4)
    let inverse = nalgebra::DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.25]);
    jacobian.update_inverse(inverse);

    assert!(jacobian.get_jacobian().is_none());

    let rhs = nalgebra::DVector::from_vec(vec![2.0, 4.0]);
    let solution = jacobian.solve(&rhs).unwrap();
    assert_eq!(solution, nalgebra::DVector::from_vec(vec![1.0, 1.0]));
}

#[test]
fn nothing_is_solved_without_a_jacobian() {
    let mut jacobian = JacobianMatrix::<nalgebra::Dyn>::new();
    let rhs = nalgebra::DVector::from_vec(vec![1.0, 1.0]);
    assert!(jacobian.solve(&rhs).is_none());

    let singular = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    assert!(jacobian.update_jacobian_with_exact_value(singular).is_err());
    assert!(jacobian.solve(&rhs).is_none());
}

/// The log writes what the resolution works with:
/// the matrix and whether it is factorized, or the inverse once it has been formed
#[test]
fn the_log_reads_what_the_resolution_holds() {
    let mut jacobian = JacobianMatrix::<nalgebra::Dyn>::new();
    assert!(jacobian.get_jacobian().is_none());
    assert!(jacobian.get_inverse().is_none());

    jacobian
        .update_jacobian_with_exact_value(matrix())
        .ok()
        .unwrap();
    assert_eq!(jacobian.get_jacobian().as_ref().unwrap(), &matrix());
    assert!(jacobian.is_factorized());
    assert!(jacobian.get_inverse().is_none());

    jacobian.form_inverse();
    let inverse = jacobian.get_inverse().clone().unwrap();
    jacobian.update_inverse(inverse);
    assert!(jacobian.get_jacobian().is_none());
    assert!(!jacobian.is_factorized());
    assert!(jacobian.get_inverse().is_some());
}

/// A singular matrix stops the resolution: it is kept, without a factorization, for the log
#[test]
fn a_singular_matrix_is_kept_for_the_log() {
    let mut jacobian = JacobianMatrix::<nalgebra::Dyn>::new();
    let singular = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    assert!(
        jacobian
            .update_jacobian_with_exact_value(singular.clone())
            .is_err()
    );
    assert_eq!(jacobian.get_jacobian().as_ref().unwrap(), &singular);
    assert!(!jacobian.is_factorized());
    assert!(jacobian.get_inverse().is_none());

    let mut factorization_only = JacobianMatrix::<nalgebra::Dyn>::factorization_only();
    assert!(
        factorization_only
            .update_jacobian_with_exact_value(singular)
            .is_err()
    );
    assert!(factorization_only.get_jacobian().is_none());
}

/// A matrix whose rows the factorization has to permute: its first pivot is zero,
/// and the largest entry of each column is never on the diagonal
#[test]
fn a_system_with_permuted_rows_is_solved() {
    let matrix = nalgebra::DMatrix::from_row_slice(
        4,
        4,
        &[
            0.0, 2.0, 1.0, -1.0, //
            1.0, 1.0, 5.0, 2.0, //
            4.0, -3.0, 0.5, 1.0, //
            2.0, 6.0, -1.0, 8.0,
        ],
    );
    let expected = nalgebra::DVector::from_vec(vec![1.0, -2.0, 3.0, 0.5]);
    let rhs = &matrix * &expected;

    let mut jacobian = JacobianMatrix::<nalgebra::Dyn>::new();
    jacobian
        .update_jacobian_with_exact_value(matrix)
        .ok()
        .unwrap();
    let solution = jacobian.solve(&rhs).unwrap();

    for i in 0..4 {
        assert!(float_cmp::approx_eq!(
            f64,
            solution[i],
            expected[i],
            epsilon = 1e-14
        ));
    }
}

/// The solution is the one of the triangular solves of `nalgebra`,
/// for the static dimensions as for the dynamic ones
#[test]
fn the_solution_is_the_one_of_the_lu_solve_of_nalgebra() {
    // Entries without structure, from a linear congruential sequence
    let mut state = 12345_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0
    };

    for size in 1..=12 {
        let matrix = nalgebra::DMatrix::from_fn(size, size, |_, _| next());
        let rhs = nalgebra::DVector::from_fn(size, |_, _| next());
        let reference = matrix.clone().lu().solve(&rhs).unwrap();

        let mut jacobian = JacobianMatrix::<nalgebra::Dyn>::new();
        jacobian
            .update_jacobian_with_exact_value(matrix)
            .ok()
            .unwrap();
        let solution = jacobian.solve(&rhs).unwrap();

        for i in 0..size {
            assert!(float_cmp::approx_eq!(
                f64,
                solution[i],
                reference[i],
                epsilon = 1e-12 * reference.amax().max(1.0)
            ));
        }
    }

    let matrix = nalgebra::SMatrix::<f64, 3, 3>::from_fn(|_, _| next());
    let rhs = nalgebra::SVector::<f64, 3>::from_fn(|_, _| next());
    let reference = matrix.lu().solve(&rhs).unwrap();

    let mut jacobian = JacobianMatrix::<nalgebra::Const<3>>::new();
    jacobian
        .update_jacobian_with_exact_value(matrix)
        .ok()
        .unwrap();
    let solution = jacobian.solve(&rhs).unwrap();

    for i in 0..3 {
        assert!(float_cmp::approx_eq!(
            f64,
            solution[i],
            reference[i],
            epsilon = 1e-12 * reference.amax().max(1.0)
        ));
    }
}

/// The resolutions that never read the matrix back keep its factorization only
#[test]
fn a_jacobian_can_keep_its_factorization_only() {
    let mut jacobian = JacobianMatrix::<nalgebra::Dyn>::factorization_only();
    jacobian
        .update_jacobian_with_exact_value(matrix())
        .ok()
        .unwrap();

    assert!(jacobian.get_jacobian().is_none());
    assert!(jacobian.get_inverse().is_none());

    let rhs = nalgebra::DVector::from_vec(vec![5.0, 10.0]);
    let solution = jacobian.solve(&rhs).unwrap();
    assert_eq!(solution, nalgebra::DVector::from_vec(vec![1.0, 3.0]));

    assert!(jacobian.is_factorized());

    // The methods updating the inverse still get it from the factorization
    jacobian.form_inverse();
    let inverse = jacobian.get_inverse().clone().unwrap();
    let identity = &inverse * matrix();
    assert!(float_cmp::approx_eq!(
        f64,
        identity[(0, 0)],
        1.0,
        epsilon = 1e-15
    ));
    assert!(float_cmp::approx_eq!(
        f64,
        identity[(1, 0)],
        0.0,
        epsilon = 1e-15
    ));
}
