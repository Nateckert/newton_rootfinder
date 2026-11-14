//! The quasi-Newton updates on a residual changing sign on the way to
//! the root: the secant of an update is taken on the signed residuals, as
//! the step is, not on their absolute values.

use newton_rootfinder as nrf;

use crate::common::run_function_case_fd;

use nrf::solver::{QuasiNewtonMethod, ResolutionMethod, UpdateQuasiNewtonMethod};

/// atan(x) = 0: from x = 1, the first Newton step overshoots the root,
/// the residual changing sign.
fn arctangent(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_vec(vec![x[0].atan()])
}

fn every_update() -> Vec<ResolutionMethod> {
    let updates = [
        UpdateQuasiNewtonMethod::BroydenFirstMethod,
        UpdateQuasiNewtonMethod::BroydenSecondMethod,
        UpdateQuasiNewtonMethod::GreenstadtFirstMethod,
        UpdateQuasiNewtonMethod::GreenstadtSecondMethod,
    ];
    updates
        .iter()
        .flat_map(|&update| {
            [
                ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(update)),
                ResolutionMethod::QuasiNewton(QuasiNewtonMethod::InverseJacobianUpdate(update)),
            ]
        })
        .collect()
}

#[test]
fn every_update_follows_a_residual_changing_sign() {
    for method in every_update() {
        for damping in [false, true] {
            run_function_case_fd(
                1,
                arctangent,
                nalgebra::DVector::from_vec(vec![1.0]),
                nalgebra::DVector::from_vec(vec![0.0]),
                method,
                damping,
            );
        }
    }
}
