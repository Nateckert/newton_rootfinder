extern crate nalgebra;
use nalgebra::linalg;

use crate::model;
use crate::util::residuals;

/// Evaluate a jacobian per forward finite difference when perturbation step eps is provided
pub fn jacobian_evaluation<T>(
    model: &mut T,
    perturbations: &nalgebra::DVector<f64>,
    update_residuals: &residuals::ResidualsConfig,
) -> nalgebra::DMatrix<f64>
where
    T: model::Model,
{
    let problem_size = model.len_problem();
    let mut jacobian: nalgebra::DMatrix<f64> = nalgebra::DMatrix::zeros(problem_size, problem_size);

    let memory_ref = model.get_memory();
    let iteratives_ref = model.get_iteratives();
    let residuals_ref = update_residuals.evaluate_update_residuals(&model.get_residuals());

    for i in 0..problem_size {
        // Finite-difference column evaluation
        let mut iteratives_perturbations = iteratives_ref.clone();
        iteratives_perturbations[i] += perturbations[i];

        model.set_iteratives(&iteratives_perturbations);
        model.evaluate();

        let residuals_perturbation =
            update_residuals.evaluate_update_residuals(&model.get_residuals());

        // First order forward difference
        let col = (residuals_perturbation - &residuals_ref) / perturbations[i];

        jacobian.set_column(i, &col);

        // Restart from reference state, needed for :
        // - next iteration of the loop :
        //          jacobian evaluation independant of column order
        // - next model evaluation after jacobian computation :
        //          in case of step rejection
        //          to make the memory of next step independant of column order
        model.set_memory(&memory_ref); // restart from reference state
    }

    jacobian
}

pub fn newton_raw_step_size(
    residuals_values: &nalgebra::DVector<f64>,
    jac: &nalgebra::DMatrix<f64>,
) -> nalgebra::DVector<f64> {
    // error management
    // It is limited to finding a null column, as it is the most common case in practice
    //
    // When modeling a physical system, it is extremly rare with floating point arithmeric
    // to have a determinant that is exactly zero
    // The most frequent case is having an iterative variable without any influence
    // Hence, a column of zero in the jacobian matrix

    let lu_jac = linalg::LU::new(jac.clone());

    let inv_jac = match lu_jac.try_inverse() {
        Some(inv_jac) => inv_jac,
        None => panic!("The jacobian matrix is non invertible"),
    };

    -inv_jac * residuals_values
}

pub fn newton_limited_step_size(
    iter_values: &nalgebra::DVector<f64>,
    raw_step: nalgebra::DVector<f64>,
) -> nalgebra::DVector<f64> {
    raw_step + iter_values - iter_values
}
