//! Jacobian evaluation per finite-differences
//!
use nalgebra;

use crate::solver_advanced::model;
use crate::solver_advanced::residuals;

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
