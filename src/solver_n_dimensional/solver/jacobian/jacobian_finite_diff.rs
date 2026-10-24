use std::fmt;

use super::super::Point;
use super::JacobianMatrix;
use crate::errors;
use crate::iteratives;
use crate::iteratives::Iterative;
use crate::model;
use crate::model::ModelError;
use crate::residuals;

/// Evaluate a jacobian per forward finite difference when perturbation step eps is provided
///
/// This function has been made public for testing purpose only
///
/// The reference point is the one the model has just been evaluated on
pub fn compute_jacobian_from_finite_difference<M, D>(
    model: &mut M,
    perturbations: &nalgebra::OVector<f64, D>,
    update_residuals: &residuals::ResidualsConfig,
) -> Result<nalgebra::OMatrix<f64, D, D>, ModelError<M, D>>
where
    M: model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    let reference = Point::of(model);
    compute_jacobian_from_finite_difference_at(model, &reference, perturbations, update_residuals)
}

/// Evaluate a jacobian per forward finite difference around the `reference` point
///
/// The model is left on the last perturbed point, with the memory of the reference point
pub(crate) fn compute_jacobian_from_finite_difference_at<M, D>(
    model: &mut M,
    reference: &Point<D>,
    perturbations: &nalgebra::OVector<f64, D>,
    update_residuals: &residuals::ResidualsConfig,
) -> Result<nalgebra::OMatrix<f64, D, D>, ModelError<M, D>>
where
    M: model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    let problem_size = model.len_problem();
    let mut jacobian: nalgebra::OMatrix<f64, D, D> =
        super::super::super::omatrix_zeros_like_ovector(perturbations);
    let memory_ref = &reference.memory;
    let iteratives_ref = &reference.iteratives;
    let residuals_ref = update_residuals.evaluate_update_residuals(&reference.residuals);

    for i in 0..problem_size {
        // Finite-difference column evaluation
        let mut iteratives_perturbations = iteratives_ref.clone();
        iteratives_perturbations[i] += perturbations[i];

        model.set_iteratives(&iteratives_perturbations);
        match model.evaluate_column(i) {
            // recovers from inaccurate values
            Ok(()) | Err(ModelError::InaccurateValuesError(_)) => (),
            Err(model_error) => {
                model.set_memory(memory_ref);
                return Err(model_error);
            }
        }

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
        model.set_memory(memory_ref); // restart from reference state
    }

    Ok(jacobian)
}

pub fn evaluate_jacobian_from_finite_difference<'a, M, D, T>(
    jacobian: &mut JacobianMatrix<D>,
    model: &mut M,
    iters_params: &'a iteratives::Iteratives<'a, T>,
    residuals_config: &'a residuals::ResidualsConfig<'a>,
) -> Result<(), crate::errors::SolverInternalError<M, D>>
where
    M: model::Model<D>,
    T: Iterative + fmt::Display,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    let reference = Point::of(model);
    evaluate_jacobian_from_finite_difference_at(
        jacobian,
        model,
        &reference,
        iters_params,
        residuals_config,
    )
}

pub(crate) fn evaluate_jacobian_from_finite_difference_at<'a, M, D, T>(
    jacobian: &mut JacobianMatrix<D>,
    model: &mut M,
    reference: &Point<D>,
    iters_params: &'a iteratives::Iteratives<'a, T>,
    residuals_config: &'a residuals::ResidualsConfig<'a>,
) -> Result<(), crate::errors::SolverInternalError<M, D>>
where
    M: model::Model<D>,
    T: Iterative + fmt::Display,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    let perturbations = iters_params.compute_perturbations(&reference.iteratives);

    let matrix = compute_jacobian_from_finite_difference_at(
        model,
        reference,
        &perturbations,
        residuals_config,
    );
    match matrix {
        Ok(valid_jacobian) => match jacobian.update_jacobian_with_exact_value(valid_jacobian) {
            Ok(()) => Ok(()),
            Err(errors::NonInvertibleJacobian) => {
                Err(errors::SolverInternalError::InvalidJacobianInverseError)
            }
        },
        Err(model_error) => Err(errors::SolverInternalError::InvalidJacobianError(
            model_error,
        )),
    }
}
