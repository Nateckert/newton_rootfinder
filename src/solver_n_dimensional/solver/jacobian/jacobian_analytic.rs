use super::JacobianMatrix;
use crate::model;
use crate::residuals;

pub fn evaluate_jacobian_from_analytical_function<'a, M, D>(
    jacobian_matrix: &mut JacobianMatrix<D>,
    model: &mut M,
    residuals_config: &'a residuals::ResidualsConfig<'a>,
) -> Result<(), crate::model::ModelError<M, D>>
where
    M: model::Model<D>,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<(usize, usize), D>,
{
    let residuals_values = model.get_residuals();

    let jacobians = model.get_jacobian();
    match jacobians {
        Ok(valid_jacobians) => {
            let normalization_method = residuals_config.get_update_methods();
            jacobian_matrix.update_jacobian(
                valid_jacobians.normalize(&residuals_values, &normalization_method),
            );
            Ok(())
        }
        Err(error) => {
            jacobian_matrix.invalidate_jacobian();
            Err(error)
        }
    }
}
