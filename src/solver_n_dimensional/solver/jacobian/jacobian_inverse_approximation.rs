use super::JacobianMatrix;

use super::super::quasi_method_update_inv_jac;
use super::super::UpdateQuasiNewtonMethod;
use super::super::{broyden_first_method_udpate_inv_jac, broyden_second_method_udpate_inv_jac};

pub fn approximate_inv_jacobian<D>(
    jacobian: &mut JacobianMatrix<D>,
    method: UpdateQuasiNewtonMethod,
    iteratives_step_size: &nalgebra::OVector<f64, D>,
    residuals_step_size: &nalgebra::OVector<f64, D>,
    residuals_values_current: &nalgebra::OVector<f64, D>,
) where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::U1, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    let inv_jac_next = match method {
        UpdateQuasiNewtonMethod::BroydenFirstMethod => broyden_first_method_udpate_inv_jac(
            jacobian.get_inverse().as_ref().unwrap(),
            iteratives_step_size,
            residuals_step_size,
        ),
        UpdateQuasiNewtonMethod::BroydenSecondMethod => broyden_second_method_udpate_inv_jac(
            jacobian.get_inverse().as_ref().unwrap(),
            iteratives_step_size,
            residuals_step_size,
        ),
        UpdateQuasiNewtonMethod::GreenstadtFirstMethod => quasi_method_update_inv_jac(
            jacobian.get_inverse().as_ref().unwrap(),
            iteratives_step_size,
            residuals_step_size,
            residuals_values_current,
        ),
        UpdateQuasiNewtonMethod::GreenstadtSecondMethod => {
            let c = jacobian.get_inverse().as_ref().unwrap().transpose()
                * jacobian.get_inverse().as_ref().unwrap()
                * residuals_step_size;
            quasi_method_update_inv_jac(
                jacobian.get_inverse().as_ref().unwrap(),
                iteratives_step_size,
                residuals_step_size,
                &c,
            )
        }
    };

    jacobian.update_inverse(inv_jac_next);
}
