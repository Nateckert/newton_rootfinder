use std::fmt;

fn compute_inverse<D>(matrix: &nalgebra::OMatrix<f64, D, D>) -> nalgebra::OMatrix<f64, D, D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<(usize, usize), D>,
{
    let lu_jac = matrix.to_owned().lu();

    match lu_jac.try_inverse() {
        Some(inv_jac) => inv_jac,
        None => panic!("The jacobian matrix is non invertible"),
    }
}

pub struct JacobianMatrix<D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<(usize, usize), D>,
{
    matrix: Option<nalgebra::OMatrix<f64, D, D>>,
    inverse: Option<nalgebra::OMatrix<f64, D, D>>,
}

impl<D> Default for JacobianMatrix<D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<(usize, usize), D>,
{
    fn default() -> Self {
        JacobianMatrix::new()
    }
}

impl<D> JacobianMatrix<D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<(usize, usize), D>,
{
    pub fn new() -> Self {
        JacobianMatrix {
            matrix: None,
            inverse: None,
        }
    }

    /// When updating the jacobian,
    /// the inverse has to be recomputed
    pub fn update_jacobian(&mut self, matrix: nalgebra::OMatrix<f64, D, D>) {
        self.inverse = Some(compute_inverse(&matrix));
        self.matrix = Some(matrix);
    }

    /// When updating the inverse,
    /// the jacobian does not have to be recomputed
    /// but becomes invalid
    pub fn update_inverse(&mut self, inverse: nalgebra::OMatrix<f64, D, D>) {
        self.matrix = None;
        self.inverse = Some(inverse);
    }

    /// Need to have Some and None for the inverse ?
    /// it is always valid !
    pub fn get_inverse(&self) -> &Option<nalgebra::OMatrix<f64, D, D>> {
        &self.inverse
    }

    pub fn get_jacobian(&self) -> &Option<nalgebra::OMatrix<f64, D, D>> {
        &self.matrix
    }
}

impl<D> fmt::Display for JacobianMatrix<D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<(usize, usize), D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut content = String::from("Jacobian matrix\n");
        content.push_str("=================\n\n");

        match &self.matrix {
            Some(jac) => {
                content.push_str("Jacobian up to date with its inverse\n\n");
                content.push_str("Jacobian:\n");
                content.push_str(&jac.to_string());
            }
            None => {
                content.push_str("Jacobian not up to date with its inverse.\n");
                content.push_str("The update of the inverse is more recent\n");
            }
        }

        content.push_str("Inverse of the jacobian Matrix:\n");

        match &self.inverse {
            Some(inv) => content.push_str(&inv.to_string()),
            None => content.push_str("Inverse jacobian matrix not yet computed"),
        }

        content.push_str(&"\n");

        write!(f, "{}", content)
    }
}

use crate::model;
use crate::residuals;

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
