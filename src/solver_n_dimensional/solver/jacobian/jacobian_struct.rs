use std::fmt;

fn compute_inverse<D>(
    matrix: &nalgebra::OMatrix<f64, D, D>,
) -> Result<nalgebra::OMatrix<f64, D, D>, crate::errors::NonInvertibleJacobian>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    let lu_jac = matrix.to_owned().lu();

    match lu_jac.try_inverse() {
        Some(inv_jac) => Ok(inv_jac),
        None => Err(crate::errors::NonInvertibleJacobian),
    }
}

pub struct JacobianMatrix<D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    matrix: Option<nalgebra::OMatrix<f64, D, D>>,
    inverse: Option<nalgebra::OMatrix<f64, D, D>>,
    compute_jacobian_at_next_iteration: bool,
    is_current_jacobian_approximated: bool,
}

impl<D> Default for JacobianMatrix<D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    fn default() -> Self {
        JacobianMatrix::new()
    }
}

impl<D> JacobianMatrix<D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    pub fn new() -> Self {
        JacobianMatrix {
            matrix: None,
            inverse: None,
            compute_jacobian_at_next_iteration: true,
            is_current_jacobian_approximated: false,
        }
    }

    pub fn force_jacobian_computation(&mut self) {
        self.compute_jacobian_at_next_iteration = true
    }

    pub fn compute_jacobian(&self) -> bool {
        self.compute_jacobian_at_next_iteration
    }

    pub fn is_jacobian_approximated(&self) -> bool {
        self.is_current_jacobian_approximated
    }

    /// When updating the jacobian,
    /// the inverse has to be recomputed
    fn update_jacobian(
        &mut self,
        matrix: nalgebra::OMatrix<f64, D, D>,
    ) -> Result<(), crate::errors::NonInvertibleJacobian> {
        match compute_inverse(&matrix) {
            Ok(inverse_matrix) => {
                self.inverse = Some(inverse_matrix);
                self.matrix = Some(matrix);
                self.compute_jacobian_at_next_iteration = false;
                Ok(())
            }
            Err(_) => {
                self.invalidate_jacobian();
                Err(crate::errors::NonInvertibleJacobian)
            }
        }
    }

    pub fn update_jacobian_with_exact_value(
        &mut self,
        matrix: nalgebra::OMatrix<f64, D, D>,
    ) -> Result<(), crate::errors::NonInvertibleJacobian> {
        self.is_current_jacobian_approximated = false;
        self.update_jacobian(matrix)
    }

    pub fn update_jacobian_with_approximated_value(
        &mut self,
        matrix: nalgebra::OMatrix<f64, D, D>,
    ) -> Result<(), crate::errors::NonInvertibleJacobian> {
        self.is_current_jacobian_approximated = true;
        self.update_jacobian(matrix)
    }

    /// When updating the inverse,
    /// the jacobian does not have to be recomputed
    /// but becomes invalid
    pub fn update_inverse(&mut self, inverse: nalgebra::OMatrix<f64, D, D>) {
        self.matrix = None;
        self.inverse = Some(inverse);
        self.is_current_jacobian_approximated = true;
    }

    /// Need to have Some and None for the inverse ?
    /// it is always valid !
    pub fn get_inverse(&self) -> &Option<nalgebra::OMatrix<f64, D, D>> {
        &self.inverse
    }

    pub fn get_jacobian(&self) -> &Option<nalgebra::OMatrix<f64, D, D>> {
        &self.matrix
    }
    /// Invalidate a jacobian
    /// For example, if there is an error computing it
    pub fn invalidate_jacobian(&mut self) {
        self.matrix = None;
        self.inverse = None;
    }
}

impl<D> fmt::Display for JacobianMatrix<D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
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

        content.push('\n');

        write!(f, "{}", content)
    }
}

impl<D> fmt::Debug for JacobianMatrix<D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Jacobian matrix")
            .field("Matrix", &self.matrix)
            .field("Matrix Inverse", &self.inverse)
            .field(
                "Compute jacobian at next iteration: ",
                &self.compute_jacobian_at_next_iteration,
            )
            .field(
                "Is current jacobian approximated: ",
                &self.is_current_jacobian_approximated,
            )
            .finish()
    }
}
