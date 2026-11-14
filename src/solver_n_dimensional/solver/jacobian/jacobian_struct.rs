use std::fmt;

/// The jacobian of the model, in the form the resolution method works with:
/// - the matrix and its LU factorization, the step being the solution of `jacobian * step = -residuals`
/// - or its inverse alone, once a method updating the inverse has taken over
///
/// The inverse is formed only for the methods of
/// [super::super::QuasiNewtonMethod::InverseJacobianUpdate], whose state it is.
pub struct JacobianMatrix<D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    matrix: Option<nalgebra::OMatrix<f64, D, D>>,
    factorization: Option<nalgebra::LU<f64, D, D>>,
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
            factorization: None,
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

    /// When updating the jacobian, it is factorized again:
    /// a matrix the factorization finds singular is refused.
    /// An inverse formed from the previous matrix is dropped
    fn update_jacobian(
        &mut self,
        matrix: nalgebra::OMatrix<f64, D, D>,
    ) -> Result<(), crate::errors::NonInvertibleJacobian> {
        let factorization = matrix.clone().lu();

        if factorization.is_invertible() {
            self.matrix = Some(matrix);
            self.factorization = Some(factorization);
            self.inverse = None;
            self.compute_jacobian_at_next_iteration = false;
            Ok(())
        } else {
            self.invalidate_jacobian();
            Err(crate::errors::NonInvertibleJacobian)
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
    /// but becomes invalid, with its factorization
    pub fn update_inverse(&mut self, inverse: nalgebra::OMatrix<f64, D, D>) {
        self.matrix = None;
        self.factorization = None;
        self.inverse = Some(inverse);
        self.is_current_jacobian_approximated = true;
    }

    /// The solution `x` of `jacobian * x = rhs`:
    /// from the factorization of the jacobian when it is up to date,
    /// otherwise as the product of the inverse by `rhs`.
    ///
    /// `None` when neither the jacobian nor its inverse is known
    pub fn solve(&self, rhs: &nalgebra::OVector<f64, D>) -> Option<nalgebra::OVector<f64, D>> {
        match (&self.factorization, &self.inverse) {
            (Some(factorization), _) => factorization.solve(rhs),
            (None, Some(inverse)) => Some(inverse * rhs),
            (None, None) => None,
        }
    }

    /// Form the inverse from the factorization of the jacobian, unless it is already held.
    ///
    /// Only the methods updating the inverse need it: the others solve with the factorization
    pub fn form_inverse(&mut self) {
        if self.inverse.is_none() {
            self.inverse = self
                .factorization
                .as_ref()
                .and_then(|factorization| factorization.try_inverse());
        }
    }

    /// The inverse of the jacobian, when it has been formed:
    /// by [JacobianMatrix::form_inverse] or given to [JacobianMatrix::update_inverse].
    ///
    /// A jacobian that has just been computed or approximated holds its factorization only
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
        self.factorization = None;
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

        // The log shows what the resolution works with, and nothing else:
        // the inverse appears only once a method updating it has formed it
        match (&self.matrix, &self.inverse) {
            (Some(jac), _) => {
                content.push_str("Step solved with the LU factorization of the jacobian\n\n");
                content.push_str("Jacobian:\n");
                content.push_str(&jac.to_string());
            }
            (None, Some(_)) => {
                content.push_str("Jacobian not up to date with its inverse.\n");
                content.push_str("The update of the inverse is more recent\n");
            }
            (None, None) => content.push_str("Jacobian not computed\n"),
        }

        if let Some(inv) = &self.inverse {
            content.push_str("Inverse of the jacobian Matrix:\n");
            content.push_str(&inv.to_string());
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
            .field("Matrix Factorization", &self.factorization)
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
