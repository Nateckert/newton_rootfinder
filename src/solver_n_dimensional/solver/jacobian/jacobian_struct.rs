use std::fmt;

use super::factorization::Factorization;

/// The jacobian of the model, in the form the resolution method works with:
/// - the matrix and its LU factorization, the step being the solution of `jacobian * step = -residuals`
/// - or its inverse alone, once a method updating the inverse has taken over
///
/// The inverse is formed only for the methods of
/// [super::super::QuasiNewtonMethod::InverseJacobianUpdate], whose state it is.
///
/// The matrix itself is read back by the methods of
/// [super::super::QuasiNewtonMethod::JacobianUpdate], which update it, and by the log:
/// a jacobian built with [JacobianMatrix::factorization_only] does not keep it.
pub(crate) struct JacobianMatrix<D>
where
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    matrix: Option<nalgebra::OMatrix<f64, D, D>>,
    factorization: Option<Factorization<D>>,
    inverse: Option<nalgebra::OMatrix<f64, D, D>>,
    keep_matrix: bool,
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
    /// A jacobian keeping the matrices it is given beside their factorization
    pub(crate) fn new() -> Self {
        JacobianMatrix {
            matrix: None,
            factorization: None,
            inverse: None,
            keep_matrix: true,
            compute_jacobian_at_next_iteration: true,
            is_current_jacobian_approximated: false,
        }
    }

    /// A jacobian keeping the factorization of the matrices it is given, and not the matrices:
    /// [JacobianMatrix::get_jacobian] returns `None`.
    ///
    /// It spares a copy of the matrix at each update, for the resolutions that never read it back:
    /// the ones of the Newton-Raphson, stationary Newton and inverse update methods, without the log
    pub(crate) fn factorization_only() -> Self {
        JacobianMatrix {
            keep_matrix: false,
            ..JacobianMatrix::new()
        }
    }

    pub(crate) fn force_jacobian_computation(&mut self) {
        self.compute_jacobian_at_next_iteration = true
    }

    pub(crate) fn compute_jacobian(&self) -> bool {
        self.compute_jacobian_at_next_iteration
    }

    pub(crate) fn is_jacobian_approximated(&self) -> bool {
        self.is_current_jacobian_approximated
    }

    /// When updating the jacobian, it is factorized again:
    /// a matrix the factorization finds singular is refused.
    /// An inverse formed from the previous matrix is dropped
    fn update_jacobian(
        &mut self,
        matrix: nalgebra::OMatrix<f64, D, D>,
    ) -> Result<(), crate::errors::NonInvertibleJacobian> {
        // The factorization overwrites the matrix it is given
        let kept_matrix = if self.keep_matrix {
            Some(matrix.clone())
        } else {
            None
        };

        match Factorization::new(matrix) {
            Some(factorization) => {
                self.matrix = kept_matrix;
                self.factorization = Some(factorization);
                self.inverse = None;
                self.compute_jacobian_at_next_iteration = false;
                Ok(())
            }
            None => {
                self.invalidate_jacobian();
                Err(crate::errors::NonInvertibleJacobian)
            }
        }
    }

    pub(crate) fn update_jacobian_with_exact_value(
        &mut self,
        matrix: nalgebra::OMatrix<f64, D, D>,
    ) -> Result<(), crate::errors::NonInvertibleJacobian> {
        self.is_current_jacobian_approximated = false;
        self.update_jacobian(matrix)
    }

    pub(crate) fn update_jacobian_with_approximated_value(
        &mut self,
        matrix: nalgebra::OMatrix<f64, D, D>,
    ) -> Result<(), crate::errors::NonInvertibleJacobian> {
        self.is_current_jacobian_approximated = true;
        self.update_jacobian(matrix)
    }

    /// When updating the inverse,
    /// the jacobian does not have to be recomputed
    /// but becomes invalid, with its factorization
    pub(crate) fn update_inverse(&mut self, inverse: nalgebra::OMatrix<f64, D, D>) {
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
    #[inline]
    pub(crate) fn solve(
        &self,
        rhs: &nalgebra::OVector<f64, D>,
    ) -> Option<nalgebra::OVector<f64, D>> {
        match (&self.factorization, &self.inverse) {
            (Some(factorization), _) => Some(factorization.solve(rhs)),
            (None, Some(inverse)) => Some(inverse * rhs),
            (None, None) => None,
        }
    }

    /// Form the inverse from the factorization of the jacobian, unless it is already held.
    ///
    /// Only the methods updating the inverse need it: the others solve with the factorization
    pub(crate) fn form_inverse(&mut self) {
        if self.inverse.is_none() {
            self.inverse = self
                .factorization
                .as_ref()
                .map(|factorization| factorization.inverse());
        }
    }

    /// The inverse of the jacobian, when it has been formed:
    /// by [JacobianMatrix::form_inverse] or given to [JacobianMatrix::update_inverse].
    ///
    /// A jacobian that has just been computed or approximated holds its factorization only
    pub(crate) fn get_inverse(&self) -> &Option<nalgebra::OMatrix<f64, D, D>> {
        &self.inverse
    }

    pub(crate) fn get_jacobian(&self) -> &Option<nalgebra::OMatrix<f64, D, D>> {
        &self.matrix
    }
    /// Invalidate a jacobian
    /// For example, if there is an error computing it
    pub(crate) fn invalidate_jacobian(&mut self) {
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
        if self.factorization.is_some() {
            content.push_str("Step solved with the LU factorization of the jacobian\n\n");
            match &self.matrix {
                Some(jac) => {
                    content.push_str("Jacobian:\n");
                    content.push_str(&jac.to_string());
                }
                None => content.push_str("Jacobian not kept beside its factorization\n"),
            }
        } else if self.inverse.is_some() {
            content.push_str("Jacobian not up to date with its inverse.\n");
            content.push_str("The update of the inverse is more recent\n");
        } else {
            content.push_str("Jacobian not computed\n");
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
            .field("Keep the matrix: ", &self.keep_matrix)
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
