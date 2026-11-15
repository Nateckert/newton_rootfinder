/// The LU factorization of an invertible jacobian, with partial pivoting,
/// kept to solve `jacobian * x = rhs`.
///
/// It is written here and not taken from `nalgebra`, whose factorization and triangular solves
/// cost more on a system of a few unknowns than the product by an inverse.
/// Both factors are stored in one matrix, the lower one without its diagonal of ones.
/// Two values are kept with them, for a solution to permute nothing in place
/// and to divide by nothing:
/// - the row permutation, as the row of the right-hand side each row of the permuted system reads
/// - the diagonal of the upper factor, as its reciprocals
#[derive(Clone, Debug)]
pub(super) struct Factorization<D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    factors: nalgebra::OMatrix<f64, D, D>,
    permutation: nalgebra::OVector<usize, D>,
    inverse_diagonal: nalgebra::OVector<f64, D>,
}

impl<D> Factorization<D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// Factorize a matrix: `None` when a pivot is zero, the matrix being singular
    #[inline]
    pub(super) fn new(mut factors: nalgebra::OMatrix<f64, D, D>) -> Option<Self> {
        let (dimension, _) = factors.shape_generic();
        let size = dimension.value();
        let mut permutation =
            nalgebra::OVector::<usize, D>::from_fn_generic(dimension, nalgebra::U1, |row, _| row);
        let mut inverse_diagonal =
            nalgebra::OVector::<f64, D>::zeros_generic(dimension, nalgebra::U1);

        // The matrix is stored column by column
        let entries = factors.as_mut_slice();

        for k in 0..size {
            // The pivot is the largest entry of the column, on the diagonal or under it
            let column = &entries[k * size..(k + 1) * size];
            let mut pivot_row = k;
            for row in k + 1..size {
                if column[row].abs() > column[pivot_row].abs() {
                    pivot_row = row;
                }
            }
            let pivot = column[pivot_row];
            if pivot == 0.0 {
                return None;
            }

            if pivot_row != k {
                for column in entries.chunks_exact_mut(size) {
                    column.swap(k, pivot_row);
                }
                permutation.as_mut_slice().swap(k, pivot_row);
            }

            let inverse_pivot = 1.0 / pivot;
            inverse_diagonal[k] = inverse_pivot;

            // The column of the lower factor, then the elimination of the columns on its right
            let (left, right) = entries.split_at_mut((k + 1) * size);
            let multipliers = &mut left[k * size + k + 1..];
            for multiplier in multipliers.iter_mut() {
                *multiplier *= inverse_pivot;
            }
            for column in right.chunks_exact_mut(size) {
                let u_kj = column[k];
                for (entry, multiplier) in column[k + 1..].iter_mut().zip(multipliers.iter()) {
                    *entry -= multiplier * u_kj;
                }
            }
        }

        Some(Factorization {
            factors,
            permutation,
            inverse_diagonal,
        })
    }

    /// The solution `x` of `matrix * x = rhs`
    #[inline]
    pub(super) fn solve(&self, rhs: &nalgebra::OVector<f64, D>) -> nalgebra::OVector<f64, D> {
        let size = rhs.len();
        let factors = self.factors.as_slice();
        let mut solution = self.permutation.map(|row| rhs[row]);
        let x = solution.as_mut_slice();

        // Forward substitution with the lower factor, whose diagonal is one
        for j in 0..size {
            let column = &factors[j * size..(j + 1) * size];
            let (head, tail) = x.split_at_mut(j + 1);
            let x_j = head[j];
            for (x_i, l_ij) in tail.iter_mut().zip(&column[j + 1..]) {
                *x_i -= l_ij * x_j;
            }
        }

        // Backward substitution with the upper factor
        for j in (0..size).rev() {
            let column = &factors[j * size..(j + 1) * size];
            let (head, tail) = x.split_at_mut(j);
            let x_j = tail[0] * self.inverse_diagonal[j];
            tail[0] = x_j;
            for (x_i, u_ij) in head.iter_mut().zip(&column[..j]) {
                *x_i -= u_ij * x_j;
            }
        }

        solution
    }

    /// The inverse of the matrix, each of its columns the solution for a column of the identity
    pub(super) fn inverse(&self) -> nalgebra::OMatrix<f64, D, D> {
        let (dimension, _) = self.factors.shape_generic();
        let mut inverse = nalgebra::OMatrix::<f64, D, D>::zeros_generic(dimension, dimension);
        let mut unit = nalgebra::OVector::<f64, D>::zeros_generic(dimension, nalgebra::U1);

        for j in 0..dimension.value() {
            unit[j] = 1.0;
            inverse.set_column(j, &self.solve(&unit));
            unit[j] = 0.0;
        }

        inverse
    }
}
