use nalgebra::storage::Storage;

pub fn ovector_zeros_like<D>(input: &nalgebra::OVector<f64, D>) -> nalgebra::OVector<f64, D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
{
    let (nrows, ncols) = input.data.shape();

    nalgebra::OMatrix::zeros_generic(nrows, ncols)
}

pub fn omatrix_zeros_like<D>(input: &nalgebra::OMatrix<f64, D, D>) -> nalgebra::OMatrix<f64, D, D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
    let (nrows, ncols) = input.data.shape();

    nalgebra::OMatrix::zeros_generic(nrows, ncols)
}

pub fn omatrix_zeros_like_ovector<D>(
    input: &nalgebra::OVector<f64, D>,
) -> nalgebra::OMatrix<f64, D, D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
    let (nrows, _ncols) = input.data.shape();

    nalgebra::OMatrix::zeros_generic(nrows, nrows)
}

pub fn ovector_zeros_from_shape<D>(nrows: D) -> nalgebra::OVector<f64, D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
{
    nalgebra::OMatrix::zeros_generic(nrows, nalgebra::Const::<1_usize>)
}

pub fn omatrix_zeros_from_shape<D>(nrows: D) -> nalgebra::OMatrix<f64, D, D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
    nalgebra::OMatrix::zeros_generic(nrows, nrows)
}
