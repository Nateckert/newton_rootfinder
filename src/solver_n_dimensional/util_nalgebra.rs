pub fn ovector_zeros_like<D>(input: &nalgebra::OVector<f64, D>) -> nalgebra::OVector<f64, D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
{
    let (nrows, ncols) = input.shape();

    nalgebra::OMatrix::zeros_generic(nrows, ncols)
}

pub fn omatrix_zeros_like<D>(input: &nalgebra::OMatrix<f64, D, D>) -> nalgebra::OMatrix<f64, D, D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    let (nrows, ncols) = input.shape();

    nalgebra::OMatrix::zeros_generic(nrows, ncols)
}

pub fn omatrix_zeros_like_ovector<D>(input: &nalgebra::OVector<f64, D>) -> nalgebra::OMatrix<f64, D, D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    let (nrows, _ncols) = input.shape();

    nalgebra::OMatrix::zeros_generic(nrows, nrows)
}

pub fn ovector_zeros_from_shape<D>(nrows: D) -> nalgebra::OVector<f64, D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
{
    nalgebra::OMatrix::zeros_generic(nrows, nalgebra::Const::<1_usize>)
}

pub fn omatrix_zeros_from_shape<D>(nrows: D) -> nalgebra::OMatrix<f64, D, D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    nalgebra::OMatrix::zeros_generic(nrows, nrows)
}
