use nalgebra::storage::Storage;

pub fn ovector_zeros_like<D>(input: &nalgebra::OVector<f64, D>) -> nalgebra::OVector<f64, D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
{
    let (nrows, ncols) = input.data.shape();

    return nalgebra::OMatrix::zeros_generic(nrows, ncols);
}
