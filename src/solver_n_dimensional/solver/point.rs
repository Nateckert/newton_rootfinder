use crate::model;
use crate::residuals;

/// The point the solver stands on: the last guess accepted, with the residuals
/// and the memory of its evaluation
///
/// The solver keeps it instead of reading it back from the model,
/// as the model holds a perturbed point after a jacobian evaluated by finite-difference.
pub(crate) struct Point<M, D>
where
    M: model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    pub(crate) iteratives: nalgebra::OVector<f64, D>,
    pub(crate) residuals: residuals::ResidualsValues<D>,
    pub(crate) memory: M::Memory,
}

impl<M, D> Point<M, D>
where
    M: model::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    /// The point the model has just been evaluated on
    pub(crate) fn of(model: &M) -> Self {
        Point {
            iteratives: model.get_iteratives(),
            residuals: model.get_residuals(),
            memory: model.get_memory(),
        }
    }
}
