use crate::model;
use crate::residuals;

/// The point the solver stands on: the last guess accepted, with the residuals
/// and the memory of its evaluation
///
/// The solver keeps it instead of reading it back from the model,
/// as the model holds a perturbed point after a jacobian evaluated by finite-difference.
pub(crate) struct Point<D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
{
    pub(crate) iteratives: nalgebra::OVector<f64, D>,
    pub(crate) residuals: residuals::ResidualsValues<D>,
    pub(crate) memory: nalgebra::DVector<f64>,
}

impl<D> Point<D>
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
{
    /// The point the model has just been evaluated on
    pub(crate) fn of<M>(model: &M) -> Self
    where
        M: model::Model<D>,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    {
        Point {
            iteratives: model.get_iteratives(),
            residuals: model.get_residuals(),
            memory: model.get_memory(),
        }
    }
}
