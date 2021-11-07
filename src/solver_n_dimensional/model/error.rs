use std::error::Error;
use std::fmt;
/// User model error definition
///
/// If the user model raise an error,
/// depending of the phase during which it happens,
/// it can be recoverable or not.
///
/// From the algorithms point of view, there are three phases:
/// - initialization
/// - iterations
/// - final evaluation
///
/// User error are categorized as the following:
///
/// - numerical values exist but are inaccurate:
///   - recoverable except if it happens in the final evaluation,
///      in this case the algorithm can continue with hope it will recover before the final evaluation
///   - the source of such an error could be an out of validity range on a component of the user model
/// - unusable numerical values such as NaN, None, defaults values or random values.
///   - recoverable if it happens during the iterations phase,
///     the algorithm will forget this iteration and try to perform a new one
///     with slightly changed inputs
///
/// The `ModelError` enum is wrapping the `Error` trait from the standard library,
/// so that the user can sub-classify its error, such as:
/// - `InaccurateValuesError(Box<OutOfValidityRangeError>)`
/// - `InaccurateValuesError(Box<ExtrapolationError>)`
///
/// The rootfinder algorithm will not act on the subclassification,
/// but the information can be reported in errors logs.
pub enum ModelError<M, D>
where
    M: super::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
    InaccurateValuesError(M::InaccurateValuesError),
    UnusableValuesError(M::UnusableValuesError),
}

impl<M, D> fmt::Display for ModelError<M, D>
where
    M: super::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InaccurateValuesError(error) => write!(f, "InaccurateValues Error: {}", error),
            Self::UnusableValuesError(error) => write!(f, "UnusableValuesError Error: {}", error),
        }
    }
}

impl<M, D> fmt::Debug for ModelError<M, D>
where
    M: super::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<M, D> Error for ModelError<M, D>
where
    M: super::Model<D>,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D, D>,
{
}
