use std;
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
/// - unrecoverable (but not a panic), the algorithm stops if that happens.
///
/// The `ModelError` enum is wrapping the `Error` trait from the standard library,
/// so that the user can sub-classify its error, such as:
/// - `InaccurateValuesError(Box<OutOfValidityRangeError>)`
/// - `InaccurateValuesError(Box<ExtrapolationError>)`
///
/// The rootfinder algorithm will not act on the subclassification,
/// but the information can be reported in errors logs.
pub enum ModelError {
    InaccurateValuesError(Box<dyn std::error::Error>),
    InvalidValuesError(Box<dyn std::error::Error>),
    UnrecoverableError(Box<dyn std::error::Error>)
}