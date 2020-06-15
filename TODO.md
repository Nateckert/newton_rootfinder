## Testing

Implement each test with 3 variants:
  - finite-differences
  - jacobian function provided
  - jacobian calculated thanks to AD, for example with https://crates.io/crates/fwd_ad

Implement the tests of the paper cited in tests/common/spedicato1966.rs

Implement benchmarking for :
  - convergence ratio on a specific domain


## Functionality

Improve Error Handling
Implement others methods : Broyden, ... (see paper in tests/common/spedicato1966.rs)
Introduce debugging : others variables computation.
Make it more generic to avoid enforcing the function having to be nalgebra vectors

## Documentation

One can always improve it !

## Datastructure

Currently, the residuals are separated in the ResidualsValues (and Jacobian Values)
and the ResidualsConfig.
The values are returned by the model and the config are field of the solver.
However, the functions to compute the residuals are in-between them:
Computing the residuals is a method of the ResidualsConfig and takes as argument a ResidualsValues.
Computing the jacobian from the jacobians are a method of JacobianValues and takes as argument a ResidualsConfig.
