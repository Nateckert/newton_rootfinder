## Testing

Implement each test with 3 variants:
  - finite-differences
  - jacobian function provided
  - jacobian calculated thanks AD, for example with https://crates.io/crates/fwd_ad

Implement the tests of the paper cited in tests/common/spedicato1966.rs

Implement benchmarking for :
  - convergence ratio on a specific domain
  - resolution speed

## Functionality

Improve Error Handling
Implement others methods : Broyden, ... (see paper in tests/common/spedicato1966.rs)
Make the solver available from Python
Introduce substitution methods
Introduce inequalities
Introduce damping
Introduce debugging : solver log and others variables computation.
Make it more generic to avoid enforcing the function having to be nalgebra vectors
Introduce reading xml configuration files for iteratives and residuals

## Documentation

Add references to literature
Document advanced solver usage
