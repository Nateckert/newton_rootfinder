# Benchmarking results and history

Test performed on a MacBook Pro mid-2012:
- CPU: 2.5 GHz Intel Core i5 double core
- Memory: 8Go, 1600 MHz DDR3

## Root square case

Trying to find the solution to x^2 = 2.

### Model definition evaluation

Functions evaluation time measurements (no changes over time expected)
As the solver is using nalgebra types (to work with n-dimensional problems),
that are more complex than the basic f64,
it can cause an overhead that is measured.

Results : f64 function is 137 times faster than DVector (expected)
- f64 :  [784.26 ps 792.64 ps 802.17 ps]
- nalgebra : [108.38 ns 109.31 ns 110.26 ns]

### Solver speeds

If the solver speed is driven by the function evaluation,
The time taken for resolution should be in the same proportion
as for the function evaluation test

Reference results :
- Solver 1D:                              [37.099 ns 37.350 ns 37.602 ns]
- Solver 1D FD:                           [60.788 ns 61.595 ns 62.524 ns]
- Advanced solver FD:                     [686.11 ns 691.50 ns 697.20 ns]
- Advanced solver FD StationaryNewton :   [712.83 ns 719.00 ns 725.14 ns]
- Advanced solver FD jacobian provided :  [718.22 ns 724.01 ns 729.89 ns]

Without derivatives is 1.6 times faster than with
Minimal solver is 11 times faster than advanced solver
Expected times was 137 times
The advanced solver is roughly 10 times faster than the minimal implementation

## Broyden case 8

Benchmarking on Broyden test case 8

### Xml parsing speed

Only one method has been benchmarked

- NewtonRaphson-FD: [397.90 us 399.15 us 400.39 us]

### Resolution speed

- NewtonRaphson-FD:             [812.33 ns 818.59 ns 825.31 ns]
- StationaryNewton-FD:          [829.89 ns 836.45 ns 843.19 ns]
- BroydenFirstMethod-FD:        [828.63 ns 834.22 ns 839.66 ns]
- BroydenSecondMethod-FD:       [814.20 ns 820.57 ns 827.17 ns]
- BroydenFirstMethod_INV-FD:    [819.56 ns 827.18 ns 835.29 ns]
- BroydenSecondMethod_INV-FD:   [826.59 ns 831.91 ns 837.23 ns]
