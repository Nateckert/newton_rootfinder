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

If the solver speed is driven by the function evaluation.
The time taken for resolution should be in the same proportion
as for the function evaluation test

Reference results :
- Minimal 1D Solver :                              [37.099 ns 37.350 ns 37.602 ns]
- Minimal 1D Solver FD (finite differences) :      [60.788 ns 61.595 ns 62.524 ns]
- Newton-Raphson FD dynamic types:                 [686.11 ns 691.50 ns 697.20 ns]
- StationaryNewton FD dynamic types :              [712.83 ns 719.00 ns 725.14 ns]
- Newton-Raphson jacobian provided dynamic types:  [718.22 ns 724.01 ns 729.89 ns]
- Newton-Raphson FD static types:                  [16.666 ns 17.390 ns 18.298 ns]

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

## Solver reset between resolutions (2026-10-24)

Measured on an 11th Gen Intel Core i9-11950H at 2.60 GHz with 30 GB of memory, rustc 1.98.1,
with `cargo bench --all-features`.

Every bench of the solver reuses one `RootFinder` in its loop. Until the solver was reset at the start of
each resolution, the iteration count was kept from one resolution to the next, so after the first ones the
solver failed at once with `NonConvergenceError`: the results above, and the "before" column below, time a
resolution that stops after one evaluation. The bench with static types, which unwraps the result, panics
in that state.

| benchmark | before the reset | after |
|---|---|---|
| Solver parsing/NR | 28.145 µs | 29.105 µs |
| Advanced solver on Broyden test case 8/NR | 212.20 ns | 42.737 µs |
| Advanced solver on Broyden test case 8/SN | 212.67 ns | 18.244 µs |
| Advanced solver on Broyden test case 8/BROY1_jac | 216.15 ns | 51.829 µs |
| Advanced solver on Broyden test case 8/BROY2_jac | 214.03 ns | 97.793 µs |
| Advanced solver on Broyden test case 8/BROY1_inv | 219.81 ns | 31.233 µs |
| Advanced solver on Broyden test case 8/BROY2_inv | 215.47 ns | 33.272 µs |
| Advanced solver on Broyden test case 8/GRST1 | 227.28 ns | 77.621 µs |
| Advanced solver on Broyden test case 8/GRST2 | 224.40 ns | 52.608 µs |
| Advanced solver on Broyden test case 8/GRST1_inv | 212.38 ns | 29.024 µs |
| Advanced solver on Broyden test case 8/GRST2_inv | 213.91 ns | 32.621 µs |
| Function evaluation/nalgebra | 9.6228 ns | 9.4210 ns |
| Advanced solver with dynamic types/Newton-Raphson with finite differences | 105.82 ns | 1.4370 µs |
| Advanced solver with dynamic types/Stationary Newton with finite differences | 106.96 ns | 2.7255 µs |
| Advanced solver with dynamic types/Newton-Raphson with derivative provided | 102.88 ns | 1.2449 µs |
| Function evaluation/f64 | 399.06 ps | 388.07 ps |
| Minimal 1D solver/With derivative provided | 15.091 ns | 13.312 ns |
| Minimal 1D solver/Finite differences | 21.571 ns | 20.229 ns |
| Advanced solver with static types/Newton-Raphson with finite differences | panics (`Convergence not reached`) | 126.19 ns |

### Starting from the iteratives of the model (same day)

With the resolution starting from the iteratives the model holds, instead of a guess given to the
solver, the benches set the model back to its guess inside the timed loop: each measurement is one
resolution and one `set_iteratives`. Same machine, same benches.

| benchmark | after the reset | starting from the model |
|---|---|---|
| Solver parsing/NR | 29.105 µs | 25.992 µs |
| Advanced solver on Broyden test case 8/NR | 42.737 µs | 42.714 µs |
| Advanced solver on Broyden test case 8/SN | 18.244 µs | 18.749 µs |
| Advanced solver on Broyden test case 8/BROY1_jac | 51.829 µs | 52.173 µs |
| Advanced solver on Broyden test case 8/BROY2_jac | 97.793 µs | 98.451 µs |
| Advanced solver on Broyden test case 8/BROY1_inv | 31.233 µs | 33.057 µs |
| Advanced solver on Broyden test case 8/BROY2_inv | 33.272 µs | 34.644 µs |
| Advanced solver on Broyden test case 8/GRST1 | 77.621 µs | 78.270 µs |
| Advanced solver on Broyden test case 8/GRST2 | 52.608 µs | 53.949 µs |
| Advanced solver on Broyden test case 8/GRST1_inv | 29.024 µs | 30.979 µs |
| Advanced solver on Broyden test case 8/GRST2_inv | 32.621 µs | 33.191 µs |
| Advanced solver with dynamic types/Newton-Raphson with finite differences | 1.4370 µs | 1.4322 µs |
| Advanced solver with dynamic types/Stationary Newton with finite differences | 2.7255 µs | 2.7927 µs |
| Advanced solver with dynamic types/Newton-Raphson with derivative provided | 1.2449 µs | 1.2519 µs |
| Minimal 1D solver/With derivative provided | 13.312 ns | 13.078 ns |
| Minimal 1D solver/Finite differences | 20.229 ns | 20.538 ns |
| Advanced solver with static types/Newton-Raphson with finite differences | 126.19 ns | 118.23 ns |

## Step solved with the LU factorization of the jacobian (2026-11-15)

Measured on an 11th Gen Intel Core i9-11950H at 2.60 GHz with 30 GB of memory, rustc 1.98.1,
with `cargo bench --all-features`, the bench pinned to one core.
Each value is the lowest median of three runs, the runs of both versions alternated.

Until this change the jacobian was factorized, then inverted, and the step was the product by the inverse.
The step is now the solution of the factorized system: the inverse is formed only by the methods updating it,
once, at their first update.

### Jacobian provided, from 5 to 100 unknowns

Bench `linear_algebra`: the function of the Broyden cases 5 to 8, defined for any dimension, with its jacobian
provided, so that the linear algebra dominates as the dimension grows.

| method | unknowns | product by the inverse | solution of the factorized system | ratio |
|---|---|---|---|---|
| NR | 5 | 2.732 µs | 2.144 µs | 1.27 |
| NR | 10 | 7.982 µs | 4.010 µs | 1.99 |
| NR | 20 | 32.60 µs | 11.47 µs | 2.84 |
| NR | 50 | 246.4 µs | 70.61 µs | 3.49 |
| NR | 100 | 1.526 ms | 424.2 µs | 3.60 |
| SN | 5 | 2.673 µs | 2.558 µs | 1.04 |
| SN | 10 | 5.375 µs | 3.987 µs | 1.35 |
| SN | 20 | 15.55 µs | 7.849 µs | 1.98 |
| SN | 50 | 96.11 µs | 32.55 µs | 2.95 |
| SN | 100 | 493.6 µs | 142.8 µs | 3.46 |
| BROY1_jac | 5 | 4.058 µs | 3.254 µs | 1.25 |
| BROY1_jac | 10 | 14.16 µs | 6.935 µs | 2.04 |
| BROY1_jac | 20 | 55.80 µs | 18.44 µs | 3.03 |
| BROY1_jac | 50 | 427.8 µs | 116.1 µs | 3.69 |
| BROY1_jac | 100 | 2.586 ms | 713.7 µs | 3.62 |
| GRST2_jac | 5 | 4.377 µs | 3.682 µs | 1.19 |
| GRST2_jac | 10 | 14.60 µs | 7.876 µs | 1.85 |
| GRST2_jac | 20 | 56.61 µs | 20.36 µs | 2.78 |
| GRST2_jac | 50 | 430.4 µs | 121.4 µs | 3.55 |
| GRST2_jac | 100 | 2.603 ms | 730.4 µs | 3.56 |
| BROY1_inv | 5 | 3.310 µs | 3.300 µs | 1.00 |
| BROY1_inv | 10 | 8.561 µs | 8.030 µs | 1.07 |
| BROY1_inv | 20 | 23.91 µs | 20.66 µs | 1.16 |
| BROY1_inv | 50 | 150.9 µs | 130.7 µs | 1.15 |
| BROY1_inv | 100 | 813.4 µs | 743.2 µs | 1.09 |

### The other benches

| benchmark | product by the inverse | solution of the factorized system | ratio |
|---|---|---|---|
| Advanced solver on Broyden test case 8/NR | 42.70 µs | 20.86 µs | 2.05 |
| Advanced solver on Broyden test case 8/SN | 18.31 µs | 10.78 µs | 1.70 |
| Advanced solver on Broyden test case 8/BROY1_jac | 58.76 µs | 21.37 µs | 2.75 |
| Advanced solver on Broyden test case 8/BROY2_jac | 65.69 µs | 27.72 µs | 2.37 |
| Advanced solver on Broyden test case 8/BROY1_inv | 26.78 µs | 23.63 µs | 1.13 |
| Advanced solver on Broyden test case 8/BROY2_inv | 19.95 µs | 17.19 µs | 1.16 |
| Advanced solver on Broyden test case 8/GRST1 | 65.96 µs | 27.45 µs | 2.40 |
| Advanced solver on Broyden test case 8/GRST2 | 59.66 µs | 23.14 µs | 2.58 |
| Advanced solver on Broyden test case 8/GRST1_inv | 19.91 µs | 17.30 µs | 1.15 |
| Advanced solver on Broyden test case 8/GRST2_inv | 27.66 µs | 24.23 µs | 1.14 |
| Advanced solver with dynamic types/Newton-Raphson with finite differences | 1.446 µs | 1.374 µs | 1.05 |
| Advanced solver with dynamic types/Stationary Newton with finite differences | 2.863 µs | 2.697 µs | 1.06 |
| Advanced solver with dynamic types/Newton-Raphson with derivative provided | 1.245 µs | 1.207 µs | 1.03 |
| Advanced solver with static types/Newton-Raphson with finite differences | 118.7 ns | 113.8 ns | 1.04 |

### Systems of a few unknowns, static types

The factorization and its substitutions are written in the crate. With `nalgebra::LU` and its `solve`,
a resolution was slower than with the inverse on the systems of one to three unknowns, up to 66 % for the
stationary Newton method: the factorization is built by a call that is not inlined, and its triangular solves
permute the right-hand side in place and divide by the diagonal, when the product by an inverse
of a static dimension stays in the registers.
The factorization of the crate keeps the permutation as indices and the diagonal as its reciprocals.

Processor cycles of two million resolutions of `x_i^2 + 0.1 * x_(i+1) = 2.1` from 1, by finite differences,
static types, counted with `perf stat` (lowest of seven runs):

| method | unknowns | product by the inverse | factorization of `nalgebra` | factorization of the crate |
|---|---|---|---|---|
| NR | 1 | 1.002 G | 1.240 G | 0.899 G |
| NR | 2 | 3.100 G | 3.338 G | 2.574 G |
| NR | 3 | 5.252 G | 5.014 G | 4.458 G |
| NR | 6 | 16.69 G | 13.32 G | 12.53 G |
| SN | 1 | 2.221 G | 2.888 G | 2.235 G |
| SN | 2 | 4.476 G | 7.446 G | 4.449 G |
| SN | 3 | 6.340 G | 9.464 G | 6.834 G |
| SN | 6 | 17.31 G | 16.87 G | 13.52 G |

The stationary Newton method on three unknowns is the one case left slower, by 8 %:
it solves many times with one factorization, and a substitution chains its operations
where the product by an inverse runs them side by side.
