

pub fn solver_1D(init_guess: f64,
                    func: fn(f64)->f64,
                    deriv: fn(f64)->f64,
                    max_iter: usize,
                    tol: f64) -> f64 {

    let mut iter = 0;
    let mut res = func(init_guess)
    let mut error = res.abs();
    let mut guess = init_guess;
    while error < tol && iter < max_iter {
        iter += 1;
        guess -= res/deriv(guess);
        res = func(guess);
        error = res.abs();
    }
    guess
}

pub fn solver_1D_fd(init_guess: f64,
                    func: fn(f64)->f64,
                    max_iter: usize,
                    tol: f64,
                    dx: f64) {

    let mut iter = 0;
    let mut res = func(init_guess)
    let mut error = res.abs();
    let mut guess = init_guess;
    while error < tol && iter < max_iter {
        iter += 1;
        guess -= res/deriv(guess, res, func, dx);
        res = func(guess);
        error = res.abs();
    }
    guess
}

fn finite_diff(x: f64, f_ref: f64, func: fn(f64)->f64, dx: f64) -> f64 {
    fx = func(x + dx);
    (fx - f_ref)/dx
}
