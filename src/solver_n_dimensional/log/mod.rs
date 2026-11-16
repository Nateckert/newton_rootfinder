//! The log of the resolutions
//!
//! A [Log] records what the solvers do, as it happens, in [JSON Lines](https://jsonlines.org/):
//! one JSON object per line, its `event` naming its kind.
//! A script reads it with its standard library (`json` in Python, `JSON.parse` in JavaScript, `jq`).
//!
//! One log serves every resolution of a program: a [Log] is a handle,
//! cheap to clone and to send to another thread, each clone writing to the same file.
//! A solver writes in it once given it with [crate::solver::RootFinder::log_to].
//!
//! The model adds what the solver does not know: the names of its iteratives and residuals
//! ([crate::model::Model::describe]), the values an evaluation computed besides them
//! ([crate::model::Model::report]), and the message of an evaluation it finds inaccurate or refuses.
//!
//! ```no_run
//! use newton_rootfinder as nrf;
//! # use nrf::iteratives;
//! # use nrf::residuals;
//! # use nrf::solver::ResolutionMethod;
//! use nrf::log::{Detail, Log};
//! # pub fn square2(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
//! #   let mut y = x * x;
//! #   y[0] -= 2.0;
//! #   y
//! # }
//! # let problem_size = 1;
//! # let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
//! # let iter_params = iteratives::Iteratives::new(&vec_iter_params);
//! # let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
//! # let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
//! # let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
//! # let mut user_model = nrf::model::UserModelFromFunction::new(problem_size, square2);
//! # nrf::model::Model::set_iteratives(&mut user_model, &nalgebra::DVector::from_vec(vec![1.0]));
//!
//! let log = Log::create("solver_log.jsonl")?.detail(Detail::Jacobians);
//! log.note("run", |fields| {
//!     fields.text("case", "square root of 2");
//! });
//!
//! let mut rf = nrf::solver::default(&iter_params, &res_config, ResolutionMethod::NewtonRaphson, false);
//! rf.log_to(&log, "square root");
//! rf.solve(&mut user_model).unwrap();
//!
//! log.close()?;
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! # The lines
//!
//! A Newton-Raphson resolution converging in one iteration, with the detail [Detail::Jacobians]:
//!
//! ```text
//! {"event":"log","format":1,"version":"0.13.0","opened":"2026-11-16T09:12:03.120Z"}
//! {"event":"start","id":1,"parent":null,"time":0.000004,"label":"block 0","method":"newton_raphson","tolerance":1e-6,"max_iterations":60,"damping":true,"iteratives":[{"settings":{"max_step_abs":"Infinity","max_step_rel":"Infinity","min_value":"-Infinity","max_value":"Infinity","perturbation":"max","dx_abs":5e-8,"dx_rel":5e-8}}],"residuals":[{"settings":{"stopping":"abs","update":"abs"}}]}
//! {"event":"evaluation","id":1,"time":0.000009,"iteration":0,"of":"guess","iteratives":[1.0],"left":[-1.0],"right":[2.0],"errors":[3.0],"max_error":3.0}
//! {"event":"jacobian","id":1,"time":0.000012,"iteration":1,"how":"finite_differences","matrix":[[-1.0]]}
//! {"event":"step","id":1,"time":0.000013,"iteration":1,"step":[-3.0]}
//! {"event":"evaluation","id":1,"time":0.000015,"iteration":1,"of":"step","iteratives":[-2.0],"left":[-4.0],"right":[-4.0],"errors":[0.0],"max_error":0.0}
//! {"event":"end","id":1,"time":0.000016,"outcome":"converged","iterations":1,"max_error":0.0}
//! ```
//!
//! | Event | Written | Fields beside `id` and `time` |
//! |---|---|---|
//! | `log` | once, first line | `format`, the version of the crate, `opened`: the time the log was opened, UTC |
//! | `scope`, `scope_end` | a [Scope] opened and closed | `parent`, `label` |
//! | `note` | [Log::note] | `parent`, `label`, `fields` |
//! | `start` | [crate::solver::RootFinder::solve] called | `parent`, `during`, `label`, `method`, `tolerance`, `max_iterations`, `damping`, for each iterative and each residual its `name` and `details` given by the model, and its `settings` |
//! | `evaluation` | the model evaluated at the guess, at a step or at a damped step | `iteration`, `of` (`guess`, `step`, `damping`), `iteratives`, `left` and `right` sides of the residuals, `errors` (the stopping criteria), `max_error`, `inaccurate`: the message of the model finding its values inaccurate, `values` the model reports; or `iteratives` and `error`, the message of the model, when the evaluation failed |
//! | `step` | a step computed | `iteration`, `step`: the solution of the linear system, before the limits of the iteratives; the evaluation that follows shows the iteratives proposed |
//! | `jacobian` | the jacobian computed, approximated or its inverse updated | `iteration`, `how` (`finite_differences`, `analytic` or the update method), `matrix` or `inverse`; `singular`: `true` for a matrix that could not be factorized |
//! | `refused` | a step of a quasi-Newton method refused | `iteration`: the jacobian is computed at the next one |
//! | `end` | `solve` returns | `outcome`, `message` for a failure, `iterations`, `max_error` of the last evaluation; `residuals`, the indices of the non-finite ones, for `non_finite_residuals` |
//!
//! - `id` names a scope or a resolution, unique in the log. `parent` is the scope or the resolution in progress
//!   on the same thread when a scope, a resolution or a note starts, `null` at the top:
//!   a resolution started by the evaluation of a model, inside another one, is placed under it,
//!   `during` the evaluation of its parent it belongs to: `{"iteration":3,"of":"jacobian","column":2}`
//!   for the evaluation of the third column of a jacobian by finite differences, at the third iteration.
//! - `time`: seconds since the log was opened, from a monotonic clock.
//! - `outcome`: `converged`, or the variant of [crate::errors::SolverError] in snake case
//!   (`non_convergence_error`, `model_evaluation_error`, ...), its message the `Display` of the error.
//!   A resolution interrupted by a panic ends with `panic`.
//! - The numbers read back to the same `f64`. A non-finite value is the string `"NaN"`, `"Infinity"` or `"-Infinity"`,
//!   which `float()` in Python and `Number()` in JavaScript read.
//! - The method is `newton_raphson`, `stationary_newton`, or the update method,
//!   `broyden_first_method` for instance, followed by `_inverse` when the inverse is updated.
//!   The normalization of a residual is `abs`, `rel` or `adapt`.
//! - A reader ignores the events and the fields it does not know:
//!   the lines may gain some without a change of `format`.
//!
//! # The failures only
//!
//! A log built with [Log::failures_only] keeps the lines of each scope and resolution in memory until it ends:
//! a resolution that converged leaves nothing, nor a scope where none failed.
//! When a resolution fails, the lines of the scopes and resolutions it is in are written first,
//! then its own: the file keeps the order of the events, and a run of many resolutions,
//! a sweep or a transient, leaves the failures alone.
//!
//! # Writing
//!
//! The log writes through a buffer, flushed when a resolution fails,
//! when the outermost resolution or scope of a thread ends, and when the last handle is dropped.
//! An error of writing stops the log, never the resolution: [Log::close] returns it.
//!
//! With the log off, a resolution pays a test per event, and keeps the jacobian as its factorization alone.

mod clock;
mod description;
mod json;
mod resolution;
#[cfg(test)]
mod tests;

pub use description::{Description, Entry};
pub(crate) use resolution::{Evaluation, JacobianSource, ResolutionLog};

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, ThreadId};
use std::time::Instant;

/// The version of the lines, in the first one: raised when the meaning of an event or a field changes
const FORMAT: u32 = 1;

/// How much a resolution writes, beside its `start` and `end` lines
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Detail {
    /// The `start` and the `end` of each resolution
    Outcome,
    /// Also each evaluation of the model, each step and each refused step
    #[default]
    Iterations,
    /// Also each jacobian: n² numbers per iteration
    Jacobians,
}

impl Detail {
    fn from_u8(value: u8) -> Detail {
        match value {
            0 => Detail::Outcome,
            1 => Detail::Iterations,
            _ => Detail::Jacobians,
        }
    }
}

/// A log of resolutions, written in JSON Lines: see the [module documentation](self)
///
/// A handle: its clones write to the same file, from any thread.
#[derive(Clone)]
pub struct Log {
    shared: Arc<Shared>,
}

struct Shared {
    opened: Instant,
    detail: AtomicU8,
    failures_only: AtomicBool,
    state: Mutex<State>,
}

struct State {
    writer: Option<Box<dyn Write + Send>>,
    error: Option<io::Error>,
    next_id: u64,
    /// The scopes and resolutions in progress, innermost last, per thread
    open: HashMap<ThreadId, Vec<Open>>,
}

impl Log {
    /// A log written to the file at `path`, created or truncated
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<Log> {
        Ok(Log::new(File::create(path)?))
    }

    /// A log written to `writer`
    pub fn new<W: Write + Send + 'static>(writer: W) -> Log {
        let log = Log {
            shared: Arc::new(Shared {
                opened: Instant::now(),
                detail: AtomicU8::new(Detail::default() as u8),
                failures_only: AtomicBool::new(false),
                state: Mutex::new(State {
                    writer: Some(Box::new(BufWriter::new(writer))),
                    error: None,
                    next_id: 1,
                    open: HashMap::new(),
                }),
            }),
        };

        let mut line = String::from("{\"event\":\"log\",\"format\":");
        json::unsigned(&mut line, u64::from(FORMAT));
        line.push_str(",\"version\":");
        json::string(&mut line, env!("CARGO_PKG_VERSION"));
        line.push_str(",\"opened\":");
        match clock::utc_now() {
            Some(time) => json::string(&mut line, &time),
            None => line.push_str("null"),
        }
        line.push('}');
        log.lock().write_line(&line);
        log
    }

    /// The detail of the resolutions started from now on, by every handle of the log
    pub fn detail(self, detail: Detail) -> Log {
        self.shared.detail.store(detail as u8, Ordering::Relaxed);
        self
    }

    /// Keep the lines of the resolutions that fail only, for the scopes and resolutions started from now on
    ///
    /// The lines of a scope or a resolution are kept in memory until it ends:
    /// a resolution that converged, or a scope where none failed, leaves nothing.
    /// When a resolution fails, the lines of the scopes and resolutions it is in are written first,
    /// these writing directly from then on.
    pub fn failures_only(self) -> Log {
        self.shared.failures_only.store(true, Ordering::Relaxed);
        self
    }

    /// Open a scope: the scopes, resolutions and notes started on this thread until it is dropped are placed under it
    #[must_use = "the scope closes when dropped"]
    pub fn scope(&self, label: &str) -> Scope {
        let id = self.open(|line, id, parent, _, time| {
            line.push_str("{\"event\":\"scope\",\"id\":");
            json::unsigned(line, id);
            json::key(line, "parent");
            json::optional_unsigned(line, parent);
            json::key(line, "time");
            json::number(line, time);
            json::key(line, "label");
            json::string(line, label);
            line.push('}');
        });
        Scope {
            log: self.clone(),
            id,
        }
    }

    /// Write facts the solver does not know, under the scope or the resolution in progress on this thread
    ///
    /// ```
    /// # let log = newton_rootfinder::log::Log::new(std::io::sink());
    /// log.note("runner", |fields| {
    ///     fields.text("os", std::env::consts::OS).number("threads", 4.0);
    /// });
    /// ```
    pub fn note<F>(&self, label: &str, fields: F)
    where
        F: FnOnce(&mut Fields),
    {
        let time = self.elapsed();
        let mut state = self.lock();
        let parent = state.innermost();
        let mut line = String::from("{\"event\":\"note\",\"parent\":");
        json::optional_unsigned(&mut line, parent);
        json::key(&mut line, "time");
        json::number(&mut line, time);
        json::key(&mut line, "label");
        json::string(&mut line, label);
        json::key(&mut line, "fields");
        let mut writer = Fields::open(&mut line);
        fields(&mut writer);
        writer.close();
        line.push('}');
        state.emit(parent, line);
    }

    /// Flush the log, and return the first error of writing, if any:
    /// the log stopped writing at that error.
    ///
    /// The other handles of the log can still write.
    pub fn close(self) -> io::Result<()> {
        let mut state = self.lock();
        state.flush();
        match state.error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub(crate) fn get_detail(&self) -> Detail {
        Detail::from_u8(self.shared.detail.load(Ordering::Relaxed))
    }

    fn elapsed(&self) -> f64 {
        self.shared.opened.elapsed().as_secs_f64()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // A panic while the lock was held leaves at worst a line cut short: the log goes on
        self.shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Open a scope or a resolution: its line is built from its id, its parent and the time
    fn open<F>(&self, build: F) -> u64
    where
        F: FnOnce(&mut String, u64, Option<u64>, Option<During>, f64),
    {
        let time = self.elapsed();
        let kept = self.shared.failures_only.load(Ordering::Relaxed);
        let mut state = self.lock();
        let id = state.next_id;
        state.next_id += 1;
        let stack = state.open.entry(thread::current().id()).or_default();
        let (parent, during) = match stack.last() {
            Some(open) => (Some(open.id), open.during),
            None => (None, None),
        };
        let mut line = String::with_capacity(256);
        build(&mut line, id, parent, during, time);
        if kept {
            stack.push(Open {
                id,
                during: None,
                kept: Some(vec![line]),
            });
        } else {
            stack.push(Open {
                id,
                during: None,
                kept: None,
            });
            state.write_line(&line);
        }
        id
    }

    /// The evaluation the resolution `id` is making, innermost on this thread:
    /// the resolutions started inside it are placed in it
    fn evaluating(&self, id: u64, during: During) {
        let mut state = self.lock();
        if let Some(open) = state
            .open
            .get_mut(&thread::current().id())
            .and_then(|stack| stack.last_mut())
            && open.id == id
        {
            open.during = Some(during);
        }
    }

    /// Write an event of the scope or the resolution `id`, in progress
    fn write(&self, id: u64, line: String) {
        self.lock().emit(Some(id), line);
    }

    /// Close a scope or a resolution with its last line: `failed` for a resolution that did not converge
    fn close_node(&self, id: u64, line: String, failed: bool) {
        let mut state = self.lock();
        match state.remove(id) {
            Some((thread, open)) => {
                match open.kept {
                    Some(mut lines) => {
                        if failed {
                            state.write_kept(thread);
                            lines.push(line);
                            for line in &lines {
                                state.write_line(line);
                            }
                        }
                    }
                    None => state.write_line(&line),
                }
                if failed || !state.open.contains_key(&thread) {
                    state.flush();
                }
            }
            None => {
                state.write_line(&line);
                state.flush();
            }
        }
    }
}

impl fmt::Debug for Log {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Log")
            .field("detail", &self.get_detail())
            .finish()
    }
}

impl State {
    fn write_line(&mut self, line: &str) {
        if let Some(writer) = &mut self.writer {
            let result = writer
                .write_all(line.as_bytes())
                .and_then(|()| writer.write_all(b"\n"));
            if let Err(error) = result {
                self.stop(error);
            }
        }
    }

    fn flush(&mut self) {
        if let Some(writer) = &mut self.writer
            && let Err(error) = writer.flush()
        {
            self.stop(error);
        }
    }

    fn stop(&mut self, error: io::Error) {
        self.writer = None;
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    /// Keep the line for the scope or the resolution `id` when it keeps its lines, else write it
    fn emit(&mut self, id: Option<u64>, line: String) {
        if let Some(id) = id
            && let Some(lines) = self.find(id).and_then(|open| open.kept.as_mut())
        {
            lines.push(line);
        } else {
            self.write_line(&line);
        }
    }

    /// The scope or the resolution `id` in progress, looked for on the current thread first
    fn find(&mut self, id: u64) -> Option<&mut Open> {
        let current = thread::current().id();
        let thread = if self
            .open
            .get(&current)
            .is_some_and(|stack| stack.iter().any(|open| open.id == id))
        {
            current
        } else {
            *self
                .open
                .iter()
                .find(|(_, stack)| stack.iter().any(|open| open.id == id))?
                .0
        };
        self.open
            .get_mut(&thread)?
            .iter_mut()
            .rev()
            .find(|open| open.id == id)
    }

    /// Write the lines the scopes and resolutions in progress on `thread` kept, the outermost first:
    /// they write directly from then on
    fn write_kept(&mut self, thread: ThreadId) {
        let mut lines = Vec::new();
        if let Some(stack) = self.open.get_mut(&thread) {
            for open in stack.iter_mut() {
                if let Some(kept) = open.kept.take() {
                    lines.extend(kept);
                }
            }
        }
        for line in &lines {
            self.write_line(line);
        }
    }

    fn innermost(&self) -> Option<u64> {
        self.open
            .get(&thread::current().id())
            .and_then(|stack| stack.last())
            .map(|open| open.id)
    }

    /// Remove a scope or a resolution from the ones in progress,
    /// looked for on the current thread first, a scope may be dropped on another one.
    /// Returns its thread, without its entry when no other one is in progress on it
    fn remove(&mut self, id: u64) -> Option<(ThreadId, Open)> {
        let current = thread::current().id();
        let thread = if self
            .open
            .get(&current)
            .is_some_and(|stack| stack.iter().any(|open| open.id == id))
        {
            Some(current)
        } else {
            self.open
                .iter()
                .find(|(_, stack)| stack.iter().any(|open| open.id == id))
                .map(|(thread, _)| *thread)
        };
        let thread = thread?;
        let stack = self.open.get_mut(&thread)?;
        let position = stack.iter().position(|open| open.id == id)?;
        let open = stack.remove(position);
        if stack.is_empty() {
            self.open.remove(&thread);
        }
        Some((thread, open))
    }
}

/// A scope or a resolution in progress
struct Open {
    id: u64,
    during: Option<During>,
    /// Its lines, kept until it ends, when the log keeps the failures only
    kept: Option<Vec<String>>,
}

/// The evaluation a resolution is making: `of` the guess, a step, a damped step or the jacobian,
/// `column` the column of a jacobian by finite differences
#[derive(Debug, Clone, Copy)]
pub(crate) struct During {
    pub(crate) iteration: usize,
    pub(crate) of: &'static str,
    pub(crate) column: Option<usize>,
}

/// A scope of a [Log], closed when dropped: see [Log::scope]
pub struct Scope {
    log: Log,
    id: u64,
}

impl Drop for Scope {
    fn drop(&mut self) {
        let mut line = String::from("{\"event\":\"scope_end\",\"id\":");
        json::unsigned(&mut line, self.id);
        json::key(&mut line, "time");
        json::number(&mut line, self.log.elapsed());
        line.push('}');
        self.log.close_node(self.id, line, false);
    }
}

impl fmt::Debug for Scope {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Scope").field("id", &self.id).finish()
    }
}

/// Named values written in a line of the log, as a JSON object: see [Log::note]
///
/// A name given twice is written twice: the readers keep one of them.
#[derive(Debug)]
pub struct Fields<'a> {
    line: &'a mut String,
    empty: bool,
}

impl<'a> Fields<'a> {
    pub(crate) fn open(line: &'a mut String) -> Self {
        line.push('{');
        Fields { line, empty: true }
    }

    pub(crate) fn close(self) {
        self.line.push('}');
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.empty
    }

    fn name(&mut self, name: &str) {
        if !self.empty {
            self.line.push(',');
        }
        self.empty = false;
        json::string(self.line, name);
        self.line.push(':');
    }

    /// A number, non-finite ones included
    pub fn number(&mut self, name: &str, value: f64) -> &mut Self {
        self.name(name);
        json::number(self.line, value);
        self
    }

    /// A text
    pub fn text(&mut self, name: &str, value: &str) -> &mut Self {
        self.name(name);
        json::string(self.line, value);
        self
    }
}
