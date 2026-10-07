//! A dedicated inference thread that owns the [`Model`].
//!
//! MLX arrays are `Send` but not `Sync`, and MLX streams are per-thread, so the
//! model lives on exactly one thread for its whole life and work is shipped to
//! it as closures. Two lanes keep the app responsive:
//!
//! - **priority**: interactive work (search queries, explain). Always drained
//!   first.
//! - **background**: indexing batches. Taken only when the priority lane is
//!   empty, so a search issued mid-import waits for at most one in-flight batch
//!   rather than the whole file.
//!
//! Results come back on a runtime-agnostic oneshot, so async callers can
//! `.await` them and blocking callers can `block_on`/`recv` them.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{anyhow, Result};
use futures_channel::oneshot;
use tokenizers::Tokenizer;

use crate::Model;

type Job = Box<dyn FnOnce(&Model) + Send>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    Priority,
    Background,
}

#[derive(Default)]
struct Queues {
    priority: VecDeque<Job>,
    background: VecDeque<Job>,
}

struct Shared {
    queues: Mutex<Queues>,
    ready: Condvar,
}

/// Cheap-to-clone handle to the inference thread.
#[derive(Clone)]
pub struct EmbedService {
    shared: Arc<Shared>,
    tokenizer: Arc<Tokenizer>,
    dim: usize,
}

impl EmbedService {
    /// Spawn the inference thread, load the model on it, and run a warmup pass
    /// (Metal pipeline compilation) before any queued job. Returns once the
    /// weights are loaded so load errors surface here; warmup continues in the
    /// background ahead of the first job.
    pub fn spawn(model_dir: PathBuf, dim: usize) -> Result<Self> {
        let shared = Arc::new(Shared {
            queues: Mutex::new(Queues::default()),
            ready: Condvar::new(),
        });
        let (loaded_tx, loaded_rx) = std::sync::mpsc::channel::<Result<(Tokenizer, usize)>>();
        let worker = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("semantra-embed".into())
            .spawn(move || {
                let model = match Model::load(&model_dir, dim) {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = loaded_tx.send(Err(e));
                        return;
                    }
                };
                let _ = loaded_tx.send(Ok((model.tokenizer().clone(), model.embedding_dim())));
                if let Err(e) = model.warmup() {
                    eprintln!("[semantra-embed] warmup failed: {e}");
                }
                run_loop(&worker, &model);
            })?;
        let (tokenizer, dim) = loaded_rx
            .recv()
            .map_err(|_| anyhow!("embedding thread exited before loading the model"))??;
        Ok(EmbedService {
            shared,
            tokenizer: Arc::new(tokenizer),
            dim,
        })
    }

    /// The model's tokenizer (shared; safe to use from any thread), for
    /// token-aware chunking without a round trip to the inference thread.
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    pub fn embedding_dim(&self) -> usize {
        self.dim
    }

    /// Queue `f` on `lane`; resolves with its result once the inference thread
    /// has run it.
    pub fn submit<R, F>(&self, lane: Lane, f: F) -> oneshot::Receiver<Result<R>>
    where
        R: Send + 'static,
        F: FnOnce(&Model) -> Result<R> + Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let job: Job = Box::new(move |m| {
            let _ = tx.send(f(m));
        });
        let mut q = self.shared.queues.lock().unwrap();
        match lane {
            Lane::Priority => q.priority.push_back(job),
            Lane::Background => q.background.push_back(job),
        }
        drop(q);
        self.shared.ready.notify_one();
        rx
    }

    /// [`submit`](Self::submit) and await the result.
    pub async fn run<R, F>(&self, lane: Lane, f: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&Model) -> Result<R> + Send + 'static,
    {
        self.submit(lane, f)
            .await
            .map_err(|_| anyhow!("embedding thread stopped"))?
    }

    /// [`submit`](Self::submit) and block the calling thread for the result.
    /// For use from blocking contexts (never from the inference thread itself).
    pub fn run_blocking<R, F>(&self, lane: Lane, f: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&Model) -> Result<R> + Send + 'static,
    {
        futures_executor::block_on(self.run(lane, f))
    }
}

/// Runs for the life of the process (the app never tears the model down).
fn run_loop(shared: &Shared, model: &Model) -> ! {
    loop {
        let job = {
            let mut q = shared.queues.lock().unwrap();
            loop {
                if let Some(j) = q.priority.pop_front().or_else(|| q.background.pop_front()) {
                    break j;
                }
                q = shared.ready.wait(q).unwrap();
            }
        };
        job(model);
    }
}
