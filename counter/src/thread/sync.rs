use super::pool::{JobScheduler, Stopped};

// ANCHOR: FutexScheduler
pub struct FutexScheduler {
    /// Number of tasks to be grabbed or request to stop
    ///
    /// In normal operation, when a job is submitted, the main thread sets this
    /// to to the number of worker threads (which is `num_threads - 1`) and
    /// worker threads decrement it as they accept their tasks. Once this
    /// reaches zero, it means all tasks to be processed have been taken care of
    /// and next wait_for_task() callers should go to sleep.
    ///
    /// To request threads to stop, this is set to `u32::MAX`. In general,
    /// anything above `Self::STOP_THRESHOLD` indicates that threads should
    /// stop, this flexibility allows making cancelation free on the fast path.
    ///
    /// Provides Acquire/Release synchronization with publisher of info read.
    ///
    task_futex: std::sync::atomic::AtomicU32,

    /// Requested or ongoing computation
    request: atomic::Atomic<u64>,
}
//
impl JobScheduler for FutexScheduler {
    fn new(num_threads: u32) -> Self {
        use atomic::Atomic;
        use std::sync::atomic::AtomicU32;
        assert!(
            num_threads <= Self::STOP_THRESHOLD,
            "{} threads ought to be enough for anybody",
            Self::STOP_THRESHOLD
        );
        Self {
            task_futex: AtomicU32::new(0),
            request: Atomic::default(),
        }
    }

    const MIN_TARGET: u64 = 0;

    fn start(&self, target: u64, num_threads: u32) {
        use atomic::Ordering;

        // Publish the target, that's not synchronization-critical
        self.request.store(target, Ordering::Relaxed);

        // Publish one task per worker thread, making sure that worker threads
        // have not stopped on the other side of the pipeline.
        debug_assert!(num_threads < Self::STOP_THRESHOLD);
        self.task_futex
            .fetch_update(Ordering::Release, Ordering::Relaxed, |old| {
                assert!(
                    old < Self::STOP_THRESHOLD,
                    "Can't schedule this new job, some workers have stopped"
                );
                debug_assert_eq!(old, Self::NO_TASK);
                Some(num_threads - 1)
            })
            .unwrap();

        // Wake up worker threads
        atomic_wait::wake_all(&self.task_futex);
    }

    fn stop(&self) {
        self.task_futex.store(u32::MAX, atomic::Ordering::Release);
        atomic_wait::wake_all(&self.task_futex);
    }

    fn stopped(&self) -> bool {
        self.task_futex.load(atomic::Ordering::Relaxed) >= Self::STOP_THRESHOLD
    }

    fn wait_for_task(&self) -> Result<u64, Stopped> {
        use atomic::Ordering;
        loop {
            // Wait for a request to come in
            if self.faillible_spin::<false>(|| !self.no_task()).is_none() {
                atomic_wait::wait(&self.task_futex, Self::NO_TASK);
                if self.no_task() {
                    continue;
                }
            }

            // Acknowledge the request, check for concurrent stop signals
            let prev_started = self.task_futex.fetch_sub(1, Ordering::Acquire);
            debug_assert!(prev_started > Self::NO_TASK);
            if prev_started < Self::STOP_THRESHOLD {
                return Ok(self.request.load(Ordering::Relaxed));
            } else {
                return Err(Stopped);
            }
        }
    }

    fn wait_for_started(&self) {
        // No need to handle the stop signal at this stage, it will be caught
        // on the next call to wait_for_task().
        std::mem::drop(self.faillible_spin_wait(|| self.no_task()));
    }
}
//
impl FutexScheduler {
    /// This value of `task_futex` means there is no work to be done and worker
    /// threads should spin then sleep waiting for tasks to come up
    const NO_TASK: u32 = 0;

    /// Values of `task_futex` higher than this means all threads should stop
    const STOP_THRESHOLD: u32 = u32::MAX / 2;

    /// Truth that all scheduled tasks have been started
    fn no_task(&self) -> bool {
        self.task_futex.load(atomic::Ordering::Relaxed) == Self::NO_TASK
    }
}
// ANCHOR_END: FutexScheduler

#[cfg(test)]
mod tests {
    use super::FutexScheduler;
    use crate::{test_utils, thread::pool::ThreadPool};
    use once_cell::sync::Lazy;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;
    use std::{panic::RefUnwindSafe, sync::Mutex};

    type CounterBox = Box<dyn Fn(u64) -> u64 + RefUnwindSafe + Send + Sync + 'static>;
    static BKG_THREADS_FUTEX: Lazy<Mutex<ThreadPool<CounterBox, FutexScheduler>>> =
        Lazy::new(|| {
            Mutex::new(ThreadPool::start(
                Box::new(crate::simd::multiversion::multiversion_avx2) as _,
            ))
        });

    #[quickcheck]
    fn thread_futex(target: u32) -> TestResult {
        let mut lock = BKG_THREADS_FUTEX.lock().unwrap();
        test_utils::test_counter(target, |target| lock.count(target))
    }
}
