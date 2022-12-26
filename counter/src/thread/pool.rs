// ANCHOR: spin_loop
/// Spin until a condition is validated
///
/// Start with busy waiting in userpace with a cheap condition check. For longer
/// waits, burn less CPU cycles by yielding to the OS and allow the waiter to
/// switch to a more expensive check along the way.
///
/// Unless INFINITE is true, give up after a while and return None to allow for
/// proper OS-controlled blocking to take place.
///
/// Otherwise, only Some(result) options may be returned.
///
fn spin_loop<const INFINITE: bool, Ready>(
    mut cheap_check: impl FnMut() -> Option<Ready>,
    mut expensive_check: impl FnMut() -> Option<Ready>,
) -> Option<Ready> {
    // Tuning parameters, empirically optimal on my machine
    use std::time::{Duration, Instant};
    const SPIN_ITERS: usize = 300;
    const MAX_BACKOFF: usize = 1 << 2;
    const OS_SPIN_DELAY: Duration = Duration::from_nanos(1);
    const OS_SPIN_BOUND: Duration = Duration::from_micros(20);

    // Start with a userspace busy loop with a bit of exponential backoff
    let mut backoff = 1;
    for _ in 0..SPIN_ITERS {
        if let Some(ready) = cheap_check() {
            return Some(ready);
        }
        for _ in 0..backoff {
            std::hint::spin_loop();
        }
        backoff = (2 * backoff).min(MAX_BACKOFF);
    }

    // Switch to yielding to the OS once it's clear it's gonna take a while to
    // reduce our CPU consumption at the cost of higher wakeup latency
    macro_rules! yield_iter {
        () => {
            // Check if the condition is now met
            if let Some(ready) = expensive_check() {
                return Some(ready);
            }

            // yield_now() would be semantically more correct for this situation
            // but is broken on Linux as the CFS scheduler just reschedules us.
            std::thread::sleep(OS_SPIN_DELAY);
        };
    }
    //
    if INFINITE {
        loop {
            yield_iter!();
        }
    } else {
        let start = Instant::now();
        while start.elapsed() < OS_SPIN_BOUND {
            yield_iter!();
        }
        expensive_check()
    }
}
// ANCHOR_END: spin_loop

// ANCHOR: JobScheduler
/// Mechanism to synchronize job startup and error handling
pub trait JobScheduler {
    /// Set up synchronization
    fn new(num_threads: u32) -> Self;

    /// Minimum accepted parallel job size, please run smaller jobs sequentially
    const MIN_TARGET: u64;

    /// Start a job
    ///
    /// `num_threads` must be the same value that was passed to `new()`.
    ///
    fn start(&self, target: u64, num_threads: u32);

    /// Request all threads to stop
    fn stop(&self);

    /// Check if the stop signal has been raised
    fn stopped(&self) -> bool;

    /// Wait for a counting job, or a stop signal
    ///
    /// This returns the full counting job, it is then up to this thread to
    /// figure out its share of work from it.
    ///
    fn wait_for_task(&self) -> Result<u64, Stopped>;

    /// Wait until all other worker threads have accepted their task
    ///
    /// Worker threads must do this before telling other threads that they are
    /// done with their current task, and `JobScheduler` implementations may
    /// rely on this to optimize `wait_for_task()`.
    ///
    fn wait_for_started(&self);

    /// Wait for a condition to be met or for the stop signal to be received
    fn faillible_spin_wait(&self, condition: impl Fn() -> bool) -> Result<Done, Stopped> {
        self.faillible_spin::<true>(condition).unwrap()
    }

    /// `spin_loop` specialization that monitors the stop signal like
    /// `faillible_spin_wait`
    fn faillible_spin<const INFINITE: bool>(
        &self,
        condition: impl Fn() -> bool,
    ) -> Option<Result<Done, Stopped>> {
        let condition = || condition().then_some(Ok(Done));
        let error = || self.stopped().then_some(Err(Stopped));
        spin_loop::<INFINITE, _>(condition, || condition().or(error()))
    }
}

/// Error type used to signal that the stop signal was raised
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Stopped;

/// Signal emitted by `faillible_spin` to tell that the condition was satisfied
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Done;
// ANCHOR_END: JobScheduler

// ANCHOR: BasicScheduler
/// Basic JobScheduler implementation, to be refined later on
pub struct BasicScheduler {
    /// Counting job (Some(target)) or request to stop (None)
    request: atomic::Atomic<Option<std::num::NonZeroU64>>,

    /// Mechanism for worker threads to await a request.
    /// This can take the form of an incoming job or a termination request.
    barrier: std::sync::Barrier,
}
//
impl JobScheduler for BasicScheduler {
    fn new(num_threads: u32) -> Self {
        use atomic::Atomic;
        use std::{num::NonZeroU64, sync::Barrier};
        assert!(Atomic::<Option<NonZeroU64>>::is_lock_free());
        Self {
            request: Atomic::new(NonZeroU64::new(u64::MAX)),
            barrier: Barrier::new(num_threads.try_into().unwrap()),
        }
    }

    // We're reserving `target` == 0 for requests to stop
    const MIN_TARGET: u64 = 1;

    fn start(&self, target: u64, _num_threads: u32) {
        use std::{num::NonZeroU64, sync::atomic::Ordering};

        // Package the counting request
        let request = NonZeroU64::new(target);
        assert!(
            request.is_some(),
            "Don't schedule jobs smaller than MIN_TARGET"
        );

        // Send it to the worker threads, making sure that worker threads have
        // not stopped on the other side of the pipeline.
        self.request
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                old.expect("Can't schedule this new job, some workers have stopped");
                Some(request)
            })
            .unwrap();

        // Wake up worker threads
        self.barrier.wait();
    }

    fn stop(&self) {
        self.request.store(None, atomic::Ordering::Relaxed);
        self.barrier.wait();
    }

    fn stopped(&self) -> bool {
        self.request.load(atomic::Ordering::Relaxed).is_none()
    }

    fn wait_for_task(&self) -> Result<u64, Stopped> {
        self.barrier.wait();
        self.request
            .load(atomic::Ordering::Relaxed)
            .map(u64::from)
            .ok_or(Stopped)
    }

    fn wait_for_started(&self) {}
}
// ANCHOR_END: BasicScheduler

// ANCHOR: Aggregator
/// Mechanism to collect processing thread results and detect termination
struct Aggregator {
    /// Spin lock used to synchronize concurrent read-modify-write operations
    ///
    /// Initially `false`, set to `true` while the lock is held.
    ///
    spin_lock: atomic::Atomic<bool>,

    /// Number of processing threads that have work to do
    ///
    /// This is set to `num_threads` when a job is submitted. Each time
    /// a thread is done with its task, it decrements this counter. So when it
    /// reaches 0, it tells the thread that last decremented it that all
    /// worker threads are done, with Acquire/Release synchronization.
    ///
    remaining_tasks: atomic::Atomic<u32>,

    /// Sum of partial computation results
    result: atomic::Atomic<u64>,
}
//
impl Aggregator {
    /// Create a new aggregator
    pub fn new() -> Self {
        use atomic::Atomic;
        Self {
            spin_lock: Atomic::new(false),
            remaining_tasks: Atomic::new(0),
            result: Atomic::default(),
        }
    }

    /// Prepare the aggregator for a new job
    ///
    /// This should be done before starting a new job, and after any previously
    /// scheduled job have finished running.
    ///
    pub fn reset(&self, num_threads: u32) {
        use atomic::Ordering;
        debug_assert!(!self.spin_lock.load(Ordering::Relaxed));
        debug_assert_eq!(self.remaining_tasks.load(Ordering::Relaxed), 0);
        self.remaining_tasks.store(num_threads, Ordering::Relaxed);
        self.result.store(0, Ordering::Relaxed);
    }

    /// Aggregate a thread's partial results, tell the job's result if finished
    pub fn task_done(&self, result: u64) -> Option<u64> {
        use atomic::Ordering;
        let mut lock = self.lock();
        let merged_result = lock.merge_result(result, Ordering::Relaxed);
        lock.notify_done(Ordering::Release).then_some(merged_result)
    }

    /// Wait for the job to be done or error out, collect the result
    pub fn wait_for_result(&self, scheduler: &impl JobScheduler) -> Result<u64, Stopped> {
        use atomic::Ordering;
        scheduler.faillible_spin_wait(|| self.remaining_tasks.load(Ordering::Relaxed) == 0)?;
        atomic::fence(Ordering::Acquire);
        Ok(self.result.load(Ordering::Relaxed))
    }

    /// Acquire spin lock
    fn lock(&self) -> AggregatorGuard {
        use atomic::Ordering;
        loop {
            // Try to opportunistically acquire the lock
            if self.spin_lock.swap(true, Ordering::Acquire) == false {
                return AggregatorGuard(self);
            }

            // Otherwise, wait for it to become available before retrying
            let check = || {
                if self.spin_lock.load(Ordering::Relaxed) {
                    None
                } else {
                    Some(())
                }
            };
            spin_loop::<true, ()>(check, check);
        }
    }
}

/// Equivalent of `MutexGuard` for the Aggregator spin lock
struct AggregatorGuard<'aggregator>(&'aggregator Aggregator);
//
impl<'aggregator> AggregatorGuard<'aggregator> {
    /// Merge partial result `result`, get the current sum of partial results
    pub fn merge_result(&mut self, result: u64, order: atomic::Ordering) -> u64 {
        self.fetch_update(
            &self.0.result,
            order,
            |old| old <= u64::MAX - result,
            |old| old + result,
        )
    }

    /// Notify that this thread is done, tell if the job is done
    pub fn notify_done(&mut self, order: atomic::Ordering) -> bool {
        self.fetch_update(&self.0.remaining_tasks, order, |old| old > 0, |old| old - 1) == 0
    }

    /// Read-Modify_Write operation that is not atomic in hardware, but
    /// logically atomic if all concurrent writes to the target atomic variable
    /// require exclusive access to the spinlock-protected AggrecatorGuard
    ///
    /// In debug builds, the `check` sanity check is first performed on the
    /// existing value, then a new value is computed through `change`, inserted
    /// into the target atomic variable, and returned.
    ///
    fn fetch_update<T: Copy>(
        &mut self,
        target: &atomic::Atomic<T>,
        order: atomic::Ordering,
        check: impl FnOnce(T) -> bool,
        change: impl FnOnce(T) -> T,
    ) -> T {
        assert!(atomic::Atomic::<T>::is_lock_free());
        let [load_order, store_order] = Self::rmw_order(order);
        let old = target.load(load_order);
        debug_assert!(check(old));
        let new = change(old);
        target.store(new, store_order);
        new
    }

    /// Load and store ordering of non-atomic read-modify-write operations
    fn rmw_order(order: atomic::Ordering) -> [atomic::Ordering; 2] {
        use atomic::Ordering;
        match order {
            Ordering::Relaxed => [Ordering::Relaxed, Ordering::Relaxed],
            Ordering::Acquire => [Ordering::Acquire, Ordering::Relaxed],
            Ordering::Release => [Ordering::Relaxed, Ordering::Release],
            Ordering::AcqRel => [Ordering::Acquire, Ordering::Release],
            Ordering::SeqCst | _ => unimplemented!(),
        }
    }
}
//
impl<'aggregator> Drop for AggregatorGuard<'aggregator> {
    fn drop(&mut self) {
        self.0.spin_lock.store(false, atomic::Ordering::Release);
    }
}
// ANCHOR_END: Aggregator

// ANCHOR: SharedState
/// State shared between the main thread and worker threads
struct SharedState<Counter: Fn(u64) -> u64 + Sync, Scheduler> {
    /// Counter implementation
    counter: Counter,

    /// Number of processing threads, including main thread
    num_threads: u32,

    /// Mechanism to synchronize job startup and error handling
    scheduler: Scheduler,

    /// Mechanism to synchronize task and job completion
    aggregator: Aggregator,
}
//
impl<Counter: Fn(u64) -> u64 + std::panic::RefUnwindSafe + Sync, Scheduler: JobScheduler>
    SharedState<Counter, Scheduler>
{
    /// Set up shared state
    pub fn new(counter: Counter, num_threads: u32) -> Self {
        Self {
            counter,
            num_threads,
            scheduler: Scheduler::new(num_threads),
            aggregator: Aggregator::new(),
        }
    }

    /// Schedule counting work and wait for the end result
    pub fn count(&self, target: u64) -> u64 {
        // Handle sequential special cases
        if self.num_threads == 1 || target < Scheduler::MIN_TARGET {
            return (self.counter)(target);
        }

        // Schedule job
        let debug_log = |action| debug_log(true, action);
        debug_log("scheduling a new job");
        self.aggregator.reset(self.num_threads);
        self.scheduler.start(target, self.num_threads);

        // Do our share of the work
        let result = self
            .process(target, Self::MAIN_THREAD as u64)
            .unwrap_or_else(|| {
                // If we're not finishing last, wait for workers to finish
                debug_log("waiting for the job's result");
                self.aggregator
                    .wait_for_result(&self.scheduler)
                    .expect("This job won't end because some workers have stopped")
            });
        debug_log("done");
        result
    }

    /// Thread index of the main thread
    const MAIN_THREAD: u32 = 0;

    /// Request worker threads to stop
    pub fn stop(&self, is_main: bool) {
        debug_log(is_main, "sending the stop signal");
        self.scheduler.stop();
    }

    /// Worker thread implementation
    pub fn worker(&self, thread_idx: u32) {
        use std::panic::AssertUnwindSafe;
        assert!(thread_idx != Self::MAIN_THREAD);
        let debug_log = |action| debug_log(false, action);
        if let Err(payload) = std::panic::catch_unwind(AssertUnwindSafe(|| {
            // Normal work loop
            let thread_idx = thread_idx as u64;
            debug_log("waiting for its first job");
            while let Ok(target) = self.scheduler.wait_for_task() {
                self.process(target, thread_idx);
                debug_log("waiting for its next job")
            }
            debug_log("shutting down normally");
        })) {
            // In case of panic, tell others to stop before unwinding
            debug_log("panicking!");
            self.stop(false);
            std::panic::resume_unwind(payload);
        }
    }

    /// Process this thread's share of a job, tell if the job is done
    fn process(&self, target: u64, thread_idx: u64) -> Option<u64> {
        // Determine which share of the counting work we'll take
        let num_threads = self.num_threads as u64;
        let base_share = target / num_threads;
        let extra = target % num_threads;
        let share = base_share + (thread_idx < extra) as u64;

        // Do the counting work
        let is_main = thread_idx == Self::MAIN_THREAD as u64;
        let is_worker = !is_main;
        let debug_log = |action| debug_log(is_main, action);
        debug_log("executing its task");
        let result = (self.counter)(share);

        // Wait for other threads to have accepted their task
        if is_worker {
            debug_log("waiting for other tasks to be started");
            self.scheduler.wait_for_started();
        }

        // Merge our partial result into the global result
        debug_log("merging its result contribution");
        self.aggregator.task_done(result)
    }
}

/// Logs to ease debugging
fn debug_log(is_main: bool, action: &str) {
    if cfg!(debug_assertions) {
        static MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _lock = MUTEX.lock();
        let header = if is_main { "Main " } else { "" };
        eprintln!("{header}{:?} is {action}", std::thread::current().id());
    }
}
// ANCHOR_END: SharedState

// ANCHOR: ThreadPool
pub struct ThreadPool<
    Counter: Fn(u64) -> u64 + std::panic::RefUnwindSafe + Sync + 'static,
    Scheduler: JobScheduler,
> {
    /// Worker threads
    _workers: Box<[std::thread::JoinHandle<()>]>,

    /// State shared with worker threads
    state: std::sync::Arc<SharedState<Counter, Scheduler>>,
}
//
impl<
        Counter: Fn(u64) -> u64 + std::panic::RefUnwindSafe + Send + Sync + 'static,
        Scheduler: JobScheduler + Send + Sync + 'static,
    > ThreadPool<Counter, Scheduler>
{
    /// Set up worker threads with a certain counter implementation
    pub fn start(counter: Counter) -> Self {
        use std::{sync::Arc, thread};

        let num_threads = u32::try_from(
            std::thread::available_parallelism()
                .map(|nzu| usize::from(nzu))
                .unwrap_or(2),
        )
        .unwrap();

        let state = Arc::new(SharedState::new(counter, num_threads));
        let _workers = (1..num_threads)
            .map(|thread_idx| {
                let state2 = state.clone();
                thread::spawn(move || state2.worker(thread_idx))
            })
            .collect();

        Self { _workers, state }
    }

    /// Count in parallel using the worker threads
    ///
    /// This wants &mut because the shared state is not meant for concurrent use
    ///
    pub fn count(&mut self, target: u64) -> u64 {
        self.state.count(target)
    }
}
//
impl<
        Counter: Fn(u64) -> u64 + std::panic::RefUnwindSafe + Sync + 'static,
        Scheduler: JobScheduler,
    > Drop for ThreadPool<Counter, Scheduler>
{
    /// Tell worker threads to exit on Drop: we won't be sending more tasks
    fn drop(&mut self) {
        self.state.stop(true)
    }
}
// ANCHOR_END: ThreadPool

#[cfg(test)]
mod tests {
    use super::{BasicScheduler, ThreadPool};
    use crate::test_utils;
    use once_cell::sync::Lazy;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;
    use std::{panic::RefUnwindSafe, sync::Mutex};

    type CounterBox = Box<dyn Fn(u64) -> u64 + RefUnwindSafe + Send + Sync + 'static>;
    static BKG_THREADS_BASIC: Lazy<Mutex<ThreadPool<CounterBox, BasicScheduler>>> =
        Lazy::new(|| {
            Mutex::new(ThreadPool::start(
                Box::new(crate::simd::multiversion::multiversion_avx2) as _,
            ))
        });

    #[quickcheck]
    fn thread_pool(target: u32) -> TestResult {
        let mut lock = BKG_THREADS_BASIC.lock().unwrap();
        test_utils::test_counter_impl(target, |target| lock.count(target))
    }
}
