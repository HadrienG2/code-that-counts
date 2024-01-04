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

    // Switch to yielding to the OS once it's clear it's gonna take a while, to
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
pub trait JobScheduler: Sync {
    /// Minimum accepted parallel job size, smaller jobs must be run sequentially
    const MIN_TARGET: u64;

    /// Start a job
    fn start(&self, target: u64);

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
impl BasicScheduler {
    /// Set up a new BasicScheduler
    pub fn new(num_threads: u32) -> Self {
        use atomic::Atomic;
        use std::{num::NonZeroU64, sync::Barrier};
        assert!(Atomic::<Option<NonZeroU64>>::is_lock_free());
        Self {
            request: Atomic::new(NonZeroU64::new(u64::MAX)),
            barrier: Barrier::new(num_threads.try_into().unwrap()),
        }
    }
}
//
impl JobScheduler for BasicScheduler {
    // We're reserving `target` == 0 for requests to stop
    const MIN_TARGET: u64 = 1;

    fn start(&self, target: u64) {
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

// ANCHOR: Reducer
/// Synchronization primitive that aggregates per-thread contributions and can
/// tell when all threads have provided their contribution.
///
/// Logically equivalent to the combination of an atomic thread counter and
/// an atomic accumulator of results, where results are accumulated before
/// decreasing the thread counter.
///
pub trait Reducer: Sync {
    /// Clear the accumulator and prepare for `num_threads` contributions
    ///
    /// This is meant for use by the main thread inbetween jobs and should not
    /// be called while worker threads may observe the state of this object.
    ///
    fn reset(&self, num_threads: u32);

    /// Truth that not all threads have submitted their contribution yet
    ///
    /// If this is used for synchronization purposes, it should be followed by
    /// an `Acquire` memory barrier.
    ///
    fn has_remaining_threads(&self) -> bool;

    /// Optional data payload that threads can contribute
    ///
    /// In the special case where this is `()`, this `Reducer` just counts down
    /// until all threads have called `thread_done`, as a barrier of sorts.
    ///
    type Contribution;

    /// Sum of aggregated contributions for all threads so far
    fn current_result(&self) -> Self::Contribution;

    /// Wait for all contributions to have been received or an error to occur
    fn wait_for_end_result(
        &self,
        scheduler: &impl JobScheduler,
    ) -> Result<Self::Contribution, Stopped> {
        let res = scheduler.faillible_spin_wait(|| !self.has_remaining_threads());
        atomic::fence(atomic::Ordering::Acquire);
        res.map(|Done| self.current_result())
    }

    /// Accumulator identifier
    ///
    /// Some Reducers have an internal structure, where threads can target
    /// multiple inner accumulators. This identifies one such accumulator.
    ///
    type AccumulatorId: Copy;

    /// Process a contribution from one thread
    ///
    /// If this was the last thread, it gets the full aggregated job result.
    ///
    /// The notification that this thread has passed by and the check for job
    /// completion are performed as if done by a single atomic read-modify-write
    /// operation with the specified memory ordering.
    ///
    fn thread_done(
        &self,
        contribution: Self::Contribution,
        ordering: atomic::Ordering,
        accumulator_idx: Self::AccumulatorId,
    ) -> Option<Self::Contribution>;
}
// ANCHOR_END: Reducer

/// Mechanism to collect processing thread results and detect job completion
// ANCHOR: BasicResultReducer
#[derive(Default)]
pub struct BasicResultReducer {
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
    remaining_threads: atomic::Atomic<u32>,

    /// Sum of partial computation results
    result: atomic::Atomic<u64>,
}
//
impl BasicResultReducer {
    /// Create a new result reducer
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquire spin lock
    fn lock(&self) -> ReducerGuard {
        use atomic::Ordering;

        // If we are the last thread, we do not need a lock
        if self.remaining_threads.load(Ordering::Relaxed) == 1 {
            atomic::fence(Ordering::Acquire);
            return ReducerGuard(self);
        }

        loop {
            // Try to opportunistically acquire the lock
            if !self.spin_lock.swap(true, Ordering::Acquire) {
                return ReducerGuard(self);
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
//
impl Reducer for BasicResultReducer {
    fn reset(&self, num_threads: u32) {
        use atomic::Ordering;
        debug_assert!(!self.spin_lock.load(Ordering::Relaxed));
        debug_assert_eq!(self.remaining_threads.load(Ordering::Relaxed), 0);
        self.remaining_threads.store(num_threads, Ordering::Relaxed);
        self.result.store(0, Ordering::Relaxed);
    }

    fn has_remaining_threads(&self) -> bool {
        self.remaining_threads.load(atomic::Ordering::Relaxed) != 0
    }

    type Contribution = u64;

    fn current_result(&self) -> u64 {
        self.result.load(atomic::Ordering::Relaxed)
    }

    type AccumulatorId = ();

    fn thread_done(&self, result: u64, mut ordering: atomic::Ordering, (): ()) -> Option<u64> {
        use atomic::Ordering;

        // Enforce a Release barrier so that threads observing this
        // notification with Acquire ordering also observe the merged results
        ordering = match ordering {
            Ordering::Relaxed => Ordering::Release,
            Ordering::Acquire => Ordering::AcqRel,
            Ordering::Release | Ordering::AcqRel | Ordering::SeqCst => ordering,
            _ => unimplemented!(),
        };

        // Merge our results, expose job results if done
        let mut lock = self.lock();
        let merged_result = lock.merge_result(result, Ordering::Relaxed);
        lock.notify_done(ordering).then_some(merged_result)
    }
}
//
/// Equivalent of `MutexGuard` for the BasicResultReducer spin lock
struct ReducerGuard<'aggregator>(&'aggregator BasicResultReducer);
//
impl<'aggregator> ReducerGuard<'aggregator> {
    /// Merge partial result `result`, get the current sum of partial results
    pub fn merge_result(&mut self, result: u64, order: atomic::Ordering) -> u64 {
        self.fetch_update(
            &self.0.result,
            order,
            |old| old <= u64::MAX - result,
            |old| old + result,
        )
    }

    /// Notify that this thread is done, tell if all threads are done
    pub fn notify_done(&mut self, order: atomic::Ordering) -> bool {
        self.fetch_update(
            &self.0.remaining_threads,
            order,
            |old| old > 0,
            |old| old - 1,
        ) == 0
    }

    /// Read-Modify-Write operation that is not atomic in hardware, but
    /// logically atomic if all concurrent writes to the target atomic variable
    /// require exclusive access to the spinlock-protected ReducerGuard
    ///
    /// In debug builds, the `check` sanity check is first performed on the
    /// existing value, then a new value is computed through `change`, inserted
    /// into the target atomic variable, and returned.
    ///
    /// Note that this is unlike the fetch_xyz functions of Atomic variables,
    /// which return the _previous_ value of the variable.
    ///
    fn fetch_update<T: bytemuck::NoUninit>(
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

    /// Load and store ordering to be used when emulating an atomic
    /// read-modify-write operation under lock protection
    fn rmw_order(order: atomic::Ordering) -> [atomic::Ordering; 2] {
        use atomic::Ordering;
        match order {
            Ordering::Relaxed => [Ordering::Relaxed, Ordering::Relaxed],
            Ordering::Acquire => [Ordering::Acquire, Ordering::Relaxed],
            Ordering::Release => [Ordering::Relaxed, Ordering::Release],
            Ordering::AcqRel => [Ordering::Acquire, Ordering::Release],
            _ => unimplemented!(),
        }
    }
}
//
impl<'aggregator> Drop for ReducerGuard<'aggregator> {
    fn drop(&mut self) {
        self.0.spin_lock.store(false, atomic::Ordering::Release);
    }
}
// ANCHOR_END: BasicResultReducer

// ANCHOR: SharedState
/// State shared between the main thread and worker threads
struct SharedState<Counter: Fn(u64) -> u64 + Sync, Scheduler, ResultReducer: Reducer> {
    /// Counter implementation
    counter: Counter,

    /// Assignment of threads to reducer slots
    thread_ids: Box<[ResultReducer::AccumulatorId]>,

    /// Mechanism to synchronize job startup and error handling
    scheduler: Scheduler,

    /// Mechanism to synchronize task and job completion
    result_reducer: ResultReducer,
}
//
impl<
        Counter: Fn(u64) -> u64 + std::panic::RefUnwindSafe + Sync,
        Scheduler: JobScheduler,
        ResultReducer: Reducer<Contribution = u64>,
    > SharedState<Counter, Scheduler, ResultReducer>
{
    /// Set up shared state
    pub fn new(
        counter: Counter,
        thread_ids: Box<[ResultReducer::AccumulatorId]>,
        scheduler: Scheduler,
        result_reducer: ResultReducer,
    ) -> Self {
        assert!(thread_ids.len() < usize::try_from(u32::MAX).unwrap());
        Self {
            counter,
            thread_ids,
            scheduler,
            result_reducer,
        }
    }

    /// Schedule counting work and wait for the end result
    pub fn count(&self, target: u64) -> u64 {
        // Handle sequential special cases
        if self.num_threads() == 1 || target < Scheduler::MIN_TARGET {
            return (self.counter)(target);
        }

        // Schedule job
        let debug_log = |action| debug_log(true, action);
        debug_log("scheduling a new job");
        self.result_reducer.reset(self.num_threads());
        self.scheduler.start(target);

        // Do our share of the work
        let result = self.process(target, Self::MAIN_THREAD).unwrap_or_else(|| {
            // If we're not finishing last, wait for workers to finish
            debug_log("waiting for the job's result");
            self.result_reducer
                .wait_for_end_result(&self.scheduler)
                .expect("This job won't end because some workers have stopped")
        });
        debug_log("done");
        result
    }

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

    /// Thread index of the main thread
    const MAIN_THREAD: u32 = 0;

    /// Number of threads managed by this SharedState
    fn num_threads(&self) -> u32 {
        self.thread_ids.len() as u32
    }

    /// Process this thread's share of a job, tell if the job is done
    fn process(&self, target: u64, thread_idx: u32) -> Option<u64> {
        // Discriminate main thread from worker thread
        let is_main = thread_idx == Self::MAIN_THREAD;
        let is_worker = !is_main;

        // Determine which share of the counting work we'll take
        let thread_idx = thread_idx as u64;
        let num_threads = self.num_threads() as u64;
        let base_share = target / num_threads;
        let extra = target % num_threads;
        let share = base_share + (thread_idx < extra) as u64;

        // Do the counting work
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
        self.result_reducer.thread_done(
            result,
            atomic::Ordering::Release,
            self.thread_ids[thread_idx as usize],
        )
    }
}

/// Logs to ease debugging
fn debug_log(is_main: bool, action: &str) {
    if cfg!(debug_assertions) {
        let header = if is_main { "Main " } else { "" };
        // While using stdout here defies Unix convention, it also ensures that
        // the test harness can capture the message, unlike unbuffered stderr.
        println!("{header}{:?} is {action}", std::thread::current().id());
    }
}
// ANCHOR_END: SharedState

// ANCHOR: BasicThreadPool
pub struct BasicThreadPool<
    Counter: Fn(u64) -> u64 + std::panic::RefUnwindSafe + Sync + 'static,
    Scheduler: JobScheduler,
> {
    /// Worker threads
    _workers: Box<[std::thread::JoinHandle<()>]>,

    /// State shared with worker threads
    state: std::sync::Arc<SharedState<Counter, Scheduler, BasicResultReducer>>,
}
//
impl<
        Counter: Fn(u64) -> u64 + std::panic::RefUnwindSafe + Send + Sync + 'static,
        Scheduler: JobScheduler + Send + 'static,
    > BasicThreadPool<Counter, Scheduler>
{
    /// Set up worker threads with a certain counter implementation
    ///
    /// `make_scheduler` takes a number of threads as a parameter and sets up a
    /// Scheduler that can work with this number of threads.
    ///
    pub fn start(counter: Counter, make_scheduler: impl FnOnce(u32) -> Scheduler) -> Self {
        use std::{sync::Arc, thread};

        let num_threads = u32::try_from(
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(2),
        )
        .expect("Number of threads must fit on 32 bits");

        let state = Arc::new(SharedState::new(
            counter,
            std::iter::repeat(()).take(num_threads as usize).collect(),
            make_scheduler(num_threads),
            BasicResultReducer::new(),
        ));
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
    > Drop for BasicThreadPool<Counter, Scheduler>
{
    /// Tell worker threads to exit on Drop: we won't be sending more tasks
    fn drop(&mut self) {
        self.state.stop(true)
    }
}
// ANCHOR_END: BasicThreadPool

#[cfg(test)]
mod tests {
    use super::{BasicScheduler, BasicThreadPool};
    use std::{
        panic::RefUnwindSafe,
        sync::{Mutex, OnceLock},
    };

    type CounterBox = Box<dyn Fn(u64) -> u64 + RefUnwindSafe + Send + Sync + 'static>;
    static BKG_THREADS_BASIC: OnceLock<Mutex<BasicThreadPool<CounterBox, BasicScheduler>>> =
        OnceLock::new();

    crate::test_counter!((thread_pool, |target| {
        BKG_THREADS_BASIC
            .get_or_init(|| {
                Mutex::new(BasicThreadPool::start(
                    Box::new(crate::simd::multiversion::multiversion_avx2) as _,
                    BasicScheduler::new,
                ))
            })
            .lock()
            .unwrap()
            .count(target)
    }));
}
