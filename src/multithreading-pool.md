# Custom thread pool

Using rayon makes it a lot cheaper to spawn parallel tasks than explicitly
spawning threads on every counting run, but cheaper is not free.

When using all 16 hyperthreads of my 8-core CPU, small counting loops take
microseconds to execute due to scheduling and synchronization overhead. As a
result, it takes more than 32 million counting iterations for this execution
configuration to beat the performance of sequential counting.

This overhead is, as one would expect, less severe when using less threads...

- With 2 threads, sequential counting is beaten above 4 million iterations
- With 4 threads pinned in such a way that they share a common CPU L3 cache
  shard, which reduces communication latency, it takes 8 million iterations. 
- With 8 threads or 4 poorly pinned threads, it takes 16 million iterations.

...but it's still a big entry cost compared to the optimizations we have done
previously, which could be problematic in latency-sensitive applications
like audio processing. Can we do better than that?

Indeed we can because rayon is a general-purpose library, which supports
features we do not need here like dynamic load balancing and arbitrary task
code. By forgoing these features and accepting to write the code ourselves, we
can implement a lighter-weight scheduling and synchronization protocol, and thus
beat rayon's performance. However, rayon is pretty well implemented, so it
actually takes a decent amount of work to outperform it.

So, what building blocks do we need to count in parallel efficiently?

- N-1 worker threads where N is the desired amount of parallelism. The thread
  which requested the counting will do its share of the work.
- A mechanism for worker threads to wait for work to come up and for the main
  thread to wake them up, which must be immune to
  [lost wake-ups](https://docs.oracle.com/cd/E19120-01/open.solaris/816-5137/sync-30/index.html).
- A mechanism for the main thread to tell worker threads what work they need to
  do.
- A mechanism for the main thread to wait for worker threads to be done and for
  worker threads to wake it up, again immune to lost wake-ups.
- A mechanism for worker threads and the main thread to aggregate their
  temporary results into shared storage, and for the main thread to collect the
  final results.
- A mechanism for worker threads and the main thread to tell each other to stop
  when the main thread exits or a worker thread panics.

That's actually a fair amount of building blocks, and since we are trying to
outperform rayon's tuned implementation, we also need to implement them using
a synchronization protocol that is relatively clever. Hence the end result is
a big chunk of code. Let's go through it step by step.


## Read-Modify-Write operations and spin locks

Our synchronization protocol requires some atomic Read-Modify-Write (RMW)
operations, where a thread must read a value from memory, compute a new
dependent value, and store it in place of its former self, without other threads
being able to intervene in the middle.

While x86 CPUs provides native support for such operations on <=64-bit words,
they are very expensive (hundreds of CPU cycles in the absence of contention,
and that grows super-linearly when contention occurs), so we want as few of
them per synchronization transaction as possible.

To support blocking, well implemented mutexes must perform a minimum of two of
these operations during a lock/unlock cycle:

- One at locking time to check if the mutex is unlocked and if so lock it.
    * If the mutex is locked, at least one more to tell the thread holding the
      lock that we're waiting if the lock is still held at that time, or else
      try to lock it again.
- One at unlocking time to check if there are threads waiting for the unlock,
  which allows avoiding the overhead of calling into the OS to unblock them in
  the uncontended fast path.

For the purpose of this simple program, any synchronization transaction should
be doable in one blocking OS transaction, so we should never need more than two
hardware RMW operations in succession on the uncontended path. If we do, it
means that we're doing something inefficiently.

There are two ways to get that down to one single hardware RMW atomic operation:

- If we have only one <=64-bit word to update, we can do it with one hardware
  RMW operation.
- If we have multiple variables to update but we expect updates to take a very
  short amount of time (<1µs), then we can use a spin lock.

This last bit warrants some explanations since [spin locks got a bit of a bad
reputation recently](https://matklad.github.io//2020/01/02/spinlocks-considered-harmful.html),
and for good reason: they are a rather specialized tool that tends to be overused.

Well implemented mutexes differ from well-implemented spin locks in the
following ways:

1. Mutexes need more expensive hardware RMW operations during lock acquisition
   to exchange more information with other threads. An RMW operation will always
   be needed, but on some hardware, not all RMW operations are born equal.
    * For example, `fetch_add` is less expensive than `compare_exchange` on x86
      because the former is implemented using an infaillible instruction,
      avoiding the need for costly retries in presence of thread contention.
      But the more complexity you pile up into a single variable, the more
      likely you are to need the full power of `compare_exchange` or the
      higher-level `fetch_update` abstraction tht Rust builds on top of it.
2. To detect other blocked threads, mutexes need a hardware RMW operation at unlock
   time, as mentioned earlier, whereas unlocking a spin lock only requires a
   write because the in-memory data only has two states, locked and unlocked,
   and only the thread holding the lock can perform the locked -> unlocked
   state transition. There is no information for it to read.
3. If they fail to acquire the lock, mutexes will start by busy-waiting in
   userspace, but then eventually tell the OS scheduler that they are waiting
   for something. The OS can use this information to schedule them out and
   priorize running the thread holding the lock that they depend on. This is not
   possible with spin locks, which are doomed to either burn CPU cycles or yield
   to random other OS threads for unknown amounts of time.

Spin locks are used to avoid costs #1 and #2 (expensive RMW operations) at the
cost of losing benefit #3 (efficient blocking during long waits).

The problem is that if long waits do happen, efficient blocking is actually very
important, much more so than cheaping out on hardware RMW operations. Therefore,
[spin locks are only efficient when long waits are exceptional, and their
performance degrades horribly as contention goes
up](https://matklad.github.io/2020/01/04/mutexes-are-faster-than-spinlocks.html).
Thus, they require a very well-controlled execution environment where you can
confidently assert that CPU cores are not oversubscribed for extended periods of
time. And as a result, they are only relevant for specialized use cases, not
general-purpose applications.

However, "specialized use case" is basically the motto of this book, so
obviously I can and will provide the right environment guarantees for the sake
of reaping those nice little spinlock perf profits.

And thus, when I'll need to do batches of read-modify-write operations, I'll
allow myself to lock them through the following spin waiting mechanism. Which,
conveniently enough, is also generic enough to be usable as part of a blocking
synchronization strategy.

```rust,no_run
{{#include ../counter/src/thread/pool.rs:spin_loop}}
```


## Scheduling

Given read-modify-write building blocks, we can start to work on the scheduling
mechanism used by the main thread to submit work to worker threads and by all
parties involved to tell each other when it's time to stop working, typically
when the main thread exits or a worker thread panics.

As a minor spoiler, I'll need to iterate on the implementation a bit later on,
so let's be generic over components implementing this logic via the following
trait...

```rust,no_run
{{#include ../counter/src/thread/pool.rs:JobScheduler}}
```

...which is easy to implement using an atomic variable and a standard Barrier.

```rust,no_run
{{#include ../counter/src/thread/pool.rs:BasicScheduler}}
```

Here, I am using a standard Barrier to have worker threads wait for job and
stop signals. The main interesting thing that's happening is that I am
forbidding zero-sized parallel jobs, as allowed by the `JobScheduler` API, to
repurpose that forbidden job size as a signal that threads should stop.

The astute reader will, however, notice that I am running afoul of my own
"not more than two hardware RMW operations per transaction" rule in
`BasicScheduler::start()`, as the blocking implementation of `Barrier::wait()`
must use at least two such operations and I am using one more for error handling.
This performance deficiency will be adressed in the next chapter.


## Collecting results

After starting jobs, we need to wait for them to finish and collect results.
Again, we'll explore several ways of doing this, so let's express what we need
as a trait:

```rust,no_run
{{#include ../counter/src/thread/pool.rs:Reducer}}
```

To be able simultaneously track the aggregated 64-bit job result and the 32-bit
counter of threads that still have to provide their contribution, we need
96 bits of state, which is more state than hardware can update in a single
atomic read-modify-write transaction (well technically some hardware can do
128-bit atomics but they are almost twice as slow as their 64-bit counterparts).

To avoid the need for multiple expensive hardware atomic transactions, we
synchronize writers using a spinlock, as we hinted at earlier.

```rust,no_run
{{#include ../counter/src/thread/pool.rs:BasicResultReducer}}
```

Besides the spinlock, the main other interesting thing in this code is the
choice of atomic memory orderings, which ensure that any thread which either
acquires the spinlock or spins waiting for `remaining_threads` to reach 0 with
`Acquire` ordering will get a consistent value of `result` at the time where
`remaining_threads` reached 0.


## Shared facilities

Now that we have ways to schedule jobs and collect results, we are ready to
define the state shared between all processing threads, as well as its basic
transactions.

```rust,no_run
{{#include ../counter/src/thread/pool.rs:SharedState}}
```

This is a bit large, let's go through it step by step.

TODO: Update

The shared state is a combination of the two synchronization primitives that we
have introduced previously with a sequential counting implementation and
a counter of processing threads.

The main thread starts a job by resetting the `Accumulator`, then waking up
workers through the `JobScheduler` mechanism. After that, it does its share of
the work by calling `process()`, which we're going to describe later. Finally
it either gets the final result directly by virtue of finishing last or waits
for worker threads to provide it.

Worker threads go through a simple loop where they wait for
`scheduler.wait_for_task()` to emit a job, then process it, then go back to
waiting, until they are requested to stop. If a panic occurs, the panicking
thread sends the stop signal so that other worker threads and the main threads
eventually stop as well, avoiding a deadlock scenario where surviving worker
threads would wait for a new job from the main thread, which itself waits for
worker threads to finish the current job.

The processing logic in `process()` starts by splitting the job into roughly
identical chunks, counts sequentially, then makes sure worker threads wait for
other workers to have started working (which is going to be needed later on),
and finally merges the result contribution using the `Accumulator`.

Finally, a little logger which is compiled out of release builds is provided,
as this is empirically very useful when debugging incorrect logic in
multi-threaded code.



## Putting it all together

At this point, we have all the logic we need. The only thing left to do is
to determine how many CPUs are available, spawn an appropriate number of worker
threads, provide the top-level counting API, and making sure that when the main
thread exits, it warns worker threads so they exit too.

```rust,no_run
{{#include ../counter/src/thread/pool.rs:BasicThreadPool}}
```

And with that, we have a full custom parallel counting implementation that we
can compare to the Rayon one. How well does it perform?

- 2 well-pinned threads beat sequential counting above 256 thousand iterations
  (16x better).
- 4 well-pinned threads do so above 1 million iterations (8x better).
- 8 threads do so above 2 million iterations (8x better).
- 16 hyperthreads do so above 8 million iterations, (4x better).
- Asymptotic performance at large amounts of iterations is comparable.

So, with this specialized implementation, we've cut down the small-task overhead
by a sizable factor that goes up as the number of threads goes down. But can we
do better still by adressing my earlier comment that the naive `JobScheduler`
above isn't very optimal?
