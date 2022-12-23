# Custom code

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

...but it's still quite a steep entry ticket. Can we do better?

Indeed we can because rayon is a general-purpose library, which supports
features we do not need here like dynamic load balancing and arbitrary task
code. By forgoing these features and accepting to write the code ourselves, we
can implement a lighter-weight scheduling and synchronization protocol, and thus
beat rayon's performance. However, rayon is pretty well implemented, so it
actually takes a decent amount of work to outperform it.

What building blocks do we need to count in parallel efficiently?

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
a synchronization protocol that is as lightweight as possible. Hence the end
result is a fairly large amount of code:

```rust,no_run
{{#include ../counter/src/lib.rs:thread_custom}}
```

What are the main performance optimizations applied here?

- Like rayon, we skip threading-related overhead when running sequentially.
- Atomic variables are used in combination with locks to enable both
  blocking and non-blocking access to most of the shared state. Blocking access
  is used to reduce the need for atomic read-modify-write operations, which are
  expensive, as well as for lost wake-up prevention. Non-blocking access is used
  to read out state or overwrite it without looking at the existing value.
- Condition variables are only used when   waiting for a job to come up, which
  takes unbounded time. Other waits, which are expected to be very short, are
  handled through spinlocks instead.

How well does this work out?

- 2 well-pinned threads beat sequential counting above 256 thousand iterations.
- 4 well-pinned threads do so above 1 million iterations.
- 8 threads do so above 4 million iterations.
- 16 hyperthreads do so above 8 million iterations.

So, with this specialized implementation, we've cut down the small-task overhead
by a factor of around 4x with respect to the rayon version at higher thread
counts, and 16x at low thread counts.
