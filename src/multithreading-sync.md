# Custom synchronization

Running the custom threaded version with a small job through a profiler tells me
that it spends a sizable fraction of its time in the Linux kernel's `futex()`
synchronization primitive, called by rustc's internal condition variable
implementation, itself triggered by our "wait for new task" Barrier.

As far as I know, `futex()` is indeed the fastest primitive provided by Linux
for programs to await events. However, a condition variable may not be the most
efficient API to use it for our purposes.

Condition variables are not designed for scalability. You need to hold a mutex
just to await one, and when it is notified, every thread awating it will be
awakened, immediately grabbing the mutex just to make sure other threads don't
get a chance to run yet.

Even when this underlying
[thundering herd problem](https://en.wikipedia.org/wiki/Thundering_herd_problem)
is correctly accounted for, this API design prevents condition variable
overheads from scaling anything better than linearly with the number of waiting
threads, which matches our observations.

To its credit, this API design makes it a little harder to shoot yourself in the
foot with lost wake-ups in naive usage. But it is unfortunate that it does so
at the cost of making this synchronization primitive a poor pick in code where
performance matters.

So if want to go faster, we'll need to skip the middleman and go lower-level,
down to `futex()` and its cousins on other OSes. Fortunately, the `atomic_wait`
crate provides a least common denominator platform abstraction layer across all
popular desktop operating systems.

```rust,no_run
{{#include ../counter/src/lib.rs:thread_sync}}
```

How does that change in synchronization strategy affect performance?

- 2 well-pinned threads beat sequential counting above 128 thousand iterations
  (2x better).
- 4 well-pinned threads do so above 128 thousand iterations (8x better).
- 8 threads do so above 512 thousand iterations (4x better).
- 16 hyperthreads do so above 1 million iterations, (8x better).
- Asymptotic performance at large amounts of iterations is comparable.
