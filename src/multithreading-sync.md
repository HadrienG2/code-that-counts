# Custom synchronization

Running the custom threaded version with a small job through a profiler tells me
that it spends a sizable fraction of its time in the Linux kernel's `futex()`
synchronization primitive, called by rustc's internal condition variable
implementation, itself triggered by our "wait for new task" barrier.

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

So if want to go faster, we'll need to cut the middleman go lower-level, down to
`futex()` and its cousins on other platforms. Fortunately, the `atomic_wait`
crate provides a least common denominator platform abstraction layer across all
popular desktop operating systems.

```rust,no_run
{{#include ../counter/src/lib.rs:thread_sync}}
```

This provides a performance improvement of up to 2x at smaller problem sizes
(less than a few millions) and for smaller number of threads (2-4). At higher
problem sizes and thread count, however, this code is actually slightly slower,
asymptotically we lose a few microseconds per call.

This most likely happens because as the number of locally managed atomics grows,
the design is getting less and less friendly to Zen 2's non-uniform L3 cache
structure, and would benefit from some cache locality optimizations like teaming
up threads by L3 cache blocks and having them work within "private" cache lines
there until they're ready to share aggregated results with other teams.

But before getting to this sort of optimizations, I think it might actually be
worthwhile to discuss the elephant in the room, which is that if the performance
profile is now close to 100% syscalls and we've worked extra hard to make sure
we call the right syscalls at the right time, then the only way forward is to
figure out another design where we rely on even less syscalls.
