# Custom synchronization

Running the custom threaded version with a small job through a profiler tells me
that it spends a sizable fraction of its time in the Linux kernel's `futex()`
synchronization primitive, called by rustc's internal condition variable
implementation, itself triggered by our "wait for new task" barrier.

As far as I know, `futex()` is indeed the fastest primitive provided by Linux
for programs to await events. However, a condition variable may not be the most
efficient API to use it for our purposes.

Condition variables do not scale well, and cannot do so due to their API design.
You need to hold a mutex to await one, and when it is notified, every thread
awating it will be awakened, immediately grabbing the mutex just to make sure
other threads don't get a chance to run yet.

Even when the
[thundering herd problem](https://en.wikipedia.org/wiki/Thundering_herd_problem)
is correctly accounted for, this API design prevents condition variable
overheads from scaling anything better than linearly with the number of waiting
threads, which matches our observations.

To its credit, this API design makes it a little harder to shoot yourself in the
foot with lost wake-ups in naive usage. But it is unfortunate that it does so
at the cost of making this synchronization primitive a poor pick in code where
performance matters.

So if want to go faster, we'll need to cut the middleman go lower-level, down to
`futex()` and its cousins on other platforms.


TODO: A condvar is not what we truly want here, a futex is
