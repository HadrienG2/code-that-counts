# Reducing launch overhead

There are two problems with the simple approach to multithreading that I used
at the start of this chapter. One that we can already see on my CPU, and another
one that we would see if we were to run the code on a higher-end compute node.

The first problem is that creating and destroying thread uses up a lot of system
resources, and `thread_simple` does that on every call. It would be better to
create threads once at the start of the program, and keep them alive across
benchmark runs, only submitting work to them through a lightweight
synchronization mechanism.

The second problem is that we are spawning threads one by one from the main
thread, which would become a bottleneck if we were to run this on a compute node
with many CPU cores. To put things in perspective, one can already build
rackmount servers with more than 200 physical CPU cores as of 2022, where this
book is written, and systems with thousands of CPU cores per node are on the
medium term horizon if semiconductor technology evolution progresses as expected.

Thankfully, I can offload both of these concerns to
[rayon](https://docs.rs/rayon), a great Rust library that provides both an
automatically managed thread pool and a scalable fork-join task spawning
mechanism, to get code that is simpler and performs better at little development
cost :

```rust,no_run
{{#include ../counter/src/lib.rs:thread_rayon}}
```

And with that, we get a 7.6x speedup on 8 CPU cores when counting up to 2^36.
This imperfect scaling largely comes from a slight reduction in CPU clock rate
when using all CPU cores, with throughput per CPU clock cycle increasing by
7.9x on its side.

Now, three thousand billion counter increments per cycle is as far as I'm going
to get on my CPU when it comes to asymptotic throughput. But this solution can
be improved in other respects.

Before getting there, though, I would like to address one more matter. Given
that my CPU has hyperthreading functionality, it actually exposes 16 hardware
threads, not 8, so you might expect a 16x speedup. But the truth is, for a
well-tuned compute-bound task, that cannot happen.

The reason is that hyperthreading works by multiplexing two instruction streams
over the execution resources of one CPU core. As a result, hyperthreading only
provides performance benefits for code that fails to saturate a CPU core's
execution resources on its own, like our initial versions without ILP. Now that
our code is tuned to do avoid this pitfall, the impact of hyperthreading on
performance can only be at best neutral, and at worst negative.

To put in an other way, if you encounter code that performs better with
hyperthreading than without, chances are high that the code is not putting the
CPU to good use to begin with and could benefit from some single-threaded
optimization. The main exception would be latency-sensitive code that spends
its time waiting for external events but must react to them very quickly as soon
as they occur, such as low-latency servers or hardware interrupt handlers in
operating systems.
