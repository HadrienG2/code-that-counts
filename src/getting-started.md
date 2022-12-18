# Getting started

Let's start simple and count with the `+=` operator, since that's the way we
increment integers in Rust, my language of choice for high-performance
programming:

```rust,no_run
{{#include ../counter/src/lib.rs:basic}}
```

Here, I use my [`pessimize` crate](https://github.com/HadrienG2/pessimize),
which is a fancier version of the recently stabilized
[`std::hint::black_box`](https://doc.rust-lang.org/std/hint/fn.black_box.html),
to ensure that the compiler doesn't optimize out the counting by directly
returning the end result. This optimization barrier pretends to the compiler
that _something_ is reading out every intermediary count, which prevents it from
skipping ahead.

How fast does this code count ? Well, actually, that depends how far you are
counting:

- For smaller counts, the counting time is dominated by the time it takes to
  set up and tear down the counting code, which is around one nanosecond on my
  CPU.
- For counts of more than about thousand elements, this overhead becomes
  negligible with respect to the actual counting rate, and thus we start
  counting once per CPU cycle, or a little more than 4 billion times per
  second on my CPU (4.25 to be precise).

In the following, I'm going to assume that we are interested in optimizing
for _asymptotic_ counting throughput when going for large counts. But I'll still
keep track of how many iterations it takes to reach baseline performance and
peak performance, just so you see what tradeoffs we make when optimizing for
throughput like this.

All the code that I'm going to present will be featured in the "counter"
directory of this book's source code, along with tests and microbenchmarks that
allow you to assert that the count is correct and replicate the results
discussed here on your own CPU.
