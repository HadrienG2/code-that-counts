# Superscalar SIMD

## By our powers combined...

As we have seen, both instruction-level parallelism and SIMD can be leveraged to
achieve extra execution performance. Now it's time to combine them and see how
the benefits add up.

```rust,no_run
{{#include ../counter/src/lib.rs:simd_ilp}}
```

This code... could be prettier. It certainly shows the need for layers of
abstraction when combining multiple layers of hardware parallelism. But since
this might be your first exposure to the marriage of SIMD and ILP, seeing it
written explicitly the first time may make understanding easier.

How fast do we expect this to get? From uops.info, my Zen 2 CPU can do 3 SSE
64-bit integer additions per clock cycle. Each SSE addition, in turn, can sum
two 64-bit integers. So our expectation would be to reach 6 integer increments
per clock cycle, which indeed we do, with this implementation reaching 25.6
billion integer increments per second on my CPU.

Another thing worth noting is that we need less ILP streams to do this, with
peak performance being reached with an ILP_WIDTH of 9. This is good for multiple
reasons:

- Narrower code uses up less instruction cache and CPU registers, thus reducing
  the risk of slowdown elsewhere in the codebase when this function is called.
- Narrower code reaches top speed with a smaller number of loop iterations.
  This implementation beats the 15-way ILP loop at any number of iterations,
  but it also beats the basic for loop's counting rate when counting only up
  to >8, and the basic SIMD loop when counting up to >16. So not only is it very
  efficient asymptotically, it also becomes efficient very quickly as problem
  size ramps up.

It is also very easy to explain why this is the case: with ILP based on scalar
increments, our previous code was sharing resources with the outer loop
management instructions. This code only relies on SIMD instructions for
counting, which are not used for outer loop management, so as soon as we have
enough work to keep the SIMD unit fed without outer loop management becoming a
bottleneck, we should operate at maximum possible speed.



## Superscalar honeymoon?

At this point, it may seem like I am claiming that SIMD is superior to scalar
processing and should be used in its place whenever practical. However, for
integer computations at least, it actually should not have to be one or the
other.

Since my Zen 2 CPU can process at most
[5 instructions per cycle](https://travisdowns.github.io/blog/2019/06/11/speed-limits.html#pipeline-width),
but only 3 SSE instructions per cycle, one has to wonder if given suitable
shaped code, it shouldn't be possible for me to also have it perform a pair of
scalar integer increments in each CPU cycle in addition to all the SSE work
it's already doing.

Of course, in doing so, I would need to be careful not to consume ILP resources
needed to handle the outer counting loop. But given enough code contortions,
that can be taken care of.

And so I did actually try out this hybrid scalar + SIMD approach...

```rust,no_run
{{#include ../counter/src/lib.rs:extreme_ilp}}
```

...but I observed no performance benefits with respect to the SIMD+ILP version,
only a slight performance degradation that varied depending on the value of
tuning parameters.

This puzzles me. Obviously I must be hitting some microarchitectural limit, but
it's not clear which. I looked up every source of microarchitectural
documentation that I know of and tried to check every relevant AMD performance
counter for this code and the previous one. But I found nothing that could
explain this lack of ILP improvements.

However, AMD's CPU performance counters are much less numerous than Intel's,
which suggests that they may have more blind spots. So it may be the case that
if I re-tried this analysis on an Intel chip, I could find the answer to this
enigma. Maybe I'll do this someday. Or maybe one of you readers will find the
answer before I do and
[tell me about it](https://github.com/HadrienG2/code-that-counts/issues) so I
update the book accordingly!

In any case, though, since this approach does not currently provide any
measurable improvement and requires even more convoluted code, I am very happy
to let it go and stick with the simpler SIMD + ILP approach for the remainder of
the current version of this book.


## Consolidation

Speaking of keeping code understandable, before we scale this code up further,
we need to do something about the cognitive complexity of the SIMD + ILP
solution. While less scary than the hybrid scalar + SIMD version, this code
still feels too complex, and we're far from being done with it.

To address this, let me separate concerns of SIMD computation and
instruction-level parallelism a bit. For SIMD, I'll introduce the following
trait...

```rust,no_run
{{#include ../counter/src/lib.rs:Accumulator}}
```

We will later come back to why this SIMD_WIDTH is a generic parameter of this
trait, as opposed to an associated const. For now, it is sufficient to say that
we may want to implement it multiple times for a single type with different
values of SIMD_WIDTH.

In any case, this trait can be trivially implemented for both our initial `u64`
scalar counter and our new `m128i` SIMD counter as follows...

```rust,no_run
{{#include ../counter/src/lib.rs:implAccumulator}}
```

...and then we can write a generic SIMD + ILP implementation that can work with
terms of any implementation of this trait, which effectively supersedes all of
our previous `_ilp` implementations:

```rust,no_run
{{#include ../counter/src/lib.rs:generic_ilp}}
```

Then we can stop worrying about ILP and offload that concern to `generic_ilp`,
only focusing our later efforts on perfecting our usage of SIMD.
