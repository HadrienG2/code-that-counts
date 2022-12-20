# When smaller is better

At this stage, we are computing 64-bit additions as fast as the CPU can, so you
might expect that we have also reached peak counting throughput. But that is
not the case yet.

On x86 as well as on many other CPU architectures, SIMD works by giving you a
fixed-size bag of bits and letting you slice it in integers of any
machine-supported size. So if you choose to slice your SIMD vector in
smaller-sized integers, then you can do work on more of these integers per
instruction. For example, given a 256-bit AVX vector, you can do either four
64-bit additions per cycle, eight 32-bit ones, sixteen 16-bit ones, or
thirty-two 8-bit ones.

The simple way to use this property would be to redesign the API to use
smaller-size counters, e.g. u32 ones. But at the rate we're already going, we
would overflow a 32-bit counter in about 20ms. So clearly, for our Big Counting
purposes, we want to keep using 64-bit accumulators somewhere.

However, that does not mean we cannot use smaller-sized integers as temporary
fast accumulators that eventually spill into these slower 64-bit ones.

Using 32-bit counters for illustration, the general approach is going to be
along these lines...

```rust,no_run
let mut accumulator = 0u64;
let mut remainder = target;
while remainder > 0 {
    let mut counter = 0u32;
    for _ in 0..remainder.min(u32::MAX as u64) {
        counter = pessimize::hide(counter + 1);
        remainder -= 1;
    }
    accumulator += counter as u64;
}
```

...but with native-width SIMD, instruction-level parallelism, and ideally a
choice of counter width as well since there's a potential tradeoff there: small
counters count more quickly but merge more often, and merging might be
expensive, so it's not absolutely certain that, say, 8-bit integers will be the
fastest choice.


## Narrow SIMD accumulators

Since the SimdAccumulator trait was designed for it, supporting narrower
integers is, on its side, just a matter of going through the `safe_arch`
documentation and filling in the right methods with appropriate conditional
compilation directives.

Here I will do it for AVX vectors only, since that's all I need for optimal
performance on Zen 2 and other modern x86 CPUs. But hopefully you can easily see
how the general approach would extend to SSE and non-x86 vector instruction sets.

```rust,no_run
{{#include ../counter/src/lib.rs:narrow_SimdAccumulator}}
```


## Simple full counter implementation

As currently written, `generic_ilp_u64` will only reliably work with 64-bit
accumulators. Narrower accumulators would overflow for larger values of
`target`. To avoid this, we need to regularly spill into larger accumulators as
outlined at the beginning of this chapter.

A simple if imperfect way to do so is to use the existing SimdAccumulator
facilities to reduce our full narrow SIMD accumulators all the way to u64
integers, then integrate these results into a set of global u64 accumulators,
every time a narrow integer overflow would happen:

```rust,no_run
{{#include ../counter/src/lib.rs:generic_ilp_simple}}
```

...and if we test this implementation, we see that each halving of the counter
width doubles our throughput until we reach u8 integers. There we only
improve over u16 integers by a factor of 1.7 because we merge every 255
increments and the merging overhead starts to become noticeable.

Now, improving by a factor of 6.7x overall is already nice, but if we want this
technique to get closer to the 8x speedup over the 64-bit SIMD version that it
aimed for, then we need to reduce the overhead of merging. And the obvious way
to do that is to refrain from reducing narrow SIMD accumulators all the way to
scalar u64s when spilling to wider SIMD accumulators would suffice.

Along the way, I'll also extract some complexity out of the full counter
implementation, as it starts to pack too much complexity into a single function
for my taste again.


## Reaching peak asymptotic throughput

As outlined above, we want to have a cascading accumulation design where we
increment SIMD vectors of 8-bit integers that spill into SIMD vectors of
16-bit integers when full, and those SIMD vectors of 16-bit integers in turn
spill into scalar 64-bit integers as they did before.

One can offload this cascading accumulation concern to a dedicated struct...

```rust,no_run
{{#include ../counter/src/lib.rs:U8Accumulator}}
```

...and then build a top-level counting function that is expressed in terms of
using this accumulator:

```rust,no_run
{{#include ../counter/src/lib.rs:narrow_u8}}
```

And with that, I get a 1.9x speedup with respect to the version that uses
16-bit integers, or a 7.7x speedup with respect to the version that does not use
narrow integers, for a final asymptotic performance of 98 integer increments
per CPU cycle or 393 billion integer increments per second.

Now, it's time to talk about non-asymptotic throughput.

All this extra counter merging we now do to achieve 64-bit counting range using
narrower integers is not free. It comes at the price of a steep reduction in
performance at low iteration counts.

Merely invoking a narrow counting function with no work to do costs 14ns with
32-bit integers, 23ns with 16-bit integers, and 31ns with 8-bit integers. And at
low iteration counts, we're also penalized by the fact that we're not trying to
keep using the widest SIMD format at all times, as we did in the non-narrow SIMD
versions, as the code for that would get quite gruesome when more and more
accumulator reduction stages start to pile up!

As a result, 32-bit hierarchical counting only becomes faster than 64-bit SIMD
counting when counting to more than 2048, then 16-bit counting catches up around
8192, and finally 8-bit counting becomes king of the hill around 16384 iterations.

But hey, at least we are _eventually_ counting as fast as a single Zen 2 CPU
core can count!

## Improving small-scale throughput

While the "entry cost" cannot be reduced without losing some benefits of
counting with smaller integers, the problem of poor performance at low iteration
counts can be worked around by simply offloading the work of small-scale
countings to our previous SIMD + ILP version, which has a very low entry cost
and much better performance than counting with scalar 64-bit integers already!

```rust,no_run
{{#include ../counter/src/lib.rs:narrow_u8_tuned}}
```

And with that, we get much more satisfactory performance at low iteration counts:

- Above 16 iterations, `narrow_u8_tuned` is faster than un-tuned `narrow_u8`
- Above 64 iterations, it is faster than `narrow_simple<u16>`
- Above 1024 iterations, it is faster than `narrow_simple<u32>`
- Above 2048 iterations, it is faster than `multiversion_avx2` alone

Basically, as soon as the timing of another solution gets above 35-40ns,
`narrow_u8_tuned` beats it. And for 4096 iterations, `narrow_u8_tuned` is a
whopping 2.6x faster than its un-tuned counterpart.

Speaking more generally, if you are trying to implement a general-purpose
utility that must perform well on a very wide range of problem sizes, like
`memcpy()` in C, it is a good idea to combine algorithms with low entry cost
and suboptimal scaling with algorithms with high entry cost and better scaling
like this. The price to pay being, of course, more implementation complexity.
