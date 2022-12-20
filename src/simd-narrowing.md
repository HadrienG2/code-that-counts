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
technique to provide the 8x speedup over the 64-bit SIMD version that it
theoretically allows for, then we need to reduce the overhead of merging. And
the obvious way to do that is to refrain from reducing narrow SIMD accumulators
all the way to scalar u64s when spilling to wider SIMD accumulators would
suffice.

Along the way, I'll also extract some complexity out of the full counter
implementation, as it starts to pack too much complexity into a single function
for my taste again.


## Reaching peak asymptotic throughput

TODO: Roll out an U8Accumulator that contains an [u8x32; ILP_WIDTH], and
      [i16x16; ILP_WIDTH] and an [u64; ILP_WIDTH] as well as two u8 counters.
      You can increment it and it increments into the u8x32, with auto-spill
      into i16x16 that itself auto-spills to [u64; ILP_WIDTH]. By calling a
      method, you extract a merged [u64; ILP_WIDTH].

TODO: Discuss throughput at lower iteration counts.

