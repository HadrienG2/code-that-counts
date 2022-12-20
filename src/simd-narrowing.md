# Smaller is better

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
choice of counter width as well since there's a tradeoff there (small counters
count more quickly but merge more often) so it's not absolutely certain that,
say, 8-bit integers will be the fastest choice.

Good thing the `SimdAccumulator` trait was designed with this in mind!


## Extending SimdAccumulator

TODO: Need to extract the equivalent of u32::MAX above and expose it. This
      should be possible without adding anything to the implementation, just by
      reasoning based on `mem::size_of::<T>()`.

TODO: Reduce will need a lot of care to combine high speed with overflow
      avoidance. Extracting array-reduce from generic_ilp might help. Also,
      could it be worthwhile to do a full reduction tree with 8-bit vectors
      spilling into 16-bit vectors, which spill into 32-bit vectors, which spill
      into 64-bit vectors ? It certainly has the potential to reduce
      merging/reduction overhead and thus make small integers more attractive...

      What if I actually could have a type ReduceResult in SimdAccumulator that
      reduces to either an SIMD type of the same size but twice the integer
      width, or the next narrower SIMD type if we're already at u64, and
      eventually u64 ? And then redesign generic_ilp so it recursively calls
      SimdAccumulator::reduce until it gets an u64 ?

      Or maybe a type ReductionTree that does the recursive work internally ?
      This will definitely need some extra code iterations before I do the
      write-up!

TODO: ...and then continue through implementation...
