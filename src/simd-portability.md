# Handling hardware heterogeneity

So far, I have been making two assumptions:

- That our CPU supports SSE2 vectorization. For x86 hardware, that is quite
  reasonable as almost every CPU released since the Intel Pentium 4 in 2000
  supports it, to the point that in fact binary releases of the Rust standard
  library and most Linux distributions assume availability of this instruction
  set on x86. But for non-x86 hardware, like say a smartphone, assuming
  availability of SSE2 would be very wrong.
- That our CPU does not support any better vectorization scheme than SSE2. That
  is very much not reasonable, as many x86 CPUs support the wider AVX
  instruction set, and some support the even wider AVX-512 instruction set.
  And even if we were to stick with SSE, SSE4.1 did contain several interesting
  upgrades that I'll want to use in later chapters of this book.

In this chapter, we'll break free from these assumptions to produce counting
code that runs everywhere and runs efficiently on most currently available x86
hardware, including of course Zen 2.


## Scaling down

To correctly handle hardware diversity, we need a way to dispatch to multiple
code paths, picking the best one for the hardware available.

One way to do this is at compile-time, using cfg attributes. And indeed, this is
the only way supported by the `safe_arch` crate that we've used so far, so
that's the way we are going to use. Simple usage looks like this:

```rust,no_run
{{#include ../counter/src/lib.rs:multiversion_sse2}}
```

...and just like that, we get rid of our "supports SSE2" assumption by using our
previous optimal SSE2 implementation where SSE2 is supported, and our previous
optimal scalar implementation where SSE2 is not supported. It may not be
optimally tuned for, say, an ARM chip, but at least it will run. The only caveat
is that users of older x86 hardware will need to adjust their compiler options
to benefit from this, as we'll demonstrate next.


## Scaling up

Now that we have a way to adapt to hardware capabilities, we can now get more
adventurous and support AVX2 vectorization, which is a wider cousin of SSE2
available on newer x86 CPUs from 2011 onwards. Using that, we can increment four
64-bit integers per instruction instead of two.

The support code should look straightforward enough if you've followed what
I've done so far. The only bit that may be mildly surprising is the binary
reduction tree in the `SimdAccumulator::reduce()` implementation, which is just
the x86 SIMD equivalent of what we've been doing with ILP for a while now.

```rust,no_run
{{#include ../counter/src/lib.rs:avx2}}
```

But to actually benefit from this, users need to compile the code with the
`RUSTFLAGS` environment variable set to `-C target-cpu=native`. This will
produce binaries specific to the host CPU, that will leverage all of its
feature, including AVX2 if available. And that's the same way one would use
SSE2 on older 32-bit x86 processors.

The AVX2 version is never slower than the SSE2 version, and it is asymptotically
twice as fast as we would expect. And with that, we reach peak 64-bit integer
increment rate on all CPUs up to the AVX-512 generation (which I cannot test on,
and thus will not cover here), which is twelve 64-bit integer increments per
cycle or 51.3 billion integer increments per second.
