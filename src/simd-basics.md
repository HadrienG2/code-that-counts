# Basic SIMD

Let's start with a simple vector loop that uses the SSE2 instruction set:

```rust,no_run
{{#include ../counter/src/simd/basic.rs:simd_basic}}
```

You may notice strong similarities with the instruction-level parallelism
approach that I used earlier, except this time instead of nice arrays I am
using a weird `m128i` x86-specific type that comes with its own weird array of
operations. And that is in spite of me using the
[`safe_arch`](https://docs.rs/safe_arch/) Rust crate, which already has
friendlier API conventions than the underlying hardware intrinsics.

If you can bear with the weirdness, however, this approach works slightly better
than two-way instruction level parallelism by virtue of generating a simpler
instruction stream that the CPU has an easier time parsing through, and also
leaving the CPU's scalar execution resources free to handle other program work
like loop counter management.

As a result, while the asymptotic throughput is the same as that of two-way
instruction-level parallelism, as you would expect, small loops will execute a
few CPU cycles faster.

Also, while readers with former SIMD experience may frown at me for using
conversions from integer arrays instead of cheaper hardware tricks for
generating SIMD vectors of zeroes and ones, I would advise such readers to give
optimizing compilers a little more credit. While this is certainly not always
true, sometimes you don't really need clever code to produce decent assembly:

```x86asm
    counter::simd_basic:
      mov     %rdi,%rax
      pxor    %xmm0,%xmm0
      cmp     $0x2,%rdi
    ↓ jb      29
      mov     %rax,%rcx
      shr     %rcx
      pxor    %xmm0,%xmm0
      pcmpeqd %xmm1,%xmm1
      nop
20:   psubq   %xmm1,%xmm0
      dec     %rcx
    ↑ jne     20
29:   movq    %xmm0,%rcx
      pshufd  $0xee,%xmm0,%xmm0
      movq    %xmm0,%rdx
      and     $0x1,%eax
      add     %rcx,%rax
      add     %rdx,%rax
    ← ret
```
