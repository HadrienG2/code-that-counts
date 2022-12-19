# SIMD

Executing multiple instructions per CPU clock cycle is one thing, but modern
CPUs have another trick up their sleeves, which is to execute multiple
computations per instruction. The technical jargon for this is "Single
Instruction Multiple Data" or SIMD for short.

While one also often speaks of "vectorization", which is definitely easier to
pronounce and remember, I will try to avoid this term here because these days it
is most often used to refer to the unrelated idea of re-expressing iterative
computations in terms of standardized operations on large arrays, in order to
work around inefficient programming language designs and implementations.

SIMD is a broad topic, so I will approach it incrementally:

- First I will demonstrate the basic concept by taking our baseline counting
  loop and translating it to simple SIMD as another way of computing multiple
  integer increments per cycle.
- Next I will show how SIMD can can be productively combined with
  instruction-level parallelism.
- After that, I will address the hardware heterogeneity issues that pervade
  SIMD programming, and tools that can be used to handle this discrepancy.
- And finally, I will demonstrate SIMD's sensitivity to data width and ways to
  make the most of it in the context of this toy counting problem.
