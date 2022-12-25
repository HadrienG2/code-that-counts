# Scaling up

So far, by virtue of working on a "small" 8-core CPU, I have had the luxury of
not caring at all about scalability to large thread counts and still getting
decent synchronization performance. But now is the right time to start caring,
be it only to cross-check my assumption that 8 cores is indeed "few" cores from
a synchronization perspective.

What have I been doing wrong from a scalability perspective so far?

* In my atomic read-modify-write transactions, all threads are interacting
  through a fixed set of atomic variables. This does not scale well on
  cache-coherent CPU architectures like x86:
    * Because coherent caches handle concurrent access to cache lines through
      the conceptual equivalent of a reader-writer lock, only one thread at a
      time can be in the process of synchronizing with other threads. A
      scalable synchronization protocol would instead allow threads to
      synchronize in parallel. This is done by using N-ary reduction trees,
      where threads start to synchronize in groups of N, then the group results
      are themselves synchronized in groups of N, and so one until a fully
      merged result is produced.
    * Cache coherence is implemented by physically transferring data from the
      L1 cache of one CPU core to that of another core through an interconnect
      whenever a write occurs. But not all CPU cores are at equal distance from
      the perspective of this interconnect. For example, my 8-core Zen 2 CPU is
      composed of two 4-core complexes, with faster data transfers within a
      complex than across complexes. Scalable synchronization protocols must
      account for this by minimizing data transfers across slower interconnects.
* I have not paid attention to cache locality and false sharing concerns so
  far. In a nutshell, data which is manipulated together by a thread should
  reside on the same cache line to amortize inter-cache data transfer costs,
  while data which is not manipulated together should reside on different cache
  lines to enable parallel access by different CPU cores. Where these goals
  conflict, parallelism should usually be favored over reduction of inter-cache
  data transfers.

Given the enormous amount of work that has been expended on non-uniform memory
access in Linux, which is a related problem that affects RAM accesses, it is
safe to assume that the `futex()` Linux system call does the right thing here
and we do not need to waste time implementing a `futex()` reduction tree.
However, any synchronization that we do ourselves using atomic shared state will
need to be reworked to follow the scalability guidelines outlined above.


## Tree-based synchronization

TODO: Use reduction trees with const MAX_ARITY + lstopo-driven splitting.
