# Multithreading

Now that I have taken single-core counting as far as I can (which is a little
less than 100 times the counting speed of a naive `for` loop), it is time to
leverage all the cores of my CPU for absolute best-in-class leading counting
performance.

On the surface, that sounds easy enough:

```rust,no_run
{{#include ../counter/src/lib.rs:thread_basic}}
```

But this version doesn't perform super impressively, only achieving a 4.8x
speed-up on 8 CPU cores even when counting to a very high 69 billion limit.
Surely we can do better than that.

TODO: Start with mutex and contended atomic, then go all the way to a parallel
      binary reduction tree for the final merging, with join-and-sum as an
      intermediary easy hack. May also benefit from sub-parts. Do not miss cheap
      shot on hyperthreading, but only after making sure we can't use it to
      put those integer ALUs to good use :)
