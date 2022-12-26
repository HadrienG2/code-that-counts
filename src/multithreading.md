# Multithreading

Now that I have taken single-core counting as far as I can (which is a little
less than 100 times the counting speed of a naive `for` loop), it is time to
leverage all the cores of my CPU for absolute best-in-class leading counting
performance.

On the surface, that sounds easy enough:

```rust,no_run
{{#include ../counter/src/thread/basic.rs:thread_basic}}
```

But this version doesn't perform super impressively, only achieving a 4.8x
speed-up on 8 CPU cores even when counting to a very high 69 billion limit.
Surely we can do better than that.
