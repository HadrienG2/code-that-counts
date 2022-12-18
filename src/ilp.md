# Instruction-Level Parallelism

Reaching the end of the previous chapter, you may think that the naive `for`
loop is optimal, since it increments the counter on every clock cycle so it
obviously keeps the CPU busy all the time. However, you would be wrong.

You see, modern CPUs are superscalar, which means that they can do multiple
things per clock cyle. In the case of my Zen 2 CPU, you can check
[over at uops.info](https://uops.info/html-instr/INC_R64.html) that it can
actually do four integer increments per cycle. So, how come we only get one?

The problem here is that the CPU may only execute multiple instructions at the
same time if these instructions are independent from each other. But here, each
counter value depends on the previous one, so the counter increments are not
independent and must await each other. To get instruction level parallelism, we
need to maintain multiple independent counters and spread the counting work
across them like this:

```rust,no_run
{{#include ../counter/src/lib.rs:ilp}}
```

Notice that we need extra code in the end to merge counters and manage those
odd elements that don't fit into an integer multiple of the number of counters.
This will be a recuring theme in the remainder of this book.

Given the above assertion that the CPU can do 4 increments per cycle, you may
expect optimal instruction-level performance to be achieved with 4 independent
counters. But due to the extra instructions needed to manage the counting
loop, and the fact that adding counters has the side-effect of forcing the
compiler to do extra unrolling which amortizes that overhead, performance
improvements are actually observed all the way up to 15 counters, which is the
maximum supported by the x86_64 architecture before counters start being spilled
from registers to RAM.

With those 15 counters, we observe a peak throughput of 3.6 increments per
cycle, a little less than the 4 we would have hoped for but still pretty close.

In terms of scaling with the number of counting iterations, the performance is
close to that of sequential counting with only one iteration of parallel
counting (15 increments), and beats it with two iterations. Here, we see that
instruction-level parallelism is actually a good way to get performance in
large loops without sacrificing the performance of smaller loops too much.
