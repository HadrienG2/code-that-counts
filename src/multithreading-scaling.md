# Scaling up

So far, by virtue of working on a "small" 8-core CPU, I have had the luxury of
not caring at all about scalability to large thread counts and still getting
decent synchronization performance. But now is the right time to start caring,
be it only to check my assumption that 8 cores is "few" cores.

TODO: Use reduction trees with const MAX_ARITY + lstopo-driven splitting.
