# Vectorization (WIP)

TODO: First basic 64-bit vectorization with SSE2, then add ILP, then add AVX via
      multiversion, then go wider by using smaller integers in vectors with
      occasional merging (need either slow extract loop on SSE2 16b or SSE 4.1
      for conversion), and finally use scalar units as well through ILP for a
      little extra. This could probably use sub-parts !
