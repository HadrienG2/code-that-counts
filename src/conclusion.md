# Conclusion (WIP)

TODO: This is most certainly a pointless calculation, but it does provide a nice
      occasion to touch on several things that matter in real calculations, and
      it is also not that far from a real calculation (just add some form of
      masking to turn pointless exhaustive counting into an actually useful
      statistical primitive). Hope you enjoyed this thxbye.

      I should definitely also send a copy of this to Paul McKenney with some
      background info on how this is a parallel programming book joke gone too
      far.

      Also, look ma no unsafe !

      General observations:
      - Know your hardware specs, question why you're not there yet
      - Parallelize, parallelize, parallelize
      - Quite a difference between ILP & SIMD where you usually need to do all
        the work yourself but it's easy, and multithreading where you can often
        be satisfied with what a generic library gives you but when you aren't
        it's an order of magnitude more tricky.