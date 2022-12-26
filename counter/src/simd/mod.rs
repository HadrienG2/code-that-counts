pub mod basic;
pub mod ilp;
pub mod multiversion;
pub mod narrow;

// ANCHOR: SimdAccumulator
/// Set of integer counters with SIMD semantics
pub trait SimdAccumulator<Counter>: Copy + Eq + pessimize::Pessimize + Sized {
    /// Number of inner accumulators
    const WIDTH: usize = std::mem::size_of::<Self>() / std::mem::size_of::<Counter>();

    /// Set up accumulators, all initialized to 0 or 1
    fn identity(one: bool) -> Self;

    /// Set up accumulators initialized to 0
    #[inline(always)]
    fn zeros() -> Self {
        Self::identity(false)
    }

    /// Set up accumulators initialized to 1
    #[inline(always)]
    fn ones() -> Self {
        Self::identity(true)
    }

    /// Merge another set of accumulators into this one
    fn add(self, other: Self) -> Self;

    /// Add one to every accumulator in the set in a manner that cannot be
    /// optimized out by the compiler
    #[inline(always)]
    fn increment(&mut self) {
        *self = pessimize::hide(Self::add(*self, Self::ones()));
    }

    /// Merge another accumulator into this one
    #[inline(always)]
    fn merge(&mut self, other: Self) {
        *self = Self::add(*self, other);
    }

    /// Doubles the size of Counter if it's not u64 yet, else stays at u64
    type ReducedCounter;

    /// Goes to another SimdAccumulator type that is half as wide if Counter is
    /// u64 and WIDTH is not yet 1, else stays at Self.
    type ReducedAccumulator: SimdAccumulator<Self::ReducedCounter>;

    /// Go to the next step down the reduction pipeline
    fn reduce_step(self) -> [Self::ReducedAccumulator; 2];

    /// Reduce all the way to a 64-bit counter
    /// Must be specialized to the identity function for u64 scalars
    #[inline(always)]
    fn reduce(self) -> u64 {
        let [mut half, half2] = self.reduce_step();
        half.merge(half2);
        half.reduce()
    }
}
// ANCHOR_END: SimdAccumulator

#[cfg(target_feature = "sse2")]
// ANCHOR: implAccumulator
impl SimdAccumulator<u64> for safe_arch::m128i {
    #[inline(always)]
    fn identity(one: bool) -> Self {
        Self::from([one as u64; <Self as SimdAccumulator<u64>>::WIDTH])
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        safe_arch::add_i64_m128i(self, other)
    }

    // SSE vectors of 64-bit integers reduce to 64-bit integers
    type ReducedCounter = u64;
    type ReducedAccumulator = u64;
    //
    #[inline(always)]
    fn reduce_step(self) -> [Self::ReducedAccumulator; 2] {
        self.into()
    }
}

impl SimdAccumulator<u64> for u64 {
    #[inline(always)]
    fn identity(one: bool) -> Self {
        one as u64
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self + other
    }

    // Scalar u64 is the end of the reduction recursion
    type ReducedCounter = u64;
    type ReducedAccumulator = u64;
    //
    #[inline(always)]
    fn reduce_step(self) -> [Self::ReducedAccumulator; 2] {
        [self, 0]
    }
    //
    #[inline(always)]
    fn reduce(self) -> u64 {
        self
    }
}
// ANCHOR_END: implAccumulator
