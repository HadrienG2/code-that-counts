use pessimize::Pessimize;

// ANCHOR: basic
pub fn basic(target: u64) -> u64 {
    let mut current = 0;
    for _ in 0..target {
        current = pessimize::hide(current + 1);
    }
    current
}
// ANCHOR_END: basic

// ANCHOR: ilp
pub fn basic_ilp<const WIDTH: usize>(target: u64) -> u64 {
    assert_ne!(WIDTH, 0, "No progress possible in this configuration");

    // Accumulate in parallel
    let mut counters = [0; WIDTH];
    for _ in 0..(target / WIDTH as u64) {
        for counter in &mut counters {
            *counter = pessimize::hide(*counter + 1);
        }
    }

    // Accumulate remaining elements
    for counter in counters.iter_mut().take(target as usize % WIDTH) {
        *counter = pessimize::hide(*counter + 1);
    }

    // Merge accumulators using parallel reduction
    let mut stride = WIDTH.next_power_of_two() / 2;
    while stride > 0 {
        for i in 0..stride.min(WIDTH - stride) {
            counters[i] += counters[i + stride];
        }
        stride /= 2;
    }
    counters[0]
}
// ANCHOR_END: ilp

#[cfg(target_feature = "sse2")]
// ANCHOR: simd_basic
pub fn simd_basic(target: u64) -> u64 {
    use safe_arch::m128i;
    const SIMD_WIDTH: usize = 2;

    // Set up SIMD counters and increment
    let mut simd_counter = m128i::from([0u64; SIMD_WIDTH]);
    let simd_increment = m128i::from([1u64; SIMD_WIDTH]);

    // Accumulate in parallel
    for _ in 0..(target / SIMD_WIDTH as u64) {
        simd_counter = pessimize::hide(safe_arch::add_i64_m128i(simd_counter, simd_increment));
    }

    // Merge the SIMD counters into a scalar counter
    let counters: [u64; SIMD_WIDTH] = simd_counter.into();
    let mut counter = counters.iter().sum();

    // Accumulate trailing element, if any
    counter = pessimize::hide(counter + target % SIMD_WIDTH as u64);
    counter
}
// ANCHOR_END: simd_basic

#[cfg(target_feature = "sse2")]
// ANCHOR: simd_ilp
pub fn simd_ilp<const ILP_WIDTH: usize>(target: u64) -> u64 {
    use safe_arch::m128i;
    const SIMD_WIDTH: usize = 2;
    assert_ne!(ILP_WIDTH, 0, "No progress possible in this configuration");

    // Set up counters and increment
    let mut simd_counters = [m128i::from([0u64; SIMD_WIDTH]); ILP_WIDTH];
    let simd_increment = m128i::from([1u64; SIMD_WIDTH]);

    // Accumulate in parallel
    let full_width = (SIMD_WIDTH * ILP_WIDTH) as u64;
    for _ in 0..(target / full_width) {
        for simd_counter in &mut simd_counters {
            *simd_counter =
                pessimize::hide(safe_arch::add_i64_m128i(*simd_counter, simd_increment));
        }
    }

    // Accumulate remaining pairs of elements
    let mut remainder = (target % full_width) as usize;
    while remainder >= SIMD_WIDTH {
        for simd_counter in simd_counters.iter_mut().take(remainder / SIMD_WIDTH) {
            *simd_counter =
                pessimize::hide(safe_arch::add_i64_m128i(*simd_counter, simd_increment));
            remainder -= SIMD_WIDTH;
        }
    }

    // Merge SIMD accumulators using parallel reduction
    let mut stride = ILP_WIDTH.next_power_of_two() / 2;
    while stride > 0 {
        for i in 0..stride.min(ILP_WIDTH - stride) {
            simd_counters[i] =
                safe_arch::add_i64_m128i(simd_counters[i], simd_counters[i + stride]);
        }
        stride /= 2;
    }
    let simd_counter = simd_counters[0];

    // Merge the SIMD counters into a scalar counter
    let counters: [u64; SIMD_WIDTH] = simd_counter.into();
    let mut counter = counters.iter().sum();

    // Accumulate trailing element, if any, the scalar way
    for _ in 0..remainder {
        counter = pessimize::hide(counter + 1);
    }
    counter
}
// ANCHOR_END: simd_ilp

#[cfg(target_feature = "sse2")]
// ANCHOR: extreme_ilp
pub fn extreme_ilp<
    // Number of SIMD operations per cycle
    const SIMD_ILP: usize,
    // Number of scalar iterations per cycle
    const SCALAR_ILP: usize,
    // Number of cycles in the inner loop
    const UNROLL_FACTOR: usize,
    // Number of scalar instructions needed for outer loop maintenance
    const LOOP_INSNS: usize,
>(
    target: u64,
) -> u64 {
    use safe_arch::m128i;
    const SIMD_WIDTH: usize = 2;
    assert_ne!(SIMD_ILP, 0, "No point in this without SIMD ops");
    assert_ne!(SCALAR_ILP, 0, "No point in this without scalar ops");

    // Set up counters and increment
    let mut simd_counters = [m128i::from([0u64; SIMD_WIDTH]); SIMD_ILP];
    let simd_increment = m128i::from([1u64; SIMD_WIDTH]);
    let mut scalar_counters = [0; SCALAR_ILP];

    // Accumulate in parallel
    let unrolled_simd_ops = SIMD_WIDTH * SIMD_ILP * UNROLL_FACTOR;
    let unrolled_scalar_ops = (SCALAR_ILP * UNROLL_FACTOR).saturating_sub(LOOP_INSNS);
    let full_width = unrolled_simd_ops + unrolled_scalar_ops;
    for _ in 0..(target / full_width as u64) {
        for unroll_iter in 0..UNROLL_FACTOR {
            for simd_counter in &mut simd_counters {
                *simd_counter =
                    pessimize::hide(safe_arch::add_i64_m128i(*simd_counter, simd_increment));
            }
            for scalar_counter in scalar_counters
                .iter_mut()
                .take(unrolled_scalar_ops - unroll_iter * SCALAR_ILP)
            {
                *scalar_counter = pessimize::hide(*scalar_counter + 1)
            }
        }
    }

    // Accumulate remaining elements using as much parallelism as possible
    let mut remainder = (target % full_width as u64) as usize;
    while remainder > SIMD_WIDTH {
        for simd_counter in simd_counters.iter_mut().take(remainder / SIMD_WIDTH) {
            *simd_counter =
                pessimize::hide(safe_arch::add_i64_m128i(*simd_counter, simd_increment));
            remainder -= SIMD_WIDTH;
        }
    }
    while remainder > 0 {
        for scalar_counter in scalar_counters.iter_mut().take(remainder) {
            *scalar_counter = pessimize::hide(*scalar_counter + 1);
            remainder -= 1;
        }
    }

    // Merge accumulators using parallel reduction
    let mut stride = (SIMD_ILP.max(SCALAR_ILP)).next_power_of_two() / 2;
    while stride > 0 {
        for i in 0..stride.min(SIMD_ILP.saturating_sub(stride)) {
            simd_counters[i] =
                safe_arch::add_i64_m128i(simd_counters[i], simd_counters[i + stride]);
        }
        for i in 0..stride.min(SCALAR_ILP.saturating_sub(stride)) {
            scalar_counters[i] += scalar_counters[i + stride];
        }
        stride /= 2;
    }
    let simd_counter = simd_counters[0];
    let mut scalar_counter = scalar_counters[0];

    // Merge the SIMD counters and scalar counter into one
    let counters: [u64; SIMD_WIDTH] = simd_counter.into();
    scalar_counter += counters.iter().sum::<u64>();
    scalar_counter
}
// ANCHOR_END: extreme_ilp

// ANCHOR: Accumulator
/// Set of integer counters with SIMD semantics
pub trait SimdAccumulator<Counter>: Copy + Eq + Pessimize + Sized {
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
// ANCHOR_END: Accumulator

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

// ANCHOR: generic_ilp
pub fn generic_ilp_u64<const ILP_WIDTH: usize, Simd: SimdAccumulator<u64>>(target: u64) -> u64 {
    assert_ne!(ILP_WIDTH, 0, "No progress possible in this configuration");

    // Set up counters
    let mut simd_accumulators = [Simd::zeros(); ILP_WIDTH];

    // Accumulate in parallel
    let full_width = (Simd::WIDTH * ILP_WIDTH) as u64;
    for _ in 0..(target / full_width) {
        for simd_accumulator in &mut simd_accumulators {
            simd_accumulator.increment();
        }
    }

    // Accumulate remaining SIMD vectors of elements
    let mut remainder = (target % full_width) as usize;
    while remainder >= Simd::WIDTH {
        for simd_accumulator in simd_accumulators.iter_mut().take(remainder / Simd::WIDTH) {
            simd_accumulator.increment();
            remainder -= Simd::WIDTH;
        }
    }

    // Merge SIMD accumulators using parallel reduction
    let mut stride = ILP_WIDTH.next_power_of_two() / 2;
    while stride > 0 {
        for i in 0..stride.min(ILP_WIDTH - stride) {
            simd_accumulators[i].merge(simd_accumulators[i + stride]);
        }
        stride /= 2;
    }
    let simd_accumulator = simd_accumulators[0];

    // Merge the SIMD counters into a scalar counter
    let mut counter = simd_accumulator.reduce();

    // Accumulate trailing element, if any, the scalar way
    for _ in 0..remainder {
        counter.increment();
    }
    counter
}
// ANCHOR_END: generic_ilp

// ANCHOR: multiversion_sse2
#[cfg(target_feature = "sse2")]
pub fn multiversion_sse2(target: u64) -> u64 {
    generic_ilp_u64::<9, safe_arch::m128i>(target)
}

#[cfg(not(target_feature = "sse2"))]
pub fn multiversion_sse2(target: u64) -> u64 {
    generic_ilp_u64::<15, u64>(target)
}
// ANCHOR_END: multiversion_sse2

// ANCHOR: avx2
#[cfg(target_feature = "avx2")]
impl SimdAccumulator<u64> for safe_arch::m256i {
    #[inline(always)]
    fn identity(one: bool) -> Self {
        Self::from([one as u64; <Self as SimdAccumulator<u64>>::WIDTH])
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        safe_arch::add_i64_m256i(self, other)
    }

    // AVX vectors of 64-bit integers reduce to SSE vectors of 64-bit integers
    type ReducedCounter = u64;
    type ReducedAccumulator = safe_arch::m128i;
    //
    #[inline(always)]
    fn reduce_step(self) -> [Self::ReducedAccumulator; 2] {
        [
            safe_arch::extract_m128i_m256i::<0>(self),
            safe_arch::extract_m128i_m256i::<1>(self),
        ]
    }
}

#[cfg(target_feature = "avx2")]
pub fn multiversion_avx2(target: u64) -> u64 {
    generic_ilp_u64::<9, safe_arch::m256i>(target)
}

#[cfg(not(target_feature = "avx2"))]
pub fn multiversion_avx2(target: u64) -> u64 {
    multiversion_sse2(target)
}
// ANCHOR_END: avx2

// ANCHOR: narrow_SimdAccumulator
#[cfg(target_feature = "avx2")]
impl SimdAccumulator<u8> for safe_arch::m256i {
    #[inline(always)]
    fn identity(one: bool) -> Self {
        Self::from([one as u8; <Self as SimdAccumulator<u8>>::WIDTH])
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        safe_arch::add_i8_m256i(self, other)
    }

    // AVX vectors of 8-bit integers reduce to AVX vectors of 16-bit integers
    type ReducedCounter = u16;
    type ReducedAccumulator = safe_arch::m256i;
    //
    #[inline(always)]
    fn reduce_step(self) -> [Self::ReducedAccumulator; 2] {
        use safe_arch::m256i;
        fn extract_u16_m256i_from_u8_m256i<const LANE: i32>(u8s: m256i) -> m256i {
            let half = safe_arch::extract_m128i_m256i::<LANE>(u8s);
            safe_arch::convert_to_i16_m256i_from_u8_m128i(half)
        }
        [
            extract_u16_m256i_from_u8_m256i::<0>(self),
            extract_u16_m256i_from_u8_m256i::<1>(self),
        ]
    }
}

#[cfg(target_feature = "avx2")]
impl SimdAccumulator<u16> for safe_arch::m256i {
    #[inline(always)]
    fn identity(one: bool) -> Self {
        Self::from([one as u16; <Self as SimdAccumulator<u16>>::WIDTH])
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        safe_arch::add_i16_m256i(self, other)
    }

    // AVX vectors of 16-bit integers reduce to AVX vectors of 32-bit integers
    type ReducedCounter = u32;
    type ReducedAccumulator = safe_arch::m256i;
    //
    #[inline(always)]
    fn reduce_step(self) -> [Self::ReducedAccumulator; 2] {
        use safe_arch::m256i;
        fn extract_u32_m256i_from_u16_m256i<const LANE: i32>(u16s: m256i) -> m256i {
            let half = safe_arch::extract_m128i_m256i::<LANE>(u16s);
            safe_arch::convert_to_i32_m256i_from_u16_m128i(half)
        }
        [
            extract_u32_m256i_from_u16_m256i::<0>(self),
            extract_u32_m256i_from_u16_m256i::<1>(self),
        ]
    }
}

#[cfg(target_feature = "avx2")]
impl SimdAccumulator<u32> for safe_arch::m256i {
    #[inline(always)]
    fn identity(one: bool) -> Self {
        Self::from([one as u32; <Self as SimdAccumulator<u32>>::WIDTH])
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        safe_arch::add_i32_m256i(self, other)
    }

    // AVX vectors of 32-bit integers reduce to AVX vectors of 64-bit integers
    type ReducedCounter = u64;
    type ReducedAccumulator = safe_arch::m256i;
    //
    #[inline(always)]
    fn reduce_step(self) -> [Self::ReducedAccumulator; 2] {
        use safe_arch::m256i;
        fn extract_u64_m256i_from_u32_m256i<const LANE: i32>(u16s: m256i) -> m256i {
            let half = safe_arch::extract_m128i_m256i::<LANE>(u16s);
            safe_arch::convert_to_i64_m256i_from_u32_m128i(half)
        }
        [
            extract_u64_m256i_from_u32_m256i::<0>(self),
            extract_u64_m256i_from_u32_m256i::<1>(self),
        ]
    }
}
// ANCHOR_END: narrow_SimdAccumulator

// ANCHOR: generic_ilp_simple
pub fn generic_narrow_simple<Counter: num_traits::PrimInt, Simd: SimdAccumulator<Counter>>(
    target: u64,
) -> u64 {
    const ILP_WIDTH: usize = 10;

    // Set up narrow SIMD counters and wide scalar counters
    let mut simd_accumulators = [Simd::zeros(); ILP_WIDTH];
    let mut scalar_accumulators = [0u64; ILP_WIDTH];

    // Set up overflow avoidance through spilling to scalar counters
    let mut counter_usage = Counter::zero();
    let spill = |simd_accumulators: &mut [Simd; ILP_WIDTH],
                 scalar_accumulators: &mut [u64; ILP_WIDTH]| {
        for (scalar, simd) in scalar_accumulators
            .iter_mut()
            .zip(simd_accumulators.into_iter())
        {
            *scalar += simd.reduce();
        }
        for simd in simd_accumulators {
            *simd = Simd::zeros();
        }
    };

    // Accumulate in parallel
    let mut remainder = target;
    let full_width = (Simd::WIDTH * ILP_WIDTH) as u64;
    while remainder >= full_width {
        // Perform a round of counting
        for simd_accumulator in &mut simd_accumulators {
            simd_accumulator.increment();
        }
        remainder -= full_width;

        // When the narrow SIMD counters fill up, spill to scalar counters
        counter_usage = counter_usage + Counter::one();
        if counter_usage == Counter::max_value() {
            spill(&mut simd_accumulators, &mut scalar_accumulators);
            counter_usage = Counter::zero();
        }
    }

    // Merge SIMD accumulators into scalar counters
    spill(&mut simd_accumulators, &mut scalar_accumulators);

    // Accumulate remaining elements in scalar counters
    while remainder > 0 {
        for scalar_accumulator in scalar_accumulators.iter_mut().take(remainder as usize) {
            scalar_accumulator.increment();
            remainder -= 1;
        }
    }

    // Merge scalar accumulators using parallel reduction
    let mut stride = ILP_WIDTH.next_power_of_two() / 2;
    while stride > 0 {
        for i in 0..stride.min(ILP_WIDTH - stride) {
            scalar_accumulators[i].merge(scalar_accumulators[i + stride]);
        }
        stride /= 2;
    }
    scalar_accumulators[0]
}

#[cfg(target_feature = "avx2")]
pub fn narrow_simple<Counter: num_traits::PrimInt>(target: u64) -> u64
where
    safe_arch::m256i: SimdAccumulator<Counter>,
{
    generic_narrow_simple::<Counter, safe_arch::m256i>(target)
}
// ANCHOR_END: generic_ilp_simple

// ANCHOR: U8Accumulator
struct U8Accumulator<const ILP_WIDTH: usize, Simd: SimdAccumulator<u8> + SimdAccumulator<u16>> {
    /// SIMD 8-bit integer accumulators
    simd_u8s: [Simd; ILP_WIDTH],

    /// Number of increments that occured in u8s
    u8s_usage: u8,

    /// SIMD 16-bit integer accumulators
    simd_u16s: [Simd; ILP_WIDTH],

    /// Number of u8s spills that occured in u16s
    u16s_usage: u8,

    /// Scalar integer accumulators
    scalars: [u64; ILP_WIDTH],
}
//
impl<
        const ILP_WIDTH: usize,
        Simd: SimdAccumulator<u8, ReducedAccumulator = Simd> + SimdAccumulator<u16>,
    > U8Accumulator<ILP_WIDTH, Simd>
{
    /// Total width including ILP
    pub const WIDTH: usize = ILP_WIDTH * <Simd as SimdAccumulator<u8>>::WIDTH;

    /// Set up accumulator
    pub fn new() -> Self {
        Self {
            simd_u8s: [<Simd as SimdAccumulator<u8>>::zeros(); ILP_WIDTH],
            u8s_usage: 0,
            simd_u16s: [<Simd as SimdAccumulator<u16>>::zeros(); ILP_WIDTH],
            u16s_usage: 0,
            scalars: [0; ILP_WIDTH],
        }
    }

    /// Increment counters
    #[inline(always)]
    pub fn increment(&mut self) {
        // Perfom a 8-bit increment
        for simd_u8 in &mut self.simd_u8s {
            <Simd as SimdAccumulator<u8>>::increment(simd_u8);
        }
        self.u8s_usage += 1;

        // Spill to 16-bit counters if it's time
        if self.u8s_usage == u8::MAX {
            self.spill_u8s_to_u16s();
            self.u8s_usage = 0;
            self.u16s_usage += 1;
        }

        // Spill to scalar counters if it's time
        if self.u16s_usage == (u16::MAX / (2 * u8::MAX as u16)) as u8 {
            self.spill_u16s_to_scalars();
            self.u16s_usage = 0;
        }
    }

    /// Spill SIMD counters and extract scalar counters
    pub fn scalarize(mut self) -> [u64; ILP_WIDTH] {
        self.spill_u8s_to_u16s();
        self.spill_u16s_to_scalars();
        self.scalars
    }

    /// Spill 8-bit SIMD accumulators into matching 16-bit ones
    #[inline(always)]
    fn spill_u8s_to_u16s(&mut self) {
        for (simd_u8, simd_u16) in self.simd_u8s.iter_mut().zip(self.simd_u16s.iter_mut()) {
            fn spill_one<SimdU8, SimdU16>(simd_u8: &mut SimdU8, simd_u16: &mut SimdU16)
            where
                SimdU8: SimdAccumulator<u8, ReducedAccumulator = SimdU16>,
                SimdU16: SimdAccumulator<u16>,
            {
                let [mut u16_contrib, u16_contrib_2] = simd_u8.reduce_step();
                u16_contrib.merge(u16_contrib_2);
                simd_u16.merge(u16_contrib);
                *simd_u8 = SimdU8::zeros();
            }
            spill_one(simd_u8, simd_u16);
        }
    }

    /// Spill 16-bit SIMD accumulators into matching scalar ones
    #[inline(always)]
    fn spill_u16s_to_scalars(&mut self) {
        for (simd_u16, scalar) in self.simd_u16s.iter_mut().zip(self.scalars.iter_mut()) {
            fn spill_one<SimdU16: SimdAccumulator<u16>>(simd_u16: &mut SimdU16, scalar: &mut u64) {
                scalar.merge(simd_u16.reduce());
                *simd_u16 = SimdU16::zeros();
            }
            spill_one(simd_u16, scalar);
        }
    }
}
// ANCHOR_END: U8Accumulator

// ANCHOR: narrow_u8
pub fn generic_narrow_u8<Simd>(target: u64) -> u64
where
    Simd: SimdAccumulator<u8, ReducedAccumulator = Simd> + SimdAccumulator<u16>,
{
    const ILP_WIDTH: usize = 10;

    // Set up accumulators
    let mut simd_accumulator = U8Accumulator::<ILP_WIDTH, Simd>::new();

    // Accumulate in parallel
    let mut remainder = target;
    let full_width = U8Accumulator::<ILP_WIDTH, Simd>::WIDTH as u64;
    while remainder >= full_width {
        simd_accumulator.increment();
        remainder -= full_width;
    }

    // Merge SIMD accumulators into scalar counters
    let mut scalar_accumulators = simd_accumulator.scalarize();

    // Accumulate remaining elements in scalar counters
    while remainder > 0 {
        for scalar_accumulator in scalar_accumulators.iter_mut().take(remainder as usize) {
            scalar_accumulator.increment();
            remainder -= 1;
        }
    }

    // Merge scalar accumulators using parallel reduction
    let mut stride = ILP_WIDTH.next_power_of_two() / 2;
    while stride > 0 {
        for i in 0..stride.min(ILP_WIDTH - stride) {
            scalar_accumulators[i].merge(scalar_accumulators[i + stride]);
        }
        stride /= 2;
    }
    scalar_accumulators[0]
}

#[cfg(target_feature = "avx2")]
pub fn narrow_u8(target: u64) -> u64 {
    generic_narrow_u8::<safe_arch::m256i>(target)
}
// ANCHOR_END: narrow_u8

// ANCHOR: narrow_u8_tuned
pub fn generic_narrow_u8_tuned<Simd>(target: u64) -> u64
where
    Simd: SimdAccumulator<u8, ReducedAccumulator = Simd> + SimdAccumulator<u16>,
{
    const ILP_WIDTH: usize = 10;

    // Set up accumulators
    let mut simd_accumulator = U8Accumulator::<ILP_WIDTH, Simd>::new();

    // Accumulate in parallel
    let mut remainder = target;
    let full_width = U8Accumulator::<ILP_WIDTH, Simd>::WIDTH as u64;
    while remainder >= full_width {
        simd_accumulator.increment();
        remainder -= full_width;
    }

    // Merge SIMD accumulators into scalar counters
    let mut scalar_accumulators = simd_accumulator.scalarize();

    // Merge scalar accumulators using parallel reduction
    let mut stride = ILP_WIDTH.next_power_of_two() / 2;
    while stride > 0 {
        for i in 0..stride.min(ILP_WIDTH - stride) {
            scalar_accumulators[i].merge(scalar_accumulators[i + stride]);
        }
        stride /= 2;
    }
    scalar_accumulators[0] + multiversion_avx2(remainder)
}

#[cfg(target_feature = "avx2")]
pub fn narrow_u8_tuned(target: u64) -> u64 {
    generic_narrow_u8_tuned::<safe_arch::m256i>(target)
}
// ANCHOR_END: narrow_u8_tuned

// ANCHOR: thread_basic
pub fn thread_basic(target: u64, sequential: impl Fn(u64) -> u64 + Sync) -> u64 {
    let num_threads = std::thread::available_parallelism()
        .map(|nzu| usize::from(nzu))
        .unwrap_or(2);

    let base_share = target / num_threads as u64;
    let extra = target % num_threads as u64;
    let sequential = &sequential;

    std::thread::scope(|s| {
        let mut threads = Vec::with_capacity(num_threads);
        for thread in 0..num_threads as u64 {
            let share = base_share + (thread < extra) as u64;
            threads.push(s.spawn(move || sequential(share)));
        }
        threads.into_iter().map(|t| t.join().unwrap()).sum()
    })
}
// ANCHOR_END: thread_basic

// ANCHOR: thread_rayon
pub fn thread_rayon(target: u64, sequential: impl Fn(u64) -> u64 + Sync) -> u64 {
    use rayon::prelude::*;
    rayon::iter::split(target, |target| {
        let half1 = target / 2;
        let half2 = target - half1;
        (half2, (half1 > 0).then_some(half1))
    })
    .map(&sequential)
    .sum()
}
// ANCHOR_END: thread_rayon

// ANCHOR: thread_custom
pub struct BackgroundThreads<Counter: Fn(u64) -> u64 + std::panic::RefUnwindSafe + Sync + 'static> {
    /// Worker threads
    _workers: Box<[std::thread::JoinHandle<()>]>,

    /// State shared with worker threads
    state: std::sync::Arc<SharedState<Counter>>,
}
//
impl<Counter: Fn(u64) -> u64 + std::panic::RefUnwindSafe + Send + Sync + 'static>
    BackgroundThreads<Counter>
{
    /// Set up worker threads with a certain counter implementation
    pub fn start(counter: Counter) -> Self {
        use std::{sync::Arc, thread};

        let num_threads = std::thread::available_parallelism()
            .map(|nzu| usize::from(nzu))
            .unwrap_or(2);

        let state = Arc::new(SharedState::new(counter, num_threads));
        let _workers = (1..num_threads)
            .map(|thread_idx| {
                let state2 = state.clone();
                thread::spawn(move || state2.worker(thread_idx))
            })
            .collect();

        Self { _workers, state }
    }

    /// Count in parallel using the worker threads
    ///
    /// This wants &mut because the shared state is not meant for concurrent use
    ///
    pub fn count(&mut self, target: u64) -> u64 {
        self.state.count(target)
    }
}
//
impl<Counter: Fn(u64) -> u64 + std::panic::RefUnwindSafe + Sync + 'static> Drop
    for BackgroundThreads<Counter>
{
    /// Tell worker threads to exit on Drop
    fn drop(&mut self) {
        self.state.stop(0)
    }
}

/// State shared between the main thread and worker threads
struct SharedState<Counter: Fn(u64) -> u64 + Sync> {
    /// Counter implementation
    counter: Counter,

    /// Number of processing threads, including main thread
    num_threads: usize,

    /// Mechanism to synchronize at the start and end of jobs
    job_barrier: BasicJobBarrier,

    /// Computation result, synchronized using `num_working`
    result: atomic::Atomic<u64>,

    /// State protection mutex
    ///
    /// This mutex is not used to protect access to fully unsynchronized state,
    /// but rather to optimize batches of atomic read-modify-write operations.
    ///
    locked: std::sync::Mutex<LockedOps>,
}
//
impl<Counter: Fn(u64) -> u64 + std::panic::RefUnwindSafe + Sync> SharedState<Counter> {
    /// Set up shared state
    pub fn new(counter: Counter, num_threads: usize) -> Self {
        use atomic::Atomic;
        use std::sync::Mutex;
        Self {
            counter,
            num_threads,
            job_barrier: BasicJobBarrier::new(num_threads),
            result: Atomic::default(),
            locked: Mutex::new(LockedOps),
        }
    }

    /// Request worker threads to stop
    pub fn stop(&self, thread_idx: usize) {
        debug_log(thread_idx == 0, "sending the stop signal");
        self.job_barrier.stop();
    }

    /// Schedule counting work and wait for the end result
    pub fn count(&self, target: u64) -> u64 {
        use atomic::Ordering;

        // Handle sequential special cases
        if self.num_threads == 1 || target <= BasicJobBarrier::MIN_PARALLEL_TARGET {
            return (self.counter)(target);
        }

        // Schedule job
        debug_log(true, "scheduling a job");
        self.result.store(0, Ordering::Relaxed);
        self.job_barrier.start(target, self.num_threads);

        // Do our share of the work and wait for results
        if !self.process(target, 0) {
            debug_log(true, "waiting for results");
            self.job_barrier.wait_for_end();
        }

        // Fetch result, piggybacking on thread::park()'s Acquire barrier
        self.result.load(Ordering::Relaxed)
    }

    /// Worker thread logic
    pub fn worker(&self, thread_idx: usize) {
        use std::panic::AssertUnwindSafe;
        if let Err(payload) = std::panic::catch_unwind(AssertUnwindSafe(|| {
            // Normal work loop
            let thread_idx = thread_idx as u64;
            debug_log(false, "waiting for the first job");
            while let Some(target) = self.job_barrier.wait_for_start() {
                self.process(target, thread_idx);
                debug_log(false, "waiting for the next job")
            }
            debug_log(false, "shutting down normally");
        })) {
            // In case of panic, tell others to stop before unwinding
            debug_log(false, "panicking");
            self.stop(thread_idx);
            std::panic::resume_unwind(payload);
        }
    }

    /// Process a task, then give away the lock and truth that the job is finished
    fn process(&self, target: u64, thread_idx: u64) -> bool {
        // Determine which share of the counting work we'll take
        let num_threads = self.num_threads as u64;
        let base_share = target / num_threads;
        let extra = target % num_threads;
        let share = base_share + (thread_idx < extra) as u64;

        // Do the counting work
        let is_main = thread_idx == 0;
        debug_log(is_main, "executing its task");
        let result = (self.counter)(share);

        // Merge our partial result into the global result
        debug_log(is_main, "merging its result contribution");
        let mut lock = self.lock_unwrap();
        lock.fetch_add(&self.result, result, atomic::Ordering::Relaxed);

        // Notify that we are done, check if the overall job is done
        debug_log(is_main, "done with its task");
        self.job_barrier.finish(lock)
    }

    /// Acquire shared state lock, propagating panics
    fn lock_unwrap(&self) -> std::sync::MutexGuard<LockedOps> {
        self.locked.lock().unwrap()
    }
}

/// Lock-protected operations
struct LockedOps;
//
impl LockedOps {
    /// Specialization of fetch_update for counter decrement that goes to zero
    pub fn fetch_dec(&mut self, target: &atomic::Atomic<usize>, order: atomic::Ordering) -> bool {
        self.fetch_update(target, order, |old| old > 0, |old| old - 1) == 1
    }

    /// Specialization of fetch_update for result aggregation
    pub fn fetch_add(&mut self, target: &atomic::Atomic<u64>, value: u64, order: atomic::Ordering) {
        self.fetch_update(
            target,
            order,
            |old| old <= u64::MAX - value,
            |old| old + value,
        );
    }

    /// Non-atomic read-modify_write operation that is actually atomic due to
    /// exclusive access to the locked state imposing it
    fn fetch_update<T: Copy>(
        &mut self,
        target: &atomic::Atomic<T>,
        order: atomic::Ordering,
        check: impl FnOnce(T) -> bool,
        change: impl FnOnce(T) -> T,
    ) -> T {
        assert!(atomic::Atomic::<T>::is_lock_free());
        let [load_order, store_order] = Self::rmw_order(order);
        let old = target.load(load_order);
        debug_assert!(check(old));
        target.store(change(old), store_order);
        old
    }

    /// Load and store ordering of non-atomic read-modify-write operations
    fn rmw_order(order: atomic::Ordering) -> [atomic::Ordering; 2] {
        use atomic::Ordering;
        match order {
            Ordering::Relaxed => [Ordering::Relaxed, Ordering::Relaxed],
            Ordering::Acquire => [Ordering::Acquire, Ordering::Relaxed],
            Ordering::Release => [Ordering::Relaxed, Ordering::Release],
            Ordering::AcqRel => [Ordering::Acquire, Ordering::Release],
            Ordering::SeqCst | _ => unimplemented!(),
        }
    }
}

/// Basic implementation of blocking synchronization at start and end of tasks
struct BasicJobBarrier {
    /// Number of worker threads that have work to do
    ///
    /// `num_working` is set to `num_threads` when work is submitted. Each time
    /// a thread is done with its task, it decrements this counter. So when it
    /// reaches 0, it tells the thread that last decremented it that all
    /// worker threads are done, with Acquire/Release synchronization.
    ///
    num_working: atomic::Atomic<usize>,

    /// Requested or ongoing computation
    ///
    /// This variable has two special values: 0 means no request and 1 means a
    /// request to stop.
    ///
    request: atomic::Atomic<u64>,

    /// Mechanism for worker threads to await a request.
    /// This can take the form of an incoming job or a termination request.
    barrier: std::sync::Barrier,
}
//
impl BasicJobBarrier {
    /// Set up barrier
    pub fn new(num_threads: usize) -> Self {
        use atomic::Atomic;
        use std::sync::Barrier;
        Self {
            num_working: Atomic::new(0),
            request: Atomic::new(Self::NO_REQUEST),
            barrier: Barrier::new(num_threads),
        }
    }

    /// Minimum accepted parallel job size, execute smaller jobs sequentially
    pub const MIN_PARALLEL_TARGET: u64 = Self::STOP_REQUEST + 1;

    /// Start a job
    pub fn start(&self, target: u64, num_threads: usize) {
        use atomic::Ordering;
        assert!(target >= Self::MIN_PARALLEL_TARGET);
        debug_assert_eq!(self.num_working.load(Ordering::Relaxed), 0);
        self.num_working.store(num_threads, Ordering::Relaxed);
        self.request.store(target, Ordering::Relaxed);
        self.barrier.wait();
    }

    /// Ask all worker threads to stop
    pub fn stop(&self) {
        self.request
            .store(Self::STOP_REQUEST, atomic::Ordering::Relaxed);
        self.barrier.wait();
    }

    /// Wait for a counting job or an exit signal
    ///
    /// This returns the full counting job, it is then up to this thread to
    /// figure out its share of work from it. None means the worker should stop.
    ///
    pub fn wait_for_start(&self) -> Option<u64> {
        let mut request = Self::NO_REQUEST;
        while request == Self::NO_REQUEST {
            self.barrier.wait();
            request = self.request.load(atomic::Ordering::Relaxed);
        }
        (request > Self::STOP_REQUEST).then_some(request)
    }

    /// Notify other threads that we are finished, tell if we were last
    pub fn finish(&self, mut lock: std::sync::MutexGuard<LockedOps>) -> bool {
        lock.fetch_dec(&self.num_working, atomic::Ordering::Release)
    }

    /// Wait for other worker threads to finish processing the job
    pub fn wait_for_end(&self) {
        use atomic::Ordering;
        assert!(
            self.spin_wait(|| self.num_working.load(Ordering::Relaxed) == 0),
            "Worker has panicked"
        );
        atomic::fence(Ordering::Acquire);
    }

    /// Special value of `request` that means there is no request
    const NO_REQUEST: u64 = 0;

    /// Special value of `request` that means threads should exit
    const STOP_REQUEST: u64 = 1;

    /// Spin on some condition, abort and return false on termination request
    fn spin_wait(&self, mut termination: impl FnMut() -> bool) -> bool {
        use atomic::Ordering;

        // Start with a spin loop with exponential backoff
        for backoff in 0..3 {
            if termination() {
                return true;
            }
            for _ in 0..1 << backoff {
                std::hint::spin_loop();
            }
        }

        // Switch to yielding to the OS once it's clear it's gonna take a while
        let mut exit = false;
        while !(termination() || exit) {
            exit = self.request.load(Ordering::Relaxed) == Self::STOP_REQUEST;
            std::thread::yield_now();
        }
        !exit
    }
}
// ANCHOR_END: thread_custom

/// Debug logging
fn debug_log(is_main: bool, action: &str) {
    if cfg!(debug_assertions) {
        static MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _lock = MUTEX.lock();
        if is_main {
            eprint!("Main thread");
        } else {
            eprint!("{:?}", std::thread::current().id());
        }
        eprintln!(" is {action}");
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    fn test_counter(target: u32, mut counter: impl FnMut(u64) -> u64) -> TestResult {
        if target > 1 << 24 {
            return TestResult::discard();
        }
        let target = target as u64;
        TestResult::from_bool(counter(target) == target)
    }

    macro_rules! test_counter {
        ($name:ident) => {
            test_counter!(($name, crate::$name));
        };
        (($name:ident, $imp:expr)) => {
            #[quickcheck]
            fn $name(target: u32) -> TestResult {
                test_counter(target, $imp)
            }
        };
    }
    //
    macro_rules! test_counters {
        ($($name:tt),+) => {
            $(
                test_counter!($name);
            )+
        };
    }
    //
    test_counters!(
        basic,
        (ilp1, crate::basic_ilp::<1>),
        (ilp14, crate::basic_ilp::<14>),
        (ilp15, crate::basic_ilp::<15>),
        (ilp16, crate::basic_ilp::<16>),
        (ilp17, crate::basic_ilp::<17>),
        multiversion_sse2,
        multiversion_avx2
    );

    #[cfg(target_feature = "sse2")]
    mod sse2 {
        use super::*;
        use safe_arch::m128i;

        test_counters!(
            simd_basic,
            (simd_ilp1, crate::simd_ilp::<1>),
            (simd_ilp9, crate::simd_ilp::<9>),
            (simd_ilp15, crate::simd_ilp::<15>),
            (simd_ilp16, crate::simd_ilp::<16>),
            (simd_ilp17, crate::simd_ilp::<17>),
            (extreme_ilp_1p1x1, crate::extreme_ilp::<1, 1, 1, 2>),
            (extreme_ilp_2p1x1, crate::extreme_ilp::<2, 1, 1, 2>),
            (extreme_ilp_3p1x1, crate::extreme_ilp::<3, 1, 1, 2>),
            (extreme_ilp_1p2x1, crate::extreme_ilp::<1, 2, 1, 2>),
            (extreme_ilp_3p2x2, crate::extreme_ilp::<3, 2, 2, 2>),
            (extreme_ilp_3p2x3, crate::extreme_ilp::<3, 2, 3, 2>),
            (extreme_ilp_3p2x4, crate::extreme_ilp::<3, 2, 4, 2>),
            (extreme_ilp_3p2x5, crate::extreme_ilp::<3, 2, 5, 2>),
            (generic_ilp1_u64, crate::generic_ilp_u64::<1, u64>),
            (generic_ilp14_u64, crate::generic_ilp_u64::<14, u64>),
            (generic_ilp15_u64, crate::generic_ilp_u64::<15, u64>),
            (generic_ilp16_u64, crate::generic_ilp_u64::<16, u64>),
            (generic_ilp17_u64, crate::generic_ilp_u64::<17, u64>),
            (generic_ilp1_u64x2, crate::generic_ilp_u64::<1, m128i>),
            (generic_ilp9_u64x2, crate::generic_ilp_u64::<9, m128i>),
            (generic_ilp15_u64x2, crate::generic_ilp_u64::<15, m128i>),
            (generic_ilp16_u64x2, crate::generic_ilp_u64::<16, m128i>),
            (generic_ilp17_u64x2, crate::generic_ilp_u64::<17, m128i>)
        );
    }

    #[cfg(target_feature = "avx2")]
    mod avx2 {
        use super::*;
        use crate::BackgroundThreads;
        use once_cell::sync::Lazy;
        use std::sync::Mutex;

        test_counters!(
            (narrow_simple_u8, crate::narrow_simple::<u8>),
            (narrow_simple_u16, crate::narrow_simple::<u16>),
            (narrow_simple_u32, crate::narrow_simple::<u32>),
            narrow_u8,
            narrow_u8_tuned,
            (thread_basic, |target| crate::thread_basic(
                target,
                crate::narrow_u8_tuned
            )),
            (thread_rayon, |target| crate::thread_rayon(
                target,
                crate::narrow_u8_tuned
            ))
        );

        #[quickcheck]
        fn thread_custom(target: u32) -> TestResult {
            use std::panic::RefUnwindSafe;
            type CounterBox = Box<dyn Fn(u64) -> u64 + RefUnwindSafe + Send + Sync + 'static>;
            static BACKGROUND: Lazy<Mutex<BackgroundThreads<CounterBox>>> = Lazy::new(|| {
                Mutex::new(BackgroundThreads::start(
                    Box::new(crate::narrow_u8_tuned) as _
                ))
            });
            let mut lock = BACKGROUND.lock().unwrap();
            test_counter(target, |target| lock.count(target))
        }
    }
}
