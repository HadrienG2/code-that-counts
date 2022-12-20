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
/// Set of integer counters of type T with SIMD semantics
pub trait SimdAccumulator<T>: Copy + Pessimize + Sized {
    /// Number of inner accumulators
    const WIDTH: usize = std::mem::size_of::<Self>() / std::mem::size_of::<T>();

    /// Set up empty accumulators
    fn zeros() -> Self;

    /// Set up accumulators all set to 1
    fn ones() -> Self;

    /// Merge another set of accumulators into this one
    fn add(self, other: Self) -> Self;

    /// Reduce into a 64-bit counter
    fn reduce(self) -> u64;

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
}
// ANCHOR_END: Accumulator

// ANCHOR: implAccumulator
impl SimdAccumulator<u64> for u64 {
    #[inline(always)]
    fn zeros() -> Self {
        0
    }

    #[inline(always)]
    fn ones() -> Self {
        1
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self + other
    }

    #[inline(always)]
    fn reduce(self) -> u64 {
        self
    }
}

impl SimdAccumulator<u64> for safe_arch::m128i {
    #[inline(always)]
    fn zeros() -> Self {
        Self::from([0u64; Self::WIDTH])
    }

    #[inline(always)]
    fn ones() -> Self {
        Self::from([1u64; Self::WIDTH])
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        safe_arch::add_i64_m128i(self, other)
    }

    #[inline(always)]
    fn reduce(self) -> u64 {
        let counters: [u64; Self::WIDTH] = self.into();
        counters.iter().sum()
    }
}
// ANCHOR_END: implAccumulator

// ANCHOR: generic_ilp
pub fn generic_ilp<const ILP_WIDTH: usize, Counter, Simd: SimdAccumulator<Counter>>(
    target: u64,
) -> u64 {
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
    generic_ilp::<9, u64, safe_arch::m128i>(target)
}

#[cfg(not(target_feature = "sse2"))]
pub fn multiversion_sse2(target: u64) -> u64 {
    generic_ilp::<15, u64, u64>(target)
}
// ANCHOR_END: multiversion_sse2

// ANCHOR: avx2
#[cfg(target_feature = "avx2")]
impl SimdAccumulator<u64> for safe_arch::m256i {
    #[inline(always)]
    fn zeros() -> Self {
        Self::from([0u64; Self::WIDTH])
    }

    #[inline(always)]
    fn ones() -> Self {
        Self::from([1u64; Self::WIDTH])
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        safe_arch::add_i64_m256i(self, other)
    }

    #[inline(always)]
    fn reduce(self) -> u64 {
        use safe_arch::m128i;
        let mut half = safe_arch::extract_m128i_m256i::<0>(self);
        let half2 = safe_arch::extract_m128i_m256i::<1>(self);
        <m128i as SimdAccumulator<u64>>::merge(&mut half, half2);
        <m128i as SimdAccumulator<u64>>::reduce(half)
    }
}

#[cfg(target_feature = "avx2")]
pub fn multiversion_avx2(target: u64) -> u64 {
    generic_ilp::<9, u64, safe_arch::m256i>(target)
}

#[cfg(all(not(target_feature = "avx2"), target_feature = "sse2"))]
pub fn multiversion_avx2(target: u64) -> u64 {
    generic_ilp::<9, u64, safe_arch::m128i>(target)
}

#[cfg(not(target_feature = "sse2"))]
pub fn multiversion_avx2(target: u64) -> u64 {
    generic_ilp::<15, u64, u64>(target)
}
// ANCHOR_END: avx2

#[cfg(test)]
mod tests {
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;
    use safe_arch::m128i;

    fn test_counter(target: u32, mut counter: impl FnMut(u64) -> u64) -> TestResult {
        if target > 1 << 24 {
            return TestResult::discard();
        }
        let target = target as u64;
        TestResult::from_bool(counter(target) == target)
    }

    macro_rules! test_counter {
        ($name:ident) => {
            test_counter!(($name, super::$name));
        };
        (($name:ident, $imp:path)) => {
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
        (ilp1, super::basic_ilp::<1>),
        (ilp14, super::basic_ilp::<14>),
        (ilp15, super::basic_ilp::<15>),
        (ilp16, super::basic_ilp::<16>),
        (ilp17, super::basic_ilp::<17>),
        simd_basic,
        (simd_ilp1, super::simd_ilp::<1>),
        (simd_ilp9, super::simd_ilp::<9>),
        (simd_ilp15, super::simd_ilp::<15>),
        (simd_ilp16, super::simd_ilp::<16>),
        (simd_ilp17, super::simd_ilp::<17>),
        (extreme_ilp_1p1x1, super::extreme_ilp::<1, 1, 1, 2>),
        (extreme_ilp_2p1x1, super::extreme_ilp::<2, 1, 1, 2>),
        (extreme_ilp_3p1x1, super::extreme_ilp::<3, 1, 1, 2>),
        (extreme_ilp_1p2x1, super::extreme_ilp::<1, 2, 1, 2>),
        (extreme_ilp_3p2x2, super::extreme_ilp::<3, 2, 2, 2>),
        (extreme_ilp_3p2x3, super::extreme_ilp::<3, 2, 3, 2>),
        (extreme_ilp_3p2x4, super::extreme_ilp::<3, 2, 4, 2>),
        (extreme_ilp_3p2x5, super::extreme_ilp::<3, 2, 5, 2>),
        (generic_ilp1_u64, super::generic_ilp::<1, 1, u64>),
        (generic_ilp14_u64, super::generic_ilp::<14, 1, u64>),
        (generic_ilp15_u64, super::generic_ilp::<15, 1, u64>),
        (generic_ilp16_u64, super::generic_ilp::<16, 1, u64>),
        (generic_ilp17_u64, super::generic_ilp::<17, 1, u64>),
        (generic_ilp1_u64x2, super::generic_ilp::<1, 2, m128i>),
        (generic_ilp9_u64x2, super::generic_ilp::<9, 2, m128i>),
        (generic_ilp15_u64x2, super::generic_ilp::<15, 2, m128i>),
        (generic_ilp16_u64x2, super::generic_ilp::<16, 2, m128i>),
        (generic_ilp17_u64x2, super::generic_ilp::<17, 2, m128i>),
        multiversion_sse2
    );

    #[cfg(target_feature = "avx2")]
    test_counter!(multiversion_avx2);
}
