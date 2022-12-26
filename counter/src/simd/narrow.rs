use super::SimdAccumulator;

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

// ANCHOR: generic_narrow_simple
pub fn generic_narrow_simple<
    Counter: num_traits::PrimInt,
    Simd: crate::simd::SimdAccumulator<Counter>,
>(
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
// ANCHOR_END: generic_narrow_simple

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
    scalar_accumulators[0] + super::multiversion::multiversion_avx2(remainder)
}

#[cfg(target_feature = "avx2")]
pub fn narrow_u8_tuned(target: u64) -> u64 {
    generic_narrow_u8_tuned::<safe_arch::m256i>(target)
}
// ANCHOR_END: narrow_u8_tuned

#[cfg(all(test, target_feature = "avx2"))]
mod tests {
    crate::test_counters!(
        (narrow_simple_u8, super::narrow_simple::<u8>),
        (narrow_simple_u16, super::narrow_simple::<u16>),
        (narrow_simple_u32, super::narrow_simple::<u32>),
        narrow_u8,
        narrow_u8_tuned
    );
}
