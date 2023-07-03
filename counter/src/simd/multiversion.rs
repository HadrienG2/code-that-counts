#[cfg(target_feature = "avx2")]
use super::SimdAccumulator;

// ANCHOR: multiversion_sse2
#[cfg(target_feature = "sse2")]
pub fn multiversion_sse2(target: u64) -> u64 {
    super::ilp::generic_ilp_u64::<9, safe_arch::m128i>(target)
}

#[cfg(not(target_feature = "sse2"))]
pub fn multiversion_sse2(target: u64) -> u64 {
    super::ilp::generic_ilp_u64::<15, u64>(target)
}
// ANCHOR_END: multiversion_sse2

// ANCHOR: avx2
#[cfg(target_feature = "avx2")]
impl SimdAccumulator<u64> for safe_arch::m256i {
    #[inline]
    fn identity(one: bool) -> Self {
        Self::from([one as u64; <Self as SimdAccumulator<u64>>::WIDTH])
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        safe_arch::add_i64_m256i(self, other)
    }

    // AVX vectors of 64-bit integers reduce to SSE vectors of 64-bit integers
    type ReducedCounter = u64;
    type ReducedAccumulator = safe_arch::m128i;
    //
    #[inline]
    fn reduce_step(self) -> [Self::ReducedAccumulator; 2] {
        [
            safe_arch::extract_m128i_m256i::<0>(self),
            safe_arch::extract_m128i_m256i::<1>(self),
        ]
    }
}

#[cfg(target_feature = "avx2")]
pub fn multiversion_avx2(target: u64) -> u64 {
    super::ilp::generic_ilp_u64::<9, safe_arch::m256i>(target)
}

#[cfg(not(target_feature = "avx2"))]
pub fn multiversion_avx2(target: u64) -> u64 {
    multiversion_sse2(target)
}
// ANCHOR_END: avx2

#[cfg(test)]
mod tests {
    crate::test_counters!(multiversion_sse2, multiversion_avx2);
}
