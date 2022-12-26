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

#[cfg(all(test, target_feature = "sse2"))]
mod tests {
    crate::test_counter!(simd_basic);
}
