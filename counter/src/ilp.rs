// ANCHOR: ilp
pub fn ilp<const WIDTH: usize>(target: u64) -> u64 {
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

#[cfg(test)]
mod tests {
    crate::test_counters!(
        (ilp1, super::ilp::<1>),
        (ilp14, super::ilp::<14>),
        (ilp15, super::ilp::<15>),
        (ilp16, super::ilp::<16>),
        (ilp17, super::ilp::<17>)
    );
}
