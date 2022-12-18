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
pub fn ilp<const WIDTH: usize>(target: u64) -> u64 {
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
    let mut stride = WIDTH;
    while stride > 1 {
        stride /= 2;
        for i in 0..stride.min(WIDTH - stride) {
            counters[i] += counters[i + stride];
        }
    }
    counters[0]
}
// ANCHOR_END: ilp

#[cfg(test)]
mod tests {
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    fn test_counter(target: u64, mut counter: impl FnMut(u64) -> u64) -> TestResult {
        if target > (1 << 32) {
            TestResult::discard()
        } else {
            TestResult::from_bool(counter(target) == target)
        }
    }

    macro_rules! test_counter {
        ($name:ident) => {
            test_counter!(($name, super::$name));
        };
        (($name:ident, $imp:path)) => {
            #[quickcheck]
            fn $name(target: u64) -> TestResult {
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
        (ilp1, super::ilp::<1>),
        (ilp2, super::ilp::<2>),
        (ilp3, super::ilp::<3>),
        (ilp4, super::ilp::<4>),
        (ilp5, super::ilp::<5>),
        (ilp6, super::ilp::<6>),
        (ilp7, super::ilp::<7>),
        (ilp8, super::ilp::<8>),
        (ilp9, super::ilp::<9>),
        (ilp10, super::ilp::<10>),
        (ilp11, super::ilp::<11>),
        (ilp12, super::ilp::<12>),
        (ilp13, super::ilp::<13>),
        (ilp14, super::ilp::<14>),
        (ilp15, super::ilp::<15>),
        (ilp16, super::ilp::<16>)
    );
}
