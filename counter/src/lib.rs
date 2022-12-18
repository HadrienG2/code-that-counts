pub fn basic(target: u64) -> u64 {
    let mut current = 0;
    for _ in 0..target {
        current += 1;
        pessimize::consume(current);
    }
    current
}

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

    macro_rules! test_counters {
        ($($name:ident),+) => {
            $(
                #[quickcheck]
                fn $name(target: u64) -> TestResult {
                    test_counter(target, super::$name)
                }
            )+
        };
    }
    //
    test_counters!(basic);
}
