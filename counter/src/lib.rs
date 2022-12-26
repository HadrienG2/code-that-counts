pub mod basic;
pub mod ilp;
pub mod simd;
pub mod thread;

/// Tools used to test the various implementations
#[cfg(test)]
pub(crate) mod test_utils {
    use quickcheck::TestResult;

    pub fn test_counter(target: u32, mut counter: impl FnMut(u64) -> u64) -> TestResult {
        if target > 1 << 24 {
            return TestResult::discard();
        }
        let target = target as u64;
        TestResult::from_bool(counter(target) == target)
    }

    #[doc(hidden)]
    #[macro_export]
    macro_rules! test_counter {
        ($name:ident) => {
            $crate::test_counter!(($name, super::$name));
        };
        (($name:ident, $imp:expr)) => {
            #[quickcheck_macros::quickcheck]
            fn $name(target: u32) -> quickcheck::TestResult {
                $crate::test_utils::test_counter(target, $imp)
            }
        };
    }

    #[doc(hidden)]
    #[macro_export]
    macro_rules! test_counters {
        ($($name:tt),+) => {
            $(
                $crate::test_counter!($name);
            )+
        };
    }
}
