pub mod basic;
pub mod ilp;
pub mod simd;
pub mod thread;

/// Tools used to test the various implementations
#[cfg(test)]
pub(crate) mod test_utils {
    use quickcheck::TestResult;

    /// Test a counter implementation by counting up to `target`
    pub fn test_counter_impl(target: u32, mut counter: impl FnMut(u64) -> u64) -> TestResult {
        if target > 1 << 24 {
            return TestResult::discard();
        }
        let target = target as u64;
        TestResult::from_bool(counter(target) == target)
    }

    /// Shorthand to generate a quickcheck test based on `test_counter_impl`
    ///
    /// If your counter implementation is a simple function, you can just do
    /// `test_counter!(my_counter)` and it will generate a test with the same
    /// name as the counter function..
    ///
    /// If it is a generic function or something more complicated, you can use
    /// the longer `test_counter!(name, counter)` where the first parameter is
    /// the test name and the second one is the counter implementation.
    ///
    #[doc(hidden)]
    #[macro_export]
    macro_rules! test_counter {
        ($name:ident) => {
            $crate::test_counter!(($name, super::$name));
        };
        (($name:ident, $imp:expr)) => {
            #[quickcheck_macros::quickcheck]
            fn $name(target: u32) -> quickcheck::TestResult {
                $crate::test_utils::test_counter_impl(target, $imp)
            }
        };
    }

    /// Shorthand to generate multiple tests using `test_counter!`.
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
