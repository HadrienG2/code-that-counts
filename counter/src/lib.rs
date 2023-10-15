pub mod basic;
pub mod ilp;
pub mod simd;
pub mod thread;

/// Tools used to test the various implementations
#[cfg(test)]
pub(crate) mod test_utils {
    /// Put a limit on how high we count to keep test durations in check
    pub(crate) const MAX_TARGET: u64 = 1 << 24;

    /// Shorthand to generate a proptest for a certain counter implementation
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
            proptest::proptest! {
                #[allow(clippy::redundant_closure_call)]
                #[test]
                fn $name(target in 0..$crate::test_utils::MAX_TARGET) {
                    proptest::prop_assert_eq!(($imp)(target), target);
                }
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
