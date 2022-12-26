// ANCHOR: basic
pub fn basic(target: u64) -> u64 {
    let mut current = 0;
    for _ in 0..target {
        current = pessimize::hide(current + 1);
    }
    current
}
// ANCHOR_END: basic

#[cfg(test)]
mod tests {
    crate::test_counter!(basic);
}
