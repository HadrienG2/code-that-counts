// ANCHOR: thread_rayon
pub fn thread_rayon(target: u64, sequential: impl Fn(u64) -> u64 + Sync) -> u64 {
    use rayon::prelude::*;
    rayon::iter::split(target, |target| {
        let half1 = target / 2;
        let half2 = target - half1;
        (half2, (half1 > 0).then_some(half1))
    })
    .map(&sequential)
    .sum()
}
// ANCHOR_END: thread_rayon

#[cfg(test)]
mod tests {
    crate::test_counters!((thread_rayon, |target| super::thread_rayon(
        target,
        crate::simd::multiversion::multiversion_avx2
    )));
}
