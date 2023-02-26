// ANCHOR: thread_basic
pub fn thread_basic(target: u64, sequential: impl Fn(u64) -> u64 + Sync) -> u64 {
    let num_threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(2);

    let base_share = target / num_threads as u64;
    let extra = target % num_threads as u64;
    let sequential = &sequential;

    std::thread::scope(|s| {
        let mut threads = Vec::with_capacity(num_threads);
        for thread in 0..num_threads as u64 {
            let share = base_share + (thread < extra) as u64;
            threads.push(s.spawn(move || sequential(share)));
        }
        threads.into_iter().map(|t| t.join().unwrap()).sum()
    })
}
// ANCHOR_END: thread_basic

#[cfg(test)]
mod tests {
    crate::test_counter!((thread_basic, |target| super::thread_basic(
        target,
        crate::simd::multiversion::multiversion_avx2
    )));
}
