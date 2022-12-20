use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use safe_arch::m128i;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut benchmarks: Vec<(&str, fn(u64) -> u64)> = vec![
        ("basic", counter::basic),
        ("ilp1", counter::basic_ilp::<1>),
        ("ilp14", counter::basic_ilp::<14>),
        ("ilp15", counter::basic_ilp::<15>),
        ("ilp16", counter::basic_ilp::<16>),
        ("simd_basic", counter::simd_basic),
        ("simd_ilp1", counter::simd_ilp::<1>),
        ("simd_ilp7", counter::simd_ilp::<7>),
        ("simd_ilp8", counter::simd_ilp::<8>),
        ("simd_ilp9", counter::simd_ilp::<9>),
        ("generic_ilp1_u64", counter::generic_ilp_u64::<1, u64>),
        ("generic_ilp15_u64", counter::generic_ilp_u64::<15, u64>),
        ("generic_ilp1_u64x2", counter::generic_ilp_u64::<1, m128i>),
        ("generic_ilp9_u64x2", counter::generic_ilp_u64::<9, m128i>),
    ];
    #[cfg(target_feature = "sse2")]
    {
        benchmarks.push(("multiversion_sse2", counter::multiversion_sse2));
    }
    #[cfg(target_feature = "avx2")]
    {
        benchmarks.push(("multiversion_avx2", counter::multiversion_avx2));
    }
    for (group, counter) in benchmarks {
        for size_pow2 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30] {
            let size = 1 << size_pow2;
            let mut group = c.benchmark_group(group);
            group.throughput(Throughput::Elements(size));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
                b.iter(|| counter(pessimize::hide(size)));
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
