use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

pub fn criterion_benchmark(c: &mut Criterion) {
    let benchmarks: &[(&str, fn(u64) -> u64)] = &[
        ("basic", counter::basic),
        ("ilp1", counter::ilp::<1>),
        ("ilp2", counter::ilp::<2>),
        ("ilp3", counter::ilp::<3>),
        ("ilp4", counter::ilp::<4>),
        ("ilp5", counter::ilp::<5>),
        ("ilp6", counter::ilp::<6>),
        ("ilp7", counter::ilp::<7>),
        ("ilp8", counter::ilp::<8>),
        ("ilp9", counter::ilp::<9>),
        ("ilp10", counter::ilp::<10>),
        ("ilp11", counter::ilp::<11>),
        ("ilp12", counter::ilp::<12>),
        ("ilp13", counter::ilp::<13>),
        ("ilp14", counter::ilp::<14>),
        ("ilp15", counter::ilp::<15>),
        ("ilp16", counter::ilp::<16>),
    ];
    for (group, counter) in benchmarks {
        for size_pow2 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30] {
            let size = 1 << size_pow2;
            let mut group = c.benchmark_group(*group);
            group.throughput(Throughput::Elements(size));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
                b.iter(|| counter(pessimize::hide(size)));
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
