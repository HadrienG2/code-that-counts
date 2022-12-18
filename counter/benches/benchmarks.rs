use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

pub fn criterion_benchmark(c: &mut Criterion) {
    for (group, counter) in [("basic", counter::basic)] {
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
