use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hwloc2::{CpuBindFlags, CpuSet, ObjectType, Topology, TopologyObject};
use rayon::ThreadPoolBuilder;
use std::fmt::Write;

pub fn criterion_benchmark(c: &mut Criterion) {
    // Construct benchmark set
    let mut benchmarks = benchmarks();

    // Discover CPU topology and generate multithreading configurations, in
    // rough order of decreasing performance (so the benchmarks that take
    // forever to run come later in an unattended cargo bench run)
    let mut cpusets = Vec::new();
    let mut topology = Topology::new().unwrap();
    let full_cpuset = topology.get_cpubind(CpuBindFlags::CPUBIND_PROCESS).unwrap();
    let num_cpus = full_cpuset.weight() as u32;
    cpusets.push(("".to_string(), full_cpuset.clone()));
    //
    let root = topology.object_at_root();
    let mut desired_parallelism = num_cpus;
    while desired_parallelism > 2 {
        desired_parallelism /= 2;
        let dense = cpuset(root, desired_parallelism, CpusetAlgorithm::Dense);
        let sparse = cpuset(root, desired_parallelism, CpusetAlgorithm::Sparse);
        if dense == sparse {
            cpusets.push((format!("{dense}"), dense));
        } else {
            cpusets.push((format!("dense:{dense}"), dense));
            cpusets.push((format!("sparse:{sparse}"), sparse));
        }
    }
    //
    let mut seq_cpuset = full_cpuset;
    seq_cpuset.singlify();
    cpusets.push((format!("{seq_cpuset}"), seq_cpuset));

    // Run benchmarks
    for size_pow2 in 0..=40 {
        let size = 1 << size_pow2;
        // Set up multithreading configuration
        for (cpuset_idx, (cpuset_name, cpuset)) in cpusets.iter().enumerate() {
            topology
                .set_cpubind(cpuset.clone(), CpuBindFlags::CPUBIND_PROCESS)
                .unwrap();
            'benchmarks: for (benchmark_name, benchmark) in &mut benchmarks {
                // Set up the benchmark
                let mut counter = match benchmark {
                    Benchmark::Sequential(counter) => {
                        if cpuset_idx > 0 {
                            continue 'benchmarks;
                        }
                        Box::new(counter) as _
                    }
                    Benchmark::Parallel(factory) => factory(),
                };

                // Set up a custom Rayon thread pool for rayon tests
                // (the global thread pool will not tolerate cpuset changes)
                let rayon_pool = benchmark_name.starts_with("thread_rayon").then(|| {
                    ThreadPoolBuilder::new()
                        .num_threads(cpuset.weight() as usize)
                        .build()
                        .unwrap()
                });

                // Run the benchmark
                let mut group_name = benchmark_name.to_string();
                if !cpuset_name.is_empty() {
                    write!(&mut group_name, "{{{cpuset_name}}}").unwrap();
                }
                let mut group = c.benchmark_group(group_name);
                group.throughput(Throughput::Elements(size));
                group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
                    let mut iter = || b.iter(|| counter(pessimize::hide(size)));
                    if let Some(pool) = &rayon_pool {
                        pool.install(iter)
                    } else {
                        iter()
                    }
                });
            }
        }
    }
}

/// Construct benchmark set
fn benchmarks() -> Vec<(&'static str, Benchmark)> {
    let mut benchmarks = Vec::new();
    let add_benchmarks = |output: &mut Vec<(&'static str, Benchmark)>,
                          input: &[(&'static str, fn(u64) -> u64)]| {
        output.extend(
            input
                .iter()
                .copied()
                .map(|(name, code)| (name, Benchmark::Sequential(code))),
        );
    };
    add_benchmarks(
        &mut benchmarks,
        &[
            ("basic", counter::basic),
            ("ilp1", counter::basic_ilp::<1>),
            ("ilp14", counter::basic_ilp::<14>),
            ("ilp15", counter::basic_ilp::<15>),
            ("ilp16", counter::basic_ilp::<16>),
            ("generic_ilp1_u64", counter::generic_ilp_u64::<1, u64>),
            ("generic_ilp15_u64", counter::generic_ilp_u64::<15, u64>),
            ("multiversion_sse2", counter::multiversion_sse2),
            ("multiversion_avx2", counter::multiversion_avx2),
        ],
    );
    #[cfg(target_feature = "sse2")]
    {
        use safe_arch::m128i;
        add_benchmarks(
            &mut benchmarks,
            &[
                ("simd_basic", counter::simd_basic),
                ("simd_ilp1", counter::simd_ilp::<1>),
                ("simd_ilp7", counter::simd_ilp::<7>),
                ("simd_ilp8", counter::simd_ilp::<8>),
                ("simd_ilp9", counter::simd_ilp::<9>),
                ("generic_ilp1_u64x2", counter::generic_ilp_u64::<1, m128i>),
                ("generic_ilp9_u64x2", counter::generic_ilp_u64::<9, m128i>),
            ],
        );
    }
    #[cfg(target_feature = "avx2")]
    {
        add_benchmarks(
            &mut benchmarks,
            &[
                ("narrow_simple_u32", counter::narrow_simple::<u32>),
                ("narrow_simple_u16", counter::narrow_simple::<u16>),
                ("narrow_simple_u8", counter::narrow_simple::<u8>),
                ("narrow_u8", counter::narrow_u8),
                ("narrow_u8_tuned", counter::narrow_u8_tuned),
            ],
        );
        benchmarks.push((
            "thread_basic",
            Benchmark::Parallel(Box::new(|| {
                Box::new(|target| counter::thread_basic(target, counter::narrow_u8_tuned))
                    as CounterBox
            }) as _),
        ));
        benchmarks.push((
            "thread_rayon",
            Benchmark::Parallel(Box::new(|| {
                Box::new(|target| counter::thread_rayon(target, counter::narrow_u8_tuned))
                    as CounterBox
            }) as _),
        ));
        benchmarks.push((
            "thread_pool",
            Benchmark::Parallel(Box::new(|| {
                let mut bkg = counter::BackgroundThreads::<_, counter::BasicScheduler>::start(
                    counter::narrow_u8_tuned,
                );
                Box::new(move |target| bkg.count(target)) as CounterBox
            }) as _),
        ));
        benchmarks.push((
            "thread_futex",
            Benchmark::Parallel(Box::new(|| {
                let mut bkg = counter::BackgroundThreads::<_, counter::FutexScheduler>::start(
                    counter::narrow_u8_tuned,
                );
                Box::new(move |target| bkg.count(target)) as CounterBox
            }) as _),
        ));
    }
    benchmarks
}
//
enum Benchmark {
    /// Simple sequential counter implementation
    Sequential(fn(u64) -> u64),

    /// Parallel counter implementation that sets up a parallel counter implementation
    Parallel(CounterFactory),
}
//
type CounterBox = Box<dyn FnMut(u64) -> u64 + Send>;
//
type CounterFactory = Box<dyn FnMut() -> CounterBox>;

// Construct a CPUset of physical CPU cores of a given length using a certain algorithm
fn cpuset(object: &TopologyObject, num_cpus: u32, algorithm: CpusetAlgorithm) -> CpuSet {
    // Validate request
    assert!(num_cpus > 0, "Request is pointless");
    debug_assert!(
        num_cpus <= physical_cpus(object),
        "Request is unsatisfiable"
    );

    // Ignore hyperthreads within cores if given the opportunity
    if object.object_type() == ObjectType::Core {
        let mut cpuset = object.cpuset().unwrap();
        cpuset.singlify();
        return cpuset;
    }

    // Handle leaf nodes and and mono-parental nodes like L1 caches
    let arity = object.arity();
    if arity == 0 {
        return object.cpuset().unwrap();
    } else if arity == 1 {
        return cpuset(object.first_child().unwrap(), num_cpus, algorithm);
    }

    // Spread num_cpus across a set of children
    type Children<'topology> = [(&'topology TopologyObject, u32)];
    let distribute_cpuset = |children: &mut Children, total_cpus: u32| -> CpuSet {
        // Balance the cpu budget across children
        let total_share = num_cpus as f64 / total_cpus as f64;
        let mut allocated_cpus = 0;
        for (_, cpus) in &mut children[..] {
            let new_cpus = (total_share * *cpus as f64).ceil() as u32;
            *cpus = new_cpus;
            allocated_cpus += new_cpus;
        }
        let excess_cpus = allocated_cpus - num_cpus;
        for (_, cpus) in children.iter_mut().rev().take(excess_cpus as usize) {
            *cpus -= 1;
        }

        // Recurse into children and merge resulting CPUsets
        children
            .iter()
            .filter(|(_, cpus)| *cpus != 0)
            .flat_map(|(child, cpus)| cpuset(child, *cpus, algorithm))
            .collect()
    };

    // If there are multiple children, follow traversal algorithm
    match algorithm {
        // Spread requested CPUs evenly across children
        CpusetAlgorithm::Sparse => {
            // Enumerate all children and count their physical cores
            let mut children = Vec::with_capacity(object.arity() as usize);
            let mut child_opt = object.first_child();
            let mut total_cpus = 0;
            while let Some(child) = child_opt {
                let child_cpus = physical_cpus(child);
                if child_cpus > 0 {
                    children.push((child, child_cpus));
                    total_cpus += child_cpus;
                }
                child_opt = child.next_sibling();
            }

            // Spread requested CPUs across all children
            distribute_cpuset(&mut children[..], total_cpus)
        }

        // Pack requested CPUs into as few consecutive children as possible
        // We want them consecutive as hwloc2-rs doesn't yet expose latency,
        // so we must trust the OS and/or hwloc to order children sensibly.
        CpusetAlgorithm::Dense => {
            // Pick the first child with enough cores, if any
            let mut children = Vec::with_capacity(object.arity() as usize);
            let mut child_opt = object.first_child();
            while let Some(child) = child_opt {
                let child_cpus = physical_cpus(child);
                if child_cpus >= num_cpus {
                    return cpuset(child, num_cpus, algorithm);
                } else if child_cpus > 0 {
                    children.push((child, child_cpus));
                }
                child_opt = child.next_sibling();
            }

            // Else look for a consecutive set of children with enough cores
            let mut window_cpus = children.iter().map(|(_, cpus)| *cpus).collect::<Vec<_>>();
            for window_len in 2.. {
                window_cpus.pop();
                for (start_idx, total_cpus) in window_cpus.iter_mut().enumerate() {
                    // Update window total CPU count
                    let window = &mut children[start_idx..start_idx + window_len];
                    *total_cpus += window.last().unwrap().1;
                    if *total_cpus < num_cpus {
                        continue;
                    }

                    // This window is large enough, spread the CPU budget evenly across it
                    return distribute_cpuset(window, *total_cpus);
                }
            }
            unreachable!()
        }
    }
}
//
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum CpusetAlgorithm {
    /// Try to use physically close CPUs. Good for communication latency.
    Dense,

    /// Try to use physically remote CPUs, Good for spreading out shared resource usage.
    Sparse,
}

/// Count physical CPU cores below a certain Topology node
fn physical_cpus(object: &TopologyObject) -> u32 {
    // Discard non-CPU resources
    let Some(cpuset) = object.cpuset() else { return 0; };

    // Ignore hyperthreads within cores if given the opportunity
    if object.object_type() == ObjectType::Core {
        return 1;
    }

    // Terminate recursion on leaf nodes
    let Some(mut child) = object.first_child() else { return cpuset.weight() as u32 };

    // On non-leaf nodes, recurse through children
    let mut result = physical_cpus(child);
    while let Some(cousin) = child.next_sibling() {
        child = cousin;
        result += physical_cpus(cousin);
    }
    result
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
