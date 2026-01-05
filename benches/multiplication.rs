//! Benchmarks comparing Cayley-Dickson vs direct multiplication.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use octonion::{Octonion, mul_direct};

fn bench_multiplication(c: &mut Criterion) {
    let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let b = Octonion::new(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

    c.bench_function("mul_cayley_dickson", |bench| {
        bench.iter(|| black_box(a) * black_box(b))
    });

    c.bench_function("mul_direct", |bench| {
        bench.iter(|| mul_direct(black_box(a), black_box(b)))
    });
}

fn bench_chained_multiplication(c: &mut Criterion) {
    let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let b = Octonion::new(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    let c_oct = Octonion::new(-1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5, 1.0);

    c.bench_function("chain3_cayley_dickson", |bench| {
        bench.iter(|| {
            let ab = black_box(a) * black_box(b);
            ab * black_box(c_oct)
        })
    });

    c.bench_function("chain3_direct", |bench| {
        bench.iter(|| {
            let ab = mul_direct(black_box(a), black_box(b));
            mul_direct(ab, black_box(c_oct))
        })
    });
}

fn bench_batch_multiplication(c: &mut Criterion) {
    // Simulate multiplying many octonions (e.g., in a physics simulation)
    let octonions: Vec<Octonion> = (0..100)
        .map(|i| {
            let f = i as f64;
            Octonion::new(
                f,
                f + 1.0,
                f + 2.0,
                f + 3.0,
                f + 4.0,
                f + 5.0,
                f + 6.0,
                f + 7.0,
            )
        })
        .collect();

    c.bench_function("batch100_cayley_dickson", |bench| {
        bench.iter(|| {
            let mut acc = Octonion::ONE;
            for oct in &octonions {
                acc = acc * *oct;
            }
            black_box(acc)
        })
    });

    c.bench_function("batch100_direct", |bench| {
        bench.iter(|| {
            let mut acc = Octonion::ONE;
            for oct in &octonions {
                acc = mul_direct(acc, *oct);
            }
            black_box(acc)
        })
    });
}

criterion_group!(
    benches,
    bench_multiplication,
    bench_chained_multiplication,
    bench_batch_multiplication
);
criterion_main!(benches);
