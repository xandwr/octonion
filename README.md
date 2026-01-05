# octonion

Minimal, dependency-free, `no_std` octonion algebra for Rust.

This crate provides `Octonion` (an 8D hypercomplex number over `f64`) with
basic arithmetic, conjugation, norms, and inversion.

## Quick start

```rust
use octonion::Octonion;

let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);

// Basis elements square to -1.
assert_eq!(Octonion::E1 * Octonion::E1, -Octonion::ONE);

// Non-commutativity.
assert_eq!(Octonion::E1 * Octonion::E2, Octonion::E3);
assert_eq!(Octonion::E2 * Octonion::E1, -Octonion::E3);

// Conjugation, norm, inverse.
let _conj = x.conj();
let _norm_sq = x.norm_sqr();
let _inv = x.try_inverse().unwrap();
```

## Notes

- Octonion multiplication is **non-associative**. If you need explicit grouping,
	enable `alloc` (on by default) and use `Paren` or `OctoExpr`.
- If an octonion lies in the quaternion subalgebra (its `e₄..e₇` coefficients are
	exactly zero), you can call `Octonion::as_quaternion` to get a `QuaternionView`
	that supports associative multiplication.

## SIMD helpers

The `octonion::simd` module includes a direct coefficient expansion
`simd::mul_direct` (often easier for the compiler to auto-vectorize than the
default implementation).

On `x86_64` with `avx`, `simd::mul_simd_avx` is available (currently a thin
wrapper around `mul_direct`).

## Features

- `alloc` (default): enables [`Paren`], [`OctoExpr`], and `simd::mul_batch`.
- `alloc` (default): enables `Paren`, `OctoExpr`, and `simd::mul_batch`.
- `e8`: enables `octonion::e8::IntegralOctonion` (Cayley integers / E8 lattice).

To use this crate without allocation support:

```toml
octonion = { version = "0.1", default-features = false }
```

## Examples and benches

- Colorized display demo: `cargo run --example display_demo`
- Benchmarks: `cargo bench`
