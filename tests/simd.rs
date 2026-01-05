//! Tests for SIMD-accelerated multiplication.
//!
//! Verifies that `mul_direct` and `mul_simd_avx` produce results identical
//! to the Cayley-Dickson implementation.

mod common;

use common::{EPS, EPS_LOOSE, approx_eq, approx_eq_oct};
use octonion::{Octonion, mul_direct, mul_simd_avx};

// =============================================================================
// EQUIVALENCE WITH CAYLEY-DICKSON
// =============================================================================

#[test]
fn mul_direct_matches_cayley_dickson() {
    let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let b = Octonion::new(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

    let cayley = a * b;
    let direct = mul_direct(a, b);

    assert!(
        approx_eq_oct(cayley, direct, EPS),
        "Cayley-Dickson: {cayley:?}\nDirect: {direct:?}"
    );
}

#[test]
fn mul_direct_matches_for_negative_coefficients() {
    let a = Octonion::new(-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0);
    let b = Octonion::new(0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5);

    let cayley = a * b;
    let direct = mul_direct(a, b);

    assert!(
        approx_eq_oct(cayley, direct, EPS),
        "Cayley-Dickson: {cayley:?}\nDirect: {direct:?}"
    );
}

#[test]
fn mul_simd_avx_matches_direct() {
    let a = Octonion::new(1.5, -2.3, 0.7, 4.1, -5.9, 6.2, -7.8, 8.4);
    let b = Octonion::new(-0.5, 3.2, -1.1, 2.9, 0.3, -4.7, 5.6, -0.9);

    let direct = mul_direct(a, b);
    let simd = mul_simd_avx(a, b);

    assert!(
        approx_eq_oct(direct, simd, EPS),
        "Direct: {direct:?}\nSIMD: {simd:?}"
    );
}

// =============================================================================
// BASIS ELEMENT PRODUCTS
// =============================================================================

#[test]
fn mul_direct_basis_elements() {
    // e1 * e2 = e3
    assert_eq!(mul_direct(Octonion::E1, Octonion::E2), Octonion::E3);

    // e2 * e1 = -e3
    assert_eq!(mul_direct(Octonion::E2, Octonion::E1), -Octonion::E3);

    // e1^2 = -1
    assert_eq!(mul_direct(Octonion::E1, Octonion::E1), -Octonion::ONE);

    // Verify against Cayley-Dickson for more products
    assert_eq!(
        mul_direct(Octonion::E4, Octonion::E5),
        Octonion::E4 * Octonion::E5
    );
    assert_eq!(
        mul_direct(Octonion::E6, Octonion::E7),
        Octonion::E6 * Octonion::E7
    );
}

#[test]
fn mul_direct_all_basis_squares() {
    let minus_one = -Octonion::ONE;
    for e in [
        Octonion::E1,
        Octonion::E2,
        Octonion::E3,
        Octonion::E4,
        Octonion::E5,
        Octonion::E6,
        Octonion::E7,
    ] {
        assert_eq!(mul_direct(e, e), minus_one);
    }
}

// =============================================================================
// IDENTITY AND SPECIAL CASES
// =============================================================================

#[test]
fn mul_direct_identity() {
    let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    assert_eq!(mul_direct(a, Octonion::ONE), a);
    assert_eq!(mul_direct(Octonion::ONE, a), a);
}

#[test]
fn mul_direct_zero() {
    let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    assert!(mul_direct(a, Octonion::ZERO).is_zero());
    assert!(mul_direct(Octonion::ZERO, a).is_zero());
}

// =============================================================================
// ALGEBRAIC PROPERTIES
// =============================================================================

#[test]
fn mul_direct_norm_multiplicative() {
    let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let b = Octonion::new(-2.0, 1.0, 0.5, -3.0, 4.0, -1.0, 0.25, 2.0);

    let product = mul_direct(a, b);
    let lhs = product.norm_sqr();
    let rhs = a.norm_sqr() * b.norm_sqr();

    assert!(
        approx_eq(lhs, rhs, EPS_LOOSE),
        "Norm not multiplicative: {lhs} vs {rhs}"
    );
}

#[test]
fn mul_direct_conjugation_reverses() {
    let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let b = Octonion::new(-2.0, 1.0, 0.5, -3.0, 4.0, -1.0, 0.25, 2.0);

    let lhs = mul_direct(a, b).conj();
    let rhs = mul_direct(b.conj(), a.conj());

    assert!(approx_eq_oct(lhs, rhs, EPS));
}

// =============================================================================
// BATCH OPERATIONS
// =============================================================================

#[test]
fn mul_batch_matches_individual() {
    use octonion::simd::mul_batch;

    let pairs = vec![
        (Octonion::E1, Octonion::E2),
        (Octonion::E3, Octonion::E4),
        (
            Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
            Octonion::new(0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5),
        ),
    ];

    let results = mul_batch(&pairs);

    for (i, ((a, b), result)) in pairs.iter().zip(results.iter()).enumerate() {
        let expected = *a * *b;
        assert!(
            approx_eq_oct(*result, expected, EPS),
            "Batch mismatch at index {i}"
        );
    }
}

#[test]
fn mul_batch_empty() {
    use octonion::simd::mul_batch;

    let pairs: Vec<(Octonion, Octonion)> = vec![];
    let results = mul_batch(&pairs);
    assert!(results.is_empty());
}

// =============================================================================
// STRESS TESTS
// =============================================================================

#[test]
fn mul_direct_many_random_products() {
    // Test many products to catch edge cases
    let values = [
        Octonion::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        Octonion::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        Octonion::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        Octonion::new(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        Octonion::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        Octonion::new(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0),
    ];

    for a in &values {
        for b in &values {
            let cayley = *a * *b;
            let direct = mul_direct(*a, *b);
            assert!(
                approx_eq_oct(cayley, direct, EPS),
                "Mismatch for {a:?} * {b:?}"
            );
        }
    }
}
