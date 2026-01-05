//! Integration tests for cross-module functionality.
//!
//! Tests that verify the consistency and interoperability of different
//! modules: Octonion, IntegralOctonion, SIMD, and Paren.

mod common;

use common::{EPS, EPS_LOOSE, approx_eq, approx_eq_oct};
use octonion::{OctoExpr, Octonion, Paren, mul_direct};

// =============================================================================
// SIMD VS CAYLEY-DICKSON VS PAREN
// =============================================================================

#[test]
fn all_multiplication_methods_agree() {
    let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let b = Octonion::new(-1.0, 0.5, -2.0, 1.5, 0.25, -0.75, 3.0, -0.5);

    let cayley = a * b;
    let direct = mul_direct(a, b);
    let paren = Paren::new(a).mul(b).eval();
    let expr = OctoExpr::new(a).mul(OctoExpr::new(b)).eval();

    assert!(approx_eq_oct(cayley, direct, EPS));
    assert!(approx_eq_oct(cayley, paren, EPS));
    assert!(approx_eq_oct(cayley, expr, EPS));
}

// =============================================================================
// INTEGRAL OCTONION <-> OCTONION CONSISTENCY
// =============================================================================

#[cfg(feature = "e8")]
mod e8_integration {
    use super::*;
    use octonion::e8::IntegralOctonion;

    #[test]
    fn integral_octonion_multiplication_matches_float() {
        let ix = IntegralOctonion::E1;
        let iy = IntegralOctonion::E2;
        let iprod = ix * iy;

        let fx = ix.to_octonion();
        let fy = iy.to_octonion();
        let fprod = fx * fy;

        let iprod_float = iprod.to_octonion();

        assert!(
            approx_eq_oct(iprod_float, fprod, EPS),
            "Integral: {iprod_float:?}\nFloat: {fprod:?}"
        );
    }

    #[test]
    fn integral_octonion_norm_matches_float() {
        let x = IntegralOctonion::integer(1, 2, 3, 0, 0, 0, 0, 0);
        let fx = x.to_octonion();

        let int_norm = x.norm() as f64;
        let float_norm = fx.norm_sqr();

        assert!(
            approx_eq(int_norm, float_norm, EPS),
            "Integral norm: {int_norm}, Float norm: {float_norm}"
        );
    }

    #[test]
    fn all_basis_products_consistent() {
        let int_bases = [
            IntegralOctonion::E1,
            IntegralOctonion::E2,
            IntegralOctonion::E3,
            IntegralOctonion::E4,
            IntegralOctonion::E5,
            IntegralOctonion::E6,
            IntegralOctonion::E7,
        ];

        let float_bases = [
            Octonion::E1,
            Octonion::E2,
            Octonion::E3,
            Octonion::E4,
            Octonion::E5,
            Octonion::E6,
            Octonion::E7,
        ];

        for i in 0..7 {
            for j in 0..7 {
                let int_prod = (int_bases[i] * int_bases[j]).to_octonion();
                let float_prod = float_bases[i] * float_bases[j];

                assert!(
                    approx_eq_oct(int_prod, float_prod, EPS),
                    "Mismatch at e{} * e{}: int={int_prod:?}, float={float_prod:?}",
                    i + 1,
                    j + 1
                );
            }
        }
    }

    #[test]
    fn roundtrip_preserves_value() {
        // Integer octonion roundtrip
        let x = IntegralOctonion::integer(1, 2, 3, 4, 5, 6, 7, 8);
        let fx = x.to_octonion();
        let back = IntegralOctonion::try_from_octonion(&fx).unwrap();
        assert_eq!(x, back);

        // Half-integer roundtrip
        let h = IntegralOctonion::half(1, 1, 1, 1, -1, -1, -1, -1);
        let fh = h.to_octonion();
        let back_h = IntegralOctonion::try_from_octonion(&fh).unwrap();
        assert_eq!(h, back_h);
    }
}

// =============================================================================
// ALGEBRAIC PROPERTY CROSS-VALIDATION
// =============================================================================

#[test]
fn norm_multiplicative_with_simd() {
    let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let b = Octonion::new(-2.0, 1.0, 0.5, -3.0, 4.0, -1.0, 0.25, 2.0);

    // Using Cayley-Dickson
    let cayley_prod = a * b;
    let cayley_norm = cayley_prod.norm_sqr();

    // Using direct/SIMD
    let direct_prod = mul_direct(a, b);
    let direct_norm = direct_prod.norm_sqr();

    // Both should give the same result
    assert!(approx_eq(cayley_norm, direct_norm, EPS));

    // And both should satisfy |ab|² = |a|²|b|²
    let expected = a.norm_sqr() * b.norm_sqr();
    assert!(approx_eq(cayley_norm, expected, EPS_LOOSE));
}

#[test]
fn conjugation_reverses_with_all_methods() {
    let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let b = Octonion::new(-1.0, 0.5, -2.0, 1.5, 0.25, -0.75, 3.0, -0.5);

    // (ab)* = b* a*
    let cayley_conj = (a * b).conj();
    let direct_conj = mul_direct(a, b).conj();
    let reversed_cayley = b.conj() * a.conj();
    let reversed_direct = mul_direct(b.conj(), a.conj());

    assert!(approx_eq_oct(cayley_conj, reversed_cayley, EPS));
    assert!(approx_eq_oct(direct_conj, reversed_direct, EPS));
    assert!(approx_eq_oct(cayley_conj, direct_conj, EPS));
}

// =============================================================================
// PAREN ASSOCIATIVITY TESTS WITH DIFFERENT BACKENDS
// =============================================================================

#[test]
fn non_associativity_consistent_across_methods() {
    let e1 = Octonion::E1;
    let e2 = Octonion::E2;
    let e4 = Octonion::E4;

    // Left associative: (e1 * e2) * e4
    let left_cayley = (e1 * e2) * e4;
    let left_direct = mul_direct(mul_direct(e1, e2), e4);
    let left_paren = Paren::left3(e1, e2, e4).eval();

    // Right associative: e1 * (e2 * e4)
    let right_cayley = e1 * (e2 * e4);
    let right_direct = mul_direct(e1, mul_direct(e2, e4));
    let right_paren = Paren::right3(e1, e2, e4).eval();

    // All left methods should agree
    assert_eq!(left_cayley, left_direct);
    assert_eq!(left_cayley, left_paren);

    // All right methods should agree
    assert_eq!(right_cayley, right_direct);
    assert_eq!(right_cayley, right_paren);

    // Left and right should differ (non-associativity)
    assert_ne!(left_cayley, right_cayley);
}

// =============================================================================
// QUATERNION SUBALGEBRA
// =============================================================================

#[test]
fn quaternion_view_multiplication_via_paren() {
    let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
    let b = Octonion::new(0.5, -1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0);
    let c = Octonion::new(-1.0, 0.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0);

    // For quaternions, associativity holds
    let left = Paren::left3(a, b, c).eval();
    let right = Paren::right3(a, b, c).eval();

    // These should be equal (quaternion subalgebra is associative)
    assert!(
        approx_eq_oct(left, right, EPS),
        "Quaternion subalgebra should be associative"
    );

    // Also verify via QuaternionView
    let qa = a.as_quaternion().unwrap();
    let qb = b.as_quaternion().unwrap();
    let qc = c.as_quaternion().unwrap();

    let ab = qa.mul(qb);
    let bc = qb.mul(qc);
    let left_q = ab.as_quaternion().unwrap().mul(qc);
    let right_q = qa.mul(bc.as_quaternion().unwrap());

    assert!(approx_eq_oct(left_q, right_q, EPS));
    assert!(approx_eq_oct(left, left_q, EPS));
}

// =============================================================================
// STRESS TESTS
// =============================================================================

#[test]
fn many_chained_multiplications_consistent() {
    let elements = [
        Octonion::E1,
        Octonion::E2,
        Octonion::E3,
        Octonion::E4,
        Octonion::E5,
    ];

    // Compute product left-associatively using different methods
    let mut cayley = elements[0];
    let mut direct = elements[0];

    for &e in &elements[1..] {
        cayley = cayley * e;
        direct = mul_direct(direct, e);
    }

    let paren_result = Paren::new(elements[0])
        .mul(elements[1])
        .mul(elements[2])
        .mul(elements[3])
        .mul(elements[4])
        .eval();

    assert!(approx_eq_oct(cayley, direct, EPS));
    assert!(approx_eq_oct(cayley, paren_result, EPS));
}

#[test]
fn inverse_roundtrip_with_different_methods() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let inv = x.try_inverse().unwrap();

    // x * x^{-1} should be 1 using any method
    let cayley = x * inv;
    let direct = mul_direct(x, inv);

    assert!(approx_eq_oct(cayley, Octonion::ONE, EPS));
    assert!(approx_eq_oct(direct, Octonion::ONE, EPS));
}
