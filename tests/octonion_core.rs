//! Core octonion algebra tests.
//!
//! Tests for basic operations: construction, arithmetic, conjugation, norm, inverse.

mod common;

use common::{EPS, EPS_LOOSE, approx_eq, approx_eq_oct};
use octonion::Octonion;

// =============================================================================
// CONSTRUCTION AND CONSTANTS
// =============================================================================

#[test]
fn constants_are_correct() {
    assert!(Octonion::ZERO.is_zero());
    assert!(!Octonion::ONE.is_zero());
    assert_eq!(Octonion::ONE.real(), 1.0);

    // Basis elements have unit norm
    assert_eq!(Octonion::E1.norm_sqr(), 1.0);
    assert_eq!(Octonion::E7.norm_sqr(), 1.0);
}

#[test]
fn from_array_roundtrip() {
    let coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = Octonion::from_array(coeffs);
    assert_eq!(x.to_array(), coeffs);
}

#[test]
fn from_f64_embeds_as_real() {
    let x: Octonion = 3.14.into();
    assert_eq!(x.real(), 3.14);
    for i in 1..8 {
        assert_eq!(x.coeff(i), 0.0);
    }
}

#[test]
fn coeff_accessor() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    assert_eq!(x.coeff(0), 1.0);
    assert_eq!(x.coeff(3), 4.0);
    assert_eq!(x.coeff(7), 8.0);
}

// =============================================================================
// MULTIPLICATION: BASIS ELEMENT PROPERTIES
// =============================================================================

#[test]
fn basis_squares_are_minus_one() {
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
        assert_eq!(e * e, minus_one, "{e:?}^2 should be -1");
    }
}

#[test]
fn some_basis_products() {
    // Quaternion subalgebra products
    assert_eq!(Octonion::E1 * Octonion::E2, Octonion::E3);
    assert_eq!(Octonion::E2 * Octonion::E1, -Octonion::E3);

    // Products involving e4+
    assert_eq!(Octonion::E1 * Octonion::E4, Octonion::E5);
    assert_eq!(Octonion::E4 * Octonion::E1, -Octonion::E5);
}

#[test]
fn multiplication_is_non_commutative() {
    let e1 = Octonion::E1;
    let e2 = Octonion::E2;

    assert_ne!(e1 * e2, e2 * e1);
    assert_eq!(e1 * e2, -(e2 * e1));
}

#[test]
fn multiplication_is_non_associative() {
    let e1 = Octonion::E1;
    let e2 = Octonion::E2;
    let e4 = Octonion::E4;

    let left = (e1 * e2) * e4;
    let right = e1 * (e2 * e4);

    assert_ne!(
        left, right,
        "octonion multiplication should be non-associative"
    );
}

#[test]
fn one_is_multiplicative_identity() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    assert_eq!(x * Octonion::ONE, x);
    assert_eq!(Octonion::ONE * x, x);
}

#[test]
fn zero_is_multiplicative_annihilator() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    assert!((x * Octonion::ZERO).is_zero());
    assert!((Octonion::ZERO * x).is_zero());
}

// =============================================================================
// CONJUGATION
// =============================================================================

#[test]
fn conjugate_negates_imaginary() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let c = x.conj();

    assert_eq!(c.real(), x.real());
    for i in 1..8 {
        assert_eq!(c.coeff(i), -x.coeff(i));
    }
}

#[test]
fn double_conjugate_is_identity() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    assert_eq!(x.conj().conj(), x);
}

#[test]
fn conjugation_reverses_products() {
    let x = Octonion::new(1.0, 2.0, 0.5, -3.0, 4.0, -1.0, 0.25, 2.0);
    let y = Octonion::new(-2.0, 1.0, 3.0, 0.0, -0.5, 2.0, -4.0, 1.5);

    assert_eq!((x * y).conj(), y.conj() * x.conj());
}

#[test]
fn conjugation_distributes_over_addition() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let y = Octonion::new(-1.0, 0.5, -2.0, 1.0, 0.0, -3.0, 2.0, -1.0);

    assert_eq!((x + y).conj(), x.conj() + y.conj());
}

// =============================================================================
// NORM
// =============================================================================

#[test]
fn norm_of_basis_elements_is_one() {
    assert_eq!(Octonion::ONE.norm_sqr(), 1.0);
    for e in [
        Octonion::E1,
        Octonion::E2,
        Octonion::E3,
        Octonion::E4,
        Octonion::E5,
        Octonion::E6,
        Octonion::E7,
    ] {
        assert_eq!(e.norm_sqr(), 1.0);
    }
}

#[test]
fn norm_of_zero_is_zero() {
    assert_eq!(Octonion::ZERO.norm_sqr(), 0.0);
}

#[test]
fn norm_is_sum_of_squares() {
    let x = Octonion::new(1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assert_eq!(x.norm_sqr(), 9.0); // 1 + 4 + 4 = 9
}

#[test]
fn norm_is_multiplicative_up_to_rounding() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let y = Octonion::new(2.0, -1.0, 0.5, 3.5, -4.25, 0.0, 1.0, -2.0);

    let lhs = (x * y).norm_sqr();
    let rhs = x.norm_sqr() * y.norm_sqr();
    assert!(approx_eq(lhs, rhs, EPS_LOOSE), "lhs={lhs} rhs={rhs}");
}

#[test]
fn x_times_conj_x_equals_norm_sqr() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let product = x * x.conj();

    // Should be a real number equal to norm_sqr
    assert!(approx_eq(product.real(), x.norm_sqr(), EPS));
    for i in 1..8 {
        assert!(approx_eq(product.coeff(i), 0.0, EPS));
    }
}

// =============================================================================
// INVERSE
// =============================================================================

#[test]
fn inverse_of_zero_is_none() {
    assert!(Octonion::ZERO.try_inverse().is_none());
}

#[test]
fn inverse_of_one_is_one() {
    let inv = Octonion::ONE.try_inverse().unwrap();
    assert_eq!(inv, Octonion::ONE);
}

#[test]
fn inverse_of_basis_elements() {
    for e in [
        Octonion::E1,
        Octonion::E2,
        Octonion::E3,
        Octonion::E4,
        Octonion::E5,
        Octonion::E6,
        Octonion::E7,
    ] {
        let inv = e.try_inverse().unwrap();
        // e * e^{-1} should be 1
        let prod = e * inv;
        assert!(
            approx_eq_oct(prod, Octonion::ONE, EPS),
            "e * e^(-1) = {prod:?}"
        );
    }
}

#[test]
fn inverse_works_for_general_case() {
    let x = Octonion::new(2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let inv = x.try_inverse().unwrap();
    let prod = x * inv;
    assert!(approx_eq_oct(prod, Octonion::ONE, EPS), "prod={prod:?}");
}

#[test]
fn inverse_of_complex_octonion() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let inv = x.try_inverse().unwrap();

    let prod = x * inv;
    assert!(approx_eq(prod.real(), 1.0, EPS));
    for i in 1..8 {
        assert!(approx_eq(prod.coeff(i), 0.0, EPS));
    }
}

// =============================================================================
// ARITHMETIC OPERATIONS
// =============================================================================

#[test]
fn addition_is_componentwise() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let y = Octonion::new(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    let sum = x + y;

    for i in 0..8 {
        assert_eq!(sum.coeff(i), x.coeff(i) + y.coeff(i));
    }
}

#[test]
fn subtraction_is_componentwise() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let y = Octonion::new(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    let diff = x - y;

    for i in 0..8 {
        assert_eq!(diff.coeff(i), x.coeff(i) - y.coeff(i));
    }
}

#[test]
fn negation_is_componentwise() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let neg = -x;

    for i in 0..8 {
        assert_eq!(neg.coeff(i), -x.coeff(i));
    }
}

#[test]
fn scalar_multiplication() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let scaled = x * 2.0;

    for i in 0..8 {
        assert_eq!(scaled.coeff(i), x.coeff(i) * 2.0);
    }
}

#[test]
fn scalar_multiplication_commutes() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    assert_eq!(x * 3.0, 3.0 * x);
}

#[test]
fn scalar_division() {
    let x = Octonion::new(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
    let halved = x / 2.0;

    for i in 0..8 {
        assert_eq!(halved.coeff(i), x.coeff(i) / 2.0);
    }
}

// =============================================================================
// COMPOUND ASSIGNMENT OPERATORS
// =============================================================================

#[test]
fn add_assign_works() {
    let mut x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let y = Octonion::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    x += y;

    assert_eq!(x, Octonion::new(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
}

#[test]
fn sub_assign_works() {
    let mut x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let y = Octonion::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    x -= y;

    assert_eq!(x, Octonion::new(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0));
}

#[test]
fn mul_assign_works() {
    let mut x = Octonion::E1;
    x *= Octonion::E2;
    assert_eq!(x, Octonion::E3);
}

#[test]
fn div_assign_works() {
    let mut x = Octonion::new(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
    x /= 2.0;
    assert_eq!(x, Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0));
}

// =============================================================================
// QUATERNION VIEW
// =============================================================================

#[test]
fn quaternion_view_for_valid_quaternion() {
    let q = Octonion::new(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
    let view = q.as_quaternion();
    assert!(view.is_some());

    let v = view.unwrap();
    assert_eq!(v.real(), 1.0);
    assert_eq!(v.i(), 2.0);
    assert_eq!(v.j(), 3.0);
    assert_eq!(v.k(), 4.0);
}

#[test]
fn quaternion_view_none_for_general_octonion() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0);
    assert!(x.as_quaternion().is_none());
}

#[test]
fn quaternion_multiplication_is_associative() {
    let a_oct = Octonion::new(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
    let b_oct = Octonion::new(0.5, -1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0);
    let c_oct = Octonion::new(-1.0, 0.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0);

    let a = a_oct.as_quaternion().unwrap();
    let b = b_oct.as_quaternion().unwrap();
    let c = c_oct.as_quaternion().unwrap();

    let ab = a.mul(b);
    let bc = b.mul(c);
    let left = ab.as_quaternion().unwrap().mul(c);
    let right = a.mul(bc.as_quaternion().unwrap());

    assert!(approx_eq_oct(left, right, EPS));
}
