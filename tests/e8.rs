//! Tests for integral octonions (Cayley integers) over the E8 lattice.
//!
//! Tests for `IntegralOctonion` including lattice membership, arithmetic,
//! units, roots, and the algebraic properties of the E8 lattice.

use octonion::e8::IntegralOctonion;

// =============================================================================
// CONSTRUCTION AND VALIDATION
// =============================================================================

#[test]
fn constants_are_valid() {
    assert!(IntegralOctonion::ZERO.is_zero());
    assert!(IntegralOctonion::ONE.is_unit());
    assert!(IntegralOctonion::E1.is_unit());
    assert!(IntegralOctonion::E7.is_unit());
    assert!(IntegralOctonion::H.is_root()); // half-integer element has norm 2
}

#[test]
fn integer_constructor() {
    let x = IntegralOctonion::integer(1, 2, 3, 0, 0, 0, 0, 0);
    assert_eq!(x, IntegralOctonion::new(2, 4, 6, 0, 0, 0, 0, 0));
    assert!(x.is_integer());
}

#[test]
fn half_constructor() {
    // sum = 1+1+1+1+1+1+1+1 = 8 ≡ 0 (mod 4)
    let h = IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 1);
    assert!(h.is_half_integer());
    assert_eq!(h, IntegralOctonion::H);
}

#[test]
fn half_with_negative_signs() {
    // sum = 1+1+1+1-1-1-1-1 = 0 ≡ 0 (mod 4)
    let h = IntegralOctonion::half(1, 1, 1, 1, -1, -1, -1, -1);
    assert!(h.is_half_integer());
    assert_eq!(h.norm(), 2);
}

#[test]
fn from_array() {
    let x = IntegralOctonion::from_array([2, 4, 0, 0, 0, 0, 0, 0]);
    assert_eq!(x, IntegralOctonion::integer(1, 2, 0, 0, 0, 0, 0, 0));
}

#[test]
#[should_panic(expected = "Invalid E8 lattice point")]
fn invalid_mixed_parity_panics() {
    // Mixed parities: 1 is odd, 2 is even
    IntegralOctonion::new(1, 2, 0, 0, 0, 0, 0, 0);
}

#[test]
#[should_panic(expected = "Invalid E8 lattice point")]
fn invalid_sum_mod_4_panics() {
    // All odd but sum = 1+1+1+1+1+1+1+3 = 10 ≡ 2 (mod 4)
    IntegralOctonion::new(1, 1, 1, 1, 1, 1, 1, 3);
}

#[test]
#[should_panic(expected = "half() requires all odd coefficients")]
fn half_with_even_coefficient_panics() {
    IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 2);
}

#[test]
#[should_panic(expected = "half() requires sum of coefficients")]
fn half_with_wrong_sum_panics() {
    // sum = 1+1+1+1+1+1+1+3 = 10 ≡ 2 (mod 4)
    IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 3);
}

// =============================================================================
// MULTIPLICATION: BASIS ELEMENTS
// =============================================================================

#[test]
fn basis_squares_are_minus_one() {
    let minus_one = -IntegralOctonion::ONE;
    for e in [
        IntegralOctonion::E1,
        IntegralOctonion::E2,
        IntegralOctonion::E3,
        IntegralOctonion::E4,
        IntegralOctonion::E5,
        IntegralOctonion::E6,
        IntegralOctonion::E7,
    ] {
        assert_eq!(e * e, minus_one, "{e:?}^2 should be -1");
    }
}

#[test]
fn basis_products() {
    // e1 * e2 = e3
    assert_eq!(
        IntegralOctonion::E1 * IntegralOctonion::E2,
        IntegralOctonion::E3
    );

    // e2 * e1 = -e3
    assert_eq!(
        IntegralOctonion::E2 * IntegralOctonion::E1,
        -IntegralOctonion::E3
    );

    // e1 * e4 = e5
    assert_eq!(
        IntegralOctonion::E1 * IntegralOctonion::E4,
        IntegralOctonion::E5
    );

    // e4 * e1 = -e5
    assert_eq!(
        IntegralOctonion::E4 * IntegralOctonion::E1,
        -IntegralOctonion::E5
    );
}

#[test]
fn multiplication_is_non_commutative() {
    let e1 = IntegralOctonion::E1;
    let e2 = IntegralOctonion::E2;

    assert_ne!(e1 * e2, e2 * e1);
    assert_eq!(e1 * e2, -(e2 * e1));
}

#[test]
fn one_is_multiplicative_identity() {
    let x = IntegralOctonion::integer(1, 2, 3, 0, 0, 0, 0, 0);
    assert_eq!(x * IntegralOctonion::ONE, x);
    assert_eq!(IntegralOctonion::ONE * x, x);
}

// =============================================================================
// NORM AND UNITS
// =============================================================================

#[test]
fn unit_norms() {
    // Integer units (16 total: ±1, ±e_i)
    assert!(IntegralOctonion::ONE.is_unit());
    assert!((-IntegralOctonion::ONE).is_unit());
    assert!(IntegralOctonion::E1.is_unit());
    assert!(IntegralOctonion::E7.is_unit());

    // All units have norm 1
    assert_eq!(IntegralOctonion::ONE.norm(), 1);
    assert_eq!(IntegralOctonion::E1.norm(), 1);
}

#[test]
fn half_integer_norms() {
    // Half-integer elements have norm 2 (they are E8 roots, not units)
    let h = IntegralOctonion::half(1, 1, 1, 1, -1, -1, -1, -1);
    assert!(!h.is_unit(), "half-integer elements have norm 2, not 1");
    assert!(h.is_root(), "half-integer elements are E8 roots");
    assert_eq!(h.norm(), 2);
}

#[test]
fn norm_is_multiplicative() {
    let x = IntegralOctonion::integer(1, 2, 0, 0, 0, 0, 0, 0);
    let y = IntegralOctonion::integer(0, 1, 1, 0, 0, 0, 0, 0);

    let lhs = (x * y).norm();
    let rhs = x.norm() * y.norm();
    assert_eq!(lhs, rhs, "norm should be multiplicative");
}

#[test]
fn count_integer_units() {
    // The 16 integer units: ±1, ±e_i
    let mut integer_units = 0;
    for sign in [-1i64, 1] {
        for i in 0..8 {
            let mut coeffs = [0i64; 8];
            coeffs[i] = sign * 2;
            let x = IntegralOctonion::new_unchecked(coeffs);
            if x.is_unit() {
                integer_units += 1;
            }
        }
    }
    assert_eq!(integer_units, 16, "should have 16 integer units");
}

#[test]
fn e8_root_example() {
    // e1 + e2 is a root (norm 2)
    let r = IntegralOctonion::integer(0, 1, 1, 0, 0, 0, 0, 0);
    assert!(r.is_root());
    assert_eq!(r.norm(), 2);
}

#[test]
fn e8_lattice_half_root_example() {
    // ½(1 + e1 + e2 + e3 - e4 - e5 - e6 - e7) has norm 2 (E8 root)
    let u = IntegralOctonion::half(1, 1, 1, 1, -1, -1, -1, -1);
    assert_eq!(u.norm(), 2);
    assert!(u.is_root());
    assert!(!u.is_unit());
}

// =============================================================================
// CONJUGATION
// =============================================================================

#[test]
fn conjugate_negates_imaginary() {
    let x = IntegralOctonion::integer(1, 2, 3, 4, 5, 6, 7, 8);
    let c = x.conj();

    assert_eq!(c.coeff(0), x.coeff(0));
    for i in 1..8 {
        assert_eq!(c.coeff(i), -x.coeff(i));
    }
}

#[test]
fn conjugation_reverses_products() {
    let x = IntegralOctonion::integer(1, 2, 0, 1, 0, 0, 0, 0);
    let y = IntegralOctonion::integer(0, 1, 1, 0, 1, 0, 0, 0);
    assert_eq!((x * y).conj(), y.conj() * x.conj());
}

// =============================================================================
// ARITHMETIC OPERATIONS
// =============================================================================

#[test]
fn addition() {
    let x = IntegralOctonion::integer(1, 2, 0, 0, 0, 0, 0, 0);
    let y = IntegralOctonion::integer(0, 1, 3, 0, 0, 0, 0, 0);
    let sum = x + y;

    assert_eq!(sum, IntegralOctonion::integer(1, 3, 3, 0, 0, 0, 0, 0));
}

#[test]
fn subtraction() {
    let x = IntegralOctonion::integer(5, 3, 0, 0, 0, 0, 0, 0);
    let y = IntegralOctonion::integer(2, 1, 0, 0, 0, 0, 0, 0);
    let diff = x - y;

    assert_eq!(diff, IntegralOctonion::integer(3, 2, 0, 0, 0, 0, 0, 0));
}

#[test]
fn negation() {
    let x = IntegralOctonion::integer(1, 2, 3, 4, 0, 0, 0, 0);
    let neg = -x;

    assert_eq!(neg, IntegralOctonion::integer(-1, -2, -3, -4, 0, 0, 0, 0));
}

#[test]
fn scalar_multiplication() {
    let x = IntegralOctonion::integer(1, 2, 0, 0, 0, 0, 0, 0);
    let scaled = x * 3;

    assert_eq!(scaled, IntegralOctonion::integer(3, 6, 0, 0, 0, 0, 0, 0));
}

#[test]
fn scalar_multiplication_commutes() {
    let x = IntegralOctonion::integer(1, 2, 0, 0, 0, 0, 0, 0);
    assert_eq!(x * 3, 3 * x);
}

// =============================================================================
// HALF-INTEGER ARITHMETIC
// =============================================================================

#[test]
fn half_integer_addition() {
    let h1 = IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 1);
    let h2 = IntegralOctonion::half(1, 1, 1, 1, -1, -1, -1, -1);

    // Sum of two half-integers with aligned pattern → integer
    let sum = h1 + h2;
    assert!(
        sum.is_integer(),
        "sum of aligned half-integers should be integer"
    );

    // h1 + h2 = ½(2, 2, 2, 2, 0, 0, 0, 0) = (1, 1, 1, 1, 0, 0, 0, 0)
    assert_eq!(sum, IntegralOctonion::integer(1, 1, 1, 1, 0, 0, 0, 0));
}

#[test]
fn half_integer_multiplication() {
    let h = IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 1);

    // h² should be in the lattice
    let h2 = h * h;
    // Verify it's a valid E8 point by checking we can compute its norm
    let _ = h2.norm();
}

#[test]
fn half_integer_subtraction() {
    let h1 = IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 1);
    let h2 = IntegralOctonion::half(1, 1, 1, 1, -1, -1, -1, -1);

    let diff = h1 - h2;
    // ½(0, 0, 0, 0, 2, 2, 2, 2) = (0, 0, 0, 0, 1, 1, 1, 1)
    assert_eq!(diff, IntegralOctonion::integer(0, 0, 0, 0, 1, 1, 1, 1));
}

// =============================================================================
// CONVERSIONS
// =============================================================================

#[test]
fn conversion_roundtrip_integer() {
    let x = IntegralOctonion::integer(1, -2, 3, 0, 0, 0, 0, 0);
    let oct = x.to_octonion();
    let back = IntegralOctonion::try_from_octonion(&oct).unwrap();
    assert_eq!(x, back);
}

#[test]
fn conversion_roundtrip_half_integer() {
    let h = IntegralOctonion::half(1, 1, 1, 1, -1, -1, -1, -1);
    let oct_h = h.to_octonion();
    let back_h = IntegralOctonion::try_from_octonion(&oct_h).unwrap();
    assert_eq!(h, back_h);
}

#[test]
fn conversion_rejects_non_lattice_point() {
    let oct = octonion::Octonion::new(0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assert!(IntegralOctonion::try_from_octonion(&oct).is_none());
}

#[test]
fn from_i64() {
    let x: IntegralOctonion = 5i64.into();
    assert_eq!(x, IntegralOctonion::integer(5, 0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn from_i32() {
    let x: IntegralOctonion = 5i32.into();
    assert_eq!(x, IntegralOctonion::integer(5, 0, 0, 0, 0, 0, 0, 0));
}

// =============================================================================
// ACCESSORS
// =============================================================================

#[test]
fn to_array() {
    let x = IntegralOctonion::integer(1, 2, 3, 0, 0, 0, 0, 0);
    let arr = x.to_array();
    assert_eq!(arr, [2, 4, 6, 0, 0, 0, 0, 0]); // doubled coefficients
}

#[test]
fn coeff_accessor() {
    let x = IntegralOctonion::integer(1, 2, 3, 4, 5, 6, 7, 8);
    assert_eq!(x.coeff(0), 2); // doubled
    assert_eq!(x.coeff(3), 8); // doubled
    assert_eq!(x.coeff(7), 16); // doubled
}

#[test]
fn is_zero_test() {
    assert!(IntegralOctonion::ZERO.is_zero());
    assert!(!IntegralOctonion::ONE.is_zero());
}

#[test]
fn is_integer_test() {
    assert!(IntegralOctonion::ONE.is_integer());
    assert!(IntegralOctonion::integer(1, 2, 3, 0, 0, 0, 0, 0).is_integer());
    assert!(!IntegralOctonion::H.is_integer());
}

#[test]
fn is_half_integer_test() {
    assert!(!IntegralOctonion::ONE.is_half_integer());
    assert!(IntegralOctonion::H.is_half_integer());
}
