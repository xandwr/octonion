//! Tests for explicit parenthesization API.
//!
//! Tests for `OctoExpr` and `Paren` builders that handle non-associative
//! multiplication with explicit grouping control.

mod common;

use common::{EPS, approx_eq_oct};
use octonion::{OctoExpr, Octonion, Paren};

// =============================================================================
// OctoExpr TESTS
// =============================================================================

#[test]
fn octoexpr_value_eval() {
    let expr = OctoExpr::new(Octonion::E1);
    assert_eq!(expr.eval(), Octonion::E1);
}

#[test]
fn octoexpr_mul_eval() {
    let expr = OctoExpr::new(Octonion::E1).mul(OctoExpr::new(Octonion::E2));
    assert_eq!(expr.eval(), Octonion::E3);
}

#[test]
fn octoexpr_mul_count() {
    let a = OctoExpr::new(Octonion::E1);
    assert_eq!(a.mul_count(), 0);

    let ab = OctoExpr::new(Octonion::E1).mul(OctoExpr::new(Octonion::E2));
    assert_eq!(ab.mul_count(), 1);

    let abc = OctoExpr::new(Octonion::E1)
        .mul(OctoExpr::new(Octonion::E2))
        .mul(OctoExpr::new(Octonion::E4));
    assert_eq!(abc.mul_count(), 2);
}

#[test]
fn octoexpr_from_octonion() {
    let oct = Octonion::E3;
    let expr: OctoExpr = oct.into();
    assert_eq!(expr.eval(), oct);
}

#[test]
fn octoexpr_from_f64() {
    let expr: OctoExpr = 3.14.into();
    assert_eq!(expr.eval(), Octonion::from(3.14));
}

#[test]
fn octoexpr_from_ref() {
    let oct = Octonion::E5;
    let expr: OctoExpr = (&oct).into();
    assert_eq!(expr.eval(), oct);
}

// =============================================================================
// PAREN BUILDER TESTS
// =============================================================================

#[test]
fn paren_new_and_eval() {
    let p = Paren::new(Octonion::E1);
    assert_eq!(p.eval(), Octonion::E1);
}

#[test]
fn paren_mul_simple() {
    let result = Paren::new(Octonion::E1).mul(Octonion::E2).eval();
    assert_eq!(result, Octonion::E1 * Octonion::E2);
}

#[test]
fn paren_mul_paren() {
    let a = Octonion::E1;
    let b = Octonion::E2;
    let c = Octonion::E4;

    // a * (b * c)
    let result = Paren::new(a).mul_paren(Paren::new(b).mul(c)).eval();
    assert_eq!(result, a * (b * c));
}

#[test]
fn paren_mul_count() {
    let a = Octonion::E1;
    let b = Octonion::E2;
    let c = Octonion::E4;

    let p1 = Paren::new(a);
    assert_eq!(p1.mul_count(), 0);

    let p2 = Paren::new(a).mul(b);
    assert_eq!(p2.mul_count(), 1);

    let p3 = Paren::left3(a, b, c);
    assert_eq!(p3.mul_count(), 2);

    let p4 = Paren::right3(a, b, c);
    assert_eq!(p4.mul_count(), 2);
}

// =============================================================================
// NON-ASSOCIATIVITY DEMONSTRATION
// =============================================================================

#[test]
fn left_and_right_differ() {
    let e1 = Octonion::E1;
    let e2 = Octonion::E2;
    let e4 = Octonion::E4;

    let left = Paren::left3(e1, e2, e4).eval();
    let right = Paren::right3(e1, e2, e4).eval();

    // (e1 * e2) * e4 != e1 * (e2 * e4)
    assert_ne!(
        left, right,
        "octonion multiplication should be non-associative"
    );
}

#[test]
fn builder_matches_direct_left() {
    let a = Octonion::E1;
    let b = Octonion::E2;
    let c = Octonion::E4;

    let built = Paren::left3(a, b, c).eval();
    let direct = (a * b) * c;
    assert_eq!(built, direct);
}

#[test]
fn builder_matches_direct_right() {
    let a = Octonion::E1;
    let b = Octonion::E2;
    let c = Octonion::E4;

    let built = Paren::right3(a, b, c).eval();
    let direct = a * (b * c);
    assert_eq!(built, direct);
}

// =============================================================================
// ARTIN'S THEOREM
// =============================================================================

#[test]
fn artin_theorem_holds() {
    // Artin's theorem: any 2 elements generate an associative subalgebra.
    // So expressions with only 2 distinct octonions should be associative.
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let y = Octonion::new(0.5, -1.0, 0.0, 2.0, -3.0, 1.0, -0.5, 0.0);

    // (x * y) * x vs x * (y * x)
    let left = Paren::left3(x, y, x).eval();
    let right = Paren::right3(x, y, x).eval();

    assert!(
        approx_eq_oct(left, right, EPS),
        "Artin's theorem: 2-element expressions should be associative\nleft:  {left:?}\nright: {right:?}"
    );
}

#[test]
fn artin_theorem_with_different_patterns() {
    let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let y = Octonion::new(0.5, -1.0, 0.0, 2.0, -3.0, 1.0, -0.5, 0.0);

    // All of these should be associative (only 2 distinct elements)
    // (x * x) * y vs x * (x * y)
    let left1 = (x * x) * y;
    let right1 = x * (x * y);
    assert!(approx_eq_oct(left1, right1, EPS));

    // (y * x) * y vs y * (x * y)
    let left2 = (y * x) * y;
    let right2 = y * (x * y);
    assert!(approx_eq_oct(left2, right2, EPS));
}

// =============================================================================
// QUAD GROUPINGS
// =============================================================================

#[test]
fn quad_groupings_can_differ() {
    let a = Octonion::new(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let b = Octonion::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    let c = Octonion::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    let d = Octonion::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

    let left = Paren::left4(a, b, c, d).eval();
    let right = Paren::right4(a, b, c, d).eval();
    let balanced = Paren::balanced4(a, b, c, d).eval();

    // At least left and right should differ for general octonions
    assert_ne!(
        left, right,
        "left4 and right4 should differ for general octonions"
    );

    // Verify they compute to expected direct expressions
    assert_eq!(left, ((a * b) * c) * d);
    assert_eq!(right, a * (b * (c * d)));
    assert_eq!(balanced, (a * b) * (c * d));
}

#[test]
fn left4_is_left_associative() {
    let a = Octonion::E1;
    let b = Octonion::E2;
    let c = Octonion::E4;
    let d = Octonion::E5;

    let built = Paren::left4(a, b, c, d).eval();
    let direct = ((a * b) * c) * d;
    assert_eq!(built, direct);
}

#[test]
fn right4_is_right_associative() {
    let a = Octonion::E1;
    let b = Octonion::E2;
    let c = Octonion::E4;
    let d = Octonion::E5;

    let built = Paren::right4(a, b, c, d).eval();
    let direct = a * (b * (c * d));
    assert_eq!(built, direct);
}

#[test]
fn balanced4_is_balanced() {
    let a = Octonion::E1;
    let b = Octonion::E2;
    let c = Octonion::E4;
    let d = Octonion::E5;

    let built = Paren::balanced4(a, b, c, d).eval();
    let direct = (a * b) * (c * d);
    assert_eq!(built, direct);
}

// =============================================================================
// CONVERSIONS
// =============================================================================

#[test]
fn paren_from_octonion() {
    let oct = Octonion::E3;
    let p: Paren = oct.into();
    assert_eq!(p.eval(), oct);
}

#[test]
fn paren_from_f64() {
    let p: Paren = 2.5.into();
    assert_eq!(p.eval(), Octonion::from(2.5));
}

#[test]
fn paren_into_expr() {
    let p = Paren::new(Octonion::E1).mul(Octonion::E2);
    let expr = p.into_expr();
    assert_eq!(expr.eval(), Octonion::E3);
}

// =============================================================================
// CHAINED OPERATIONS
// =============================================================================

#[test]
fn long_left_chain() {
    let a = Octonion::E1;
    let b = Octonion::E2;
    let c = Octonion::E3;
    let d = Octonion::E4;
    let e = Octonion::E5;

    // (((a * b) * c) * d) * e
    let result = Paren::new(a).mul(b).mul(c).mul(d).mul(e).eval();
    let expected = (((a * b) * c) * d) * e;
    assert_eq!(result, expected);
}

#[test]
fn mixed_grouping() {
    let a = Octonion::E1;
    let b = Octonion::E2;
    let c = Octonion::E4;
    let d = Octonion::E5;

    // (a * b) * (c * d) vs other groupings
    let balanced = Paren::new(a).mul(b).mul_paren(Paren::new(c).mul(d)).eval();

    // ((a * b) * c) * d
    let left = Paren::new(a).mul(b).mul(c).mul(d).eval();

    // These may or may not be equal depending on the specific elements
    // What matters is that they both compute correctly
    assert_eq!(balanced, (a * b) * (c * d));
    assert_eq!(left, ((a * b) * c) * d);
}
