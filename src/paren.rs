//! Builder API for explicitly parenthesized octonion expressions.
//!
//! This module provides [`Paren`], a fluent builder for constructing
//! multiplication expressions with explicit grouping control.

use crate::{OctoExpr, Octonion};

/// A builder for explicitly parenthesized octonion expressions.
///
/// This type provides a fluent API for constructing multiplication
/// expressions with explicit grouping control. Because octonion
/// multiplication is non-associative, `(a * b) * c` may differ from
/// `a * (b * c)`, and this builder makes the grouping explicit.
///
/// # Examples
///
/// ```
/// use octonion::{Octonion, Paren};
///
/// let a = Octonion::E1;
/// let b = Octonion::E2;
/// let c = Octonion::E4;
///
/// // Left-associative: (a * b) * c
/// let left = Paren::new(a).mul(b).mul(c).eval();
///
/// // Right-associative: a * (b * c)
/// let right = Paren::new(a).mul_paren(Paren::new(b).mul(c)).eval();
///
/// // These may differ due to non-associativity!
/// assert_ne!(left, right);
/// ```
///
/// # Method Chaining
///
/// The [`mul`](Paren::mul) method chains left-associatively by default:
///
/// ```
/// use octonion::{Octonion, Paren};
///
/// let a = Octonion::E1;
/// let b = Octonion::E2;
/// let c = Octonion::E4;
/// let d = Octonion::E5;
///
/// // This produces ((a * b) * c) * d
/// let result = Paren::new(a).mul(b).mul(c).mul(d).eval();
/// ```
///
/// To create right-associative or balanced groupings, nest `Paren` builders
/// or use the convenience methods like [`right3`](Paren::right3).
#[derive(Clone, Debug)]
pub struct Paren {
    expr: OctoExpr,
}

impl Paren {
    /// Creates a new parenthesized expression from an octonion.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, Paren};
    ///
    /// let p = Paren::new(Octonion::E1);
    /// ```
    #[inline]
    pub fn new(value: impl Into<Octonion>) -> Self {
        Self {
            expr: OctoExpr::Value(value.into()),
        }
    }

    /// Creates a parenthesized expression from an expression tree.
    #[inline]
    pub fn from_expr(expr: OctoExpr) -> Self {
        Self { expr }
    }

    /// Multiplies by an octonion, grouping as `(self) * other`.
    ///
    /// This chains left-associatively by default:
    /// `Paren::new(a).mul(b).mul(c)` produces `((a * b) * c)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, Paren};
    ///
    /// let a = Octonion::E1;
    /// let b = Octonion::E2;
    ///
    /// let result = Paren::new(a).mul(b).eval();
    /// assert_eq!(result, a * b);
    /// ```
    #[inline]
    pub fn mul(self, other: Octonion) -> Self {
        Self {
            expr: self.expr.mul(OctoExpr::Value(other)),
        }
    }

    /// Multiplies by a parenthesized expression, grouping as `(self) * (other)`.
    ///
    /// Use this to create right-associative or custom groupings.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, Paren};
    ///
    /// let a = Octonion::E1;
    /// let b = Octonion::E2;
    /// let c = Octonion::E4;
    ///
    /// // a * (b * c) - right associative
    /// let result = Paren::new(a).mul_paren(Paren::new(b).mul(c)).eval();
    /// assert_eq!(result, a * (b * c));
    /// ```
    #[inline]
    pub fn mul_paren(self, other: Paren) -> Self {
        Self {
            expr: self.expr.mul(other.expr),
        }
    }

    /// Evaluates the expression tree and returns the resulting octonion.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, Paren};
    ///
    /// let result = Paren::new(Octonion::E1).mul(Octonion::E2).eval();
    /// assert_eq!(result, Octonion::E3);
    /// ```
    #[inline]
    pub fn eval(self) -> Octonion {
        self.expr.eval()
    }

    /// Returns the underlying expression tree.
    #[inline]
    pub fn into_expr(self) -> OctoExpr {
        self.expr
    }

    /// Returns the number of multiplications in this expression.
    #[inline]
    pub fn mul_count(&self) -> usize {
        self.expr.mul_count()
    }

    // =========================================================================
    // Convenience methods for common grouping patterns
    // =========================================================================

    /// Creates a left-grouped triple product: `(a * b) * c`.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, Paren};
    ///
    /// let a = Octonion::E1;
    /// let b = Octonion::E2;
    /// let c = Octonion::E4;
    ///
    /// let result = Paren::left3(a, b, c).eval();
    /// assert_eq!(result, (a * b) * c);
    /// ```
    #[inline]
    pub fn left3(a: impl Into<Octonion>, b: impl Into<Octonion>, c: impl Into<Octonion>) -> Self {
        Paren::new(a).mul(b.into()).mul(c.into())
    }

    /// Creates a right-grouped triple product: `a * (b * c)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, Paren};
    ///
    /// let a = Octonion::E1;
    /// let b = Octonion::E2;
    /// let c = Octonion::E4;
    ///
    /// let result = Paren::right3(a, b, c).eval();
    /// assert_eq!(result, a * (b * c));
    /// ```
    #[inline]
    pub fn right3(a: impl Into<Octonion>, b: impl Into<Octonion>, c: impl Into<Octonion>) -> Self {
        Paren::new(a).mul_paren(Paren::new(b).mul(c.into()))
    }

    /// Creates a fully left-grouped quad product: `((a * b) * c) * d`.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, Paren};
    ///
    /// let a = Octonion::E1;
    /// let b = Octonion::E2;
    /// let c = Octonion::E4;
    /// let d = Octonion::E5;
    ///
    /// let result = Paren::left4(a, b, c, d).eval();
    /// assert_eq!(result, ((a * b) * c) * d);
    /// ```
    #[inline]
    pub fn left4(
        a: impl Into<Octonion>,
        b: impl Into<Octonion>,
        c: impl Into<Octonion>,
        d: impl Into<Octonion>,
    ) -> Self {
        Paren::new(a).mul(b.into()).mul(c.into()).mul(d.into())
    }

    /// Creates a fully right-grouped quad product: `a * (b * (c * d))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, Paren};
    ///
    /// let a = Octonion::E1;
    /// let b = Octonion::E2;
    /// let c = Octonion::E4;
    /// let d = Octonion::E5;
    ///
    /// let result = Paren::right4(a, b, c, d).eval();
    /// assert_eq!(result, a * (b * (c * d)));
    /// ```
    #[inline]
    pub fn right4(
        a: impl Into<Octonion>,
        b: impl Into<Octonion>,
        c: impl Into<Octonion>,
        d: impl Into<Octonion>,
    ) -> Self {
        Paren::new(a).mul_paren(Paren::new(b).mul_paren(Paren::new(c).mul(d.into())))
    }

    /// Creates a balanced quad product: `(a * b) * (c * d)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, Paren};
    ///
    /// let a = Octonion::E1;
    /// let b = Octonion::E2;
    /// let c = Octonion::E4;
    /// let d = Octonion::E5;
    ///
    /// let result = Paren::balanced4(a, b, c, d).eval();
    /// assert_eq!(result, (a * b) * (c * d));
    /// ```
    #[inline]
    pub fn balanced4(
        a: impl Into<Octonion>,
        b: impl Into<Octonion>,
        c: impl Into<Octonion>,
        d: impl Into<Octonion>,
    ) -> Self {
        Paren::new(a)
            .mul(b.into())
            .mul_paren(Paren::new(c).mul(d.into()))
    }
}

impl From<Octonion> for Paren {
    #[inline]
    fn from(value: Octonion) -> Self {
        Paren::new(value)
    }
}

impl From<f64> for Paren {
    #[inline]
    fn from(value: f64) -> Self {
        Paren::new(Octonion::from(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    fn approx_eq_oct(a: Octonion, b: Octonion, eps: f64) -> bool {
        let aa = a.to_array();
        let bb = b.to_array();
        for i in 0..8 {
            if !approx_eq(aa[i], bb[i], eps) {
                return false;
            }
        }
        true
    }

    #[test]
    fn left_and_right_differ() {
        // The canonical example of non-associativity
        let e1 = Octonion::E1;
        let e2 = Octonion::E2;
        let e4 = Octonion::E4;

        let left = Paren::left3(e1, e2, e4).eval();
        let right = Paren::right3(e1, e2, e4).eval();

        // (e1 * e2) * e4 ≠ e1 * (e2 * e4)
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
            approx_eq_oct(left, right, 1e-10),
            "Artin's theorem: 2-element expressions should be associative\nleft:  {left:?}\nright: {right:?}"
        );
    }

    #[test]
    fn quad_groupings_can_differ() {
        // Use general octonions to demonstrate that groupings matter
        let a = Octonion::new(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let b = Octonion::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let c = Octonion::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let d = Octonion::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

        let left = Paren::left4(a, b, c, d).eval();
        let right = Paren::right4(a, b, c, d).eval();
        let balanced = Paren::balanced4(a, b, c, d).eval();

        // At least left and right should differ for general octonions
        // (some specific combinations may happen to be equal)
        assert_ne!(
            left, right,
            "left4 and right4 should differ for general octonions"
        );

        // Verify they all compute to the expected direct expressions
        assert_eq!(left, ((a * b) * c) * d);
        assert_eq!(right, a * (b * (c * d)));
        assert_eq!(balanced, (a * b) * (c * d));
    }

    #[test]
    fn mul_count_correct() {
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
}
