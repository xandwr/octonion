//! Expression tree for explicit parenthesization of octonion multiplication.
//!
//! This module provides [`OctoExpr`], an enum representing unevaluated
//! multiplication expressions where parenthesization is explicit in
//! the tree structure.

use alloc::boxed::Box;

use crate::Octonion;

/// An unevaluated octonion expression tree.
///
/// This type represents multiplication expressions where parenthesization
/// is explicit in the tree structure. Evaluation only happens when [`.eval()`](OctoExpr::eval)
/// is called.
///
/// # Examples
///
/// ```
/// use octonion::{Octonion, OctoExpr};
///
/// let a = OctoExpr::new(Octonion::E1);
/// let b = OctoExpr::new(Octonion::E2);
/// let c = OctoExpr::new(Octonion::E4);
///
/// // Build (a * b) * c
/// let left = a.mul(b).mul(c);
/// let result = left.eval();
/// ```
#[derive(Clone, Debug)]
pub enum OctoExpr {
    /// A leaf node holding a concrete octonion value.
    Value(Octonion),
    /// A multiplication node: `left * right`.
    Mul(Box<OctoExpr>, Box<OctoExpr>),
}

impl OctoExpr {
    /// Creates a new expression from a concrete octonion.
    #[inline]
    pub fn new(value: Octonion) -> Self {
        OctoExpr::Value(value)
    }

    /// Evaluates the expression tree, respecting the parenthesization
    /// encoded in the tree structure.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, OctoExpr};
    ///
    /// let expr = OctoExpr::new(Octonion::E1).mul(OctoExpr::new(Octonion::E2));
    /// let result = expr.eval();
    /// assert_eq!(result, Octonion::E3);
    /// ```
    pub fn eval(&self) -> Octonion {
        match self {
            OctoExpr::Value(v) => *v,
            OctoExpr::Mul(left, right) => left.eval() * right.eval(),
        }
    }

    /// Multiplies this expression by another, creating a new tree node.
    ///
    /// The result represents `(self) * other`, grouping `self` on the left.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, OctoExpr};
    ///
    /// let a = OctoExpr::new(Octonion::E1);
    /// let b = OctoExpr::new(Octonion::E2);
    ///
    /// // Creates the expression (a * b)
    /// let product = a.mul(b);
    /// ```
    #[inline]
    pub fn mul(self, other: OctoExpr) -> Self {
        OctoExpr::Mul(Box::new(self), Box::new(other))
    }

    /// Returns the number of multiplication operations in this expression.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::{Octonion, OctoExpr};
    ///
    /// let a = OctoExpr::new(Octonion::E1);
    /// assert_eq!(a.mul_count(), 0);
    ///
    /// let b = OctoExpr::new(Octonion::E2);
    /// let c = OctoExpr::new(Octonion::E4);
    /// let expr = a.mul(b).mul(c);
    /// assert_eq!(expr.mul_count(), 2);
    /// ```
    pub fn mul_count(&self) -> usize {
        match self {
            OctoExpr::Value(_) => 0,
            OctoExpr::Mul(l, r) => 1 + l.mul_count() + r.mul_count(),
        }
    }
}

impl From<Octonion> for OctoExpr {
    #[inline]
    fn from(value: Octonion) -> Self {
        OctoExpr::Value(value)
    }
}

impl From<f64> for OctoExpr {
    #[inline]
    fn from(value: f64) -> Self {
        OctoExpr::Value(Octonion::from(value))
    }
}

impl From<&Octonion> for OctoExpr {
    #[inline]
    fn from(value: &Octonion) -> Self {
        OctoExpr::Value(*value)
    }
}
