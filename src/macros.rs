//! Macros for explicit parenthesization of octonion expressions.
//!
//! These macros make multiplication grouping explicit in the syntax,
//! addressing the non-associativity of octonion multiplication.

/// Evaluates an octonion expression with explicit parenthesization.
///
/// This macro passes through to standard Rust evaluation, making it clear
/// that the parentheses in your expression control the evaluation order.
/// It serves as documentation that you've considered the grouping.
///
/// # Syntax
///
/// The macro accepts expressions using `*` for multiplication.
/// Parentheses in the input directly control evaluation order.
///
/// # Examples
///
/// ```
/// use octonion::{Octonion, octo_eval};
///
/// let a = Octonion::E1;
/// let b = Octonion::E2;
/// let c = Octonion::E4;
///
/// // Left-associative: (a * b) * c
/// let left = octo_eval!((a * b) * c);
///
/// // Right-associative: a * (b * c)
/// let right = octo_eval!(a * (b * c));
///
/// // These differ due to non-associativity
/// assert_ne!(left, right);
///
/// // Balanced: (a * b) * (c * d)
/// let d = Octonion::E5;
/// let balanced = octo_eval!((a * b) * (c * d));
/// ```
///
/// # Why Use This Macro?
///
/// While this macro simply evaluates its argument, it serves as:
/// 1. **Documentation** - Makes clear you've considered parenthesization
/// 2. **Clarity** - Signals to readers that grouping matters here
/// 3. **Grep-ability** - Easy to find all places where explicit grouping is used
#[macro_export]
macro_rules! octo_eval {
    ($expr:expr) => {
        $expr
    };
}

/// Evaluates an octonion expression with keyword-based grouping.
///
/// This macro provides explicit grouping keywords that make parenthesization
/// unambiguous without relying solely on nested parentheses.
///
/// # Syntax
///
/// - `left[a, b, c]` produces `(a * b) * c`
/// - `right[a, b, c]` produces `a * (b * c)`
/// - `left[a, b, c, d]` produces `((a * b) * c) * d`
/// - `right[a, b, c, d]` produces `a * (b * (c * d))`
/// - `balanced[a, b, c, d]` produces `(a * b) * (c * d)`
///
/// # Examples
///
/// ```
/// use octonion::{Octonion, octo_group};
///
/// let a = Octonion::E1;
/// let b = Octonion::E2;
/// let c = Octonion::E4;
///
/// // Left-associative triple
/// let left = octo_group!(left[a, b, c]);
/// assert_eq!(left, (a * b) * c);
///
/// // Right-associative triple
/// let right = octo_group!(right[a, b, c]);
/// assert_eq!(right, a * (b * c));
///
/// // These differ!
/// assert_ne!(left, right);
/// ```
///
/// ```
/// use octonion::{Octonion, octo_group};
///
/// let a = Octonion::E1;
/// let b = Octonion::E2;
/// let c = Octonion::E4;
/// let d = Octonion::E5;
///
/// // Balanced quad: (a * b) * (c * d)
/// let balanced = octo_group!(balanced[a, b, c, d]);
/// assert_eq!(balanced, (a * b) * (c * d));
///
/// // Left quad: ((a * b) * c) * d
/// let left = octo_group!(left[a, b, c, d]);
/// assert_eq!(left, ((a * b) * c) * d);
///
/// // Right quad: a * (b * (c * d))
/// let right = octo_group!(right[a, b, c, d]);
/// assert_eq!(right, a * (b * (c * d)));
/// ```
#[macro_export]
macro_rules! octo_group {
    // Left-associative triple: (a * b) * c
    (left[$a:expr, $b:expr, $c:expr]) => {
        (($a) * ($b)) * ($c)
    };

    // Right-associative triple: a * (b * c)
    (right[$a:expr, $b:expr, $c:expr]) => {
        ($a) * (($b) * ($c))
    };

    // Fully left quad: ((a * b) * c) * d
    (left[$a:expr, $b:expr, $c:expr, $d:expr]) => {
        ((($a) * ($b)) * ($c)) * ($d)
    };

    // Fully right quad: a * (b * (c * d))
    (right[$a:expr, $b:expr, $c:expr, $d:expr]) => {
        ($a) * (($b) * (($c) * ($d)))
    };

    // Balanced quad: (a * b) * (c * d)
    (balanced[$a:expr, $b:expr, $c:expr, $d:expr]) => {
        (($a) * ($b)) * (($c) * ($d))
    };

    // Single value passthrough
    ($val:expr) => {
        $val
    };
}
