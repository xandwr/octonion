//! Common test utilities and helper functions.

use octonion::Octonion;

/// Approximate equality for floating-point numbers.
pub fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

/// Approximate equality for octonions (component-wise).
pub fn approx_eq_oct(a: Octonion, b: Octonion, eps: f64) -> bool {
    let aa = a.to_array();
    let bb = b.to_array();
    for i in 0..8 {
        if !approx_eq(aa[i], bb[i], eps) {
            return false;
        }
    }
    true
}

/// Default epsilon for floating-point comparisons.
pub const EPS: f64 = 1e-12;

/// Looser epsilon for tests involving many operations.
#[allow(dead_code)]
pub const EPS_LOOSE: f64 = 1e-9;
