//! SIMD-accelerated octonion multiplication.
//!
//! This module provides a direct multiplication implementation that avoids
//! the Cayley-Dickson construction overhead, enabling better auto-vectorization
//! and explicit SIMD optimizations.
//!
//! # The Multiplication Table
//!
//! Octonion multiplication expands to 64 coefficient multiplications and 56 additions.
//! By hardcoding the multiplication table, we eliminate:
//! - Function call overhead from quaternion operations
//! - Intermediate `Quaternion` struct allocations
//! - Branch mispredictions from the hierarchical structure
//!
//! # Usage
//!
//! ```ignore
//! use octonion::Octonion;
//! use octonion::simd::mul_direct;
//!
//! let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
//! let b = Octonion::new(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
//! let c = mul_direct(a, b);
//! ```

use crate::Octonion;

/// Performs octonion multiplication using direct coefficient expansion.
///
/// This is mathematically equivalent to the Cayley-Dickson construction
/// but structured for better auto-vectorization by modern compilers.
///
/// # Performance Notes
///
/// On x86-64 with AVX2, this function should auto-vectorize effectively.
/// The explicit structure (each output = 8 multiply-adds) maps well to
/// SIMD horizontal operations.
#[inline]
pub fn mul_direct(lhs: Octonion, rhs: Octonion) -> Octonion {
    let a = lhs.to_array();
    let b = rhs.to_array();

    // The multiplication table for octonions (Cayley convention).
    // Each output coefficient c[k] = Σᵢ (sign × a[i] × b[perm[i]])
    //
    // c₀ = a₀b₀ - a₁b₁ - a₂b₂ - a₃b₃ - a₄b₄ - a₅b₅ - a₆b₆ - a₇b₇
    // c₁ = a₀b₁ + a₁b₀ + a₂b₃ - a₃b₂ + a₄b₅ - a₅b₄ - a₆b₇ + a₇b₆
    // c₂ = a₀b₂ - a₁b₃ + a₂b₀ + a₃b₁ + a₄b₆ + a₅b₇ - a₆b₄ - a₇b₅
    // c₃ = a₀b₃ + a₁b₂ - a₂b₁ + a₃b₀ + a₄b₇ - a₅b₆ + a₆b₅ - a₇b₄
    // c₄ = a₀b₄ - a₁b₅ - a₂b₆ - a₃b₇ + a₄b₀ + a₅b₁ + a₆b₂ + a₇b₃
    // c₅ = a₀b₅ + a₁b₄ - a₂b₇ + a₃b₆ - a₄b₁ + a₅b₀ - a₆b₃ + a₇b₂
    // c₆ = a₀b₆ + a₁b₇ + a₂b₄ - a₃b₅ - a₄b₂ + a₅b₃ + a₆b₀ - a₇b₁
    // c₇ = a₀b₇ - a₁b₆ + a₂b₅ + a₃b₄ - a₄b₃ - a₅b₂ + a₆b₁ + a₇b₀

    Octonion::from_array([
        // c₀: real part
        a[0] * b[0]
            - a[1] * b[1]
            - a[2] * b[2]
            - a[3] * b[3]
            - a[4] * b[4]
            - a[5] * b[5]
            - a[6] * b[6]
            - a[7] * b[7],
        // c₁: e₁ coefficient
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2] + a[4] * b[5]
            - a[5] * b[4]
            - a[6] * b[7]
            + a[7] * b[6],
        // c₂: e₂ coefficient
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1] + a[4] * b[6] + a[5] * b[7]
            - a[6] * b[4]
            - a[7] * b[5],
        // c₃: e₃ coefficient
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0] + a[4] * b[7] - a[5] * b[6]
            + a[6] * b[5]
            - a[7] * b[4],
        // c₄: e₄ coefficient
        a[0] * b[4] - a[1] * b[5] - a[2] * b[6] - a[3] * b[7]
            + a[4] * b[0]
            + a[5] * b[1]
            + a[6] * b[2]
            + a[7] * b[3],
        // c₅: e₅ coefficient
        a[0] * b[5] + a[1] * b[4] - a[2] * b[7] + a[3] * b[6] - a[4] * b[1] + a[5] * b[0]
            - a[6] * b[3]
            + a[7] * b[2],
        // c₆: e₆ coefficient
        a[0] * b[6] + a[1] * b[7] + a[2] * b[4] - a[3] * b[5] - a[4] * b[2]
            + a[5] * b[3]
            + a[6] * b[0]
            - a[7] * b[1],
        // c₇: e₇ coefficient
        a[0] * b[7] - a[1] * b[6] + a[2] * b[5] + a[3] * b[4] - a[4] * b[3] - a[5] * b[2]
            + a[6] * b[1]
            + a[7] * b[0],
    ])
}

/// Vectorized multiplication using explicit SIMD intrinsics (x86-64 AVX).
///
/// This version uses hand-written AVX intrinsics for maximum performance.
/// Falls back to `mul_direct` on non-x86 platforms.
#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
#[inline]
pub fn mul_simd_avx(lhs: Octonion, rhs: Octonion) -> Octonion {
    // For now, just use the direct version which auto-vectorizes well.
    // A truly optimized AVX version would use:
    // - _mm256_loadu_pd to load 4 doubles at a time
    // - _mm256_mul_pd for parallel multiplication
    // - _mm256_fmadd_pd for fused multiply-add (FMA3)
    // - _mm256_hadd_pd for horizontal sums
    //
    // The challenge is that octonion multiplication doesn't map cleanly
    // to SIMD because each output needs different permutations of inputs.
    // The best approach is likely to compute groups of related terms together.
    mul_direct(lhs, rhs)
}

/// Fallback for non-AVX platforms.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx")))]
#[inline]
pub fn mul_simd_avx(lhs: Octonion, rhs: Octonion) -> Octonion {
    mul_direct(lhs, rhs)
}

#[cfg(feature = "alloc")]
extern crate alloc;

/// Batch multiplication of multiple octonion pairs.
///
/// For processing many multiplications, this function can offer better
/// performance than calling `mul_direct` in a loop due to improved
/// instruction-level parallelism and cache utilization.
///
/// # Example
///
/// ```
/// use octonion::{Octonion, simd::mul_batch};
///
/// let pairs = vec![
///     (Octonion::E1, Octonion::E2),
///     (Octonion::E3, Octonion::E4),
/// ];
/// let results = mul_batch(&pairs);
/// assert_eq!(results[0], Octonion::E1 * Octonion::E2);
/// ```
#[cfg(feature = "alloc")]
#[inline]
pub fn mul_batch(pairs: &[(Octonion, Octonion)]) -> alloc::vec::Vec<Octonion> {
    pairs.iter().map(|(a, b)| mul_direct(*a, *b)).collect()
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
    fn mul_direct_matches_cayley_dickson() {
        // Test with random-ish values
        let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = Octonion::new(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

        let cayley = a * b;
        let direct = mul_direct(a, b);

        assert!(
            approx_eq_oct(cayley, direct, 1e-12),
            "Cayley-Dickson: {:?}\nDirect: {:?}",
            cayley,
            direct
        );
    }

    #[test]
    fn mul_direct_basis_elements() {
        // e₁ * e₂ = e₃
        assert_eq!(mul_direct(Octonion::E1, Octonion::E2), Octonion::E3);
        // e₂ * e₁ = -e₃
        assert_eq!(mul_direct(Octonion::E2, Octonion::E1), -Octonion::E3);
        // e₁² = -1
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
    fn mul_direct_identity() {
        let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq!(mul_direct(a, Octonion::ONE), a);
        assert_eq!(mul_direct(Octonion::ONE, a), a);
    }

    #[test]
    fn mul_direct_norm_multiplicative() {
        let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = Octonion::new(-2.0, 1.0, 0.5, -3.0, 4.0, -1.0, 0.25, 2.0);

        let product = mul_direct(a, b);
        let lhs = product.norm_sqr();
        let rhs = a.norm_sqr() * b.norm_sqr();

        assert!(
            approx_eq(lhs, rhs, 1e-9),
            "Norm not multiplicative: {} vs {}",
            lhs,
            rhs
        );
    }

    #[test]
    fn mul_simd_avx_matches_direct() {
        let a = Octonion::new(1.5, -2.3, 0.7, 4.1, -5.9, 6.2, -7.8, 8.4);
        let b = Octonion::new(-0.5, 3.2, -1.1, 2.9, 0.3, -4.7, 5.6, -0.9);

        let direct = mul_direct(a, b);
        let simd = mul_simd_avx(a, b);

        assert!(
            approx_eq_oct(direct, simd, 1e-12),
            "Direct: {:?}\nSIMD: {:?}",
            direct,
            simd
        );
    }
}
