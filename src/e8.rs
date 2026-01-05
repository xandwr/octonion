//! Integral Octonions (Cayley integers) over the E8 lattice.
//!
//! This module provides the [`IntegralOctonion`] type, representing elements of the
//! **Cayley integers** (also known as integral octonions or octavians). These form a
//! maximal order in the octonion algebra and correspond to the famous **E8 lattice**,
//! which achieves the densest possible sphere packing in 8 dimensions.
//!
//! # Mathematical Background
//!
//! The Cayley integers are not simply octonions with integer coefficients—that would
//! miss the beautiful E8 structure. Instead, they include "half-integer" elements
//! analogous to how the Hurwitz quaternions extend the Lipschitz integers.
//!
//! An integral octonion can be written in two equivalent forms:
//!
//! 1. **Integer coordinates**: All 8 coefficients are integers
//! 2. **Half-integer coordinates**: All 8 coefficients are half-integers (n + ½),
//!    subject to specific parity constraints
//!
//! The half-integer elements have the form:
//! ```text
//! ½(a₀ + a₁e₁ + a₂e₂ + a₃e₃ + a₄e₄ + a₅e₅ + a₆e₆ + a₇e₇)
//! ```
//! where all `aᵢ` are odd integers and their sum is even (≡ 0 mod 4).
//!
//! # The E8 Lattice
//!
//! The Cayley integers form the E8 root lattice, famous for:
//! - Being the **densest sphere packing** in 8 dimensions (proven by Viazovska, 2016)
//! - Having **240 units** (elements of norm 1), corresponding to the E8 root system
//! - Being an **even unimodular lattice** (unique in 8D)
//! - Connections to string theory, exceptional Lie groups, and the Monster group
//!
//! # Representation
//!
//! We store elements using doubled coordinates: `[i64; 8]` where each value represents
//! twice the actual coefficient. This avoids floating-point while supporting half-integers:
//! - Even values → integer coefficients (e.g., 4 represents 2)
//! - Odd values → half-integer coefficients (e.g., 3 represents 3/2)
//!
//! # Examples
//!
//! ```
//! use octonion::e8::IntegralOctonion;
//!
//! // Integer octonion (all coefficients doubled)
//! let x = IntegralOctonion::new(2, 4, 0, 0, 0, 0, 0, 0); // = 1 + 2e₁
//!
//! // Half-integer element (a unit in E8)
//! let h = IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 1); // = ½(1 + e₁ + ... + e₇)
//!
//! // Basis elements
//! let e1 = IntegralOctonion::E1;
//! let e2 = IntegralOctonion::E2;
//!
//! // Multiplication (non-commutative, non-associative)
//! assert_eq!(e1 * e2, IntegralOctonion::E3);
//! assert_eq!(e2 * e1, -IntegralOctonion::E3);
//!
//! // Norm (always an integer for Cayley integers)
//! assert_eq!(e1.norm(), 1);
//! assert_eq!(h.norm(), 2); // |½(1+1+1+1+1+1+1+1)|² = 8/4 = 2
//! ```
//!
//! # Units and Roots
//!
//! The Cayley integers have exactly **16 units** (elements with norm 1):
//! - ±1, ±e₁, ..., ±e₇
//!
//! The **E8 root system** consists of **240 elements** with norm 2:
//! - ±eᵢ ± eⱼ for i < j (112 roots)
//! - ½(±1 ± e₁ ± ... ± e₇) with an even number of minus signs (128 roots)
//!
//! ```
//! use octonion::e8::IntegralOctonion;
//!
//! let e1 = IntegralOctonion::E1;
//! assert!(e1.is_unit());
//! assert_eq!(e1.norm(), 1);
//!
//! // Half-integer elements have norm 2 (E8 roots, not units)
//! let h = IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 1);
//! assert_eq!(h.norm(), 2);
//! assert!(h.is_root()); // E8 root
//! ```

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// An integral octonion (Cayley integer) in the E8 lattice.
///
/// Coefficients are stored as doubled values to represent both integers and half-integers
/// without floating point. For example, storing `3` represents the coefficient `3/2`.
///
/// # Invariant
///
/// A valid `IntegralOctonion` must satisfy one of:
/// 1. All coefficients are even (integer octonion)
/// 2. All coefficients are odd AND their sum ≡ 0 (mod 4) (half-integer octonion)
///
/// The constructors enforce this invariant.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct IntegralOctonion {
    /// Doubled coefficients: actual value = coeffs[i] / 2
    coeffs: [i64; 8],
}

impl IntegralOctonion {
    /// The additive identity (zero).
    pub const ZERO: Self = Self { coeffs: [0; 8] };

    /// The multiplicative identity (one).
    pub const ONE: Self = Self {
        coeffs: [2, 0, 0, 0, 0, 0, 0, 0],
    };

    /// The first imaginary basis element `e₁`.
    pub const E1: Self = Self {
        coeffs: [0, 2, 0, 0, 0, 0, 0, 0],
    };

    /// The second imaginary basis element `e₂`.
    pub const E2: Self = Self {
        coeffs: [0, 0, 2, 0, 0, 0, 0, 0],
    };

    /// The third imaginary basis element `e₃`.
    pub const E3: Self = Self {
        coeffs: [0, 0, 0, 2, 0, 0, 0, 0],
    };

    /// The fourth imaginary basis element `e₄`.
    pub const E4: Self = Self {
        coeffs: [0, 0, 0, 0, 2, 0, 0, 0],
    };

    /// The fifth imaginary basis element `e₅`.
    pub const E5: Self = Self {
        coeffs: [0, 0, 0, 0, 0, 2, 0, 0],
    };

    /// The sixth imaginary basis element `e₆`.
    pub const E6: Self = Self {
        coeffs: [0, 0, 0, 0, 0, 0, 2, 0],
    };

    /// The seventh imaginary basis element `e₇`.
    pub const E7: Self = Self {
        coeffs: [0, 0, 0, 0, 0, 0, 0, 2],
    };

    /// A canonical half-integer unit: `½(1 + e₁ + e₂ + e₃ + e₄ + e₅ + e₆ + e₇)`.
    ///
    /// This is one of the 224 "exotic" units that make the E8 lattice special.
    pub const H: Self = Self {
        coeffs: [1, 1, 1, 1, 1, 1, 1, 1],
    };

    /// Creates an integral octonion from doubled integer coefficients.
    ///
    /// Each argument represents twice the actual coefficient, allowing representation
    /// of both integers (even values) and half-integers (odd values).
    ///
    /// # Panics
    ///
    /// Panics if the coefficients don't form a valid E8 lattice point:
    /// - All even (integer octonion), OR
    /// - All odd with sum ≡ 0 (mod 4) (half-integer octonion)
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::e8::IntegralOctonion;
    ///
    /// // Integer octonion: 1 + 2e₁ (coefficients are 2, 4)
    /// let x = IntegralOctonion::new(2, 4, 0, 0, 0, 0, 0, 0);
    ///
    /// // Half-integer: ½(1 + e₁ + e₂ + e₃ + e₄ + e₅ + e₆ + e₇)
    /// let h = IntegralOctonion::new(1, 1, 1, 1, 1, 1, 1, 1);
    /// ```
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        a0: i64,
        a1: i64,
        a2: i64,
        a3: i64,
        a4: i64,
        a5: i64,
        a6: i64,
        a7: i64,
    ) -> Self {
        let coeffs = [a0, a1, a2, a3, a4, a5, a6, a7];

        // Check E8 lattice membership
        let parity =
            (a0 & 1) | (a1 & 1) | (a2 & 1) | (a3 & 1) | (a4 & 1) | (a5 & 1) | (a6 & 1) | (a7 & 1);

        if parity == 0 {
            // All even: valid integer octonion
            Self { coeffs }
        } else {
            // Must be all odd with sum ≡ 0 (mod 4)
            let all_odd = (a0 & 1)
                & (a1 & 1)
                & (a2 & 1)
                & (a3 & 1)
                & (a4 & 1)
                & (a5 & 1)
                & (a6 & 1)
                & (a7 & 1);
            let sum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;

            if all_odd == 1 && (sum & 3) == 0 {
                Self { coeffs }
            } else {
                panic!(
                    "Invalid E8 lattice point: coefficients must be all even, or all odd with sum ≡ 0 (mod 4)"
                )
            }
        }
    }

    /// Creates an integral octonion from an array of doubled coefficients.
    ///
    /// # Panics
    ///
    /// Panics if the coefficients don't form a valid E8 lattice point.
    #[inline]
    pub const fn from_array(coeffs: [i64; 8]) -> Self {
        Self::new(
            coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6], coeffs[7],
        )
    }

    /// Creates an integral octonion from integer coefficients (not doubled).
    ///
    /// This is a convenience constructor that automatically doubles the values.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::e8::IntegralOctonion;
    ///
    /// let x = IntegralOctonion::integer(1, 2, 3, 0, 0, 0, 0, 0);
    /// assert_eq!(x, IntegralOctonion::new(2, 4, 6, 0, 0, 0, 0, 0));
    /// ```
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub const fn integer(
        a0: i64,
        a1: i64,
        a2: i64,
        a3: i64,
        a4: i64,
        a5: i64,
        a6: i64,
        a7: i64,
    ) -> Self {
        Self {
            coeffs: [
                a0 * 2,
                a1 * 2,
                a2 * 2,
                a3 * 2,
                a4 * 2,
                a5 * 2,
                a6 * 2,
                a7 * 2,
            ],
        }
    }

    /// Creates a half-integer octonion from odd integer coefficients.
    ///
    /// This creates `½(a₀ + a₁e₁ + ... + a₇e₇)` where all `aᵢ` should be odd.
    ///
    /// # Panics
    ///
    /// Panics if any coefficient is even or if their sum is not ≡ 0 (mod 4).
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::e8::IntegralOctonion;
    ///
    /// // ½(1 + 1 + 1 + 1 + 1 + 1 + 1 + 1) - sum = 8 ≡ 0 (mod 4) ✓
    /// let h = IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 1);
    ///
    /// // ½(1 - 1 + 1 - 1 + 1 - 1 + 1 - 1) - sum = 0 ≡ 0 (mod 4) ✓
    /// let h2 = IntegralOctonion::half(1, -1, 1, -1, 1, -1, 1, -1);
    /// ```
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub const fn half(
        a0: i64,
        a1: i64,
        a2: i64,
        a3: i64,
        a4: i64,
        a5: i64,
        a6: i64,
        a7: i64,
    ) -> Self {
        // Verify all odd
        let all_odd =
            (a0 & 1) & (a1 & 1) & (a2 & 1) & (a3 & 1) & (a4 & 1) & (a5 & 1) & (a6 & 1) & (a7 & 1);
        if all_odd != 1 {
            panic!("half() requires all odd coefficients");
        }

        // Verify sum ≡ 0 (mod 4)
        let sum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
        if (sum & 3) != 0 {
            panic!("half() requires sum of coefficients ≡ 0 (mod 4)");
        }

        Self {
            coeffs: [a0, a1, a2, a3, a4, a5, a6, a7],
        }
    }

    /// Creates an integral octonion without validating E8 membership.
    ///
    /// # Safety
    ///
    /// The caller must ensure the coefficients form a valid E8 lattice point.
    /// Using invalid coefficients may produce mathematically incorrect results.
    #[inline]
    pub const fn new_unchecked(coeffs: [i64; 8]) -> Self {
        Self { coeffs }
    }

    /// Returns the doubled coefficients as an array.
    #[inline]
    pub const fn to_array(self) -> [i64; 8] {
        self.coeffs
    }

    /// Returns the doubled coefficient at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= 8`.
    #[inline]
    pub const fn coeff(self, index: usize) -> i64 {
        self.coeffs[index]
    }

    /// Returns `true` if this is an integer octonion (all coefficients even).
    #[inline]
    pub const fn is_integer(&self) -> bool {
        (self.coeffs[0]
            | self.coeffs[1]
            | self.coeffs[2]
            | self.coeffs[3]
            | self.coeffs[4]
            | self.coeffs[5]
            | self.coeffs[6]
            | self.coeffs[7])
            & 1
            == 0
    }

    /// Returns `true` if this is a half-integer octonion (all coefficients odd).
    #[inline]
    pub const fn is_half_integer(&self) -> bool {
        (self.coeffs[0]
            & self.coeffs[1]
            & self.coeffs[2]
            & self.coeffs[3]
            & self.coeffs[4]
            & self.coeffs[5]
            & self.coeffs[6]
            & self.coeffs[7])
            & 1
            == 1
    }

    /// Returns `true` if all coefficients are exactly zero.
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.coeffs[0] == 0
            && self.coeffs[1] == 0
            && self.coeffs[2] == 0
            && self.coeffs[3] == 0
            && self.coeffs[4] == 0
            && self.coeffs[5] == 0
            && self.coeffs[6] == 0
            && self.coeffs[7] == 0
    }

    /// Returns the conjugate of this integral octonion.
    ///
    /// The conjugate negates all imaginary components while preserving the real part.
    #[inline]
    pub const fn conj(self) -> Self {
        Self {
            coeffs: [
                self.coeffs[0],
                -self.coeffs[1],
                -self.coeffs[2],
                -self.coeffs[3],
                -self.coeffs[4],
                -self.coeffs[5],
                -self.coeffs[6],
                -self.coeffs[7],
            ],
        }
    }

    /// Returns the squared norm as a raw (4× scaled) value.
    ///
    /// Since we store doubled coefficients, the sum of squares is 4× the actual norm².
    /// This method returns that raw value for internal use.
    #[inline]
    const fn norm_sqr_raw(&self) -> i64 {
        self.coeffs[0] * self.coeffs[0]
            + self.coeffs[1] * self.coeffs[1]
            + self.coeffs[2] * self.coeffs[2]
            + self.coeffs[3] * self.coeffs[3]
            + self.coeffs[4] * self.coeffs[4]
            + self.coeffs[5] * self.coeffs[5]
            + self.coeffs[6] * self.coeffs[6]
            + self.coeffs[7] * self.coeffs[7]
    }

    /// Returns the squared norm (always an integer for Cayley integers).
    ///
    /// For integral octonions, the squared norm is always an integer. This is one
    /// of the magical properties of the E8 lattice.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::e8::IntegralOctonion;
    ///
    /// assert_eq!(IntegralOctonion::E1.norm(), 1);
    /// assert_eq!(IntegralOctonion::ONE.norm(), 1);
    ///
    /// // Half-integer unit: ½(1+1+1+1+1+1+1+1), norm² = 8×1/4 = 2
    /// let h = IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 1);
    /// assert_eq!(h.norm(), 2);
    /// ```
    #[inline]
    pub const fn norm(&self) -> i64 {
        // Raw sum of squares is 4× actual norm² (since coefficients are doubled)
        self.norm_sqr_raw() / 4
    }

    /// Returns `true` if this is a unit (norm = 1).
    ///
    /// The Cayley integers have exactly 16 units: ±1, ±e₁, ..., ±e₇.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::e8::IntegralOctonion;
    ///
    /// assert!(IntegralOctonion::ONE.is_unit());
    /// assert!(IntegralOctonion::E1.is_unit());
    /// assert!((-IntegralOctonion::E7).is_unit());
    ///
    /// // Half-integer elements have norm 2, so they're not units
    /// let h = IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 1);
    /// assert!(!h.is_unit()); // norm = 2, not 1
    /// ```
    #[inline]
    pub const fn is_unit(&self) -> bool {
        // norm² = 1 iff raw sum of squares = 4
        self.norm_sqr_raw() == 4
    }

    /// Returns `true` if this is an E8 root (norm = 2).
    ///
    /// The E8 root system consists of 240 elements with norm 2:
    /// - 112 of the form ±eᵢ ± eⱼ for i < j
    /// - 128 half-integer elements ½(±1 ± e₁ ± ... ± e₇) with even number of minus signs
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::e8::IntegralOctonion;
    ///
    /// // e1 + e2 is a root
    /// let r = IntegralOctonion::integer(0, 1, 1, 0, 0, 0, 0, 0);
    /// assert!(r.is_root());
    ///
    /// // Half-integer elements with norm 2 are roots
    /// let h = IntegralOctonion::half(1, 1, 1, 1, 1, 1, 1, 1);
    /// assert!(h.is_root());
    /// ```
    #[inline]
    pub const fn is_root(&self) -> bool {
        // norm² = 2 iff raw sum of squares = 8
        self.norm_sqr_raw() == 8
    }

    /// Converts this integral octonion to a floating-point [`Octonion`].
    ///
    /// [`Octonion`]: crate::Octonion
    #[inline]
    pub fn to_octonion(&self) -> crate::Octonion {
        crate::Octonion::new(
            self.coeffs[0] as f64 / 2.0,
            self.coeffs[1] as f64 / 2.0,
            self.coeffs[2] as f64 / 2.0,
            self.coeffs[3] as f64 / 2.0,
            self.coeffs[4] as f64 / 2.0,
            self.coeffs[5] as f64 / 2.0,
            self.coeffs[6] as f64 / 2.0,
            self.coeffs[7] as f64 / 2.0,
        )
    }

    /// Attempts to create an integral octonion from a floating-point [`Octonion`].
    ///
    /// Returns `None` if the octonion doesn't lie on the E8 lattice (i.e., if
    /// the coefficients aren't integers or valid half-integers).
    ///
    /// [`Octonion`]: crate::Octonion
    pub fn try_from_octonion(oct: &crate::Octonion) -> Option<Self> {
        let arr = oct.to_array();
        let mut doubled = [0i64; 8];

        for (i, &c) in arr.iter().enumerate() {
            // Multiply by 2 and check if result is an integer
            let d = c * 2.0;
            // no_std-compatible rounding: truncate toward zero and check
            let rounded = if d >= 0.0 {
                (d + 0.5) as i64
            } else {
                (d - 0.5) as i64
            };
            let diff = d - (rounded as f64);
            if diff.abs() > 1e-10 {
                return None;
            }
            doubled[i] = rounded;
        }

        // Check E8 membership
        let parity = doubled.iter().fold(0i64, |acc, &x| acc | (x & 1));

        if parity == 0 {
            // All even: valid
            Some(Self { coeffs: doubled })
        } else {
            // Check if all odd with sum ≡ 0 (mod 4)
            let all_odd = doubled.iter().fold(1i64, |acc, &x| acc & (x & 1));
            let sum: i64 = doubled.iter().sum();

            if all_odd == 1 && (sum & 3) == 0 {
                Some(Self { coeffs: doubled })
            } else {
                None
            }
        }
    }
}

// =============================================================================
// ARITHMETIC OPERATIONS
// =============================================================================

impl Add for IntegralOctonion {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // Addition of E8 lattice points stays in E8
        // (even + even = even, odd + odd = even, and the sum constraint is preserved)
        Self {
            coeffs: [
                self.coeffs[0] + rhs.coeffs[0],
                self.coeffs[1] + rhs.coeffs[1],
                self.coeffs[2] + rhs.coeffs[2],
                self.coeffs[3] + rhs.coeffs[3],
                self.coeffs[4] + rhs.coeffs[4],
                self.coeffs[5] + rhs.coeffs[5],
                self.coeffs[6] + rhs.coeffs[6],
                self.coeffs[7] + rhs.coeffs[7],
            ],
        }
    }
}

impl AddAssign for IntegralOctonion {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for IntegralOctonion {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            coeffs: [
                self.coeffs[0] - rhs.coeffs[0],
                self.coeffs[1] - rhs.coeffs[1],
                self.coeffs[2] - rhs.coeffs[2],
                self.coeffs[3] - rhs.coeffs[3],
                self.coeffs[4] - rhs.coeffs[4],
                self.coeffs[5] - rhs.coeffs[5],
                self.coeffs[6] - rhs.coeffs[6],
                self.coeffs[7] - rhs.coeffs[7],
            ],
        }
    }
}

impl SubAssign for IntegralOctonion {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for IntegralOctonion {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            coeffs: [
                -self.coeffs[0],
                -self.coeffs[1],
                -self.coeffs[2],
                -self.coeffs[3],
                -self.coeffs[4],
                -self.coeffs[5],
                -self.coeffs[6],
                -self.coeffs[7],
            ],
        }
    }
}

/// Octonion multiplication for integral octonions.
///
/// Uses the same Cayley-Dickson construction as floating-point octonions,
/// but with integer arithmetic. The product of two E8 lattice points is
/// always another E8 lattice point (the lattice is closed under multiplication).
impl Mul for IntegralOctonion {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // We need to compute the product and divide by 4 (since both operands
        // have doubled coefficients, the raw product is 4× the actual product).
        //
        // Using direct coefficient expansion (octonion multiplication table).
        // Let a = self.coeffs, b = rhs.coeffs (both doubled).
        // Product c = a × b needs c[i] = (sum of products) / 2
        //
        // Since we store doubled values, the actual multiplication:
        // (a/2) × (b/2) = (a×b)/4, and we store 2×result, so we need (a×b)/2

        let a = self.coeffs;
        let b = rhs.coeffs;

        // Octonion multiplication table (using Cayley-Dickson / Fano plane conventions)
        // e_i × e_j = ±e_k according to the multiplication table
        //
        // The standard multiplication table:
        //      1   e1  e2  e3  e4  e5  e6  e7
        // 1    1   e1  e2  e3  e4  e5  e6  e7
        // e1   e1  -1  e3 -e2  e5 -e4 -e7  e6
        // e2   e2 -e3  -1  e1  e6  e7 -e4 -e5
        // e3   e3  e2 -e1  -1  e7 -e6  e5 -e4
        // e4   e4 -e5 -e6 -e7  -1  e1  e2  e3
        // e5   e5  e4 -e7  e6 -e1  -1 -e3  e2
        // e6   e6  e7  e4 -e5 -e2  e3  -1 -e1
        // e7   e7 -e6  e5  e4 -e3 -e2  e1  -1

        // Compute each coefficient of the product
        // c[0] = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
        let c0 = a[0] * b[0]
            - a[1] * b[1]
            - a[2] * b[2]
            - a[3] * b[3]
            - a[4] * b[4]
            - a[5] * b[5]
            - a[6] * b[6]
            - a[7] * b[7];

        // c[1] = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6
        let c1 = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2] + a[4] * b[5]
            - a[5] * b[4]
            - a[6] * b[7]
            + a[7] * b[6];

        // c[2] = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5
        let c2 = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1] + a[4] * b[6] + a[5] * b[7]
            - a[6] * b[4]
            - a[7] * b[5];

        // c[3] = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4
        let c3 = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0] + a[4] * b[7] - a[5] * b[6]
            + a[6] * b[5]
            - a[7] * b[4];

        // c[4] = a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3
        let c4 = a[0] * b[4] - a[1] * b[5] - a[2] * b[6] - a[3] * b[7]
            + a[4] * b[0]
            + a[5] * b[1]
            + a[6] * b[2]
            + a[7] * b[3];

        // c[5] = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2
        let c5 = a[0] * b[5] + a[1] * b[4] - a[2] * b[7] + a[3] * b[6] - a[4] * b[1] + a[5] * b[0]
            - a[6] * b[3]
            + a[7] * b[2];

        // c[6] = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1
        let c6 = a[0] * b[6] + a[1] * b[7] + a[2] * b[4] - a[3] * b[5] - a[4] * b[2]
            + a[5] * b[3]
            + a[6] * b[0]
            - a[7] * b[1];

        // c[7] = a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0
        let c7 = a[0] * b[7] - a[1] * b[6] + a[2] * b[5] + a[3] * b[4] - a[4] * b[3] - a[5] * b[2]
            + a[6] * b[1]
            + a[7] * b[0];

        // Divide by 2 (since doubled × doubled = 4× actual, but we want 2× actual)
        // The E8 lattice closure guarantees these are all divisible by 2.
        Self {
            coeffs: [
                c0 / 2,
                c1 / 2,
                c2 / 2,
                c3 / 2,
                c4 / 2,
                c5 / 2,
                c6 / 2,
                c7 / 2,
            ],
        }
    }
}

impl MulAssign for IntegralOctonion {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Scalar multiplication by an integer.
impl Mul<i64> for IntegralOctonion {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self {
            coeffs: [
                self.coeffs[0] * rhs,
                self.coeffs[1] * rhs,
                self.coeffs[2] * rhs,
                self.coeffs[3] * rhs,
                self.coeffs[4] * rhs,
                self.coeffs[5] * rhs,
                self.coeffs[6] * rhs,
                self.coeffs[7] * rhs,
            ],
        }
    }
}

impl Mul<IntegralOctonion> for i64 {
    type Output = IntegralOctonion;

    fn mul(self, rhs: IntegralOctonion) -> Self::Output {
        rhs * self
    }
}

// =============================================================================
// DISPLAY AND DEBUG
// =============================================================================

/// ANSI color codes (reusing the same scheme as Octonion).
mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const DIM: &str = "\x1b[2m";
    pub const WHITE: &str = "\x1b[97m";
    pub const RED: &str = "\x1b[91m";
    pub const GREEN: &str = "\x1b[92m";
    pub const BLUE: &str = "\x1b[94m";
    pub const YELLOW: &str = "\x1b[93m";
    pub const CYAN: &str = "\x1b[96m";
    pub const MAGENTA: &str = "\x1b[95m";
    pub const VIOLET: &str = "\x1b[35m";
}

const fn basis_color(index: usize) -> &'static str {
    match index {
        0 => ansi::WHITE,
        1 => ansi::RED,
        2 => ansi::GREEN,
        3 => ansi::BLUE,
        4 => ansi::YELLOW,
        5 => ansi::CYAN,
        6 => ansi::MAGENTA,
        7 => ansi::VIOLET,
        _ => ansi::RESET,
    }
}

const fn subscript(n: usize) -> char {
    match n {
        0 => '₀',
        1 => '₁',
        2 => '₂',
        3 => '₃',
        4 => '₄',
        5 => '₅',
        6 => '₆',
        7 => '₇',
        _ => '?',
    }
}

impl fmt::Display for IntegralOctonion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let use_color = !f.alternate();

        if self.is_zero() {
            if use_color {
                return write!(f, "{}{}0{}", ansi::DIM, ansi::WHITE, ansi::RESET);
            } else {
                return write!(f, "0");
            }
        }

        let is_half = self.is_half_integer();
        let mut first = true;

        if is_half && use_color {
            write!(f, "{}½{}", ansi::DIM, ansi::RESET)?;
        } else if is_half {
            write!(f, "½")?;
        }

        if is_half {
            write!(f, "(")?;
        }

        for (i, &c) in self.coeffs.iter().enumerate() {
            if c == 0 {
                continue;
            }

            let color = if use_color { basis_color(i) } else { "" };
            let reset = if use_color { ansi::RESET } else { "" };

            // For half-integers, display the raw (odd) coefficient
            // For integers, display the halved coefficient
            let display_val = if is_half { c } else { c / 2 };

            let sign = if display_val < 0 {
                " - "
            } else if first {
                ""
            } else {
                " + "
            };

            let abs_val = display_val.abs();

            if i == 0 {
                // Real part
                write!(f, "{sign}{color}{abs_val}{reset}")?;
            } else {
                // Imaginary part
                let unit = match i {
                    1 => "i",
                    2 => "j",
                    3 => "k",
                    _ => "",
                };

                if abs_val == 1 {
                    if i <= 3 {
                        write!(f, "{sign}{color}{unit}{reset}")?;
                    } else {
                        write!(f, "{sign}{color}e{}{reset}", subscript(i))?;
                    }
                } else if i <= 3 {
                    write!(f, "{sign}{color}{abs_val}{unit}{reset}")?;
                } else {
                    write!(f, "{sign}{color}{abs_val}e{}{reset}", subscript(i))?;
                }
            }

            first = false;
        }

        if is_half {
            write!(f, ")")?;
        }

        Ok(())
    }
}

impl fmt::Debug for IntegralOctonion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let use_color = !f.alternate();

        if use_color {
            write!(f, "{}IntegralOctonion{} {{ ", ansi::DIM, ansi::RESET)?;
        } else {
            write!(f, "IntegralOctonion {{ ")?;
        }

        const UNIT_NAMES: [&str; 8] = ["", "i", "j", "k", "e₄", "e₅", "e₆", "e₇"];

        for (i, &c) in self.coeffs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let color = if use_color { basis_color(i) } else { "" };
            let reset = if use_color { ansi::RESET } else { "" };

            if i == 0 {
                write!(f, "{color}{c}/2{reset}")?;
            } else {
                write!(f, "{color}{}={c}/2{reset}", UNIT_NAMES[i])?;
            }
        }

        write!(f, " }}")
    }
}

// =============================================================================
// CONVERSIONS
// =============================================================================

impl From<i64> for IntegralOctonion {
    fn from(value: i64) -> Self {
        Self::integer(value, 0, 0, 0, 0, 0, 0, 0)
    }
}

impl From<i32> for IntegralOctonion {
    fn from(value: i32) -> Self {
        Self::integer(value as i64, 0, 0, 0, 0, 0, 0, 0)
    }
}
