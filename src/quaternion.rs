//! Internal quaternion implementation for Cayley-Dickson multiplication.
//!
//! This module provides a minimal quaternion type used internally to implement
//! octonion multiplication via the Cayley-Dickson construction. The quaternion
//! type is not exposed publicly.

use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// A quaternion over `f64`, used internally for Cayley-Dickson multiplication.
///
/// Represented as `w + xi + yj + zk` where `w` is the real part and
/// `x`, `y`, `z` are the imaginary coefficients.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub(crate) struct Quaternion {
    /// Real (scalar) part.
    pub(crate) w: f64,
    /// Coefficient of `i`.
    pub(crate) x: f64,
    /// Coefficient of `j`.
    pub(crate) y: f64,
    /// Coefficient of `k`.
    pub(crate) z: f64,
}

impl Quaternion {
    /// Creates a new quaternion from its four components.
    pub(crate) const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Returns the conjugate of this quaternion.
    ///
    /// The conjugate negates all imaginary components: `conj(w + xi + yj + zk) = w - xi - yj - zk`.
    pub(crate) const fn conj(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Component-wise addition.
impl Add for Quaternion {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

/// In-place addition.
impl AddAssign for Quaternion {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Component-wise subtraction.
impl Sub for Quaternion {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

/// In-place subtraction.
impl SubAssign for Quaternion {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// Negation of all components.
impl Neg for Quaternion {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Hamilton product (quaternion multiplication).
///
/// Uses the standard quaternion multiplication rules where `i² = j² = k² = ijk = -1`.
impl Mul for Quaternion {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bi + cj + dk)(e + fi + gj + hk)
        // = (ae - bf - cg - dh)
        // + (af + be + ch - dg)i
        // + (ag - bh + ce + df)j
        // + (ah + bg - cf + de)k
        let a = self.w;
        let b = self.x;
        let c = self.y;
        let d = self.z;

        let e = rhs.w;
        let f = rhs.x;
        let g = rhs.y;
        let h = rhs.z;

        Self {
            w: a * e - b * f - c * g - d * h,
            x: a * f + b * e + c * h - d * g,
            y: a * g - b * h + c * e + d * f,
            z: a * h + b * g - c * f + d * e,
        }
    }
}
