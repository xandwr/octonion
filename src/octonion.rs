use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::quaternion::Quaternion;

/// An octonion over `f64`, represented in the standard basis:
///
/// $x = a_0 + a_1 e_1 + a_2 e_2 + a_3 e_3 + a_4 e_4 + a_5 e_5 + a_6 e_6 + a_7 e_7$
///
/// Internally, multiplication is implemented via the Cayley–Dickson construction
/// using a pair of quaternions.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Octonion {
    coeffs: [f64; 8],
}

impl Octonion {
    pub const ZERO: Self = Self { coeffs: [0.0; 8] };
    pub const ONE: Self = Self {
        coeffs: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };

    pub const E1: Self = Self {
        coeffs: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };
    pub const E2: Self = Self {
        coeffs: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };
    pub const E3: Self = Self {
        coeffs: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    };
    pub const E4: Self = Self {
        coeffs: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    };
    pub const E5: Self = Self {
        coeffs: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    };
    pub const E6: Self = Self {
        coeffs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };
    pub const E7: Self = Self {
        coeffs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    };

    #[inline]
    pub const fn new(
        a0: f64,
        a1: f64,
        a2: f64,
        a3: f64,
        a4: f64,
        a5: f64,
        a6: f64,
        a7: f64,
    ) -> Self {
        Self {
            coeffs: [a0, a1, a2, a3, a4, a5, a6, a7],
        }
    }

    #[inline]
    pub const fn from_array(coeffs: [f64; 8]) -> Self {
        Self { coeffs }
    }

    #[inline]
    pub const fn to_array(self) -> [f64; 8] {
        self.coeffs
    }

    #[inline]
    pub const fn real(self) -> f64 {
        self.coeffs[0]
    }

    #[inline]
    pub const fn coeff(self, index: usize) -> f64 {
        self.coeffs[index]
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.coeffs.iter().all(|&c| c == 0.0)
    }

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

    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.coeffs.iter().map(|c| c * c).sum()
    }

    /// Returns the multiplicative inverse, or `None` if this is exactly zero.
    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        let n2 = self.norm_sqr();
        if n2 == 0.0 {
            return None;
        }
        Some(self.conj() / n2)
    }

    #[inline]
    fn split(self) -> (Quaternion, Quaternion) {
        let a = Quaternion::new(
            self.coeffs[0],
            self.coeffs[1],
            self.coeffs[2],
            self.coeffs[3],
        );
        let b = Quaternion::new(
            self.coeffs[4],
            self.coeffs[5],
            self.coeffs[6],
            self.coeffs[7],
        );
        (a, b)
    }

    #[inline]
    fn join(a: Quaternion, b: Quaternion) -> Self {
        Self {
            coeffs: [a.w, a.x, a.y, a.z, b.w, b.x, b.y, b.z],
        }
    }
}

impl From<f64> for Octonion {
    fn from(value: f64) -> Self {
        Self::new(value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
}

impl Add for Octonion {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.coeffs[i] + rhs.coeffs[i];
        }
        Self { coeffs: out }
    }
}

impl AddAssign for Octonion {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Octonion {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.coeffs[i] - rhs.coeffs[i];
        }
        Self { coeffs: out }
    }
}

impl SubAssign for Octonion {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Octonion {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = -self.coeffs[i];
        }
        Self { coeffs: out }
    }
}

impl Mul for Octonion {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Cayley–Dickson: treat octonions as pairs of quaternions (a, b).
        // (a, b) (c, d) = (ac - conj(d)b, da + b conj(c))
        let (a, b) = self.split();
        let (c, d) = rhs.split();

        let left = a * c - d.conj() * b;
        let right = d * a + b * c.conj();

        Self::join(left, right)
    }
}

impl MulAssign for Octonion {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Mul<f64> for Octonion {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.coeffs[i] * rhs;
        }
        Self { coeffs: out }
    }
}

impl Mul<Octonion> for f64 {
    type Output = Octonion;

    fn mul(self, rhs: Octonion) -> Self::Output {
        rhs * self
    }
}

impl Div<f64> for Octonion {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.coeffs[i] / rhs;
        }
        Self { coeffs: out }
    }
}

impl DivAssign<f64> for Octonion {
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
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
            assert_eq!(e * e, minus_one);
        }
    }

    #[test]
    fn some_basis_products() {
        assert_eq!(Octonion::E1 * Octonion::E2, Octonion::E3);
        assert_eq!(Octonion::E2 * Octonion::E1, -Octonion::E3);

        assert_eq!(Octonion::E1 * Octonion::E4, Octonion::E5);
        assert_eq!(Octonion::E4 * Octonion::E1, -Octonion::E5);
    }

    #[test]
    fn conjugation_reverses_products() {
        let x = Octonion::new(1.0, 2.0, 0.5, -3.0, 4.0, -1.0, 0.25, 2.0);
        let y = Octonion::new(-2.0, 1.0, 3.0, 0.0, -0.5, 2.0, -4.0, 1.5);
        assert_eq!((x * y).conj(), y.conj() * x.conj());
    }

    #[test]
    fn norm_is_multiplicative_up_to_rounding() {
        let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let y = Octonion::new(2.0, -1.0, 0.5, 3.5, -4.25, 0.0, 1.0, -2.0);

        let lhs = (x * y).norm_sqr();
        let rhs = x.norm_sqr() * y.norm_sqr();
        assert!(approx_eq(lhs, rhs, 1e-9), "lhs={lhs} rhs={rhs}");
    }

    #[test]
    fn inverse_works_for_simple_case() {
        let x = Octonion::new(2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let inv = x.try_inverse().unwrap();
        let prod = x * inv;
        assert!(approx_eq_oct(prod, Octonion::ONE, 1e-12), "prod={prod:?}");
    }
}
