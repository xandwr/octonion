use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub(crate) struct Quaternion {
    pub(crate) w: f64,
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
}

impl Quaternion {
    pub(crate) const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub(crate) const fn conj(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

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

impl AddAssign for Quaternion {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

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

impl SubAssign for Quaternion {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

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
