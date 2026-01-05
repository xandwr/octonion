use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::quaternion::Quaternion;

/// A view into an [`Octonion`] that is known to have zero coefficients for `e₄` through `e₇`.
///
/// This type provides access to associative multiplication operations that are only valid
/// when the octonion lies within the quaternion subalgebra. Unlike general octonion
/// multiplication, quaternion multiplication is associative: `(xy)z = x(yz)`.
///
/// # Obtaining a QuaternionView
///
/// Use [`Octonion::as_quaternion`] to obtain a `QuaternionView`. This returns `Some` only
/// if the `e₄`, `e₅`, `e₆`, and `e₇` coefficients are exactly zero.
///
/// ```
/// use octonion::Octonion;
///
/// // A pure quaternion (e4-e7 are zero)
/// let q = Octonion::new(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
/// let view = q.as_quaternion().expect("should be a quaternion");
///
/// // Can use associative multiplication
/// let q2 = Octonion::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
/// let v2 = q2.as_quaternion().unwrap();
/// let product = view.mul(v2);
///
/// // A general octonion cannot be viewed as a quaternion
/// let oct = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0);
/// assert!(oct.as_quaternion().is_none());
/// ```
///
/// # Mathematical Background
///
/// The quaternions form a 4-dimensional subalgebra of the octonions, embedded as those
/// octonions with zero coefficients for `e₄` through `e₇`. While octonion multiplication
/// is non-associative, the quaternion subalgebra retains associativity.
#[derive(Copy, Clone, Debug)]
pub struct QuaternionView<'a> {
    octonion: &'a Octonion,
}

impl<'a> QuaternionView<'a> {
    /// Returns a reference to the underlying octonion.
    #[inline]
    pub const fn as_octonion(&self) -> &'a Octonion {
        self.octonion
    }

    /// Returns the real (scalar) part of the quaternion.
    #[inline]
    pub const fn real(&self) -> f64 {
        self.octonion.coeffs[0]
    }

    /// Returns the coefficient of `i` (same as `e₁`).
    #[inline]
    pub const fn i(&self) -> f64 {
        self.octonion.coeffs[1]
    }

    /// Returns the coefficient of `j` (same as `e₂`).
    #[inline]
    pub const fn j(&self) -> f64 {
        self.octonion.coeffs[2]
    }

    /// Returns the coefficient of `k` (same as `e₃`).
    #[inline]
    pub const fn k(&self) -> f64 {
        self.octonion.coeffs[3]
    }

    /// Performs associative quaternion multiplication, returning a new octonion.
    ///
    /// Unlike general octonion multiplication, this operation is associative:
    /// `(a.mul(b)).mul(c) = a.mul(b.mul(c))` when all operands are quaternions.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::Octonion;
    ///
    /// let i = Octonion::E1.as_quaternion().unwrap();
    /// let j = Octonion::E2.as_quaternion().unwrap();
    ///
    /// // i * j = k
    /// assert_eq!(i.mul(j), Octonion::E3);
    ///
    /// // Associativity holds for quaternion multiplication
    /// let a_oct = Octonion::new(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
    /// let b_oct = Octonion::new(0.5, -1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0);
    /// let c_oct = Octonion::new(-1.0, 0.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0);
    ///
    /// let a = a_oct.as_quaternion().unwrap();
    /// let b = b_oct.as_quaternion().unwrap();
    /// let c = c_oct.as_quaternion().unwrap();
    ///
    /// let ab = a.mul(b);
    /// let bc = b.mul(c);
    /// let left = ab.as_quaternion().unwrap().mul(c);
    /// let right = a.mul(bc.as_quaternion().unwrap());
    /// // left == right (up to floating-point precision)
    /// ```
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: QuaternionView<'_>) -> Octonion {
        let a = Quaternion::new(
            self.octonion.coeffs[0],
            self.octonion.coeffs[1],
            self.octonion.coeffs[2],
            self.octonion.coeffs[3],
        );
        let b = Quaternion::new(
            rhs.octonion.coeffs[0],
            rhs.octonion.coeffs[1],
            rhs.octonion.coeffs[2],
            rhs.octonion.coeffs[3],
        );
        let result = a * b;
        Octonion::new(result.w, result.x, result.y, result.z, 0.0, 0.0, 0.0, 0.0)
    }

    /// Returns the conjugate as an octonion.
    ///
    /// The conjugate negates the imaginary components: `conj(w + xi + yj + zk) = w - xi - yj - zk`.
    #[inline]
    pub const fn conj(&self) -> Octonion {
        Octonion::new(
            self.octonion.coeffs[0],
            -self.octonion.coeffs[1],
            -self.octonion.coeffs[2],
            -self.octonion.coeffs[3],
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Returns the squared norm of this quaternion.
    #[inline]
    pub fn norm_sqr(&self) -> f64 {
        self.octonion.coeffs[0] * self.octonion.coeffs[0]
            + self.octonion.coeffs[1] * self.octonion.coeffs[1]
            + self.octonion.coeffs[2] * self.octonion.coeffs[2]
            + self.octonion.coeffs[3] * self.octonion.coeffs[3]
    }
}

/// An octonion over `f64`, represented in the standard basis.
///
/// An octonion is an 8-dimensional hypercomplex number of the form:
///
/// ```text
/// x = a₀ + a₁e₁ + a₂e₂ + a₃e₃ + a₄e₄ + a₅e₅ + a₆e₆ + a₇e₇
/// ```
///
/// where `a₀` is the real (scalar) part and `a₁` through `a₇` are the
/// imaginary coefficients. The basis elements `e₁` through `e₇` satisfy
/// `eᵢ² = -1` and follow specific multiplication rules.
///
/// # Construction
///
/// Octonions can be created in several ways:
///
/// ```
/// use octonion::Octonion;
///
/// // From individual coefficients
/// let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
///
/// // From an array
/// let y = Octonion::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
///
/// // From a real number (embeds as scalar part)
/// let z = Octonion::from(3.14);
///
/// // Using predefined constants
/// let one = Octonion::ONE;
/// let zero = Octonion::ZERO;
/// let e1 = Octonion::E1;
/// ```
///
/// # Arithmetic Operations
///
/// The standard arithmetic operators are implemented:
///
/// ```
/// use octonion::Octonion;
///
/// let x = Octonion::new(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
/// let y = Octonion::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
///
/// let sum = x + y;
/// let diff = x - y;
/// let prod = x * y;
/// let neg = -x;
///
/// // Scalar multiplication and division
/// let scaled = x * 2.0;
/// let halved = x / 2.0;
/// ```
///
/// # Non-Associativity
///
/// Unlike real numbers, complex numbers, or quaternions, octonion
/// multiplication is **not associative**:
///
/// ```
/// use octonion::Octonion;
///
/// let e1 = Octonion::E1;
/// let e2 = Octonion::E2;
/// let e4 = Octonion::E4;
///
/// // (e1 * e2) * e4 ≠ e1 * (e2 * e4) in general
/// let left = (e1 * e2) * e4;
/// let right = e1 * (e2 * e4);
/// // These may differ!
/// ```
///
/// However, octonions are *alternative*, meaning `x(xy) = x²y` and
/// `(xy)y = xy²` always hold.
///
/// # Implementation Details
///
/// Internally, multiplication uses the Cayley-Dickson construction,
/// representing an octonion as a pair of quaternions for efficient
/// computation.
#[derive(Copy, Clone, Default, PartialEq)]
pub struct Octonion {
    coeffs: [f64; 8],
}

impl Octonion {
    /// The additive identity (zero octonion).
    ///
    /// All coefficients are zero: `0 + 0e₁ + 0e₂ + ... + 0e₇`.
    pub const ZERO: Self = Self { coeffs: [0.0; 8] };

    /// The multiplicative identity (one).
    ///
    /// Equal to the real number 1: `1 + 0e₁ + 0e₂ + ... + 0e₇`.
    pub const ONE: Self = Self {
        coeffs: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };

    /// The first imaginary basis element `e₁`.
    ///
    /// Satisfies `e₁² = -1` and `e₁ * e₂ = e₃`.
    pub const E1: Self = Self {
        coeffs: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };

    /// The second imaginary basis element `e₂`.
    ///
    /// Satisfies `e₂² = -1` and `e₂ * e₁ = -e₃`.
    pub const E2: Self = Self {
        coeffs: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };

    /// The third imaginary basis element `e₃`.
    ///
    /// Satisfies `e₃² = -1`. Equal to `e₁ * e₂`.
    pub const E3: Self = Self {
        coeffs: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    };

    /// The fourth imaginary basis element `e₄`.
    ///
    /// Satisfies `e₄² = -1` and `e₁ * e₄ = e₅`.
    pub const E4: Self = Self {
        coeffs: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    };

    /// The fifth imaginary basis element `e₅`.
    ///
    /// Satisfies `e₅² = -1`. Equal to `e₁ * e₄`.
    pub const E5: Self = Self {
        coeffs: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    };

    /// The sixth imaginary basis element `e₆`.
    ///
    /// Satisfies `e₆² = -1`.
    pub const E6: Self = Self {
        coeffs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };

    /// The seventh imaginary basis element `e₇`.
    ///
    /// Satisfies `e₇² = -1`.
    pub const E7: Self = Self {
        coeffs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    };

    /// Creates a new octonion from its eight coefficients.
    ///
    /// The coefficients correspond to:
    /// - `a0`: real (scalar) part
    /// - `a1` through `a7`: coefficients of `e₁` through `e₇`
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::Octonion;
    ///
    /// let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    /// assert_eq!(x.real(), 1.0);
    /// assert_eq!(x.coeff(1), 2.0);
    /// ```
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

    /// Creates a new octonion from an array of coefficients.
    ///
    /// The array indices correspond to the basis elements:
    /// - Index 0: real part
    /// - Indices 1-7: coefficients of `e₁` through `e₇`
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::Octonion;
    ///
    /// let coeffs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// let x = Octonion::from_array(coeffs);
    /// assert_eq!(x, Octonion::ONE);
    /// ```
    #[inline]
    pub const fn from_array(coeffs: [f64; 8]) -> Self {
        Self { coeffs }
    }

    /// Returns the coefficients as an array.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::Octonion;
    ///
    /// let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    /// let arr = x.to_array();
    /// assert_eq!(arr[0], 1.0);
    /// assert_eq!(arr[7], 8.0);
    /// ```
    #[inline]
    pub const fn to_array(self) -> [f64; 8] {
        self.coeffs
    }

    /// Returns the real (scalar) part of the octonion.
    ///
    /// This is the coefficient of the identity element (index 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::Octonion;
    ///
    /// let x = Octonion::new(3.14, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    /// assert_eq!(x.real(), 3.14);
    /// ```
    #[inline]
    pub const fn real(self) -> f64 {
        self.coeffs[0]
    }

    /// Returns the coefficient at the given index.
    ///
    /// - Index 0 returns the real part
    /// - Indices 1-7 return the coefficients of `e₁` through `e₇`
    ///
    /// # Panics
    ///
    /// Panics if `index >= 8`.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::Octonion;
    ///
    /// let x = Octonion::E3;
    /// assert_eq!(x.coeff(0), 0.0);  // real part
    /// assert_eq!(x.coeff(3), 1.0);  // e₃ coefficient
    /// ```
    #[inline]
    pub const fn coeff(self, index: usize) -> f64 {
        self.coeffs[index]
    }

    /// Returns `true` if all coefficients are exactly zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::Octonion;
    ///
    /// assert!(Octonion::ZERO.is_zero());
    /// assert!(!Octonion::ONE.is_zero());
    /// assert!(!Octonion::E1.is_zero());
    /// ```
    #[inline]
    pub fn is_zero(self) -> bool {
        self.coeffs.iter().all(|&c| c == 0.0)
    }

    /// Returns the conjugate of this octonion.
    ///
    /// The conjugate negates all imaginary components while preserving
    /// the real part:
    ///
    /// ```text
    /// conj(a₀ + a₁e₁ + ... + a₇e₇) = a₀ - a₁e₁ - ... - a₇e₇
    /// ```
    ///
    /// # Properties
    ///
    /// - `conj(conj(x)) = x`
    /// - `conj(x + y) = conj(x) + conj(y)`
    /// - `conj(xy) = conj(y) * conj(x)` (note the reversal)
    /// - `x * conj(x) = |x|²` (a non-negative real number)
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::Octonion;
    ///
    /// let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    /// let c = x.conj();
    ///
    /// assert_eq!(c.real(), 1.0);  // real part unchanged
    /// assert_eq!(c.coeff(1), -2.0);  // imaginary parts negated
    ///
    /// // Double conjugate is identity
    /// assert_eq!(x.conj().conj(), x);
    /// ```
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

    /// Returns the squared norm (squared magnitude) of this octonion.
    ///
    /// The squared norm is the sum of squares of all coefficients:
    ///
    /// ```text
    /// |x|² = a₀² + a₁² + a₂² + a₃² + a₄² + a₅² + a₆² + a₇²
    /// ```
    ///
    /// This is equivalent to `x * conj(x)`, which always yields a
    /// non-negative real number.
    ///
    /// # Properties
    ///
    /// - `norm_sqr(x) >= 0`
    /// - `norm_sqr(x) = 0` if and only if `x = 0`
    /// - `norm_sqr(xy) = norm_sqr(x) * norm_sqr(y)` (multiplicative)
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::Octonion;
    ///
    /// let x = Octonion::new(1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// assert_eq!(x.norm_sqr(), 9.0);  // 1² + 2² + 2² = 9
    ///
    /// // Basis elements have unit norm
    /// assert_eq!(Octonion::E1.norm_sqr(), 1.0);
    /// ```
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.coeffs.iter().map(|c| c * c).sum()
    }

    /// Returns the multiplicative inverse, or `None` if this is exactly zero.
    ///
    /// For a non-zero octonion `x`, the inverse is:
    ///
    /// ```text
    /// x⁻¹ = conj(x) / |x|²
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::Octonion;
    ///
    /// let x = Octonion::new(2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// let inv = x.try_inverse().unwrap();
    ///
    /// // x * x⁻¹ ≈ 1 (up to floating-point precision)
    /// let prod = x * inv;
    /// assert!((prod.real() - 1.0).abs() < 1e-10);
    ///
    /// // Zero has no inverse
    /// assert!(Octonion::ZERO.try_inverse().is_none());
    /// ```
    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        let n2 = self.norm_sqr();
        if n2 == 0.0 {
            return None;
        }
        Some(self.conj() / n2)
    }

    /// Returns a [`QuaternionView`] if this octonion lies in the quaternion subalgebra.
    ///
    /// This method checks whether the coefficients for `e₄`, `e₅`, `e₆`, and `e₇` are
    /// exactly zero. If so, it returns a view that provides access to associative
    /// multiplication operations.
    ///
    /// # Returns
    ///
    /// - `Some(QuaternionView)` if `coeff(4) == coeff(5) == coeff(6) == coeff(7) == 0.0`
    /// - `None` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use octonion::Octonion;
    ///
    /// // A pure quaternion
    /// let q = Octonion::new(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
    /// assert!(q.as_quaternion().is_some());
    ///
    /// // Basis elements e1, e2, e3 are quaternions
    /// assert!(Octonion::E1.as_quaternion().is_some());
    /// assert!(Octonion::E2.as_quaternion().is_some());
    /// assert!(Octonion::E3.as_quaternion().is_some());
    ///
    /// // Basis elements e4-e7 are not in the quaternion subalgebra
    /// assert!(Octonion::E4.as_quaternion().is_none());
    /// assert!(Octonion::E5.as_quaternion().is_none());
    ///
    /// // Use associative multiplication
    /// let a = Octonion::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// let b = Octonion::new(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    /// if let (Some(qa), Some(qb)) = (a.as_quaternion(), b.as_quaternion()) {
    ///     let product = qa.mul(qb);
    ///     // Quaternion multiplication is associative!
    /// }
    /// ```
    #[inline]
    pub fn as_quaternion(&self) -> Option<QuaternionView<'_>> {
        if self.coeffs[4] == 0.0
            && self.coeffs[5] == 0.0
            && self.coeffs[6] == 0.0
            && self.coeffs[7] == 0.0
        {
            Some(QuaternionView { octonion: self })
        } else {
            None
        }
    }

    /// Splits the octonion into a pair of quaternions for Cayley-Dickson multiplication.
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

    /// Joins a pair of quaternions into an octonion.
    #[inline]
    fn join(a: Quaternion, b: Quaternion) -> Self {
        Self {
            coeffs: [a.w, a.x, a.y, a.z, b.w, b.x, b.y, b.z],
        }
    }
}

/// Embeds a real number as an octonion with zero imaginary parts.
///
/// # Examples
///
/// ```
/// use octonion::Octonion;
///
/// let x: Octonion = 3.14.into();
/// assert_eq!(x.real(), 3.14);
/// assert_eq!(x.coeff(1), 0.0);
/// ```
impl From<f64> for Octonion {
    fn from(value: f64) -> Self {
        Self::new(value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
}

/// Component-wise addition of two octonions.
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

/// In-place addition.
impl AddAssign for Octonion {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Component-wise subtraction of two octonions.
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

/// In-place subtraction.
impl SubAssign for Octonion {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// Negation of all coefficients.
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

/// Octonion multiplication using the Cayley-Dickson construction.
///
/// **Note:** Octonion multiplication is neither commutative nor associative.
///
/// # Examples
///
/// ```
/// use octonion::Octonion;
///
/// // Non-commutativity: e1 * e2 ≠ e2 * e1
/// assert_eq!(Octonion::E1 * Octonion::E2, Octonion::E3);
/// assert_eq!(Octonion::E2 * Octonion::E1, -Octonion::E3);
/// ```
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

/// In-place octonion multiplication.
impl MulAssign for Octonion {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Scalar multiplication (octonion * scalar).
///
/// Scales all coefficients by the given factor.
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

/// Scalar multiplication (scalar * octonion).
///
/// Scales all coefficients by the given factor.
impl Mul<Octonion> for f64 {
    type Output = Octonion;

    fn mul(self, rhs: Octonion) -> Self::Output {
        rhs * self
    }
}

/// Scalar division.
///
/// Divides all coefficients by the given divisor.
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

/// In-place scalar division.
impl DivAssign<f64> for Octonion {
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

// ============================================================================
// DISPLAY AND DEBUG: Fano Plane Color-Coded Output
// ============================================================================
//
// The Fano plane organizes the 7 imaginary units into 7 lines (triples):
//   Line 1: e₁, e₂, e₃  (the quaternion subalgebra)
//   Line 2: e₁, e₄, e₅
//   Line 3: e₁, e₆, e₇  (the "outer circle" through e₁)
//   Line 4: e₂, e₄, e₆
//   Line 5: e₂, e₅, e₇
//   Line 6: e₃, e₄, e₇
//   Line 7: e₃, e₅, e₆
//
// We color by "depth" from the real axis:
//   - Real (e₀):     White/Bold     - the scalar
//   - e₁, e₂, e₃:    RGB primary    - quaternion subalgebra
//   - e₄, e₅, e₆, e₇: CMY secondary - the "new" octonion directions

/// ANSI color codes for Fano plane visualization.
mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";

    // Real part: bright white
    pub const WHITE: &str = "\x1b[97m";

    // Quaternion subalgebra (e₁, e₂, e₃): RGB primaries
    pub const RED: &str = "\x1b[91m"; // e₁ (i)
    pub const GREEN: &str = "\x1b[92m"; // e₂ (j)
    pub const BLUE: &str = "\x1b[94m"; // e₃ (k)

    // Octonion-specific (e₄, e₅, e₆, e₇): CMY secondaries + violet
    pub const YELLOW: &str = "\x1b[93m"; // e₄
    pub const CYAN: &str = "\x1b[96m"; // e₅
    pub const MAGENTA: &str = "\x1b[95m"; // e₆
    pub const VIOLET: &str = "\x1b[35m"; // e₇ (darker magenta)
}

/// Returns the ANSI color for a basis element index.
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

/// Returns the subscript character for a digit 0-7.
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

/// Dimensional classification of an octonion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OctonionKind {
    /// All coefficients zero
    Zero,
    /// Only real part non-zero
    Real,
    /// Real + one imaginary (complex-like)
    Complex(usize),
    /// Real + e₁ + e₂ + e₃ (quaternion subalgebra)
    Quaternion,
    /// General octonion
    Octonion,
}

impl Octonion {
    /// Determines the "kind" of this octonion for dimensional folding.
    fn kind(&self) -> OctonionKind {
        // Count nonzero coefficients and track which ones (no_std friendly)
        let mut nonzero_count = 0u8;
        let mut nonzero_mask = 0u8; // Bitmask of which indices are nonzero
        let mut first_nonzero = 8usize; // Index of first nonzero (8 = none)
        let mut second_nonzero = 8usize;

        for (i, &c) in self.coeffs.iter().enumerate() {
            if c != 0.0 {
                nonzero_mask |= 1 << i;
                nonzero_count += 1;
                if first_nonzero == 8 {
                    first_nonzero = i;
                } else if second_nonzero == 8 {
                    second_nonzero = i;
                }
            }
        }

        match nonzero_count {
            0 => OctonionKind::Zero,
            1 => {
                if first_nonzero == 0 {
                    OctonionKind::Real
                } else {
                    OctonionKind::Complex(first_nonzero)
                }
            }
            2 if first_nonzero == 0 => OctonionKind::Complex(second_nonzero),
            _ => {
                // Check if it's in the quaternion subalgebra (only bits 0-3 set)
                if nonzero_mask & 0b1111_0000 == 0 {
                    OctonionKind::Quaternion
                } else {
                    OctonionKind::Octonion
                }
            }
        }
    }

    /// Formats a coefficient with sign handling.
    fn fmt_coeff(
        f: &mut fmt::Formatter<'_>,
        coeff: f64,
        index: usize,
        is_first: bool,
        use_color: bool,
    ) -> fmt::Result {
        let color = if use_color { basis_color(index) } else { "" };
        let reset = if use_color { ansi::RESET } else { "" };

        let sign = if coeff < 0.0 {
            " - "
        } else if is_first {
            ""
        } else {
            " + "
        };

        let abs_coeff = coeff.abs();

        // Unit name: i, j, k for quaternion subalgebra, e₄-e₇ for the rest
        let write_unit = |f: &mut fmt::Formatter<'_>| -> fmt::Result {
            match index {
                1 => write!(f, "i"),
                2 => write!(f, "j"),
                3 => write!(f, "k"),
                n => write!(f, "e{}", subscript(n)),
            }
        };

        if index == 0 {
            // Real part
            write!(f, "{sign}{color}{abs_coeff}{reset}")
        } else if (abs_coeff - 1.0).abs() < f64::EPSILON {
            // Coefficient is ±1, omit the "1"
            write!(f, "{sign}{color}")?;
            write_unit(f)?;
            write!(f, "{reset}")
        } else {
            write!(f, "{sign}{color}{abs_coeff}")?;
            write_unit(f)?;
            write!(f, "{reset}")
        }
    }
}

impl fmt::Display for Octonion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Check if we should use colors (alternate flag '#' disables them)
        let use_color = !f.alternate();

        match self.kind() {
            OctonionKind::Zero => {
                if use_color {
                    write!(f, "{}{}0{}", ansi::DIM, ansi::WHITE, ansi::RESET)
                } else {
                    write!(f, "0")
                }
            }

            OctonionKind::Real => {
                let color = if use_color { ansi::WHITE } else { "" };
                let bold = if use_color { ansi::BOLD } else { "" };
                let reset = if use_color { ansi::RESET } else { "" };
                write!(f, "{bold}{color}{}{reset}", self.coeffs[0])
            }

            OctonionKind::Complex(i) => {
                // Print as "a + bi" style (like complex numbers)
                let color = if use_color { basis_color(i) } else { "" };
                let white = if use_color { ansi::WHITE } else { "" };
                let reset = if use_color { ansi::RESET } else { "" };

                let a = self.coeffs[0];
                let b = self.coeffs[i];

                // Helper to write the unit name (i, j, k for quaternion subalgebra)
                let write_unit = |f: &mut fmt::Formatter<'_>| -> fmt::Result {
                    match i {
                        1 => write!(f, "i"),
                        2 => write!(f, "j"),
                        3 => write!(f, "k"),
                        _ => write!(f, "e{}", subscript(i)),
                    }
                };

                if a == 0.0 {
                    // Pure imaginary
                    if (b.abs() - 1.0).abs() < f64::EPSILON {
                        if b > 0.0 {
                            write!(f, "{color}")?;
                            write_unit(f)?;
                            write!(f, "{reset}")
                        } else {
                            write!(f, "-{color}")?;
                            write_unit(f)?;
                            write!(f, "{reset}")
                        }
                    } else {
                        write!(f, "{color}{b}")?;
                        write_unit(f)?;
                        write!(f, "{reset}")
                    }
                } else {
                    // a + bi form
                    let sign = if b < 0.0 { " - " } else { " + " };
                    let abs_b = b.abs();
                    if (abs_b - 1.0).abs() < f64::EPSILON {
                        write!(f, "{white}{a}{reset}{sign}{color}")?;
                        write_unit(f)?;
                        write!(f, "{reset}")
                    } else {
                        write!(f, "{white}{a}{reset}{sign}{color}{abs_b}")?;
                        write_unit(f)?;
                        write!(f, "{reset}")
                    }
                }
            }

            OctonionKind::Quaternion => {
                // Print as "a + bi + cj + dk"
                let white = if use_color { ansi::WHITE } else { "" };
                let red = if use_color { ansi::RED } else { "" };
                let green = if use_color { ansi::GREEN } else { "" };
                let blue = if use_color { ansi::BLUE } else { "" };
                let reset = if use_color { ansi::RESET } else { "" };

                let [a, b, c, d, ..] = self.coeffs;
                let mut first = true;

                if a != 0.0 {
                    write!(f, "{white}{a}{reset}")?;
                    first = false;
                }

                for (coeff, color, name) in [(b, red, "i"), (c, green, "j"), (d, blue, "k")] {
                    if coeff != 0.0 {
                        let sign = if coeff < 0.0 {
                            " - "
                        } else if first {
                            ""
                        } else {
                            " + "
                        };
                        let abs_c = coeff.abs();
                        if (abs_c - 1.0).abs() < f64::EPSILON {
                            write!(f, "{sign}{color}{name}{reset}")?;
                        } else {
                            write!(f, "{sign}{color}{abs_c}{name}{reset}")?;
                        }
                        first = false;
                    }
                }

                if first {
                    // All were zero (shouldn't happen given our kind detection)
                    write!(f, "0")?;
                }

                Ok(())
            }

            OctonionKind::Octonion => {
                // Full octonion: show all non-zero terms with Fano coloring
                let mut first = true;

                for (i, &coeff) in self.coeffs.iter().enumerate() {
                    if coeff != 0.0 {
                        Self::fmt_coeff(f, coeff, i, first, use_color)?;
                        first = false;
                    }
                }

                if first {
                    write!(f, "0")?;
                }

                Ok(())
            }
        }
    }
}

impl fmt::Debug for Octonion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Debug shows the full structure with type annotation
        let use_color = !f.alternate();

        if use_color {
            write!(f, "{}Octonion{} {{ ", ansi::DIM, ansi::RESET)?;
        } else {
            write!(f, "Octonion {{ ")?;
        }

        // Unit names for debug output
        const UNIT_NAMES: [&str; 8] = ["", "i", "j", "k", "e₄", "e₅", "e₆", "e₇"];

        // Always show all components in debug
        for (i, &coeff) in self.coeffs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let color = if use_color { basis_color(i) } else { "" };
            let reset = if use_color { ansi::RESET } else { "" };

            if i == 0 {
                write!(f, "{color}{coeff}{reset}")?;
            } else {
                write!(f, "{color}{}={coeff}{reset}", UNIT_NAMES[i])?;
            }
        }

        write!(f, " }}")
    }
}
