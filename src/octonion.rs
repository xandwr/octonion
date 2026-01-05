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
#[derive(Copy, Clone, Debug, Default, PartialEq)]
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
