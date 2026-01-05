#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

//! # Quick Start
//!
//! ```
//! use octonion::Octonion;
//!
//! // Create an octonion from coefficients
//! let x = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
//!
//! // Use basis elements
//! let e1 = Octonion::E1;
//! let e2 = Octonion::E2;
//!
//! // Basis elements square to -1
//! assert_eq!(e1 * e1, -Octonion::ONE);
//!
//! // Octonion multiplication is non-commutative
//! assert_eq!(e1 * e2, Octonion::E3);
//! assert_eq!(e2 * e1, -Octonion::E3);
//!
//! // Conjugation, norm, and inverse
//! let conj = x.conj();
//! let norm_sq = x.norm_sqr();
//! let inv = x.try_inverse().unwrap();
//! ```
//!
//! # Mathematical Background
//!
//! The octonions are an 8-dimensional normed division algebra over the reals.
//! They extend the quaternions in the same way that the quaternions extend
//! the complex numbers. An octonion can be written as:
//!
//! ```text
//! x = a₀ + a₁e₁ + a₂e₂ + a₃e₃ + a₄e₄ + a₅e₅ + a₆e₆ + a₇e₇
//! ```
//!
//! where `a₀` through `a₇` are real numbers and `e₁` through `e₇` are the
//! imaginary basis units.
//!
//! ## Properties
//!
//! - **Non-commutative**: `xy ≠ yx` in general
//! - **Non-associative**: `(xy)z ≠ x(yz)` in general
//! - **Alternative**: `x(xy) = x²y` and `(xy)y = xy²` (weaker than associativity)
//! - **Normed**: `|xy| = |x||y|` (the norm is multiplicative)
//! - **Division algebra**: Every non-zero octonion has a multiplicative inverse
//!
//! ## Cayley-Dickson Construction
//!
//! This crate implements octonion multiplication using the Cayley-Dickson
//! construction, which represents an octonion as a pair of quaternions `(a, b)`:
//!
//! ```text
//! (a, b)(c, d) = (ac - d*b, da + bc*)
//! ```
//!
//! where `*` denotes quaternion conjugation.
//!
//! # Features
//!
//! - `no_std` compatible
//! - Zero dependencies
//! - No unsafe code
//! - Compile-time constructors via `const fn`

mod octonion;
mod quaternion;

pub use octonion::Octonion;

#[cfg(test)]
extern crate std;
