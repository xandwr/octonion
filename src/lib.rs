#![no_std]
#![forbid(unsafe_code)]

//! Minimal, dependency-free, `no_std` octonion algebra.
//!
//! The main type is [`Octonion`], an octonion over `f64` with basic operations
//! (addition/subtraction, multiplication, conjugation, norm-squared, inverse).

mod octonion;
mod quaternion;

pub use octonion::Octonion;

#[cfg(test)]
extern crate std;
