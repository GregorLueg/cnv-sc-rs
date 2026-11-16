//! Errors in `cnv-sc-rs`.
//!
//! One enum for the whole crate, grouped by subsystem. Add new variants to the
//! matching section rather than the bottom. Anything a caller could plausibly
//! hit returns `Err`; panics are reserved for broken invariants.

use thiserror::Error;

/// All error variants that can occur across `cnv-sc-rs` operations.
#[derive(Debug, Error)]
pub enum CnvErrors {
    // -- input validation --
    /// Two inputs that must agree in length or shape do not.
    #[error("Shape mismatch for {what}: expected {expected}, got {got}.")]
    ShapeMismatch {
        /// Which input or dimension disagrees
        what: &'static str,
        /// Expected length or size
        expected: usize,
        /// Supplied length or size
        got: usize,
    },

    /// An input with no cells, genes or segments where at least one is needed.
    #[error("Empty input: {what}.")]
    EmptyInput {
        /// What was empty
        what: &'static str,
    },

    /// A parameter outside its valid range.
    #[error("Invalid parameter `{name}` = {value}; expected {expected}.")]
    InvalidParameter {
        /// Parameter name
        name: &'static str,
        /// Supplied value, formatted
        value: String,
        /// Description of the valid range
        expected: &'static str,
    },

    // -- genome --

    // -- reference --

    // -- emission models --
    /// An optimiser hit its iteration cap without meeting its tolerance.
    #[error("{what} did not converge within {iterations} iterations.")]
    NonConvergence {
        /// What was being optimised
        what: &'static str,
        /// Iterations spent
        iterations: usize,
    },
    // -- hmm and segments --
    /// Genes, segments or chromosomes are not in genomic order.
    #[error("Input not in genomic order: {what}.")]
    NotGenomicOrder {
        /// What is out of order
        what: &'static str,
    },
    // -- posteriors --
}
