//! Error type shared by every scatrs computation.
//!
//! Every fallible function returns [`Result`], whose [`Error`] says what went wrong:
//!
//! * [`Error::InvalidInput`]: an argument is outside its valid range or a documented limit
//!   (for example a negative imaginary part of the refractive index). Nothing is computed.
//! * [`Error::OutsideValidatedDomain`]: the inputs are valid, but outside the domain where the
//!   accuracy has been validated. Set `allow_unvalidated` to compute anyway.
//! * [`Error::NotConverged`]: an iteration missed the accuracy it promises.
//!
//! An ensemble whose size integration does not meet its tolerance is not an error: its result
//! is returned, with `converged` false.

/// Errors returned by scatrs computations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An input is outside its mathematically valid range (for example `|x| > 1` for `x = cos θ`).
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// An iterative or truncated computation did not reach the requested accuracy.
    #[error("not converged: achieved {achieved:e}, requested {requested:e}")]
    NotConverged {
        /// Accuracy actually reached (method-specific measure).
        achieved: f64,
        /// Accuracy that was requested.
        requested: f64,
    },
    /// The inputs are valid but outside the domain where accuracy has been validated.
    #[error("outside validated domain: {0}")]
    OutsideValidatedDomain(String),
}

/// Result alias using [`Error`].
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_informative() {
        assert_eq!(
            Error::InvalidInput("n must be >= 1".into()).to_string(),
            "invalid input: n must be >= 1"
        );
        assert_eq!(
            Error::NotConverged {
                achieved: 1e-3,
                requested: 1e-6
            }
            .to_string(),
            "not converged: achieved 1e-3, requested 1e-6"
        );
        assert_eq!(
            Error::OutsideValidatedDomain("x > 2e4".into()).to_string(),
            "outside validated domain: x > 2e4"
        );
    }
}
