//! Error type shared by every scatrs computation.

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
