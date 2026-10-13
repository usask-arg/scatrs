//! The committed Arb reference data, for unit tests (test builds only).

use std::io::ErrorKind;

/// The text of `reference/data/<name>`, or `None` if the file does not exist.
///
/// The published crate does not include the reference data, so its unit tests skip the
/// reference checks (with a message) rather than fail. With `SCATRS_REQUIRE_REFERENCE_DATA=1`,
/// which CI sets, a missing file fails the test instead.
pub(crate) fn reference_file(name: &str) -> Option<String> {
    let path = format!("{}/../reference/data/{name}", env!("CARGO_MANIFEST_DIR"));
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(e)
            if e.kind() == ErrorKind::NotFound
                && std::env::var("SCATRS_REQUIRE_REFERENCE_DATA").as_deref() != Ok("1") =>
        {
            eprintln!(
                "skipping the check against {path}: the file does not exist (as in the \
                 published crate); set SCATRS_REQUIRE_REFERENCE_DATA=1 to fail instead"
            );
            None
        }
        Err(e) => panic!("cannot read reference file {path}: {e}"),
    }
}
