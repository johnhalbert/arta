//! Converts token byte pieces into streamable text. A token can split a UTF-8 character or a
//! stop string, so the iterator carries an undecided suffix between pieces.

use anyhow::Result;
use std::{iter::from_fn, str};

/// Decode token bytes without exposing partial characters or stop sequences.
pub(super) fn text_chunks(
    pieces: impl Iterator<Item = Result<Vec<u8>>>,
    stops: Vec<String>,
) -> impl Iterator<Item = Result<String>> {
    text_chunks_with_stop(pieces, stops, || {})
}

/// Retain the longest suffix matching the start of a stop string and any incomplete UTF-8
/// character. Discard bytes from the earliest full stop onward, calling the hook once. At
/// ordinary EOF flush pending bytes with lossy UTF-8 decoding.
pub(super) fn text_chunks_with_stop(
    mut pieces: impl Iterator<Item = Result<Vec<u8>>>,
    mut stops: Vec<String>,
    mut on_stop: impl FnMut(),
) -> impl Iterator<Item = Result<String>> {
    stops.retain(|stop| !stop.is_empty());
    let mut pending = Vec::new();
    let mut finished = false;

    from_fn(move || {
        while !finished {
            match pieces.next() {
                Some(Ok(piece)) => pending.extend(piece),
                Some(Err(error)) => {
                    finished = true;
                    return Some(Err(error));
                }
                None => {
                    finished = true;
                    return (!pending.is_empty())
                        .then(|| Ok(String::from_utf8_lossy(&pending).into_owned()));
                }
            }

            if let Some(position) = stops
                .iter()
                .filter_map(|stop| {
                    pending
                        .windows(stop.len())
                        .position(|part| part == stop.as_bytes())
                })
                .min()
            {
                on_stop();
                finished = true;
                return (position != 0)
                    .then(|| Ok(String::from_utf8_lossy(&pending[..position]).into_owned()));
            }

            // Retain only bytes that could still complete a stop string or character.
            let stop_prefix = stops
                .iter()
                .filter_map(|stop| {
                    let stop = stop.as_bytes();
                    let max = pending.len().min(stop.len() - 1);
                    (1..=max).rev().find(|&len| pending.ends_with(&stop[..len]))
                })
                .max()
                .unwrap_or(0);
            let mut end = pending.len() - stop_prefix;
            if let Some(last) = pending[..end].utf8_chunks().last() {
                let invalid = last.invalid();
                if str::from_utf8(invalid).is_err_and(|error| error.error_len().is_none()) {
                    end -= invalid.len();
                }
            }
            if end != 0 {
                let text = String::from_utf8_lossy(&pending[..end]).into_owned();
                pending.drain(..end);
                return Some(Ok(text));
            }
        }
        None
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn emits_text_before_completion_and_hides_a_split_stop_string() -> Result<()> {
        let read = Cell::new(0);
        let pieces: &[&[u8]] = &[
            b"ready ",
            b"\xe2",
            b"\x82\xac",
            b" text <s",
            b"top> ignored",
            b"unread",
        ];
        let pieces = pieces.iter().map(|piece| {
            read.set(read.get() + 1);
            Ok(piece.to_vec())
        });
        let mut chunks = text_chunks(pieces, vec!["<stop>".into()]);
        assert_eq!(chunks.next().transpose()?.as_deref(), Some("ready "));
        assert_eq!(read.get(), 1);
        assert_eq!(chunks.next().transpose()?.as_deref(), Some("€"));
        assert_eq!(chunks.next().transpose()?.as_deref(), Some(" text "));
        assert!(chunks.next().is_none());
        assert_eq!(read.get(), 5);
        Ok(())
    }

    #[test]
    fn flushes_a_stop_prefix_and_a_truncated_character_at_the_end() -> Result<()> {
        let pieces: &[&[u8]] = &[b"prefix<st", b"ill<st", b"\xe2"];
        let pieces = pieces.iter().map(|piece| Ok(piece.to_vec()));
        let text: String = text_chunks(pieces, vec!["<stop>".into()]).collect::<Result<_>>()?;
        assert_eq!(text, "prefix<still<st�");

        let pieces = [Ok(b"prefix<st".to_vec())];
        let text: String =
            text_chunks(pieces.into_iter(), vec!["<stop>".into()]).collect::<Result<_>>()?;
        assert_eq!(text, "prefix<st");
        Ok(())
    }
}
