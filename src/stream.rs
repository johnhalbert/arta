//! Text crosses the library boundary as a fallible chunk stream. Framing allows multiple
//! messages on one stream and distinguishes an empty message from end-of-input.

use anyhow::Result;
use futures_util::{
    Stream as AsyncStream, StreamExt, TryStreamExt,
    stream::{self, BoxStream},
};
use std::{
    future,
    pin::Pin,
    task::{Context, Poll, ready},
};

/// One part of a framed text message.
#[derive(Debug, PartialEq, Eq)]
pub enum Frame {
    /// Start of one message, including an empty one.
    Begin,
    /// Text to append to the current message in arrival order.
    Chunk(String),
    /// Successful completion of the current message.
    End,
}

/// The asynchronous text chunks of one message.
pub struct Message {
    /// Owned, fallible text source that may outlive the request that produced it.
    chunks: BoxStream<'static, Result<String>>,
}

impl Message {
    /// Own a sendable chunk source so inference can continue after the provider call returns.
    pub fn new(chunks: impl AsyncStream<Item = Result<String>> + Send + 'static) -> Self {
        Self {
            chunks: chunks.boxed(),
        }
    }

    /// Frame this message with a begin marker and, on success, an end marker.
    pub fn into_stream(self) -> Stream<'static> {
        Stream::new(
            stream::iter([Ok(Frame::Begin)])
                .chain(self.chunks.map_ok(Frame::Chunk))
                .chain(stream::iter([Ok(Frame::End)])),
        )
    }

    /// Consume all chunks in order. Return the first error if the message ends
    /// unsuccessfully.
    pub async fn into_text(self) -> Result<String> {
        self.chunks
            .try_fold(String::new(), |mut text, chunk| {
                text.push_str(&chunk);
                future::ready(Ok(text))
            })
            .await
    }
}

/// An asynchronous stream of framed messages. An error terminates the stream.
pub struct Stream<'a> {
    /// Active frame source. Taking it on error or EOF makes subsequent polls terminal.
    frames: Option<BoxStream<'a, Result<Frame>>>,
}

impl<'a> Stream<'a> {
    /// Wrap a frame source and fuse it on the first error or end-of-stream.
    pub fn new(frames: impl AsyncStream<Item = Result<Frame>> + Send + 'a) -> Self {
        Self {
            frames: Some(frames.boxed()),
        }
    }
}

impl AsyncStream for Stream<'_> {
    type Item = Result<Frame>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let Some(frames) = self.frames.as_mut() else {
            return Poll::Ready(None);
        };
        let frame = ready!(frames.as_mut().poll_next(context));
        if !matches!(&frame, Some(Ok(_))) {
            self.frames = None;
        }
        Poll::Ready(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[tokio::test]
    async fn a_failed_message_does_not_emit_an_end_frame() -> Result<()> {
        let chunks = [
            Ok("partial".into()),
            Err(anyhow!("decode failed")),
            Ok("unread".into()),
        ];
        let mut stream = Message::new(stream::iter(chunks)).into_stream();
        assert_eq!(stream.next().await.transpose()?, Some(Frame::Begin));
        assert_eq!(
            stream.next().await.transpose()?,
            Some(Frame::Chunk("partial".into()))
        );
        assert_eq!(
            stream.next().await.unwrap().unwrap_err().to_string(),
            "decode failed"
        );
        assert!(stream.next().await.is_none());
        Ok(())
    }
}
