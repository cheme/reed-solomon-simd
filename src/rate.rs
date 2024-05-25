//! Advanced encoding/decoding using chosen [`Engine`] and [`Rate`].
//!
//! **This is an advanced module which is not needed for [simple usage] or [basic usage].**
//!
//! This module is relevant if you want to
//! - encode/decode using other [`Engine`] than [`DefaultEngine`].
//! - re-use working space of one encoder/decoder in another.
//! - understand/benchmark/test high or low rate directly.
//!
//! # Rates
//!
//! See [algorithm > Rate] for details about high/low rate.
//!
//! - [`DefaultRate`], [`DefaultRateEncoder`], [`DefaultRateDecoder`]
//!     - Encoding/decoding using high or low rate as appropriate.
//!     - These are basically same as [`ReedSolomonEncoder`]
//!       and [`ReedSolomonDecoder`] except with slightly different API
//!       which allows specifying [`Engine`] and working space.
//! - [`HighRate`], [`HighRateEncoder`], [`HighRateDecoder`]
//!     - Encoding/decoding using only high rate.
//! - [`LowRate`], [`LowRateEncoder`], [`LowRateDecoder`]
//!     - Encoding/decoding using only low rate.
//!
//! [simple usage]: crate#simple-usage
//! [basic usage]: crate#basic-usage
//! [algorithm > Rate]: crate::algorithm#rate
//! [`ReedSolomonEncoder`]: crate::ReedSolomonEncoder
//! [`ReedSolomonDecoder`]: crate::ReedSolomonDecoder
//! [`DefaultEngine`]: crate::engine::DefaultEngine

use crate::{engine::Engine, DecoderResult, EncoderResult, Error};

pub use self::{
    decoder_work::DecoderWork,
    encoder_work::EncoderWork,
    rate_default::{DefaultRate, DefaultRateDecoder, DefaultRateEncoder},
    rate_high::{HighRate, HighRateDecoder, HighRateEncoder},
    rate_low::{LowRate, LowRateDecoder, LowRateEncoder},
};

mod decoder_work;
mod encoder_work;
mod rate_default;
mod rate_high;
mod rate_low;

// ======================================================================
// Rate - PUBLIC

/// Reed-Solomon encoder/decoder generator using specific rate.
pub trait Rate<E: Engine> {
    // ============================================================
    // REQUIRED

    /// Encoder of this rate.
    type RateEncoder: RateEncoder<E>;
    /// Decoder of this rate.
    type RateDecoder: RateDecoder<E>;

    /// Returns `true` if given `original_count` / `recovery_count`
    /// combination is supported.
    fn supports(original_count: usize, recovery_count: usize) -> bool;

    // ============================================================
    // PROVIDED

    /// Creates new encoder. This is same as [`RateEncoder::new`].
    fn encoder(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<EncoderWork>,
    ) -> Result<Self::RateEncoder, Error> {
        Self::RateEncoder::new(original_count, recovery_count, shard_bytes, engine, work)
    }

    /// Creates new decoder. This is same as [`RateDecoder::new`].
    fn decoder(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<DecoderWork>,
    ) -> Result<Self::RateDecoder, Error> {
        Self::RateDecoder::new(original_count, recovery_count, shard_bytes, engine, work)
    }

    /// Returns `Ok(())` if given `original_count` / `recovery_count`
    /// combination is supported and given `shard_bytes` is valid.
    fn validate(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        if !Self::supports(original_count, recovery_count) {
            Err(Error::UnsupportedShardCount {
                original_count,
                recovery_count,
            })
        } else if shard_bytes == 0 {
            Err(Error::InvalidShardSize { shard_bytes })
        } else {
            Ok(())
        }
    }
}

// ======================================================================
// RateEncoder - PUBLIC

/// Reed-Solomon encoder using specific rate.
pub trait RateEncoder<E: Engine>
where
    Self: Sized,
{
    // ============================================================
    // REQUIRED

    /// Rate of this encoder.
    type Rate: Rate<E>;

    /// Like [`ReedSolomonEncoder::add_original_shard`](crate::ReedSolomonEncoder::add_original_shard).
    fn add_original_shard<T: AsRef<[u8]>>(&mut self, original_shard: T) -> Result<(), Error>;

    /// Like [`ReedSolomonEncoder::encode`](crate::ReedSolomonEncoder::encode).
    fn encode(&mut self) -> Result<EncoderResult, Error>;

    /// Consumes this encoder returning its [`Engine`] and [`EncoderWork`]
    /// so that they can be re-used by another encoder.
    fn into_parts(self) -> (E, EncoderWork);

    /// Like [`ReedSolomonEncoder::new`](crate::ReedSolomonEncoder::new)
    /// with [`Engine`] to use and optional working space to be re-used.
    fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<EncoderWork>,
    ) -> Result<Self, Error>;

    /// Like [`ReedSolomonEncoder::reset`](crate::ReedSolomonEncoder::reset).
    fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error>;

    // ============================================================
    // PROVIDED

    /// Returns `true` if given `original_count` / `recovery_count`
    /// combination is supported.
    ///
    /// This is same as [`Rate::supports`].
    fn supports(original_count: usize, recovery_count: usize) -> bool {
        Self::Rate::supports(original_count, recovery_count)
    }

    /// Returns `Ok(())` if given `original_count` / `recovery_count`
    /// combination is supported and given `shard_bytes` is valid.
    ///
    /// This is same as [`Rate::validate`].
    fn validate(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        Self::Rate::validate(original_count, recovery_count, shard_bytes)
    }
}

// ======================================================================
// RateDecoder - PUBLIC

/// Reed-Solomon decoder using specific rate.
pub trait RateDecoder<E: Engine>
where
    Self: Sized,
{
    // ============================================================
    // REQUIRED

    /// Rate of this decoder.
    type Rate: Rate<E>;

    /// Like [`ReedSolomonDecoder::add_original_shard`](crate::ReedSolomonDecoder::add_original_shard).
    fn add_original_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        original_shard: T,
    ) -> Result<(), Error>;

    /// Like [`ReedSolomonDecoder::add_recovery_shard`](crate::ReedSolomonDecoder::add_recovery_shard).
    fn add_recovery_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        recovery_shard: T,
    ) -> Result<(), Error>;

    /// Like [`ReedSolomonDecoder::decode`](crate::ReedSolomonDecoder::decode).
    fn decode(&mut self) -> Result<DecoderResult, Error>;

    /// Consumes this decoder returning its [`Engine`] and [`DecoderWork`]
    /// so that they can be re-used by another decoder.
    fn into_parts(self) -> (E, DecoderWork);

    /// Like [`ReedSolomonDecoder::new`](crate::ReedSolomonDecoder::new)
    /// with [`Engine`] to use and optional working space to be re-used.
    fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<DecoderWork>,
    ) -> Result<Self, Error>;

    /// Like [`ReedSolomonDecoder::reset`](crate::ReedSolomonDecoder::reset).
    fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error>;

    // ============================================================
    // PROVIDED

    /// Returns `true` if given `original_count` / `recovery_count`
    /// combination is supported.
    ///
    /// This is same as [`Rate::supports`].
    fn supports(original_count: usize, recovery_count: usize) -> bool {
        Self::Rate::supports(original_count, recovery_count)
    }

    /// Returns `Ok(())` if given `original_count` / `recovery_count`
    /// combination is supported and given `shard_bytes` is valid.
    ///
    /// This is same as [`Rate::validate`].
    fn validate(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        Self::Rate::validate(original_count, recovery_count, shard_bytes)
    }
}
