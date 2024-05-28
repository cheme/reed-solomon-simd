use crate::engine::{
    self,
    tables::{self, Mul128, Multiply128lutT, Skew},
    Engine, GfElement, ShardsRefMut, GF_MODULUS, GF_ORDER,
};
use std::arch::aarch64::*;
use std::iter::zip;

// ======================================================================
// Neon - PUBLIC

/// Optimized [`Engine`] using Arm Neon instructions.
///
/// [`Neon`] is an optimized engine that follows the same algorithm as
/// [`NoSimd`] but takes advantage of the Arm Neon SIMD instructions.
///
/// [`NoSimd`]: crate::engine::NoSimd
#[derive(Clone)]
pub struct Neon {
    mul128: &'static Mul128,
    skew: &'static Skew,
}

impl Neon {
    /// Creates new [`Neon`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        let mul128 = tables::initialize_mul128();
        let skew = tables::initialize_skew();

        Self { mul128, skew }
    }
}

impl Engine for Neon {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.fft_private_neon(data, pos, size, truncated_size, skew_delta);
        }
    }

    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.ifft_private_neon(data, pos, size, truncated_size, skew_delta);
        }
    }

    fn mul(&self, x: &mut [u8], log_m: GfElement) {
        unsafe {
            self.mul_neon(x, log_m);
        }
    }

    fn mul2(&self, x: [&mut [u8]; 2], log_m: [GfElement; 2]) {
        unimplemented!()
    }

    fn mul4(&self, x: [&mut [u8]; 4], log_m: [GfElement; 4]) {
        unimplemented!()
    }

    fn mul6(&self, x: [&mut [u8]; 6], log_m: [GfElement; 6]) {
        unimplemented!()
    }

    fn xor(x: &mut [u8], y: &[u8]) {
        let x: &mut [u32] = bytemuck::cast_slice_mut(x);
        let y: &[u32] = bytemuck::cast_slice(y);

        for (x64, y64) in zip(x.iter_mut(), y.iter()) {
            *x64 ^= y64;
        }
    }

    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        unsafe { Self::eval_poly_neon(erasures, truncated_size) }
    }
}

// ======================================================================
// Neon - IMPL Default

impl Default for Neon {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// Neon - PRIVATE
//
//

impl Neon {
    #[target_feature(enable = "neon")]
    unsafe fn mul_neon(&self, x: &mut [u8], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];

        for chunk in x.chunks_exact_mut(64) {
            let x_ptr: *mut u8 = chunk.as_mut_ptr();
            unsafe {
                let x0_lo = vld1q_u8(x_ptr);
                let x1_lo = vld1q_u8(x_ptr.add(16));
                let x0_hi = vld1q_u8(x_ptr.add(16 * 2));
                let x1_hi = vld1q_u8(x_ptr.add(16 * 3));

                let (prod0_lo, prod0_hi) = Self::mul_128(x0_lo, x0_hi, lut);
                let (prod1_lo, prod1_hi) = Self::mul_128(x1_lo, x1_hi, lut);

                vst1q_u8(x_ptr, prod0_lo);
                vst1q_u8(x_ptr.add(16), prod1_lo);
                vst1q_u8(x_ptr.add(16 * 2), prod0_hi);
                vst1q_u8(x_ptr.add(16 * 3), prod1_hi);
            }
        }
    }

    // Impelemntation of LEO_MUL_128
    #[inline(always)]
    fn mul_128(
        value_lo: uint8x16_t,
        value_hi: uint8x16_t,
        lut: &Multiply128lutT,
    ) -> (uint8x16_t, uint8x16_t) {
        let mut prod_lo: uint8x16_t;
        let mut prod_hi: uint8x16_t;

        unsafe {
            let t0_lo = vld1q_u8(&lut.lo[0] as *const u128 as *const u8);
            let t1_lo = vld1q_u8(&lut.lo[1] as *const u128 as *const u8);
            let t2_lo = vld1q_u8(&lut.lo[2] as *const u128 as *const u8);
            let t3_lo = vld1q_u8(&lut.lo[3] as *const u128 as *const u8);

            let t0_hi = vld1q_u8(&lut.hi[0] as *const u128 as *const u8);
            let t1_hi = vld1q_u8(&lut.hi[1] as *const u128 as *const u8);
            let t2_hi = vld1q_u8(&lut.hi[2] as *const u128 as *const u8);
            let t3_hi = vld1q_u8(&lut.hi[3] as *const u128 as *const u8);

            let clr_mask = vdupq_n_u8(0x0f);

            let data_0 = vandq_u8(value_lo, clr_mask);
            prod_lo = vqtbl1q_u8(t0_lo, data_0);
            prod_hi = vqtbl1q_u8(t0_hi, data_0);

            let data_1 = vshrq_n_u8(value_lo, 4);
            prod_lo = veorq_u8(prod_lo, vqtbl1q_u8(t1_lo, data_1));
            prod_hi = veorq_u8(prod_hi, vqtbl1q_u8(t1_hi, data_1));

            let data_0 = vandq_u8(value_hi, clr_mask);
            prod_lo = veorq_u8(prod_lo, vqtbl1q_u8(t2_lo, data_0));
            prod_hi = veorq_u8(prod_hi, vqtbl1q_u8(t2_hi, data_0));

            let data_1 = vshrq_n_u8(value_hi, 4);
            prod_lo = veorq_u8(prod_lo, vqtbl1q_u8(t3_lo, data_1));
            prod_hi = veorq_u8(prod_hi, vqtbl1q_u8(t3_hi, data_1));
        }

        (prod_lo, prod_hi)
    }

    //// {x_lo, x_hi} ^= {y_lo, y_hi} * log_m
    // Implementation of LEO_MULADD_128
    #[inline(always)]
    fn muladd_128(
        mut x_lo: uint8x16_t,
        mut x_hi: uint8x16_t,
        y_lo: uint8x16_t,
        y_hi: uint8x16_t,
        lut: &Multiply128lutT,
    ) -> (uint8x16_t, uint8x16_t) {
        let (prod_lo, prod_hi) = Self::mul_128(y_lo, y_hi, lut);
        unsafe {
            x_lo = veorq_u8(x_lo, prod_lo);
            x_hi = veorq_u8(x_hi, prod_hi);
        }
        (x_lo, x_hi)
    }
}

// ======================================================================
// Neon - PRIVATE - FFT (fast Fourier transform)

impl Neon {
    // Implementation of LEO_FFTB_128
    #[inline(always)]
    fn fftb_128(&self, x: &mut [u8; 64], y: &mut [u8; 64], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];
        let x_ptr: *mut u8 = x.as_mut_ptr();
        let y_ptr: *mut u8 = y.as_mut_ptr();
        unsafe {
            let mut x0_lo = vld1q_u8(x_ptr);
            let mut x1_lo = vld1q_u8(x_ptr.add(16));
            let mut x0_hi = vld1q_u8(x_ptr.add(16 * 2));
            let mut x1_hi = vld1q_u8(x_ptr.add(16 * 3));

            let mut y0_lo = vld1q_u8(y_ptr);
            let mut y1_lo = vld1q_u8(y_ptr.add(16));
            let mut y0_hi = vld1q_u8(y_ptr.add(16 * 2));
            let mut y1_hi = vld1q_u8(y_ptr.add(16 * 3));

            (x0_lo, x0_hi) = Self::muladd_128(x0_lo, x0_hi, y0_lo, y0_hi, lut);
            (x1_lo, x1_hi) = Self::muladd_128(x1_lo, x1_hi, y1_lo, y1_hi, lut);

            vst1q_u8(x_ptr, x0_lo);
            vst1q_u8(x_ptr.add(16), x1_lo);
            vst1q_u8(x_ptr.add(16 * 2), x0_hi);
            vst1q_u8(x_ptr.add(16 * 3), x1_hi);

            y0_lo = veorq_u8(y0_lo, x0_lo);
            y1_lo = veorq_u8(y1_lo, x1_lo);
            y0_hi = veorq_u8(y0_hi, x0_hi);
            y1_hi = veorq_u8(y1_hi, x1_hi);

            vst1q_u8(y_ptr, y0_lo);
            vst1q_u8(y_ptr.add(16), y1_lo);
            vst1q_u8(y_ptr.add(16 * 2), y0_hi);
            vst1q_u8(y_ptr.add(16 * 3), y1_hi);
        }
    }

    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn fft_butterfly_partial(&self, x: &mut [u8], y: &mut [u8], log_m: GfElement) {
        // While we wait for array_chunks/slice_as_chunks (#74985) to become stable,
        // we have to try_into().unwrap() (which cannot fail in this case)
        for (x_chunk, y_chunk) in zip(x.chunks_exact_mut(64), y.chunks_exact_mut(64)) {
            self.fftb_128(
                x_chunk.try_into().unwrap(),
                y_chunk.try_into().unwrap(),
                log_m,
            );
        }
    }

    #[inline(always)]
    fn fft_butterfly_two_layers(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m02 == GF_MODULUS {
            Self::xor(s2, s0);
            Self::xor(s3, s1);
        } else {
            self.fft_butterfly_partial(s0, s2, log_m02);
            self.fft_butterfly_partial(s1, s3, log_m02);
        }

        // SECOND LAYER

        if log_m01 == GF_MODULUS {
            Self::xor(s1, s0);
        } else {
            self.fft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            Self::xor(s3, s2);
        } else {
            self.fft_butterfly_partial(s2, s3, log_m23);
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn fft_private_neon(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // Drop unsafe privileges
        self.fft_private(data, pos, size, truncated_size, skew_delta);
    }

    #[inline(always)]
    fn fft_private(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist4 = size;
        let mut dist = size >> 2;
        while dist != 0 {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = self.skew[base];
                let log_m02 = self.skew[base + dist];
                let log_m23 = self.skew[base + dist * 2];

                for i in r..r + dist {
                    self.fft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02)
                }

                r += dist4;
            }
            dist4 = dist;
            dist >>= 2;
        }

        // FINAL ODD LAYER

        if dist4 == 2 {
            let mut r = 0;
            while r < truncated_size {
                let log_m = self.skew[r + skew_delta];

                let (x, y) = data.dist2_mut(pos + r, 1);

                if log_m == GF_MODULUS {
                    Self::xor(y, x);
                } else {
                    self.fft_butterfly_partial(x, y, log_m)
                }

                r += 2;
            }
        }
    }
}

// ======================================================================
// Neon - PRIVATE - IFFT (inverse fast Fourier transform)

impl Neon {
    // Implementation of LEO_IFFTB_128
    #[inline(always)]
    fn ifftb_128(&self, x: &mut [u8; 64], y: &mut [u8; 64], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];
        let x_ptr: *mut u8 = x.as_mut_ptr();
        let y_ptr: *mut u8 = y.as_mut_ptr();

        unsafe {
            let mut x0_lo = vld1q_u8(x_ptr);
            let mut x1_lo = vld1q_u8(x_ptr.add(16));
            let mut x0_hi = vld1q_u8(x_ptr.add(16 * 2));
            let mut x1_hi = vld1q_u8(x_ptr.add(16 * 3));

            let mut y0_lo = vld1q_u8(y_ptr);
            let mut y1_lo = vld1q_u8(y_ptr.add(16));
            let mut y0_hi = vld1q_u8(y_ptr.add(16 * 2));
            let mut y1_hi = vld1q_u8(y_ptr.add(16 * 3));

            y0_lo = veorq_u8(y0_lo, x0_lo);
            y1_lo = veorq_u8(y1_lo, x1_lo);
            y0_hi = veorq_u8(y0_hi, x0_hi);
            y1_hi = veorq_u8(y1_hi, x1_hi);

            vst1q_u8(y_ptr, y0_lo);
            vst1q_u8(y_ptr.add(16), y1_lo);
            vst1q_u8(y_ptr.add(16 * 2), y0_hi);
            vst1q_u8(y_ptr.add(16 * 3), y1_hi);

            (x0_lo, x0_hi) = Self::muladd_128(x0_lo, x0_hi, y0_lo, y0_hi, lut);
            (x1_lo, x1_hi) = Self::muladd_128(x1_lo, x1_hi, y1_lo, y1_hi, lut);

            vst1q_u8(x_ptr, x0_lo);
            vst1q_u8(x_ptr.add(16), x1_lo);
            vst1q_u8(x_ptr.add(16 * 2), x0_hi);
            vst1q_u8(x_ptr.add(16 * 3), x1_hi);
        }
    }

    #[inline(always)]
    fn ifft_butterfly_partial(&self, x: &mut [u8], y: &mut [u8], log_m: GfElement) {
        // While we wait for array_chunks/slice_as_chunks (#74985) to become stable,
        // we'll have to try_into() to array
        for (x_chunk, y_chunk) in zip(x.chunks_exact_mut(64), y.chunks_exact_mut(64)) {
            self.ifftb_128(
                x_chunk.try_into().unwrap(),
                y_chunk.try_into().unwrap(),
                log_m,
            );
        }
    }

    #[inline(always)]
    fn ifft_butterfly_two_layers(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m01 == GF_MODULUS {
            Self::xor(s1, s0);
        } else {
            self.ifft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            Self::xor(s3, s2);
        } else {
            self.ifft_butterfly_partial(s2, s3, log_m23);
        }

        // SECOND LAYER

        if log_m02 == GF_MODULUS {
            Self::xor(s2, s0);
            Self::xor(s3, s1);
        } else {
            self.ifft_butterfly_partial(s0, s2, log_m02);
            self.ifft_butterfly_partial(s1, s3, log_m02);
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn ifft_private_neon(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // Drop unsafe privileges
        self.ifft_private(data, pos, size, truncated_size, skew_delta)
    }

    #[inline(always)]
    fn ifft_private(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist = 1;
        let mut dist4 = 4;
        while dist4 <= size {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = self.skew[base];
                let log_m02 = self.skew[base + dist];
                let log_m23 = self.skew[base + dist * 2];

                for i in r..r + dist {
                    self.ifft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02)
                }

                r += dist4;
            }
            dist = dist4;
            dist4 <<= 2;
        }

        // FINAL ODD LAYER

        if dist < size {
            let log_m = self.skew[dist + skew_delta - 1];
            if log_m == GF_MODULUS {
                Self::xor_within(data, pos + dist, pos, dist);
            } else {
                let (mut a, mut b) = data.split_at_mut(pos + dist);
                for i in 0..dist {
                    self.ifft_butterfly_partial(
                        &mut a[pos + i], // data[pos + i]
                        &mut b[i],       // data[pos + i + dist]
                        log_m,
                    );
                }
            }
        }
    }
}

// ======================================================================
// Neon - PRIVATE - Evaluate polynomial

impl Neon {
    #[target_feature(enable = "neon")]
    unsafe fn eval_poly_neon(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        engine::eval_poly(erasures, truncated_size)
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
