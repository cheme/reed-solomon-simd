use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use reed_solomon_simd::{
    engine::{DefaultEngine, Engine, ShardsRefMut},
    rate::{
        HighRateDecoder, HighRateEncoder, LowRateDecoder, LowRateEncoder, RateDecoder, RateEncoder,
    },
    ReedSolomonDecoder, ReedSolomonEncoder,
};


#[cfg(target_arch = "aarch64")]
use reed_solomon_simd::engine::Neon;

// ======================================================================
// CONST

const SHARD_BYTES: usize = 64;

// ======================================================================
// UTIL

fn generate_shards(shard_count: usize, shard_bytes: usize, seed: u8) -> Vec<Vec<u8>> {
    let mut rng = ChaCha8Rng::from_seed([seed; 32]);
    let mut shards = vec![vec![0u8; shard_bytes]; shard_count];
    for shard in &mut shards {
        rng.fill::<[u8]>(shard);
    }
    shards
}

// ======================================================================
// BENCHMARKS - MAIN

fn benchmarks_main(c: &mut Criterion) {
    let mut group = c.benchmark_group("main");

    for (original_count, recovery_count) in [
        (16, 32),
			/*
        // 2^n. original_count == recovery_count
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (32768, 32768),
        // And some other combinations
        (128, 1024),
        (1000, 100),
        (1000, 10000),
        (1024, 128),
        (1024, 8192),
        (8192, 1024),
        (8192, 16384),
        (8192, 57344),
        (10000, 1000),
        (16384, 8192),
        (16385, 16385), // 2^n + 1
        (57344, 8192),
			*/
    ] {
        if original_count >= 1000 && recovery_count >= 1000 {
            group.sample_size(10);
        } else {
            group.sample_size(100);
        }

        let original = generate_shards(original_count, SHARD_BYTES, 0);
        let recovery =
            reed_solomon_simd::encode(original_count, recovery_count, &original).unwrap();


				let mut decoder =
                ReedSolomonDecoder::new(original_count, recovery_count, SHARD_BYTES).unwrap();
                        for index in 0..16 {
													println!(":{:?}", original[index]);
                        }
                        for index in 16..32 {
                            decoder.add_recovery_shard(index, &recovery[index]).unwrap();
                        }
                        let r = decoder.decode().unwrap();
												for i in 0..16 {
													println!("{:?}", r.restored_original(i));
												}
												panic!("i");


        group.throughput(Throughput::Bytes(
            ((original_count + recovery_count) * SHARD_BYTES) as u64,
        ));

        // ReedSolomonEncoder

        let mut encoder =
            ReedSolomonEncoder::new(original_count, recovery_count, SHARD_BYTES).unwrap();

        let id = format!("{}:{}", original_count, recovery_count);

        group.bench_with_input(
            BenchmarkId::new("ReedSolomonEncoder", &id),
            &original,
            |b, original| {
                b.iter(|| {
                    for original in original {
                        encoder.add_original_shard(original).unwrap();
                    }
                    encoder.encode().unwrap();
                });
            },
        );

        // ReedSolomonDecoder

        let max_original_loss_count = std::cmp::min(original_count, recovery_count);

        for loss_percent in [1, 100] {
            // We round up to make sure at least one shard is lost for low shard counts.
            // '+ 99' as div_ceil() is not in stable yet (int_roundings #88581).
            let original_loss_count = (max_original_loss_count * loss_percent + 99) / 100;
            let original_provided_count = original_count - original_loss_count;
            let recovery_provided_count = original_loss_count;

            let mut decoder =
                ReedSolomonDecoder::new(original_count, recovery_count, SHARD_BYTES).unwrap();

            let id = format!("{}:{} ({}%)", original_count, recovery_count, loss_percent);


                        group.bench_with_input(
                BenchmarkId::new("ReedSolomonDecoder", &id),
                &recovery,
                |b, recovery| {
                    b.iter(|| {
                        for index in 0..original_provided_count {
                            decoder.add_original_shard(index, &original[index]).unwrap();
                        }
                        for index in 0..recovery_provided_count {
                            decoder.add_recovery_shard(index, &recovery[index]).unwrap();
                        }
                        decoder.decode().unwrap();
                    });
                },
            );
        }
    }

    group.finish();
}

// ======================================================================
// BENCHMARKS - RATE

fn benchmarks_rate(c: &mut Criterion) {
    // benchmarks_rate_one(c, "rate-Naive", Naive::new);
    benchmarks_rate_one(c, "rate", DefaultEngine::new);
}

fn benchmarks_rate_one<E: Engine>(c: &mut Criterion, name: &str, new_engine: fn() -> E) {
    let mut group = c.benchmark_group(name);
    group.sample_size(10);

    for (original_count, recovery_count) in [
        (1024, 1024),
        (1024, 1025),
        (1025, 1024),
        (1024, 2048),
        (2048, 1024),
        (1025, 1025),
        (1025, 2048),
        (2048, 1025),
        (2048, 2048),
    ] {
        let original = generate_shards(original_count, SHARD_BYTES, 0);
        let recovery =
            reed_solomon_simd::encode(original_count, recovery_count, &original).unwrap();

        group.throughput(Throughput::Bytes(
            ((original_count + recovery_count) * SHARD_BYTES) as u64,
        ));

        // ENCODE

        let id = format!("{}:{}", original_count, recovery_count);

        // HighRateEncoder

        let mut encoder = HighRateEncoder::new(
            original_count,
            recovery_count,
            SHARD_BYTES,
            new_engine(),
            None,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("HighRateEncoder", &id),
            &original,
            |b, original| {
                b.iter(|| {
                    for original in original {
                        encoder.add_original_shard(original).unwrap();
                    }
                    encoder.encode().unwrap();
                });
            },
        );

        // LowRateEncoder

        let mut encoder = LowRateEncoder::new(
            original_count,
            recovery_count,
            SHARD_BYTES,
            new_engine(),
            None,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("LowRateEncoder", &id),
            &original,
            |b, original| {
                b.iter(|| {
                    for original in original {
                        encoder.add_original_shard(original).unwrap();
                    }
                    encoder.encode().unwrap();
                });
            },
        );

        // DECODE

        let original_loss_count = std::cmp::min(original_count, recovery_count);
        let original_provided_count = original_count - original_loss_count;
        let recovery_provided_count = original_loss_count;

        // HighRateDecoder

        let mut decoder = HighRateDecoder::new(
            original_count,
            recovery_count,
            SHARD_BYTES,
            new_engine(),
            None,
        )
        .unwrap();

        let id = format!("{}:{}", original_count, recovery_count);

        group.bench_with_input(
            BenchmarkId::new("HighRateDecoder", &id),
            &recovery,
            |b, recovery| {
                b.iter(|| {
                    for index in 0..original_provided_count {
                        decoder.add_original_shard(index, &original[index]).unwrap();
                    }
                    for index in 0..recovery_provided_count {
                        decoder.add_recovery_shard(index, &recovery[index]).unwrap();
                    }
                    decoder.decode().unwrap();
                });
            },
        );

        // LowRateDecoder

        let mut decoder = LowRateDecoder::new(
            original_count,
            recovery_count,
            SHARD_BYTES,
            new_engine(),
            None,
        )
        .unwrap();

        let id = format!("{}:{}", original_count, recovery_count);

        group.bench_with_input(
            BenchmarkId::new("LowRateDecoder", &id),
            &recovery,
            |b, recovery| {
                b.iter(|| {
                    for index in 0..original_provided_count {
                        decoder.add_original_shard(index, &original[index]).unwrap();
                    }
                    for index in 0..recovery_provided_count {
                        decoder.add_recovery_shard(index, &recovery[index]).unwrap();
                    }
                    decoder.decode().unwrap();
                });
            },
        );
    }

    group.finish();
}

// ======================================================================
// MAIN

criterion_group!(benches_main, benchmarks_main);
criterion_group!(benches_rate, benchmarks_rate);
criterion_main!(benches_main, benches_rate);
