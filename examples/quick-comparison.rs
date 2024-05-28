use std::{thread::yield_now, time::Instant};

use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use reed_solomon_erasure::galois_16::ReedSolomon as ReedSolomon16;
use reed_solomon_erasure::galois_8::ReedSolomon as ReedSolomon8;
use reed_solomon_novelpoly::{CodeParams, WrappedShard};

// ======================================================================
// CONST

const SHARD_BYTES: usize = 64;

// ======================================================================
// MAIN

fn main() {
    #[cfg(debug_assertions)]
    {
        eprintln!("Warning: Running in debug mode! Please run like this instead: cargo run --release --example quick-comparison");
    }

    println!("                           µs (init)   µs (encode)   µs (decode)");
    println!("                           ---------   -----------   -----------");

    for count in [3, 32, 64, 128, 256, 512, 1024, 4 * 1024, 32 * 1024] {
        println!("\n{}:{} ({} kiB)", count, count, SHARD_BYTES / 1024);
        test_reed_solomon_simd(count);
        test_reed_solomon_16(count);
        test_reed_solomon_novelpoly(count);
        if count <= 128 {
            test_reed_solomon_erasure_8(count);
        }
        if count <= 512 {
            test_reed_solomon_erasure_16(count);
        }
    }
}

// ======================================================================
// reed-solomon-simd

fn test_reed_solomon_simd(count: usize) {
    // INIT

    let start = Instant::now();
    // This initializes all the needed tables.
    reed_solomon_simd::engine::DefaultEngine::new();
    let elapsed = start.elapsed();
    print!("> reed-solomon-simd        {:9}", elapsed.as_micros());

    // CREATE ORIGINAL
    const POINT_SIZE: usize = 2;
    const N_SHARD: usize = 342;
    const N_POINT_BATCH: usize = 32;
    const ORIGINAL_POINTS: usize = N_SHARD * N_POINT_BATCH;
    const INPUT_DATA_LEN: usize = POINT_SIZE * N_SHARD * N_POINT_BATCH;
    const RECOVERY_POINTS: usize = ORIGINAL_POINTS * 2;
    let mut rng = SmallRng::from_seed([0; 32]);

    let mut original_data = vec![0u8; INPUT_DATA_LEN];
    rng.fill::<[u8]>(&mut original_data);

    let mut original = Vec::new();
    let mut ori_shard = vec![0; SHARD_BYTES];
    let mut o = 0;
    for i in 0..INPUT_DATA_LEN / 2 {
        ori_shard[o] = original_data[i * 2];
        ori_shard[o + 32] = original_data[i * 2 + 1];
        o += 1;
        if o == 32 {
            original.push(ori_shard);
            ori_shard = vec![0; SHARD_BYTES];
            o = 0;
        }
    }
    assert!(o == 0);
    assert!(original.len() == 342);

    println!("Original: {:?}", original);

    let count = original.len();

    // ENCODE

    let start = Instant::now();
    let recovery = reed_solomon_simd::encode(count, 2 * count, &original).unwrap();
    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    println!("{:?}", recovery);
		assert_eq!(recovery.len(), 2 * N_SHARD);
    let decoder_recovery: Vec<_> = recovery.iter().enumerate().collect();
    let mut recovery_points = Vec::with_capacity(RECOVERY_POINTS);
    for r in recovery.iter() {
        for i in 0..32 {
            recovery_points.push(r[i]);
            recovery_points.push(r[i + 32]);
        }
    }
    assert_eq!(recovery_points.len(), RECOVERY_POINTS * 2);
    println!("Recovery: {:?}", recovery_points);

    // DECODE
    let start = Instant::now();
    let mut reco_vectors = vec![vec![0; SHARD_BYTES]; N_SHARD];
    for s in 0..32 {
        // add shard to every recovery vector. // TODO done on rec only: TODO include original too.
        let mut rand: Vec<usize> = (0..N_SHARD * 2).collect();
        rand.shuffle(&mut rng);
        rand.truncate(N_SHARD);
        for j in 0..N_SHARD {
            let mut r = vec![0; SHARD_BYTES];
//            reco_vectors[j][s] = recovery_points[rand[j] * 64 + s];
//            reco_vectors[j][s + 32] = recovery_points[rand[j] * 64 + s + 1];
            reco_vectors[j][s] = recovery[rand[j]][s];
            reco_vectors[j][s + 32] = recovery[rand[j]][s + 32];
        }
    }
    let recovery_points_shards = recovery_points
        .chunks(SHARD_BYTES)
        .map(|c| c.to_vec())
        .collect::<Vec<_>>();
    // use a point per 64 byte shard indices from
    let restored = reed_solomon_simd::decode(count, 2 * count, [(0, ""); 0], reco_vectors.iter().enumerate()).unwrap();
    for i in 0..count {
        assert_eq!(restored[&i], original[i]);
    }
    panic!("Done");
}

// ======================================================================
// reed-s)olomon-16

fn test_reed_solomon_16(count: usize) {
    // INIT

    let start = Instant::now();
    // This initializes all the needed tables.
    reed_solomon_16::engine::DefaultEngine::new();
    let elapsed = start.elapsed();
    print!("> reed-solomon-16          {:9}", elapsed.as_micros());

    // CREATE ORIGINAL

    let mut original = vec![vec![0u8; SHARD_BYTES]; count];
    let mut rng = SmallRng::from_seed([0; 32]);
    for original in &mut original {
        rng.fill::<[u8]>(original);
    }

    // ENCODE

    let start = Instant::now();
    let recovery = reed_solomon_16::encode(count, count, &original).unwrap();
    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    let decoder_recovery: Vec<_> = recovery.iter().enumerate().collect();

    // DECODE

    let start = Instant::now();
    let restored = reed_solomon_16::decode(count, count, [(0, ""); 0], decoder_recovery).unwrap();
    let elapsed = start.elapsed();
    println!("{:14}", elapsed.as_micros());

    // CHECK

    for i in 0..count {
        assert_eq!(restored[&i], original[i]);
    }
}

// ======================================================================
// reed-solomon-erasure

fn test_reed_solomon_erasure_8(count: usize) {
    // INIT

    let start = Instant::now();
    let r = ReedSolomon8::new(count, count).unwrap();
    let elapsed = start.elapsed();
    print!("> reed-solomon-erasure/8   {:9}", elapsed.as_micros());

    // CREATE ORIGINAL

    let mut original = vec![vec![0u8; SHARD_BYTES]; count];
    let mut rng = SmallRng::from_seed([0; 32]);
    for shard in &mut original {
        rng.fill::<[u8]>(shard);
    }

    // ENCODE

    let mut recovery = vec![vec![0; SHARD_BYTES]; count];

    let start = Instant::now();
    r.encode_sep(&original, &mut recovery).unwrap();
    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    let mut decoder_shards = Vec::with_capacity(2 * count);
    for _ in 0..count {
        decoder_shards.push(None);
    }
    for i in 0..count {
        decoder_shards.push(Some(recovery[i].clone()));
    }

    // DECODE

    let start = Instant::now();
    r.reconstruct(&mut decoder_shards).unwrap();
    let elapsed = start.elapsed();
    println!("{:14}", elapsed.as_micros());

    // CHECK

    for i in 0..count {
        assert_eq!(decoder_shards[i].as_ref(), Some(&original[i]));
    }
}

fn test_reed_solomon_erasure_16(count: usize) {
    // INIT

    let start = Instant::now();
    let r = ReedSolomon16::new(count, count).unwrap();
    let elapsed = start.elapsed();
    print!("> reed-solomon-erasure/16  {:9}", elapsed.as_micros());

    // CREATE ORIGINAL

    let mut original = vec![vec![[0u8; 2]; SHARD_BYTES / 2]; count];
    let mut rng = SmallRng::from_seed([0; 32]);
    for shard in &mut original {
        for element in shard.iter_mut() {
            element[0] = rng.gen();
            element[1] = rng.gen();
        }
    }

    // ENCODE

    let mut recovery = vec![vec![[0; 2]; SHARD_BYTES / 2]; count];

    let start = Instant::now();
    r.encode_sep(&original, &mut recovery).unwrap();
    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    let mut decoder_shards = Vec::with_capacity(2 * count);
    for _ in 0..count {
        decoder_shards.push(None);
    }
    for i in 0..count {
        decoder_shards.push(Some(recovery[i].clone()));
    }

    // DECODE

    let start = Instant::now();
    r.reconstruct(&mut decoder_shards).unwrap();
    let elapsed = start.elapsed();
    println!("{:14}", elapsed.as_micros());

    // CHECK

    for i in 0..count {
        assert_eq!(decoder_shards[i].as_ref(), Some(&original[i]));
    }
}

// ======================================================================
// reed-solomon-novelpoly

fn test_reed_solomon_novelpoly(count: usize) {
    // INIT

    let start = Instant::now();
    let r = CodeParams::derive_parameters(2 * count, count)
        .unwrap()
        .make_encoder();
    let elapsed = start.elapsed();
    print!("> reed-solomon-novelpoly   {:9}", elapsed.as_micros());

    // CREATE ORIGINAL

    let mut original = vec![0u8; count * SHARD_BYTES];
    let mut rng = SmallRng::from_seed([0; 32]);
    rng.fill::<[u8]>(&mut original);

    // ENCODE

    let start = Instant::now();
    let encoded = r.encode::<WrappedShard>(&original).unwrap();
    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    let mut decoder_shards = Vec::with_capacity(2 * count);
    for _ in 0..count {
        decoder_shards.push(None);
    }
    for i in 0..count {
        decoder_shards.push(Some(encoded[count + i].clone()));
    }

    // DECODE

    let start = Instant::now();
    let reconstructed = r.reconstruct(decoder_shards).unwrap();
    let elapsed = start.elapsed();
    println!("{:14}", elapsed.as_micros());

    // CHECK

    assert_eq!(reconstructed, original);
}
