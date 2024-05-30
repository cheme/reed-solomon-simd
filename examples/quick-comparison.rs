use std::time::Instant;

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
    for count in [1, 2, 4, 5, 10, 16, 32, 50] {
        scenarii(count);
    }
    /*
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
        */
}

// ======================================================================
// reed-solomon-simd

fn test_reed_solomon_simd(count: usize) {
    const POINT_SIZE: usize = 2;
    const N_SHARD: usize = 342;
    const N_POINT_BATCH: usize = 32;
    const INPUT_DATA_LEN: usize = POINT_SIZE * N_SHARD * N_POINT_BATCH;

    let count = N_SHARD;

    // INIT

    let start = Instant::now();
    // This initializes all the needed tables.
    reed_solomon_simd::engine::DefaultEngine::new();
    let elapsed = start.elapsed();
    print!("> reed-solomon-simd        {:9}", elapsed.as_micros());

    // CREATE ORIGINAL

    let mut original = vec![vec![0u8; SHARD_BYTES]; count];
    let mut rng = SmallRng::from_seed([0; 32]);
    for original in &mut original {
        rng.fill::<[u8]>(original);
    }

    println!("Original: {:?}", original);

    // ENCODE

    let start = Instant::now();
    let recovery = reed_solomon_simd::encode(count, 2 * count, &original).unwrap();
    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    println!("{:?}", recovery);
    let decoder_recovery: Vec<_> = recovery.iter().enumerate().collect();
    println!("Recovery: {:?}", original);

    // DECODE
    let start = Instant::now();
    let restored =
        reed_solomon_simd::decode(count, 2 * count, [(0, ""); 0], decoder_recovery).unwrap();
    let elapsed = start.elapsed();
    println!("{:14}", elapsed.as_micros());

    //const SMALL_SHARD: usize = 2;
    //nb of point
    // CHECK
    //for i in 0..count {
    let decoder_recovery: Vec<_> = recovery.iter().enumerate().collect();

    println!("Recovery: {:?}", decoder_recovery);
    //    for i in 0..32 {
    let mut r: Vec<usize> = (0..(2 * count)).collect();
    r.shuffle(&mut rng);
    r.truncate(count);
    let rand_reco = r;
    //   }

    let restored = reed_solomon_simd::decode(
        count,
        2 * count,
        [(0, ""); 0],
        (0..count).map(|i| {
            let mut r = vec![0; SHARD_BYTES];
            // one shard at each pos from any of the distribute points.
            let ir = rand_reco[i];
            for j in 0..32 {
                //                let ir = rand_reco[j][i];
                r[j] = decoder_recovery[ir].1[j];
                r[j + 32] = decoder_recovery[ir].1[j + 32];
            }

            //					r=decoder_recovery[i].1.clone();
            (ir, r)
        }),
    )
    .unwrap();

    //println!("{}: {} == {}", i, restored[&i][j], original[i][j]);
    assert_eq!(restored.len(), count);
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

const SEGMENT_SIZE: usize = 4096;
const SEGMENT_SIZE_PADDING: usize = 8;
const SEGMENT_SIZE_PADDED: usize = SEGMENT_SIZE + SEGMENT_SIZE_PADDING;
const NB_POINT_PER_SEGMENTS: usize = (SEGMENT_SIZE + SEGMENT_SIZE_PADDING) / N_SHARDS / POINT_SIZE; // 6
const POINT_SIZE: usize = 2;
const N_SHARDS: usize = 342;
const N_POINT_BATCH: usize = 32;
const SHARD_BATCH_SIZE: usize = N_POINT_BATCH * POINT_SIZE * N_SHARDS;

fn build_original(
    original_data_segments: usize,
    rng: &mut SmallRng,
		padded: bool,
) -> (Vec<u8>, Vec<[u8; SHARD_BYTES]>) {
	let segment_size = if padded { SEGMENT_SIZE_PADDED } else { SEGMENT_SIZE };
    let mut original = vec![0; original_data_segments * segment_size];
    rng.fill::<[u8]>(&mut original);

    // every 2 byte chunks get in a point of SHARDBYTES. (losing a few byte, can be optimize later it is
    // just 8 byte per chunks (342 * 2 * 6 = 4104)).

    // So for testing best perf have original data chunks multiple of 32.
    let number_shards_batch =
        (((original_data_segments * segment_size) - 1) / SHARD_BATCH_SIZE) + 1;

    println!("x{:?}", number_shards_batch);
    let mut shards = vec![[0u8; SHARD_BYTES]; number_shards_batch * N_SHARDS];

    //	let mut shards = vec![[0u8; SHARD_BYTES]; N_SHARDS];
    let mut shard_i = 0;
    let mut shard_a = 0;
    let mut full_i = 0;
    for i_p in 0..original.len() / 2 {
        //		println!("{}: {} {} {} {} {}", i_p, number_shards_batch, original.len(), SHARD_BATCH_SIZE * number_shards_batch, shard_a, original.len()/2);
        let point = (original[i_p * 2], original[i_p * 2 + 1]);
        shards[shard_a][shard_i] = point.0;
        shards[shard_a][32 + shard_i] = point.1;
        shard_a += 1;
        if shard_a % N_SHARDS == 0 {
            shard_i += 1;
            if shard_i == N_POINT_BATCH {
                shard_i = 0;
                full_i += 1;
                //shards.extend((0..N_SHARDS).into_iter().map(|_| [0u8; SHARD_BYTES]));
            }
            shard_a = full_i * N_SHARDS;
        }
    }

    (original, shards)
}

fn data_to_dist(data: &[u8]) -> Vec<Vec<(u8, u8)>> {
    let mut res = vec![Vec::new(); N_SHARDS];
    for i in 0..data.len() / 2 {
        let point = (data[i * 2], data[i * 2 + 1]);
        res[i % N_SHARDS].push(point);
    }
    res
}

fn ori_chunk_to_data(
    chunks: &Vec<[u8; SHARD_BYTES]>,
    start_data: usize,
    end_data: usize,
) -> Vec<u8> {
    let mut data = Vec::new();
    assert!(chunks.len() % N_SHARDS == 0);
    let mut n_full = chunks.len() / N_SHARDS;
    println!("xd{:?}", n_full);
    let (mut full_i, mut shard_i, mut shard_a) = data_index_to_chunk_index(start_data);
    println!("st{:?}", (full_i, shard_i, shard_a));
		let mut full_i_offset =  full_i * N_SHARDS;
    loop {
        let l = chunks[full_i_offset + shard_a][shard_i];
        let r = chunks[full_i_offset + shard_a][32 + shard_i];
        data.push(l);
        data.push(r);
        shard_a += 1;
        if shard_a % N_SHARDS == 0 {
            shard_i += 1;
            if shard_i == N_POINT_BATCH {
                shard_i = 0;
                full_i += 1;
								full_i_offset = full_i * N_SHARDS;
                if full_i == n_full {
                    break;
                }
            }
            shard_a = 0;
        }
    }
    data
}

// return chunk index among N_SHARDS (group of n , ix in slice, ix in n)
fn data_index_to_chunk_index(index: usize) -> (usize, usize, usize) {
	println!("x{:?}", index);
    let a = index % SHARD_BATCH_SIZE;
    let b = a % (N_SHARDS * POINT_SIZE);
	println!("x{:?}", b/N_SHARDS);
    (
        index / SHARD_BATCH_SIZE,
        a / (N_SHARDS * POINT_SIZE),
        b / POINT_SIZE,
    )
}

fn chunks_to_dist(chunks: &Vec<[u8; SHARD_BYTES]>) -> Vec<Vec<(u8, u8)>> {
    // we want
    assert!(chunks.len() % N_SHARDS == 0);
    let mut dists = vec![Vec::new(); N_SHARDS];
    let nb_val_cycle = chunks.len() / N_SHARDS;
    for c in 0..nb_val_cycle {
        for v in 0..N_SHARDS {
            let shard = &chunks[c * N_SHARDS + v];
            for b in 0..SHARD_BYTES / 2 {
                let point = (shard[b], shard[b + 32]);
                dists[v].push(point);
            }
        }
    }
    dists
}
fn chunks_to_dist2(chunks: &[Vec<u8>]) -> Vec<Vec<(u8, u8)>> {
    assert!(chunks.len() % N_SHARDS == 0);
    let mut dists = vec![Vec::new(); N_SHARDS];
    let nb_val_cycle = chunks.len() / N_SHARDS;
    for c in 0..nb_val_cycle {
        for v in 0..N_SHARDS {
            let shard = &chunks[c * N_SHARDS + v];
            for b in 0..SHARD_BYTES / 2 {
                let point = (shard[b], shard[b + 32]);
                dists[v].push(point);
            }
        }
    }
    dists
}

/*
fn reco_to_dist(reco: &[Vec<u8>]) -> Vec<Vec<(u8, u8)>> {
    // rec is n time 2*342 64byte shard
    assert_eq!(reco.len() % N_SHARDS, 0);

    let nb_cycle = reco.len() / N_SHARDS;

    for c in 0..nb_cycle() {
    }
    // same dist as ori warn need to take point.
    let mut dist1 = Vec::new(); // TODO capacity same as ori

    // simply point by point distributed amongst val

    let mut d1 = Vec::with_capacity(N_SHARDS);
    let mut i_val = 0;
    for r in 0..reco.len() {
        let r1 = &reco[r];
        for i in 0..32 {
            let point1 = (r1[i], r1[32 + i]);
            d1.push(point1);
            i_val += 1;
            if i_val == N_SHARDS {
                dist1.push(d1);
                d1 = Vec::with_capacity(N_SHARDS);
                i_val = 0;
            }
        }
    }
    (dist1, dist2)
}
*/

fn scenarii(data_chunks: usize) {
	/*
	let mut s = 0;
	for _ in 0..100 {
		s += 4096; // 7 point and in theory rarely 6
		//s += 4104; exact 6 point
		let i = data_index_to_chunk_index(s);

		println!("ax{:?}", i);
	}
	panic!("done");
	*/
		let padded_segments = false;
    let mut rng = SmallRng::from_seed([0; 32]);
    let (original, o_shards) = build_original(data_chunks, &mut rng, padded_segments);
    let original2 = ori_chunk_to_data(&o_shards, 0, 0);
    assert_eq!(original[0..original.len()], original2[0..original.len()]);
		let a = original.len() * 3 / 4;
		println!("a{:?} - {:?}", a, original.len());
    let original3 = ori_chunk_to_data(&o_shards, a, 0);
    assert_eq!(
        original[a..],
        original3[0..original.len() * 1 / 4]
    );
    let o_dist = chunks_to_dist(&o_shards);
    let count = o_shards.len();
    let r_shards = reed_solomon_simd::encode(count, 2 * count, &o_shards).unwrap();
    let r_dist1 = chunks_to_dist2(&r_shards[..r_shards.len() / 2]);
    let r_dist2 = chunks_to_dist2(&r_shards[r_shards.len() / 2..]);
    println!("o data size: {:?}", original.len());
    println!("o sharded size: {:?}", o_shards.len() * SHARD_BYTES);
    /*
    let count = o_shards.len();
    let r_shards = reed_solomon_simd::encode(count, 2 * count, &o_shards).unwrap();
    assert_eq!(r_shards.len(), o_shards.len() * 2);
    */
    /*
        let r_dist1 = reco_to_dist(&r_shards[..r_shards.len()/2]);
        let r_dist2 = reco_to_dist(&r_shards[r_shards.len()/2..]);
        assert_eq!(r_dist1.len(), o_dist.len());
        assert_eq!(r_dist2.len(), o_dist.len());
    */
}

struct RecoConf {
    nb_chunks: usize,
    nb_percent_val_off: usize, // TODO random , then we want to get a fewer number of call so get as
    // many as possible sub shard from chunk to rebuild on all 341. ->
    // just count.
    favor_reco: bool, // otherwhise we go first on the originals.
}
