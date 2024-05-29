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
const NB_POINT_PER_SEGMENTS: usize = (SEGMENT_SIZE + SEGMENT_SIZE_PADDING) / N_SHARDS / POINT_SIZE; // 6
const POINT_SIZE: usize = 2;
const N_SHARDS: usize = 342;
const N_POINT_BATCH: usize = 32;
const SHARD_BATCH_SIZE: usize = N_POINT_BATCH * POINT_SIZE * N_SHARDS;


fn build_original(original_data_segments: usize, rng: &mut SmallRng) -> (Vec<u8>, Vec<[u8; SHARD_BYTES]>, Vec<Vec<(u8, u8)>>, Vec<((usize, usize), (usize, usize))>) {
	let mut original = vec![0; original_data_segments * SEGMENT_SIZE];
	rng.fill::<[u8]>(&mut original);

	// every 2 byte chunks get in a point of SHARDBYTES. (losing a few byte, can be optimize later it is
	// just 8 byte per chunks (342 * 2 * 6 = 4104)).
	
	// So for testing best perf have original data chunks multiple of 32.
	let number_shards_batch = (((original_data_segments * SEGMENT_SIZE) - 1) / SHARD_BATCH_SIZE) + 1;

	println!("{:?}", number_shards_batch);
	let mut original_shards = vec![[0u8; SHARD_BYTES]; number_shards_batch * N_SHARDS];

	let mut dist = Vec::new();
	let mut segment_bound = Vec::new();
	let mut d = Vec::new(); 
	let mut segment_batch = 0;
	let mut at = 0; // pos in segment batch
	let mut at2 = 0; // 32 point pos in 64b shards
	for i in 0..original_data_segments {
		// runing over segment is totally not necessary: could justrun over original pairs of bytes.
		let ch = &original[i * SEGMENT_SIZE..(i + 1) * SEGMENT_SIZE];
		// interleave over N_POINT_BATCH so a single segment can be recover from m fix offset of 64b
		// shards.

		let start_ch = (segment_batch, dist.len());
		// this goes over 3*points over 342 batch since
		// 32 is not a multiple it can be on 2 different segment_batch.
		for b in 0..(ch.len() / 2) + 4 {
				let s = segment_batch * N_SHARDS + at;
				let point = if b >= (ch.len() / 2) {
					// padding
					(0, 0)
				} else {
					(ch[b * 2], ch[b*2 + 1])
				};
				original_shards[s][at2] = point.0;
				original_shards[s][at2 + 32] = point.1;
				d.push(point);

				at += 1;
				if at == N_SHARDS {
					dist.push(d);
					d = Vec::new();
					at = 0;
					at2 += 1;
					if at2 == 32 {
						at2 = 0;
						segment_batch += 1;
					}
				}
		}
		let end_ch = (segment_batch, dist.len());
		assert_eq!(at, 0);
//		println!("{:?}", (start_ch, end_ch));
		segment_bound.push((start_ch, end_ch));
	}

	// distributed chunks (and hash in segment) will
	// be cat(dist(i)(v).. 6time) with v the validator ix
	// and i start chunk dist to end chunk dist.
	(original, original_shards, dist, segment_bound)
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
  let mut rng = SmallRng::from_seed([0; 32]);
	let (original, o_shards, o_dist, o_dist_bound) = build_original(data_chunks, &mut rng);
	println!("o data size: {:?}", original.len());
	println!("o sharded size: {:?}", o_shards.len() * SHARD_BYTES);
	let count = o_shards.len();
	let r_shards = reed_solomon_simd::encode(count, 2 * count, &o_shards).unwrap();
	assert_eq!(r_shards.len(), o_shards.len() * 2);
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
