use std::{mem::MaybeUninit, time::Instant};

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
// we run over 3 time 3 * 342 points.
// So minimal unaligned size is 342 * 64 = 21888.
// So minimal aligned size is 342 * 64 * 3 = 65664.
// We want the aligned size to have chunk distribution to actual
// indexes.
// as restoring a single segment of 4k means 6 points amongst one redundant.
// or amongst 2 redundant. Here max is 3 so we chunk al processes over thes
// 3 maximum cycle (bigger content do not make sense as they will be similarilly aligned).
// TODO when restore content on smaller we can use CHUNK_SIZE of 64 or 64 * 2.
const CHUNK_SIZE: usize = 64 * 3;
// it is 16 segments.
const CHUNK_SIZE_DATA: usize = 64 * 3 * N_SHARDS;

#[derive(Debug, PartialEq, Eq)]
struct ChunkedData {
    // Vec<u8> is n time 64 byte shard (must be aligned).
    shards: [Vec<u8>; N_SHARDS],
    // TODO store last indexes of relevant data to be able to limit
    // dist size.
}

impl ChunkedData {
    fn from_decode(data: std::collections::BTreeMap<usize, Vec<u8>>) -> Self {
        assert!(data.len() == N_SHARDS);
        assert!(*data.keys().last().unwrap() == N_SHARDS - 1);
        let mut shards = Self::init_sized(0);
        for (k, v) in data {
            shards.shards[k] = v;
        }
        shards
    }
    fn init(original_size: usize) -> Self {
        // So for testing best perf have original data chunks multiple of 32.
        debug_assert!(original_size % N_SHARDS == 0); // size from padded segments.
        let size_per_dist = original_size / N_SHARDS;
        let size_per_dist = ((size_per_dist - 1) / 64) + 1;
        println!("spd{:?}", (size_per_dist, size_per_dist * 64));
        Self::init_sized(size_per_dist)
    }

    fn init_sized(size_per_dist: usize) -> Self {
        let mut shards: [MaybeUninit<Vec<u8>>; N_SHARDS] =
            unsafe { MaybeUninit::uninit().assume_init() };

        let mut tot_size = 0;
        for i in 0..N_SHARDS {
            shards[i].write(vec![0; size_per_dist * 64]);
            tot_size += size_per_dist * 64;
        }
        unsafe {
            ChunkedData {
                shards: std::mem::transmute(shards),
            }
        }
    }

    fn n_full(&self) -> usize {
        self.shards[0].len() / (N_POINT_BATCH * POINT_SIZE)
    }

    fn to_dist(&self) -> DistData {
        let nb = ((self.shards[0].len() - 1) / 12) + 1;
        let mut dist: [MaybeUninit<Vec<[u8; 12]>>; N_SHARDS] =
            unsafe { MaybeUninit::uninit().assume_init() };

        for i in 0..N_SHARDS {
            let mut d = vec![[0; 12]; nb];
            let mut ix = 0;
            let mut ix2 = 0;
            for chunk in self.shards[i].chunks_exact(64) {
                for a in 0..32 {
                    let p = (chunk[a], chunk[a + 32]);
                    d[ix][ix2] = p.0;
                    d[ix][ix2 + 1] = p.1;
                    ix2 += 2;
                    if ix2 == 12 {
                        ix2 = 0;
                        ix += 1;
                    }
                }
            }
            dist[i].write(d);
            //assert_eq!(ix, nb); not true, some padding
        }
        unsafe {
            DistData {
                shards: std::mem::transmute(dist),
            }
        }
    }
}

struct DistData {
    // earch Vec<[u8; 6]> is a segment index.
    // so we indeed have each segment amongs N_SHARDS.
    shards: [Vec<[u8; 12]>; N_SHARDS],
}

impl DistData {
    fn new(shard_len: usize) -> Self {
        let nb = ((shard_len - 1) / 12) + 1;
        let mut dist: [MaybeUninit<Vec<[u8; 12]>>; N_SHARDS] =
            unsafe { MaybeUninit::uninit().assume_init() };

        for i in 0..N_SHARDS {
            dist[i].write(vec![[0; 12]; nb]);
        }
        unsafe {
            DistData {
                shards: std::mem::transmute(dist),
            }
        }
    }

    fn to_chunked(&self) -> ChunkedData {
        // do not round up (we already pad dist data so chunked data is lower)
        let chunk_len = self.shards[0].len() * 12 / 64;

        let mut shards = ChunkedData::init_sized(chunk_len).shards;
        for dest in 0..N_SHARDS {
            //			println!("{:?}", self.shards[dest]);
            let mut ch_ix = 0;
            let mut ix = 0;
            for (i, c) in self.shards[dest].iter().enumerate() {
                for b in 0..6 {
                    if ix == 32 {
                        ix = 0;
                        ch_ix += 1;
                    }
                    let b = b * 2;
                    //			println!("{:?}", (ch_ix * 64 + ix, b, c[b]));
                    shards[dest].get_mut(ch_ix * 64 + ix).map(|s| *s = c[b]);
                    shards[dest]
                        .get_mut(ch_ix * 64 + ix + 32)
                        .map(|s| *s = c[b + 1]);
                    ix += 1;
                }
            }
            //			panic!("d{:?}", shards[dest]);
        }
        ChunkedData { shards }
    }
}
fn build_original(
    original_data_segments: usize,
    rng: &mut SmallRng,
    padded: bool,
) -> (Vec<u8>, ChunkedData) {
    let segment_size = if padded {
        SEGMENT_SIZE_PADDED
    } else {
        SEGMENT_SIZE
    };
    let mut original = vec![0; original_data_segments * segment_size];
    rng.fill::<[u8]>(&mut original);

    // every 2 byte chunks get in a point of SHARDBYTES. (losing a few byte, can be optimize later it is
    // just 8 byte per chunks (342 * 2 * 6 = 4104)).

    let mut shards = ChunkedData::init(original.len()).shards;
    //	let mut shards = vec![[0u8; SHARD_BYTES]; N_SHARDS];
    let mut shard_i = 0;
    let mut shard_i_offset = 0;
    let mut shard_a = 0;
    let mut full_i = 0;
    for i_p in 0..original.len() / 2 {
        //		println!("{}: {} {} {} {} {}", i_p, number_shards_batch, original.len(), SHARD_BATCH_SIZE * number_shards_batch, shard_a, original.len()/2);
        let point = (original[i_p * 2], original[i_p * 2 + 1]);
        //				println!("{:?}", (i_p, original.len() / 2));
        shards[shard_a][shard_i_offset + shard_i] = point.0;
        shards[shard_a][shard_i_offset + 32 + shard_i] = point.1;
        shard_a += 1;
        if shard_a % N_SHARDS == 0 {
            shard_i += 1;
            if shard_i == N_POINT_BATCH {
                shard_i = 0;
                full_i += 1;
                shard_i_offset = full_i * (N_POINT_BATCH * POINT_SIZE);
            }
            shard_a = 0;
        }
    }

    (original, ChunkedData { shards })
}

fn build_rec(reco: Vec<Vec<u8>>) -> (ChunkedData, ChunkedData) {
    assert_eq!(reco.len(), N_SHARDS * 2);
    assert_eq!(reco[0].len() % 64, 0);

    // TODO can run on uninit here (using size 0 for now).
    let mut shards1 = ChunkedData::init_sized(0);
    let mut shards2 = ChunkedData::init_sized(0);
    for (i, r) in reco.into_iter().enumerate() {
        if i >= N_SHARDS {
            shards2.shards[i - N_SHARDS] = r;
        } else {
            shards1.shards[i] = r;
        }
    }
    (shards1, shards2)
}

fn data_to_dist(data: &[u8]) -> Vec<Vec<(u8, u8)>> {
    let mut res = vec![Vec::new(); N_SHARDS];
    for i in 0..data.len() / 2 {
        let point = (data[i * 2], data[i * 2 + 1]);
        res[i % N_SHARDS].push(point);
    }
    res
}

fn ori_chunk_to_data(chunks: &ChunkedData, start_data: usize, data_len: Option<usize>) -> Vec<u8> {
    let mut data = Vec::new(); // TODO capacity.
    let n_full = chunks.n_full();
    println!("spb{}", n_full * 64);

    let shards = &chunks.shards;
    let mut shard_i = 0;
    let mut shard_a = 0;
    let mut full_i = 0;
    let (mut full_i, mut shard_i, mut shard_a) = data_index_to_chunk_index(start_data);
    let mut shard_i_offset = full_i * (N_POINT_BATCH * POINT_SIZE);
    loop {
        let l = shards[shard_a][shard_i_offset + shard_i];
        data.push(l);
        let r = shards[shard_a][shard_i_offset + shard_i + 32];
        data.push(r);
        if data_len.map(|m| data.len() >= m).unwrap_or(false) {
            break;
        }
        shard_a += 1;
        if shard_a % N_SHARDS == 0 {
            shard_i += 1;
            if shard_i == N_POINT_BATCH {
                shard_i = 0;
                full_i += 1;
                if full_i == n_full {
                    break;
                }

                shard_i_offset = full_i * (N_POINT_BATCH * POINT_SIZE);
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
    println!("x{:?}", b / N_SHARDS);
    (
        index / SHARD_BATCH_SIZE,
        a / (N_SHARDS * POINT_SIZE),
        b / POINT_SIZE,
    )
}

// ret index of dist 12 bytes.
fn data_index_to_dist_index(index: usize) -> usize {
    let (full_i, shard_i, shard_a) = data_index_to_chunk_index(index);
    assert_eq!(shard_a, 0); // we run on aligned segment and this should only be use for segments
                            // for now. TODO swith to index_segment instead of index?.
    let i = full_i * 64 + shard_i * 2;
    i / 12
    //		let chunk_ix = i / 12;
    //		let chunk_ix2 = i % 12;
}

// take n * 342 chunks and distribute by 12 byte subshards.
// To get distribution over 342 dest, the result subshards can be
// split by its length / 342 as 342 chunks are processed sequentially.
// If using same indexes, then list of subshards can be split by
fn chunks_to_dist(chunks: &Vec<[u8; SHARD_BYTES]>) -> Vec<[u8; 12]> {
    // we want
    assert!(chunks.len() % N_SHARDS == 0);
    let mut res = Vec::new();
    let mut sub_shard = [0u8; 12];
    let mut sub_ix = 0;
    let nb_val_cycle = chunks.len() / N_SHARDS;
    for v in 0..N_SHARDS {
        for c in 0..nb_val_cycle {
            let chunk = &chunks[c * N_SHARDS + v];

            let mut chunk_ix = 0;
            for chunk_ix in 0..SHARD_BYTES / 2 {
                sub_shard[sub_ix] = chunk[chunk_ix];
                sub_shard[sub_ix + 1] = chunk[chunk_ix + 32];
                sub_ix += 2;
                if sub_ix == 12 {
                    res.push(sub_shard);
                    sub_ix = 0;
                }
            }
        }
        if sub_ix != 0 {
            for i in sub_ix..12 {
                // buff is not reset
                sub_shard[i] = 0;
            }
            res.push(sub_shard);
        }
    }
    assert_eq!(res.len() % N_SHARDS, 0);
    println!("dist {:?}", (chunks.len() * SHARD_BYTES, res.len() * 12));
    res
}

// ?? more clear when testing.
fn chunk_dist_at(dists: &[[u8; 12]], at: usize) -> &[u8; 12] {
    let step = dists.len() / N_SHARDS;
    let at_step = at / step;
    let at_step2 = at % step;
    &dists[at_step * 342 + at_step2]
}

fn chunks_to_dist2(chunks: &[Vec<u8>]) -> Vec<[u8; 12]> {
    // we want
    assert!(chunks.len() % N_SHARDS == 0);
    let mut res = Vec::new();
    let mut sub_shard = [0u8; 12];
    let mut sub_ix = 0;
    let nb_val_cycle = chunks.len() / N_SHARDS;
    for v in 0..N_SHARDS {
        for c in 0..nb_val_cycle {
            let chunk = &chunks[c * N_SHARDS + v];

            let mut chunk_ix = 0;
            for chunk_ix in 0..SHARD_BYTES / 2 {
                sub_shard[sub_ix] = chunk[chunk_ix];
                sub_shard[sub_ix + 1] = chunk[chunk_ix + 32];
                sub_ix += 2;
                if sub_ix == 12 {
                    res.push(sub_shard);
                    sub_ix = 0;
                }
            }
        }
        if sub_ix != 0 {
            for i in sub_ix..12 {
                // buff is not reset
                sub_shard[i] = 0;
            }
            res.push(sub_shard);
        }
    }
    assert_eq!(res.len() % N_SHARDS, 0);
    println!("dist {:?}", (chunks.len() * SHARD_BYTES, res.len() * 12));
    res
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
        //s += 4096; // 7 point and in theory rarely 6
        s += 4104; // exact 6 point
        let i = data_index_to_chunk_index(s);

        println!("ax{:?}", i);
    }
    panic!("done");
        */
    let padded_segments = true; // keep it this way for simple distribution.(6poinst of the 342.
                                // Even if some time on two 742 dist and some time even on 2
                                // batches.
                                // !! CHEME: this distribution will force
                                // validator to have same chunks (all 64 bytes).
                                // or make things rather awkward.
                                // Meaning all segment of a package being sent to same validator
                                // distribution (likely dist is fix on an epoch so fine?).
                                // generally this is a must have: otherwhise difficult combination.

    let mut rng = SmallRng::from_seed([0; 32]);
    let (original, o_shards) = build_original(data_chunks, &mut rng, padded_segments);
    let o_dist = o_shards.to_dist();
    let o_shards2 = o_dist.to_chunked();
    assert_eq!(o_shards.shards.len(), o_shards2.shards.len());
    //		println!("{:?}", o_dist.shards[0].len() * 12);
    //		println!("{:?}", o_dist.shards[0]);
    assert_eq!(o_shards.shards[0].len(), o_shards2.shards[0].len());
    assert_eq!(o_shards.shards[0], o_shards2.shards[0]);
    assert_eq!(o_shards, o_shards2);
    let original2 = ori_chunk_to_data(&o_shards, 0, None);
    assert_eq!(original[0..original.len()], original2[0..original.len()]);
    let a = original.len() * 3 / 4;
    println!("a{:?} - {:?}", a, original.len());
    let original3 = ori_chunk_to_data(&o_shards, a, None);
    let original4 = ori_chunk_to_data(&o_shards, a, Some(10));
    assert_eq!(original[a..], original3[0..original.len() * 1 / 4]);
    assert_eq!(original[a..a + 10], original4[0..10]);
    let r_shards =
        reed_solomon_simd::encode(N_SHARDS, 2 * N_SHARDS, o_shards.shards.iter()).unwrap();
    let (r_shards1, r_shards2) = build_rec(r_shards);
    let r_dist1 = r_shards1.to_dist();
    let r_dist2 = r_shards2.to_dist();
    assert_eq!(r_dist1.to_chunked(), r_shards1);
    assert_eq!(r_dist2.to_chunked(), r_shards2);

    // build single segment.
    let segment_ix = data_chunks / 2;
    let chunk_start = segment_ix * SEGMENT_SIZE_PADDED;
    let chunk_start_i = data_index_to_chunk_index(chunk_start);
    let chunk_end = (segment_ix + 1) * SEGMENT_SIZE_PADDED;
    let chunk_end_i = data_index_to_chunk_index(chunk_start);
    assert!(chunk_start_i.2 == 0);
    assert!(chunk_end_i.2 == 0);

    // TODO rebuild only from chunk index, or use orig size.
    let l = o_dist.shards[0].len();
    let mut o_dist_test = DistData::new(l * 12);

    let dist_st = data_index_to_dist_index(chunk_start);
    let dist_end = data_index_to_dist_index(chunk_end);
    assert_eq!(dist_end - dist_st, 1);
    for d in 0..N_SHARDS {
        for i in dist_st..dist_end {
            o_dist_test.shards[d][i] = o_dist.shards[d][i];
        }
    }
    let o_shards_test = o_dist_test.to_chunked();
    let mut ori_map: std::collections::BTreeMap<_, _> = o_shards_test
        .shards
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.clone()))
        .collect();
    let ori_ret = reed_solomon_simd::decode(
        N_SHARDS,
        2 * N_SHARDS,
        ori_map.iter().map(|(k, v)| (*k, v)),
        [(0, ""); 0],
    )
    .unwrap();
    assert_eq!(ori_ret.len(), 0);
    ori_ret.into_iter().for_each(|(i, s)| {
        ori_map.insert(i, s);
    });
    assert_eq!(ori_map.len(), N_SHARDS);
    // TODO avoid instantiating this shards.
    let shards = ChunkedData::from_decode(ori_map);
    let ori_test = ori_chunk_to_data(&shards, chunk_start, Some(SEGMENT_SIZE_PADDED));
    assert_eq!(original[chunk_start..chunk_end], ori_test);

    // rebuild from all ori

    /*
                panic!("d");

        let o_dist = chunks_to_dist(&o_shards);
        let count = o_shards.len();
        let r_shards = reed_solomon_simd::encode(count, 2 * count, &o_shards).unwrap();
        let r_dist1 = chunks_to_dist2(&r_shards[..r_shards.len() / 2]);
        let r_dist2 = chunks_to_dist2(&r_shards[r_shards.len() / 2..]);
            assert_eq!(o_dist.len(), r_dist1.len());
        println!("o data size: {:?}", original.len());
        println!("o sharded size: {:?}", o_shards.len() * SHARD_BYTES);
            // test a single segment reco.
            let segment_ix = data_chunks / 2;

            let chunk_start = segment_ix * SEGMENT_SIZE_PADDED;
            let chunk_start_i = data_index_to_chunk_index(chunk_start);
            let chunk_end = (segment_ix + 1) * SEGMENT_SIZE_PADDED;
            let chunk_end_i = data_index_to_chunk_index(chunk_start);
            assert!(chunk_start_i.2 == 0);
            assert!(chunk_end_i.2 == 0);

    */
    /*
    let mut oris = BTreeMap::new();
    for v in 0..342 {
    }
    */

    // reco from ori

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
