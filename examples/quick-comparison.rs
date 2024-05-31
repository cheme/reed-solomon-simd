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
    for count in [1, 2, 4, 5, 10, 16] {
        ec::test_sc(count);
    }
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
    let (r_shards1, r_shards2) = build_rec(r_shards.clone());
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

    // from ori
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

    // from first half red
    let mut o_dist_test = DistData::new(l * 12);
    let mut r_dist_test1 = DistData::new(l * 12);
    for d in 0..N_SHARDS {
        for i in dist_st..dist_end {
            r_dist_test1.shards[d][i] = r_dist1.shards[d][i];
        }
    }
    let o_shards_test = o_dist_test.to_chunked();
    let r_shards_test1 = r_dist_test1.to_chunked();
    let mut ori_map: std::collections::BTreeMap<_, _> = Default::default();
    let ori_ret = reed_solomon_simd::decode(
        N_SHARDS,
        2 * N_SHARDS,
        ori_map.iter().map(|(k, v)| (*k, v)),
        //r_shards.iter().enumerate(),
        //r_shards1.shards.iter().enumerate(),
        r_shards_test1.shards.iter().enumerate(),
    )
    .unwrap();
    assert_eq!(ori_ret.len(), N_SHARDS);
    ori_ret.into_iter().for_each(|(i, s)| {
        ori_map.insert(i, s);
    });
    assert_eq!(ori_map.len(), N_SHARDS);
    // TODO avoid instantiating this shards.
    let shards = ChunkedData::from_decode(ori_map);
    let ori_test = ori_chunk_to_data(&shards, chunk_start, Some(SEGMENT_SIZE_PADDED));
    assert_eq!(original[chunk_start..chunk_end], ori_test);

    // from second half red
    let mut o_dist_test = DistData::new(l * 12);
    let mut r_dist_test2 = DistData::new(l * 12);
    for d in 0..N_SHARDS {
        for i in dist_st..dist_end {
            r_dist_test2.shards[d][i] = r_dist2.shards[d][i];
        }
    }
    let o_shards_test = o_dist_test.to_chunked();
    let r_shards_test2 = r_dist_test2.to_chunked();
    let mut ori_map: std::collections::BTreeMap<_, _> = Default::default();
    let ori_ret = reed_solomon_simd::decode(
        N_SHARDS,
        2 * N_SHARDS,
        ori_map.iter().map(|(k, v)| (*k, v)),
        //r_shards.iter().enumerate(),
        //r_shards1.shards.iter().enumerate(),
        r_shards_test2
            .shards
            .iter()
            .enumerate()
            .map(|(i, s)| (i + N_SHARDS, s)),
    )
    .unwrap();
    assert_eq!(ori_ret.len(), N_SHARDS);
    ori_ret.into_iter().for_each(|(i, s)| {
        ori_map.insert(i, s);
    });
    assert_eq!(ori_map.len(), N_SHARDS);
    // TODO avoid instantiating this shards.
    let shards = ChunkedData::from_decode(ori_map);
    let ori_test = ori_chunk_to_data(&shards, chunk_start, Some(SEGMENT_SIZE_PADDED));
    assert_eq!(original[chunk_start..chunk_end], ori_test);

    // from 33% mix
    let mut o_dist_test = DistData::new(l * 12);
    let mut r_dist_test1 = DistData::new(l * 12);
    let mut r_dist_test2 = DistData::new(l * 12);
    for d in 0..N_SHARDS / 3 {
        for i in dist_st..dist_end {
            o_dist_test.shards[d][i] = o_dist.shards[d][i];
            r_dist_test1.shards[d][i] = r_dist1.shards[d][i];
            r_dist_test2.shards[d][i] = r_dist2.shards[d][i];
        }
    }
    let o_shards_test = o_dist_test.to_chunked();
    let r_shards_test1 = r_dist_test1.to_chunked();
    let r_shards_test2 = r_dist_test2.to_chunked();
    let mut ori_map: std::collections::BTreeMap<_, _> = o_shards_test
        .shards
        .iter()
        .take(N_SHARDS / 3)
        .enumerate()
        .map(|(i, s)| (i, s.clone()))
        .collect();
    assert_eq!(ori_map.len(), N_SHARDS / 3);
    let ori_ret = reed_solomon_simd::decode(
        N_SHARDS,
        2 * N_SHARDS,
        ori_map.iter().map(|(k, v)| (*k, v)),
        //r_shards.iter().enumerate(),
        //r_shards1.shards.iter().enumerate(),
        r_shards_test1
            .shards
            .iter()
            .take(N_SHARDS / 3)
            .enumerate()
            .chain(
                r_shards_test2
                    .shards
                    .iter()
                    .take(N_SHARDS / 3)
                    .enumerate()
                    .map(|(i, s)| (i + N_SHARDS, s)),
            ),
    )
    .unwrap();
    assert_eq!(ori_ret.len(), N_SHARDS / 3 * 2);
    ori_ret.into_iter().for_each(|(i, s)| {
        ori_map.insert(i, s);
    });
    assert_eq!(ori_map.len(), N_SHARDS);
    // TODO avoid instantiating this shards.
    let shards = ChunkedData::from_decode(ori_map);
    let ori_test = ori_chunk_to_data(&shards, chunk_start, Some(SEGMENT_SIZE_PADDED));
    assert_eq!(original[chunk_start..chunk_end], ori_test);

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

mod ec {

    //! This library provides methods for encoding the data into chunks and
    //! reconstructing the original data from chunks as well as verifying
    //! individual chunks against an erasure root.

    //mod error;
    //mod merklize;

    //pub use self::{
    //error::Error,
    //merklize::{ErasureRoot, MerklizedChunks, Proof},
    //};

    use super::ChunkedData;
    use reed_solomon_simd as reed_solomon;
    use scale::{Decode, Encode};
    use std::collections::BTreeMap;
    use std::ops::AddAssign;
    use thiserror::Error;

    pub const MAX_CHUNKS: u16 = 16384;

    // The reed-solomon library requires each shards to be 64 bytes aligned.
    const CHUNKS_ALIGNMENT: usize = 64;
    const N_POINT_BATCH: usize = CHUNKS_ALIGNMENT / SIZE_POINT;
    // The reed-solomon library requires each shards to be 64 bytes aligned.
    const CHUNKS_MIN_SHARD: usize = CHUNKS_ALIGNMENT;

    // The number of chunk of data needed to rebuild.
    const N_CHUNKS: usize = 342;

    // The number of time the erasure coded chunk we want.
    const N_REDUNDANCY: usize = 2;

    // Total of chunks, original and recovery, as will be distributed.
    const TOTAL_CHUNKS: usize = N_CHUNKS * (1 + N_REDUNDANCY);

    // Data chunked int o N_CHUNK need to be this big.
    const MIN_CHUNKED_DATA_SIZE: usize = N_CHUNKS * CHUNKS_MIN_SHARD; // 21_888

    // 3 * 12 is aligned with 64
    const SUBSHARD_BATCH_MUL: usize = 3;

    // 3 time 64 aligned subshard batch.
    const SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL: usize =
        SUBSHARD_BATCH_MUL * CHUNKS_MIN_SHARD / SUBSHARD_SIZE; // 16

    const MAX_SUB_OPTIMAL_SIZE_DATA: usize = SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL * SEGMENT_SIZE;
    //
    const SIZE_SUBSHARD_BATCH_OPTIMAL: usize = MIN_CHUNKED_DATA_SIZE * SUBSHARD_BATCH_MUL; // 65664 (mod12 = 0) and (mod 21888 = 0)

    // Segement of data that is part of a bigger content.
    const SEGMENT_SIZE: usize = 4096;

    // Segment size with added padding to allow being
    // erasure coded in batch while staying on same points indexes.
    const SEGMENT_SIZE_ALIGNED: usize = SUBSHARD_PER_SEGMENT * SUBSHARD_SIZE; // 4104 byte

    const SUBSHARD_PER_SEGMENT: usize = ((SEGMENT_SIZE - 1) / SUBSHARD_SIZE) + 1;

    const SIZE_POINT: usize = 2; // gf16

    const SUBSHARD_SIZE: usize = SIZE_POINT * SUBSHARD_POINTS; // 12bytes

    const SUBSHARD_POINTS: usize = 6;

    /// Proof of an erasure chunk which can be verified against [`ErasureRoot`]. TODO @cheme get the
    /// one from merklized of andronik dep
    #[derive(PartialEq, Eq, Clone, Debug, Encode, Decode)]
    pub struct Proof(Vec<u8>);
    /// Errors in erasure coding. TODO rem (is in module)
    #[derive(Debug, Clone, PartialEq, Error)]
    #[non_exhaustive]
    pub enum Error {
        #[error("There are too many chunks in total")]
        TooManyTotalChunks,
        #[error("Expected at least 1 chunk")]
        NotEnoughTotalChunks,
        #[error("Not enough chunks to reconstruct message")]
        NotEnoughChunks,
        #[error("Chunks are not uniform, mismatch in length or are zero sized")]
        NonUniformChunks,
        #[error("Unaligned chunk length")]
        UnalignedChunk,
        #[error("Chunk is out of bounds: {chunk_index} not included in 0..{n_chunks}")]
        ChunkIndexOutOfBounds { chunk_index: u16, n_chunks: u16 },
        #[error("Reconstructed payload invalid")]
        BadPayload,
        #[error("Invalid chunk proof")]
        InvalidChunkProof,
        #[error("The proof is too large")]
        TooLargeProof,
        #[error("Unexpected behavior of the reed-solomon library")]
        Bug,
        #[error("An unknown error has appeared when (re)constructing erasure code chunks")]
        Unknown,
    }
    //TODO rem
    impl From<reed_solomon::Error> for Error {
        fn from(error: reed_solomon::Error) -> Self {
            use reed_solomon::Error::*;

            match error {
                NotEnoughShards { .. } => Self::NotEnoughChunks,
                InvalidShardSize { .. } => Self::UnalignedChunk,
                TooManyOriginalShards { .. } => Self::TooManyTotalChunks,
                TooFewOriginalShards { .. } => Self::NotEnoughTotalChunks,
                DifferentShardSize { .. } => Self::NonUniformChunks,
                _ => Self::Unknown,
            }
        }
    }

    /// The index of an erasure chunk.
    /// Chunk indexes are split into `N_CHUNKS` original piece of data, and `N_REDUNDANCY`
    /// Original chunk will be in the first `N_CHUNKS` indicies, redundancy will be the next
    /// ones batches by `N_CHUNKS` sequence (this redundancy sequence is not strictly
    /// usefull but can be use to extract subchunks.
    #[derive(Eq, Ord, PartialEq, PartialOrd, Copy, Clone, Encode, Decode, Hash, Debug)]
    pub struct ChunkIndex(pub u16);

    impl From<u16> for ChunkIndex {
        fn from(n: u16) -> Self {
            ChunkIndex(n)
        }
    }

    impl AddAssign<u16> for ChunkIndex {
        fn add_assign(&mut self, rhs: u16) {
            self.0 += rhs
        }
    }

    /// A chunk of erasure-encoded block data. This target short lived EC content distribution (var len
    /// data).
    /// TODO @cheme in case the erasure chunk is from a data
    /// is < MIN_CHUNCKED_DATA_SIZE then 3 options:
    /// - pad original: sounds like a waste.
    /// - split into subshards, but need var len subshards.
    /// - pad to first n multiple of 4104 (padded page) or 4096 then use
    /// same subshards for all pages, grouping all n subshards by n.
    /// if is < 4096: then likely we just redundancy this to all validator. TODO @cheme
    /// not that clear, generally should have a min size for this erasure chunk (the short lived one).
    /// And at this min size we just distribute original content to everyone.
    #[derive(PartialEq, Eq, Clone, Encode, Decode, Debug)]
    pub struct ErasureChunk {
        /// The encoded chunk of data belonging to either the original data of an ec encoded data.
        pub chunk: Vec<u8>,
    }

    #[derive(PartialEq, Eq, Clone, Encode, Decode, Debug)]
    pub struct ErasureChunkWithProof {
        /// The encoded chunk of data belonging to either the original data of an ec encoded data.
        pub chunk: ErasureChunk,
        /// The index of this chunk of data.
        pub index: ChunkIndex,
        /// Proof for this chunk against an erasure root.
        pub proof: Proof,
    }

    /// A small chunk of erasure-encoded block data. This target long lived EC content distribution
    /// (fix segments sized).
    /// The size of this chunk is a portion of the original content
    /// length and var size.
    #[derive(PartialEq, Eq, Clone, Encode, Decode, Debug)]
    pub struct ErasureSubChunk {
        /// The erasure-encoded chunk of data belonging to the candidate block.
        pub chunk: [u8; SUBSHARD_SIZE],
    }

    /// Subchunk being small they are usually batched.
    #[derive(PartialEq, Eq, Clone, Encode, Decode, Debug)]
    pub struct ErasureSubChunkBatch {
        /// The erasure-encoded chunk of data belonging to the candidate block.
        pub chunks: Vec<(ChunkIndex, ErasureSubChunk)>,
    }

    #[derive(PartialEq, Eq, Clone, Encode, Decode, Debug)]
    pub struct Segment {
        /// Fix size chunk of data.
        pub data: Box<[u8; SEGMENT_SIZE]>,
        /// The index of this segment against its full data.
        pub index: u32,
    }

    /// Construct erasure-coded chunks.
    pub fn construct_chunks(data: &[u8]) -> Result<Vec<ErasureChunk>, Error> {
        if data.is_empty() {
            return Err(Error::BadPayload);
        }
        let original_data = make_original_chunks(data);

        let recovery = reed_solomon::encode(
            N_CHUNKS,
            N_REDUNDANCY * N_CHUNKS,
            original_data.iter().map(|c| &c.chunk),
        )?;

        let mut result = original_data;
        result.extend(recovery.into_iter().map(|chunk| ErasureChunk { chunk }));

        Ok(result)
    }

    fn next_aligned(n: usize, alignment: usize) -> usize {
        ((n + alignment - 1) / alignment) * alignment
    }

    fn chunk_bytes(data_len: usize) -> usize {
        let shards_min_len = data_len.div_ceil(N_CHUNKS);
        next_aligned(shards_min_len, CHUNKS_ALIGNMENT)
    }

    // The reed-solomon library takes sharded data as input.
    fn make_original_chunks(data: &[u8]) -> Vec<ErasureChunk> {
        debug_assert!(!data.is_empty(), "data must be non-empty");

        let chunk_bytes = chunk_bytes(data.len());

        let mut result = Vec::with_capacity(N_CHUNKS);
        for chunk in data.chunks(chunk_bytes) {
            let mut chunk = chunk.to_vec();
            chunk.resize(chunk_bytes, 0);
            result.push(ErasureChunk { chunk });
        }

        result
    }

    /// Reconstruct the original data from a set of chunks.
    /// Chunks must be sorted by index.
    ///
    /// Provide an iterator containing chunk data and the corresponding index.
    /// The indices of the present chunks must be indicated. If too few chunks
    /// are provided, recovery is not possible.
    ///
    /// Works only for 1..65536 chunks.
    ///
    /// Due to the internals of the erasure coding algorithm, the output might be
    /// larger than the original data and padded with zeroes; passing `data_len`
    /// allows to truncate the output to the original data size.
    pub fn reconstruct<'a, I>(chunks: &'a I, data_len: usize) -> Result<Vec<u8>, Error>
    where
        &'a I: IntoIterator<Item = (ChunkIndex, &'a ErasureChunk)>,
    {
        let original = chunks
            .into_iter()
            .take(N_CHUNKS)
            .map(|(i, v)| (i.0 as usize, &v.chunk));
        let recovery = chunks
            .into_iter()
            .skip(N_CHUNKS)
            .map(|(i, v)| (i.0 as usize - N_CHUNKS, &v.chunk));

        let mut recovered =
            reed_solomon::decode(N_CHUNKS, N_REDUNDANCY * N_CHUNKS, original, recovery)?;

        let shard_bytes = chunk_bytes(data_len);
        let mut bytes = Vec::with_capacity(shard_bytes * N_CHUNKS);

        let mut original = chunks
            .into_iter()
            .take(N_CHUNKS)
            .map(|(i, v)| (i.0 as usize, &v.chunk));
        for i in 0..N_CHUNKS {
            let chunk = recovered.get(&i).map(AsRef::as_ref).unwrap_or_else(|| {
                let (j, v) = original
                    .next()
                    .expect("what is not recovered must be present; qed");
                debug_assert_eq!(i, j); // input iterator Must be sorted.
                v
            });
            bytes.extend_from_slice(chunk);
        }

        bytes.truncate(data_len);

        Ok(bytes)
    }

    /// Subchunk uses some temp memory, so these should be used multiple time instead of allocating.
    /// These run on the smallest unpadded buffers (3 time the theoric smallest buffers).
    pub struct SubChunkEncoder {
        encoder: reed_solomon::ReedSolomonEncoder,
        // TODO @cheme should be able to remove this buffer?
        chunked_data: ChunkedData,
    }

    impl SubChunkEncoder {
        pub fn new() -> Result<Self, Error> {
            Ok(Self {
                encoder: reed_solomon::ReedSolomonEncoder::new(
                    N_CHUNKS,
                    N_REDUNDANCY * N_CHUNKS,
                    CHUNKS_MIN_SHARD * SUBSHARD_BATCH_MUL,
                )?,
                chunked_data: ChunkedData::init_sized(SUBSHARD_BATCH_MUL),
            })
        }

        /// Construct erasure-coded chunks.
        /// Segement input must be ordered by index and consecutive.
        /// Data must be less than MAX_SUB_OPTIMAL_SIZE_DATA.
        pub fn construct_chunks(
            &mut self,
            data: &[Segment],
        ) -> Result<Vec<Box<[[u8; SUBSHARD_SIZE]; N_CHUNKS * 3]>>, Error> {
            if data.len() > SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL {
                return Err(Error::BadPayload);
            }
            let mut next = 0;
            for s in data.iter() {
                if s.index != next {
                    return Err(Error::BadPayload);
                }
                next += 1;
            }
            /*
            if data.len() % SEGMENT_SIZE != 0 {
                return Err(Error::BadPayload);
            }
                        */

            let mut shards = &mut self.chunked_data.shards;
            //	let mut shards = vec![[0u8; SHARD_BYTES]; N_SHARDS];
            let mut shard_i = 0;
            let mut shard_i_offset = 0;
            let mut shard_a = 0;
            let mut full_i = 0;
            for original in data.iter().map(|s| &s.data) {
                for point in (0..SEGMENT_SIZE / 2)
                    .map(|i_p| (original[i_p * 2], original[i_p * 2 + 1]))
                    .chain((0..(SEGMENT_SIZE_ALIGNED - SEGMENT_SIZE) / 2).map(|_| (0, 0)))
                {
                    //		println!("{}: {} {} {} {} {}", i_p, number_shards_batch, original.len(), SHARD_BATCH_SIZE * number_shards_batch, shard_a, original.len()/2);
                    //				println!("{:?}", (i_p, original.len() / 2));
                    shards[shard_a][shard_i_offset + shard_i] = point.0;
                    shards[shard_a][shard_i_offset + 32 + shard_i] = point.1;
                    shard_a += 1;
                    if shard_a % N_CHUNKS == 0 {
                        shard_i += 1;
                        if shard_i == N_POINT_BATCH {
                            shard_i = 0;
                            full_i += 1;
                            shard_i_offset = full_i * (N_POINT_BATCH * SIZE_POINT);
                        }
                        shard_a = 0;
                    }
                }
                debug_assert_eq!(shard_a % N_CHUNKS, 0);
            }

            for shard in self.chunked_data.shards.iter() {
                self.encoder.add_original_shard(&shard)?;
            }

            let enco_res = self.encoder.encode()?;
            let r_shards = enco_res.recovery_iter().map(|s| s.to_vec()).collect();
            let (r_shards1, r_shards2) = super::build_rec(r_shards);
            // TODO rem those dist, directly result TODO buff res?
            let o_dist = self.chunked_data.to_dist();
            let r_dist1 = r_shards1.to_dist();
            let r_dist2 = r_shards2.to_dist();
            assert_eq!(o_dist.shards[0].len(), SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL); // TODO @cheme change o_dist type to 16 point fix
            let mut result = vec![
                Box::new([[0u8; SUBSHARD_SIZE]; N_CHUNKS * 3]);
                SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL
            ];
            for i in 0..SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL {
                for j in 0..N_CHUNKS {
                    result[i][j] = o_dist.shards[j][i];
                    result[i][j + N_CHUNKS] = r_dist1.shards[j][i];
                    result[i][j + (N_CHUNKS * 2)] = r_dist2.shards[j][i];
                }
            }
            return Ok(result);
        }
    }

    pub struct SubChunkDecoder {
        decoder: reed_solomon::ReedSolomonDecoder,
    }

    impl SubChunkDecoder {
        pub fn new() -> Result<Self, Error> {
            Ok(Self {
                decoder: reed_solomon::ReedSolomonDecoder::new(
                    N_CHUNKS,
                    N_REDUNDANCY * N_CHUNKS,
                    CHUNKS_MIN_SHARD * SUBSHARD_BATCH_MUL,
                )?,
            })
        }

        // u8 is the segment number.
        pub fn reconstruct<'a, I>(&mut self, chunks: &'a mut I) -> Result<Vec<(u8, Segment)>, Error>
        where
            I: Iterator<Item = (u8, ChunkIndex, &'a [u8; SUBSHARD_SIZE])>,
        {
            use super::DistData;
            let mut o_dist_test = DistData::new(SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL * 12);
            let mut r_dist_test1 = DistData::new(SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL * 12);
            let mut r_dist_test2 = DistData::new(SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL * 12);
            let mut nb_chunk = 0; // support a single seg index first.
            let mut map_chunk: [(BTreeMap<usize, ()>, BTreeMap<usize, ()>);
                SEGMENTS_PER_SUBSHARD_BATCH_OPTIMAL] = Default::default();
            for (segment, chunk_index, chunk) in chunks {
                let chunk_index = chunk_index.0 as usize;
                let segment = segment as usize;
                if chunk_index < N_CHUNKS {
                    map_chunk[segment].0.insert(chunk_index, ());
                    o_dist_test.shards[chunk_index][segment] = *chunk;
                } else if chunk_index < 2 * N_CHUNKS {
                    r_dist_test1.shards[chunk_index - N_CHUNKS][segment] = *chunk;
                    map_chunk[segment].1.insert(chunk_index - N_CHUNKS, ());
                } else {
                    debug_assert!(chunk_index < 3 * N_CHUNKS);
                    r_dist_test2.shards[chunk_index - (2 * N_CHUNKS)][segment] = *chunk;
                    map_chunk[segment].1.insert(chunk_index - N_CHUNKS, ());
                }
            }
            /*
            for s in map_chunk {
                // TODO this is for single segment we should have
                // matching segment and process them at once.
                // TODO would make sense to have longer iter.
                assert_eq!(map_chunk.0.len() + map_chunk.1.len() == 342);
            }
                        */
            let o_shards_test = o_dist_test.to_chunked();
            let r_shards_test1 = r_dist_test1.to_chunked();
            let r_shards_test2 = r_dist_test2.to_chunked();

            let mut result = Vec::new();
            for (segment, map_chunk) in map_chunk.iter().enumerate() {
                // TODO this is for single segment we should have
                // matching segment and process them at once.
                // TODO would make sense to have longer iter.
                let mut ori_map: std::collections::BTreeMap<usize, Vec<u8>> = Default::default();
                if map_chunk.0.len() + map_chunk.1.len() >= N_CHUNKS {
                    for i in map_chunk.0.keys() {
                        let i = *i as usize;
                        self.decoder.add_original_shard(i, &o_shards_test.shards[i]);
                        ori_map.insert(i, o_shards_test.shards[i].clone());
                    }
                    for i in map_chunk.1.keys() {
                        let i = *i as usize;
                        if i >= N_CHUNKS {
                            self.decoder
                                .add_recovery_shard(i, &r_shards_test2.shards[i - N_CHUNKS]);
                        } else {
                            self.decoder
                                .add_recovery_shard(i, &r_shards_test1.shards[i]);
                        }
                    }

                    let ori_ret = self.decoder.decode()?;
                    for (i, o) in ori_ret.restored_original_iter() {
                        ori_map.insert(i, o.to_vec());
                    }
                    assert_eq!(ori_map.len(), N_CHUNKS);
                    // TODO avoid instantiating this shards.
                    let shards = ChunkedData::from_decode(ori_map);
                    let chunk_start = segment * SEGMENT_SIZE_ALIGNED;
                    // TODO direct copy on segment chunk
                    let original =
                        super::ori_chunk_to_data(&shards, chunk_start, Some(SEGMENT_SIZE));
                    result.push((
                        segment as u8,
                        Segment {
                            data: original.try_into().unwrap(),
                            index: segment as u32,
                        },
                    ));
                }
            }
            Ok(result)
        }
    }

    pub fn test_sc(nb_seg: usize) {
        use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
        let mut rng = SmallRng::from_seed([0; 32]);
        let segments: Vec<_> = (0..nb_seg)
            .map(|s| {
                let mut se = [0u8; SEGMENT_SIZE];
                rng.fill::<_>(&mut se[..]);

                Segment {
                    data: Box::new(se),
                    index: s as u32,
                }
            })
            .collect();

        let mut encoder = SubChunkEncoder::new().unwrap();
        let mut decoder = SubChunkDecoder::new().unwrap();
        let chunks = encoder.construct_chunks(&segments).unwrap();
        let i_seg = nb_seg / 2;

        let mut it = (&chunks[i_seg][0..N_CHUNKS / 3])
            .iter()
            .enumerate()
            .map(|(i, c)| (i_seg as u8, ChunkIndex(i as u16), c))
            .chain(
                (&chunks[i_seg][N_CHUNKS..N_CHUNKS + N_CHUNKS / 3])
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i_seg as u8, ChunkIndex(i as u16 + N_CHUNKS as u16), c)),
            )
            .chain(
                (&chunks[i_seg][N_CHUNKS * 2..N_CHUNKS * 2 + N_CHUNKS / 3])
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i_seg as u8, ChunkIndex(i as u16 + N_CHUNKS as u16 * 2), c)),
            );
        let s = decoder.reconstruct(&mut it).unwrap();
        assert_eq!((i_seg as u8, segments[i_seg].clone()), s[0]);
    }
}
