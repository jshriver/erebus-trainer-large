//! erebus-trainer-large -- bullet trainer for the erebus-large NNUE arch.
//!
//! Network topology:
//!
//!   Inputs per perspective:
//!     - HalfKAv2_hm (768 × 32 king-pos buckets = 24 576 features)
//!     - Full_Threats + PP_3Wide  (62 320 features)
//!     Total: 86 896 inputs per perspective
//!
//!   Feature transformer (per perspective):
//!     Sparse Linear [86896 → 2×L0_SIZE]
//!     Pairwise CReLU: split-and-multiply → [L0_SIZE]
//!
//!   Concat STM + NTM → [2×L0_SIZE]
//!
//!   Main subnet:
//!     Dense [2×L0_SIZE → L1]  ──► SqrCReLU + CReLU concat → [2×L1]
//!     Dense [2×L1      → L2]  ──► SqrCReLU + CReLU concat → [2×L2]
//!     Skip: concat(act1, act2) → [2×(L1+L2) = 128]
//!     Linear [128 → 1], 8 output buckets (material count)
//!
//! Produces `<OUT_DIR>/<NET_ID>-<N>/quantised.bin`.
//!
//! Usage:
//!   cargo build --release --features cuda
//!   ./erebus-trainer-large <data.binpack | data-dir> [more paths...]

mod inputs;

use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::SparseInputType,
        outputs::{MaterialCount, OutputBuckets},
    },
    trainer::schedule::lr::{self, LrScheduler},
    value::{
        loader::sfbinpack::{MoveType, PieceType, SfBinpackLoader, TrainingDataEntry},
        save::save_to_checkpoint,
    },
};
use bullet_trainer::{
    model::{InitSettings, ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat},
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
    reader::ReadMapLoader,
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, train},
};
use inputs::{HalfKAv2Hm, PawnPawnInputs};

// ========================= CONFIG -- edit, then `cargo build --release` =========================

/// Accumulator size PER PERSPECTIVE after the pairwise activation.
/// The sparse linear layer outputs 2×L0_SIZE; pairwise halves it to L0_SIZE.
const L0_SIZE: usize = 512;

/// Width of the first dense hidden layer.
const L1_SIZE: usize = 32;

/// Width of the second dense hidden layer.
const L2_SIZE: usize = 32;

/// Material-count output buckets. MUST equal the engine constant.
const OUTPUT_BUCKETS: usize = 8;

/// Feature-transformer quantisation. Saved as i16.
const QA: i16 = 255;

/// Quantisation for the dense layers following the FT (i8 save).
const QB: i16 = 64;

/// Weight clip for the FT weights and the shared factoriser.
const L0_CLIP: f32 = 0.99;

const NET_ID: &str = "erebus-large";
const OUT_DIR: &str = "checkpoints";

const TOTAL_PASSES: f64 = 1.0;
const PASS_FRACTION_PER_FILE: f64 = 1.0;

/// Known raw position counts keyed by binpack basename.
/// Sum = 218_849_949_380 over 41 files (from binpack_counter).
const POSITION_COUNTS: &[(&str, u64)] = &[
    ("test60-2021-11-nov-12tb7p.min-v2.relabel-BT4-tf13tune.binpack", 1_452_424_355),
    ("test60-2021-12-dec-12tb7p.min-v2.relabel-BT4-tf13tune.binpack", 1_363_206_227),
    ("test77-2021-12-dec-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 6_286_624_013),
    ("test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 7_419_909_666),
    ("test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 3_574_170_531),
    ("test79-2022-04-apr-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 2_930_087_205),
    ("test79-2022-05-may-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 2_284_179_270),
    ("test80-2022-06-jun-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_488_679_928),
    ("test80-2022-07-jul-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_835_573_847),
    ("test80-2022-08-aug-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 3_801_599_910),
    ("test80-2022-09-sep-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_171_904_814),
    ("test80-2022-10-oct-16tb7p.v6-dd.relabel-BT4-tf13tune.part_0.binpack", 2_030_804_185),
    ("test80-2022-10-oct-16tb7p.v6-dd.relabel-BT4-tf13tune.part_1.binpack", 2_030_725_513),
    ("test80-2022-11-nov-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_608_318_891),
    ("test80-2023-01-jan-16tb7p.v6-sk20.min.relabel-BT4-tf13tune.binpack", 4_707_093_556),
    ("test80-2023-02-feb-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 3_626_845_354),
    ("test80-2023-03-mar-2tb7p.v6-sk16.min.relabel-BT4-tf13tune.binpack", 5_520_899_664),
    ("test80-2023-04-apr-2tb7p.v6-sk16.min.relabel-BT4-tf13tune.binpack", 5_653_619_110),
    ("test80-2023-05-may-2tb7p.v6.min.relabel-BT4-tf13tune.binpack", 5_600_480_538),
    ("test80-2023-06-jun-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 6_756_356_195),
    ("test80-2023-07-jul-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 6_212_977_488),
    ("test80-2023-08-aug-2tb7p.v6.min.relabel-BT4-tf13tune.binpack", 2_693_519_136),
    ("test80-2023-09-sep-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 3_257_611_143),
    ("test80-2023-10-oct-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 3_012_783_968),
    ("test80-2023-11-nov-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 2_724_311_169),
    ("test80-2023-12-dec-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 3_016_184_922),
    ("leela96-filt-v2.min.split_0.relabel-BT4-tf13tune.binpack", 5_681_356_602),
    ("leela96-filt-v2.min.split_1.relabel-BT4-tf13tune.binpack", 5_679_303_898),
    ("leela96-filt-v2.min.split_2.relabel-BT4-tf13tune.binpack", 5_680_096_474),
    ("leela96-filt-v2.min.split_3.relabel-BT4-tf13tune.binpack", 5_681_363_129),
    ("leela96-filt-v2.min.split_4.relabel-BT4-tf13tune.binpack", 5_680_919_670),
    ("T60T70wIsRightFarseerT60T74T75T76.split_0.relabel-BT4-tf13tune.binpack", 9_133_725_682),
    ("T60T70wIsRightFarseerT60T74T75T76.split_1.relabel-BT4-tf13tune.binpack", 9_150_872_906),
    ("T60T70wIsRightFarseerT60T74T75T76.split_2.relabel-BT4-tf13tune.binpack", 9_136_446_438),
    ("T60T70wIsRightFarseerT60T74T75T76.split_3.relabel-BT4-tf13tune.binpack", 9_128_728_654),
    ("T60T70wIsRightFarseerT60T74T75T76.split_4.relabel-BT4-tf13tune.binpack", 9_160_229_495),
    ("dfrc_n5000.relabel-BT4-tf13tune.binpack", 12_353_351_142),
    ("fishpack32.relabel-BT4-tf13tune.binpack", 2_555_358_353),
    ("multinet_pv-2_diff-100_nodes-5000.relabel-BT4-tf13tune.binpack", 9_485_503_089),
    ("nodes5000pv2_UHO.relabel-BT4-tf13tune.binpack", 13_937_427_120),
    ("wrongIsRight_nodes5000pv2.relabel-BT4-tf13tune.binpack", 2_344_376_130),
];

const FILTER_KEEP_FRAC: f64 = 1.0;
const MAX_SUPERBATCH: usize = 5000;

const BATCH_SIZE: usize = 16_384;
const BATCHES_PER_SUPERBATCH: usize = 6104;
const SAVE_RATE: usize = 10;

const LR_START: f32 = 0.001;
const LR_FINAL: f32 = 2.5e-6;

/// Linear WDL blend: lambda = WDL_START + t*(WDL_END-WDL_START) over 1..=global_end.
const WDL_START: f32 = 0.2;
const WDL_END: f32 = 0.6;

const EVAL_SCALE: f32 = 400.0;

const ROTATE_DATA_EACH_SESSION: bool = true;

const DATA_THREADS: usize = 4;
const SHUFFLE_BUFFER_MB: usize = 4096;

// =============================================================================================

/// Number of inputs for HalfKAv2_hm (used to size the weight matrix).
const NUM_HALFKA_INPUTS: usize = 768 * inputs::NUM_HALFKA_BUCKETS;

fn die(msg: impl AsRef<str>) -> ! {
    eprintln!("erebus-trainer-large: {}", msg.as_ref());
    std::process::exit(1);
}

fn nothing_to_do(msg: impl AsRef<str>) -> ! {
    println!("{}", msg.as_ref());
    std::process::exit(3);
}

fn usage() -> ! {
    eprintln!(
        "erebus-trainer-large -- erebus-large NNUE trainer\n\
         \n\
         usage:  erebus-trainer-large <PATH>...\n\
         \n\
         Each PATH is a .binpack file or a directory (all *.binpack inside,\n\
         sorted by name).  All hyper-parameters are compiled in -- see the\n\
         CONFIG block in src/main.rs.  Resume from {OUT_DIR}/{NET_ID}-<N>/ is\n\
         automatic."
    );
    std::process::exit(2);
}

fn collect_data_paths(inputs: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for inp in inputs {
        let p = std::path::Path::new(inp);
        if p.is_dir() {
            let mut found: Vec<String> = std::fs::read_dir(p)
                .unwrap_or_else(|e| die(format!("read_dir {inp}: {e}")))
                .flatten()
                .map(|e| e.path())
                .filter(|q| q.extension().map(|x| x == "binpack").unwrap_or(false))
                .map(|q| q.to_string_lossy().into_owned())
                .collect();
            found.sort();
            if found.is_empty() { die(format!("no *.binpack files in directory {inp}")); }
            out.extend(found);
        } else if p.is_file() {
            out.push(inp.clone());
        } else {
            die(format!("data path does not exist: {inp}"));
        }
    }
    out
}

fn count_for(path: &str) -> u64 {
    let side = format!("{path}.count");
    if let Ok(raw) = std::fs::read_to_string(&side) {
        let digits: String = raw.chars().filter(char::is_ascii_digit).collect();
        return digits.parse().unwrap_or_else(|e| die(format!("{side}: no valid integer ({e})")));
    }
    let base = std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    match POSITION_COUNTS.iter().find(|(name, _)| *name == base) {
        Some((_, n)) => *n,
        None => die(format!(
            "no position count for '{base}': add it to POSITION_COUNTS in src/main.rs, \
             or drop a '{side}' sidecar file next to it, or set EREBUS_LARGE_END_SB=N"
        )),
    }
}

fn corpus_positions() -> u64 {
    POSITION_COUNTS.iter().map(|(_, n)| *n).sum()
}

fn session_path() -> String { format!("{OUT_DIR}/{NET_ID}.session") }

fn read_session() -> Option<(usize, usize)> {
    let s = std::fs::read_to_string(session_path()).ok()?;
    let mut it = s.split_whitespace();
    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
}

fn write_session(began: usize, stop: usize) {
    let _ = std::fs::create_dir_all(OUT_DIR);
    if let Err(e) = std::fs::write(session_path(), format!("{began} {stop}\n")) {
        eprintln!("erebus-trainer-large: warning: could not write session file: {e}");
    }
}

fn latest_checkpoint() -> Option<(String, usize)> {
    let prefix = format!("{NET_ID}-");
    let mut best: Option<(String, usize)> = None;
    for ent in std::fs::read_dir(OUT_DIR).ok()?.flatten() {
        let path = ent.path();
        if !path.is_dir() { continue; }
        let name = ent.file_name();
        let name = name.to_string_lossy();
        let Some(num) = name.strip_prefix(&prefix) else { continue };
        let Ok(n) = num.parse::<usize>() else { continue };
        if !path.join("optimiser_state").is_dir() { continue; }
        if best.as_ref().map_or(true, |(_, b)| n > *b) {
            best = Some((path.to_string_lossy().into_owned(), n));
        }
    }
    best
}

fn filter(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

fn wdl_lambda(superbatch: usize, global_end: usize) -> f32 {
    let denom = global_end.saturating_sub(1).max(1) as f32;
    let t = (superbatch.saturating_sub(1) as f32 / denom).clamp(0.0, 1.0);
    WDL_START + t * (WDL_END - WDL_START)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") { usage(); }

    // ---- feature-set instances ----
    let pp = PawnPawnInputs::new(inputs::three_file_band_mask());
    let psqt = HalfKAv2Hm::new();
    let output_buckets = MaterialCount::<OUTPUT_BUCKETS>;

    // ---- resume ----
    let (resume_dir, start_superbatch) = match latest_checkpoint() {
        Some((dir, n)) => {
            println!("resume: {dir} (completed superbatch {n}) -> continuing from {}", n + 1);
            (Some(dir), n + 1)
        }
        None => {
            println!("resume: no '{OUT_DIR}/{NET_ID}-<N>' checkpoint -> starting fresh");
            (None, 1)
        }
    };

    // ---- data files ----
    let mut files = collect_data_paths(&args);
    if ROTATE_DATA_EACH_SESSION && files.len() > 1 {
        let k = (start_superbatch - 1) % files.len();
        if k != 0 {
            files.rotate_left(k);
            println!("rotated {} file(s) left by {k}", files.len());
        }
    }
    let paths_ref: Vec<&str> = files.iter().map(String::as_str).collect();

    // ---- plan + session window ----
    let pos_per_superbatch = BATCHES_PER_SUPERBATCH * BATCH_SIZE;
    let global_end = ((TOTAL_PASSES * corpus_positions() as f64 * FILTER_KEEP_FRAC
        / pos_per_superbatch as f64)
        .round() as usize)
        .max(1);

    let end_superbatch: usize = if let Ok(v) = std::env::var("EREBUS_LARGE_END_SB") {
        let n = v.trim().parse::<usize>().unwrap_or_else(|e| die(format!("EREBUS_LARGE_END_SB not a number: {e}")));
        if start_superbatch > n {
            nothing_to_do(format!("'{NET_ID}': EREBUS_LARGE_END_SB={n} but already at {start_superbatch}."));
        }
        write_session(start_superbatch, n);
        println!("session {start_superbatch}..={n}  (EREBUS_LARGE_END_SB override; global end {global_end})");
        n
    } else if start_superbatch > global_end {
        nothing_to_do(format!(
            "'{NET_ID}' completed its {TOTAL_PASSES}-pass plan ({global_end} superbatches). \
             Raise TOTAL_PASSES and rebuild to train longer."
        ))
    } else if let Some((began, stop)) =
        read_session().filter(|&(b, s)| start_superbatch >= b && start_superbatch <= s)
    {
        let stop = stop.min(global_end);
        println!("session {began}..={stop} resumed at {start_superbatch}  (global end {global_end})");
        stop
    } else {
        let passed: u64 = files.iter().map(|f| count_for(f)).sum();
        let budget = ((PASS_FRACTION_PER_FILE * passed as f64 * FILTER_KEEP_FRAC
            / pos_per_superbatch as f64)
            .round() as usize)
            .clamp(1, MAX_SUPERBATCH);
        let stop = (start_superbatch - 1 + budget).min(global_end);
        write_session(start_superbatch, stop);
        println!(
            "session {start_superbatch}..={stop}  (+{budget} sb from {passed} positions in {} file(s); global end {global_end})",
            files.len()
        );
        stop
    };

    // ---- arch summary ----
    let pp_inputs = pp.num_inputs();
    let halfka_inputs = NUM_HALFKA_INPUTS;
    let total_inputs = pp_inputs + halfka_inputs;
    println!("--------------------------------------------------------------");
    println!("arch          : erebus-large  [{total_inputs} inputs] ({halfka_inputs} HalfKAv2_hm + {pp_inputs} PP) × 2  FT({L0_SIZE}×2→{L0_SIZE}) → L1({L1_SIZE}) → L2({L2_SIZE}) → 1");
    println!("quantisation  : QA={QA} QB={QB} eval_scale={EVAL_SCALE}");
    println!("plan          : {TOTAL_PASSES} pass(es) -> global end {global_end} superbatches");
    println!("this session  : {start_superbatch}..={end_superbatch}  ({BATCHES_PER_SUPERBATCH} × {BATCH_SIZE})");
    println!("lr            : {LR_START} -> {LR_FINAL} cosine over 1..={global_end}");
    println!("wdl lambda    : {WDL_START} -> {WDL_END} linear over 1..={global_end}");
    println!("save rate     : every {SAVE_RATE} superbatches -> {OUT_DIR}/");
    println!("data          : {} files, {DATA_THREADS} decode threads, {SHUFFLE_BUFFER_MB} MiB buffer", files.len());
    println!("--------------------------------------------------------------");

    // ---- model inputs spec ----
    // Add order: stm_pp, ntm_pp, stm_halfka, ntm_halfka, bucket, target
    // Destructs as: (((((stm_pp, ntm_pp), stm_halfka), ntm_halfka), bucket), target)
    let model_inputs = ModelInputs::default()
        .add_sparse("stm/pp",    (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("ntm/pp",    (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("stm/halfka",(psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("ntm/halfka",(psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("bucket",   (OUTPUT_BUCKETS, 1), 1)
        .add_dense( "target",   (1, 1));

    // ---- model definition ----
    let halfka_init = InitSettings::Normal {
        mean: 0.0,
        stdev: (2.0_f32 / NUM_HALFKA_INPUTS as f32).sqrt(),
    };
    let pp_num = pp.num_inputs();

    let defn = ModelDefinition::build(
        &model_inputs,
        move |builder, (((((stm_pp, ntm_pp), stm_halfka), ntm_halfka), bucket), target)| {
            // ---- Feature transformer weights ----
            // Shared factoriser: 768 columns broadcast across all king buckets.
            // Merged into l0hw at save time; never used by the engine.
            let l0f = builder.new_weights("l0f", (2 * L0_SIZE, 768), InitSettings::Zeroed);
            let mut l0_halfka = builder.new_weights("l0hw", (2 * L0_SIZE, NUM_HALFKA_INPUTS), halfka_init);
            l0_halfka = l0_halfka + l0f.repeat(inputs::NUM_HALFKA_BUCKETS);

            // PP projection (includes bias for the full FT output).
            let l0_pp = builder.new_affine("l0p/", pp_num, 2 * L0_SIZE);

            // ---- Dense layers ----
            let l1    = builder.new_affine("l1/",    2 * L0_SIZE,              OUTPUT_BUCKETS * L1_SIZE);
            let l2    = builder.new_affine("l2/",    2 * L1_SIZE,              OUTPUT_BUCKETS * L2_SIZE);
            let l_out = builder.new_affine("l_out/", 2 * (L1_SIZE + L2_SIZE),  OUTPUT_BUCKETS);

            // ---- Per-perspective FT with pairwise CReLU activation ----
            let stm_pre = l0_halfka.matmul(stm_halfka) + l0_pp.forward(stm_pp);
            let stm_raw = stm_pre.crelu();
            let stm_ft  = stm_raw.slice_rows(0, L0_SIZE) * stm_raw.slice_rows(L0_SIZE, 2 * L0_SIZE);

            let ntm_pre = l0_halfka.matmul(ntm_halfka) + l0_pp.forward(ntm_pp);
            let ntm_raw = ntm_pre.crelu();
            let ntm_ft  = ntm_raw.slice_rows(0, L0_SIZE) * ntm_raw.slice_rows(L0_SIZE, 2 * L0_SIZE);

            let acc = stm_ft.concat(ntm_ft); // [2 * L0_SIZE]

            // ---- Main subnet ----
            let h1   = l1.forward(acc).select(bucket);
            let act1 = h1.screlu().concat(h1.crelu());     // [2 * L1_SIZE]

            let h2   = l2.forward(act1).select(bucket);
            let act2 = h2.screlu().concat(h2.crelu());     // [2 * L2_SIZE]

            let skip   = act1.concat(act2);                // [2 * (L1_SIZE + L2_SIZE)]
            let output = l_out.forward(skip).select(bucket);

            let loss = output.sigmoid().squared_error(target);
            (Some(loss.reduce_sum_batch()), vec![("output".to_string(), output)])
        },
    );

    // ---- optimiser ----
    let weights = ModelWeights::new(&defn, 12345678);
    let device  = DefaultDevice::new(0).unwrap_or_else(|e| die(format!("GPU init failed: {e:?}")));
    let mut optimiser = Optimiser::<_, AdamW<_>>::new(defn, weights, device, AdamWParams::default())
        .unwrap_or_else(|e| die(format!("optimiser init: {e:?}")));

    let l0_clip = AdamWParams { max_weight: L0_CLIP, min_weight: -L0_CLIP, ..Default::default() };
    optimiser.set_params_for_weight("l0hw", l0_clip);
    optimiser.set_params_for_weight("l0f",  l0_clip);

    if let Some(dir) = &resume_dir {
        optimiser.load_from_checkpoint(dir)
            .unwrap_or_else(|e| die(format!("load checkpoint '{dir}': {e:?}")));
    }

    // ---- save format ----
    // Merge factoriser into l0hw at save time so the engine sees a single weight.
    let saved_format = vec![
        SavedFormat::id("l0hw")
            .transform(|store, weights| {
                let fac = store.get("l0f").values.f32().repeat(inputs::NUM_HALFKA_BUCKETS);
                assert_eq!(weights.len(), fac.len());
                weights.iter().zip(fac).map(|(&a, b)| a + b).collect()
            })
            .round()
            .quantise::<i16>(QA),
        SavedFormat::id("l0p/w").round().quantise::<i16>(QA),
        SavedFormat::id("l0p/b").round().quantise::<i16>(QA),
        SavedFormat::id("l1/w").round().quantise::<i8>(QB).transpose(),
        SavedFormat::id("l1/b"),
        SavedFormat::id("l2/w").round().quantise::<i8>(QB).transpose(),
        SavedFormat::id("l2/b"),
        SavedFormat::id("l_out/w").round().quantise::<i8>(QB).transpose(),
        SavedFormat::id("l_out/b"),
    ];

    // ---- mapper (fills all 6 input slices per position) ----
    let pp_map    = pp.clone();
    let psqt_map  = psqt.clone();

    let mapper = ModelInputsMapper::build(
        &model_inputs,
        move |pos: &ChessBoard, step, (((((stm_pp, ntm_pp), stm_halfka), ntm_halfka), bucket), target)| {
            // HalfKA (symmetric: same count each side)
            let mut cnt = 0;
            psqt_map.map_features(pos, |stm, ntm| {
                stm_halfka[cnt] = stm.try_into().unwrap();
                ntm_halfka[cnt] = ntm.try_into().unwrap();
                cnt += 1;
            });
            if cnt < psqt_map.max_active() {
                stm_halfka[cnt] = -1;
                ntm_halfka[cnt] = -1;
            }

            // PP (asymmetric: separate stm/ntm callbacks)
            let mut stm_cnt = 0;
            let mut ntm_cnt = 0;
            pp_map.map_features(
                pos,
                |f| { stm_pp[stm_cnt] = f.try_into().unwrap(); stm_cnt += 1; },
                |f| { ntm_pp[ntm_cnt] = f.try_into().unwrap(); ntm_cnt += 1; },
            );
            if stm_cnt < pp_map.max_active() { stm_pp[stm_cnt] = -1; }
            if ntm_cnt < pp_map.max_active() { ntm_pp[ntm_cnt] = -1; }

            bucket[0] = i32::from(output_buckets.bucket(pos));

            let result = f32::from(pos.result) / 2.0;
            let score  = 1.0 / (1.0 + (f32::from(-pos.score) / EVAL_SCALE).exp());
            let lambda = wdl_lambda(step.superbatch(), global_end);
            target[0]  = lambda * result + (1.0 - lambda) * score;
        },
    );

    // ---- training loop ----
    let data_loader =
        SfBinpackLoader::new_concat_multiple(&paths_ref, SHUFFLE_BUFFER_MB, DATA_THREADS, filter);

    let schedule = TrainingSchedule {
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES_PER_SUPERBATCH,
            start_superbatch,
            end_superbatch,
        },
        lr_schedule: lr::CosineDecayLR {
            initial_lr:      LR_START,
            final_lr:        LR_FINAL,
            final_superbatch: global_end,
        }
        .boxed(),
        log_rate: 128,
    };

    train(
        &mut optimiser,
        schedule,
        ReadMapLoader::new(data_loader, mapper, DATA_THREADS as u8),
        |_, _, _| {},
        |optimiser, step| {
            let sb = step.superbatch();
            if sb.is_multiple_of(SAVE_RATE) || sb == step.final_superbatch() {
                let name = format!("{NET_ID}-{sb}");
                let path = format!("{OUT_DIR}/{name}");
                std::fs::create_dir_all(&path).ok();
                save_to_checkpoint(optimiser, &saved_format, &path);
                println!("Saved [{name}]");
            }
        },
    )
    .unwrap_or_else(|e| die(format!("training error: {e:?}")));

    // ---- verify completion ----
    let reached = latest_checkpoint().map_or(0, |(_, n)| n);
    if reached < end_superbatch {
        die(format!(
            "session ended early: last checkpoint is superbatch {reached}, expected {end_superbatch}. \
             Check the log for a data-loader panic (e.g. a truncated binpack)."
        ));
    }

    let _ = std::fs::remove_file(session_path());

    if end_superbatch >= global_end {
        println!(
            "done -- {TOTAL_PASSES}-pass plan complete at superbatch {global_end}.\n\
             net: {OUT_DIR}/{NET_ID}-{global_end}/quantised.bin"
        );
    } else {
        println!("session done at {end_superbatch}/{global_end}. run the next binpack to continue.");
    }
}
