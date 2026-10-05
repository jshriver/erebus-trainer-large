//! Resumable binpack loader.
//!
//! A copy of bullet's `SfBinpackLoader` with two additions:
//!   - it can start partway through the file list, at a byte offset inside a file;
//!   - it publishes how far training has got through the data (file index + byte
//!     offset), so the save callback can record it next to each checkpoint.
//!
//! The reader tags each batch of chunks with the file position just after it, and
//! that tag travels down the pipeline behind the data.  The published position is
//! the end of the last *fully consumed* shuffle buffer, so it never runs ahead of
//! training: a resume re-reads at most one shuffle buffer (~67M positions at
//! 4096 MiB) and never skips data.  Offsets are always chunk boundaries, so resuming
//! is a plain seek with no decoding.

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    sync::{
        Arc, Mutex,
        mpsc::{self, SyncSender},
    },
    thread,
};

use bullet_lib::{
    game::formats::bulletformat::ChessBoard,
    value::loader::sfbinpack::{Color, PieceType, TrainingDataEntry},
};
use bullet_trainer::reader::DataReader;
use oorandom::Rand64;
use sfbinpack::{ChunkReader, read_chunk_into};

/// (index into the file list, byte offset of the next unread chunk).
pub type ReadPos = (usize, u64);

#[derive(Clone)]
pub struct ResumableBinpackLoader<T: Fn(&TrainingDataEntry) -> bool> {
    file_paths: Vec<String>,
    start: ReadPos,
    buffer_size: usize,
    threads: usize,
    filter: T,
    progress: Arc<Mutex<ReadPos>>,
}

impl<T: Fn(&TrainingDataEntry) -> bool> ResumableBinpackLoader<T> {
    pub fn new(
        paths: &[&str],
        start: ReadPos,
        buffer_size_mb: usize,
        threads: usize,
        filter: T,
        progress: Arc<Mutex<ReadPos>>,
    ) -> Self {
        *progress.lock().unwrap() = start;
        Self {
            file_paths: paths.iter().map(|x| x.to_string()).collect(),
            start,
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<ChessBoard>() / 2,
            threads,
            filter,
            progress,
        }
    }
}

/// Opens `path` positioned at `offset`, falling back to the start of the file if
/// `offset` is not a chunk boundary.  Returns `None` if `offset` is at/after EOF.
fn open_at(path: &str, offset: u64) -> Option<BufReader<File>> {
    let mut reader = BufReader::new(File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}")));
    if offset == 0 {
        return Some(reader);
    }

    let len = reader.get_ref().metadata().map(|m| m.len()).unwrap_or(0);
    if offset >= len {
        return None;
    }

    let mut magic = [0u8; 4];
    let ok = reader.seek(SeekFrom::Start(offset)).is_ok() && reader.read_exact(&mut magic).is_ok() && &magic == b"BINP";
    if ok {
        reader.seek(SeekFrom::Start(offset)).unwrap();
        println!("data          : resuming {path} at byte {offset} of {len} ({:.1}%)", 100.0 * offset as f64 / len as f64);
    } else {
        eprintln!("erebus-trainer-large: warning: byte {offset} of {path} is not a chunk start; reading from the beginning");
        reader.seek(SeekFrom::Start(0)).unwrap();
    }
    Some(reader)
}

/// Converted positions, or a marker that everything read before `ReadPos` has
/// already been sent down the channel.
enum Converted {
    Data(Vec<ChessBoard>),
    Mark(ReadPos),
}

impl<T> DataReader<ChessBoard> for ResumableBinpackLoader<T>
where
    T: Fn(&TrainingDataEntry) -> bool + Clone + Send + Sync + 'static,
{
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let file_paths = self.file_paths.clone();
        let (start_file, start_offset) = self.start;
        let progress = self.progress.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let filter = self.filter.clone();
        let reader_buffer_size = threads.max(1);

        let (reader_sender, reader_receiver) = mpsc::sync_channel::<(Vec<Vec<u8>>, ReadPos)>(4);
        let (reader_msg_sender, reader_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut games = Vec::new();
            let mut first_pass = true;

            'dataloading: loop {
                for (idx, file_path) in file_paths.iter().enumerate() {
                    let offset = if first_pass {
                        if idx < start_file { continue; }
                        if idx == start_file { start_offset } else { 0 }
                    } else {
                        0
                    };

                    let Some(mut reader) = open_at(file_path, offset) else { continue };

                    let mut chunk = Vec::new();

                    while read_chunk_into(&mut reader, &mut chunk).unwrap() {
                        games.push(std::mem::take(&mut chunk));

                        if games.len() == reader_buffer_size {
                            let pos = (idx, reader.stream_position().unwrap_or(0));
                            if reader_msg_receiver.try_recv().unwrap_or(false) || reader_sender.send((games, pos)).is_err() {
                                break 'dataloading;
                            }

                            games = Vec::new();
                        }
                    }

                    // Whole file handed off: next read is the start of the following file.
                    let pos = ((idx + 1) % file_paths.len(), 0);
                    if reader_msg_receiver.try_recv().unwrap_or(false)
                        || reader_sender.send((std::mem::take(&mut games), pos)).is_err()
                    {
                        break 'dataloading;
                    }
                }

                first_pass = false;
            }
        });

        let (converted_sender, converted_receiver) = mpsc::sync_channel::<Converted>(4 * threads);
        let (converted_msg_sender, converted_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let filter = &filter;
            let mut should_break = false;
            'dataloading: while let Ok((chunks, pos)) = reader_receiver.recv() {
                if should_break || converted_msg_receiver.try_recv().unwrap_or(false) {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                // convert_buffer joins its workers before returning, so the marker
                // is queued strictly after all of this batch's positions.
                should_break = convert_buffer(threads, &converted_sender, &chunks, filter)
                    || converted_sender.send(Converted::Mark(pos)).is_err();

                if should_break {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut shuffle_buffer = Vec::with_capacity(buffer_size);
            // Position after the last batch fully inside a buffer handed to training.
            let mut marked = None;
            // That position for the buffer currently being trained on.
            let mut in_training = None;

            'dataloading: while let Ok(msg) = converted_receiver.recv() {
                let converted = match msg {
                    Converted::Mark(pos) => {
                        marked = Some(pos);
                        continue;
                    }
                    Converted::Data(converted) => converted,
                };

                for entry in converted {
                    shuffle_buffer.push(entry);

                    if shuffle_buffer.len() == buffer_size {
                        shuffle(&mut shuffle_buffer);

                        if buffer_msg_receiver.try_recv().unwrap_or(false)
                            || buffer_sender.send(shuffle_buffer).is_err()
                        {
                            converted_msg_sender.send(true).unwrap();
                            break 'dataloading;
                        }

                        // The rendezvous send returning means training has finished the
                        // previous buffer, so everything up to its mark is trained.
                        if let Some(pos) = in_training {
                            *progress.lock().unwrap() = pos;
                        }
                        in_training = marked;

                        shuffle_buffer = Vec::with_capacity(buffer_size);
                    }
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
            if f(&shuffle_buffer) {
                buffer_msg_sender.send(true).unwrap();
                break 'dataloading;
            }
        }
    }
}

fn convert_buffer<T>(threads: usize, sender: &SyncSender<Converted>, chunks: &[Vec<u8>], filter: &T) -> bool
where
    T: Fn(&TrainingDataEntry) -> bool + Sync,
{
    if chunks.is_empty() {
        return false;
    }

    let chunk_size = chunks.len().div_ceil(threads);
    let mut should_break = false;

    thread::scope(|s| {
        let mut handles = Vec::new();

        for chunk_group in chunks.chunks(chunk_size) {
            let this_sender = sender.clone();
            let handle = s.spawn(move || {
                let mut buffer = Vec::new();

                for chunk in chunk_group {
                    let mut reader = ChunkReader::default();

                    while reader.has_next(chunk) {
                        let entry = reader.next(chunk);
                        if filter(&entry) {
                            buffer.push(convert_to_bulletformat(&entry));
                        }
                    }
                }

                this_sender.send(Converted::Data(buffer)).is_err()
            });

            handles.push(handle);
        }

        for handle in handles {
            if handle.join().unwrap() {
                should_break = true;
            }
        }
    });

    should_break
}

fn convert_to_bulletformat(entry: &TrainingDataEntry) -> ChessBoard {
    let mut bbs = [0; 8];

    let stm = usize::from(entry.pos.side_to_move().ordinal());
    let pc_bb =
        |pt| entry.pos.pieces_bb_color(Color::Black, pt).bits() | entry.pos.pieces_bb_color(Color::White, pt).bits();

    bbs[0] = entry.pos.pieces_bb(Color::White).bits();
    bbs[1] = entry.pos.pieces_bb(Color::Black).bits();
    bbs[2] = pc_bb(PieceType::Pawn);
    bbs[3] = pc_bb(PieceType::Knight);
    bbs[4] = pc_bb(PieceType::Bishop);
    bbs[5] = pc_bb(PieceType::Rook);
    bbs[6] = pc_bb(PieceType::Queen);
    bbs[7] = pc_bb(PieceType::King);

    let mut score = entry.score;
    let mut result = f32::from(1 + entry.result) / 2.0;

    if stm > 0 {
        score = -score;
        result = 1.0 - result;
    }

    ChessBoard::from_raw(bbs, stm, score, result).expect("Binpack must be malformed!")
}

fn shuffle(data: &mut [ChessBoard]) {
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros();
    let mut rng = Rand64::new(seed);

    for i in (0..data.len()).rev() {
        let idx = rng.rand_range(0..i as u64 + 1) as usize;
        data.swap(idx, i);
    }
}

