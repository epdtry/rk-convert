use std::io;
use std::path::Path;
use csv;

pub struct AnimRange {
    pub name: String,
    pub start: usize,
    pub end: usize,
    pub frame_rate: u32,
}

pub fn read_anim_csv(path: impl AsRef<Path>) -> io::Result<Vec<AnimRange>> {
    let mut ranges = Vec::new();
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)?;
    for row in reader.deserialize() {
        let (name, start, end, frame_rate): (String, usize, usize, u32) = row?;
        ranges.push(AnimRange { name, start, end, frame_rate });
    }

    Ok(ranges)
}

