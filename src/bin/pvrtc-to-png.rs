use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use rkengine::pvr::PvrFile;

fn io_main() -> io::Result<()> {
    let args = env::args_os().collect::<Vec<_>>();
    assert!(args.len() == 3, "usage: {} <file.pvr> <out.png>", args[0].to_string_lossy());
    let mut pf = PvrFile::new(File::open(&args[1])?);
    let mut img = pf.read_image()?;

    let mut out_file = File::create(&args[2])?;
    img.write_raw(out_file)?;

    Ok(())
}

fn main() {
    io_main().unwrap();
}
