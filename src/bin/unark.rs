use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use rk_convert::ark::ArkFile;

fn io_main() -> io::Result<()> {
    let args = env::args_os().collect::<Vec<_>>();
    assert!(args.len() == 2, "usage: {} <file.ark>", args[0].to_string_lossy());
    let mut ark = ArkFile::new(File::open(&args[1])?);
    let files = ark.read_metadata()?;
    println!("found {} files", files.len());
    for entry in files {
        let path: PathBuf;
        if entry.directory.len() > 0 {
            fs::create_dir_all(&entry.directory)?;
            path = Path::new(&entry.directory).join(&entry.filename);
        } else {
            path = Path::new(&entry.filename).to_owned();
        }
        eprintln!("{}", path.display());
        let mut out_file = File::create(path)?;
        let content = ark.read_file(&entry)?;
        out_file.write_all(&content)?;
    }
    Ok(())
}

fn main() {
    io_main().unwrap();
}
