use std::fs;
use std::io;
use cc::Build;

fn main() -> io::Result<()> {
    let mut b = Build::new();
    b.cpp(true);

    b.include("deps/astc-encoder/Source");
    for entry in fs::read_dir("deps/astc-encoder/Source")? {
        let entry = entry?;
        let path = entry.path();
        if path.extension() != Some("cpp".as_ref()) {
            continue;
        }
        b.file(&path);
    }

    b.file("src/astcwrap.cpp");
    b.define("ASTCENC_SSE", "0");
    b.define("ASTCENC_AVX", "0");
    b.define("ASTCENC_POPCNT", "0");
    b.define("ASTCENC_VECALIGN", "16");
    b.opt_level(3);
    b.compile("astcwrap");

    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}
