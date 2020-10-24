fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-lib=astcwrap");
    println!("cargo:rustc-link-search=target/misc");
}
