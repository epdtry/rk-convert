Tools for converting rkengine .rk (model) and .anim (animation) files.
Also includes a tool for extracting .ark files.

# Building

Requires [Rust](https://rustup.rs/) and a C++ compiler (for use with
[`cc`](https://lib.rs/crates/cc)).

To build:

```sh
cargo build --release
```

# Usage

To extract a .ark file:

```sh
cargo run --release --bin unark -- path/to/file.ark
```

This will extract the contents of the .ark into the current directory.
You should probably run this in a new, empty directory.

To convert a model to glTF:

```sh
# Without animation:
cargo run --release --bin model-to-gltf -- path/to/model.rk
# With animation:
cargo run --release --bin model-to-gltf -- path/to/model.rk path/to/anim.csv
cargo run --release --bin model-to-gltf -- path/to/model.rk path/to/anim.anim
```

For a model named `XXX_YYY_lodN.rk`, there will usually be accompanying
animation files `XXX.anim`, `XXX.xml`, and `XXX.csv` that define animations for
that category of models.  Pass the `.csv` file if there is one - this file
divides the frames of the `.anim` file into separate, named animations.
Otherwise, pass the `.anim` file to get all frames as a single giant animation.
For models that aren't animated, don't pass an animation file at all.

The output will be saved as `out.glb` in glTF binary format.
