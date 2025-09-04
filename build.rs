extern crate cc;

use cc::Build;

fn main() {
    println!("cargo:rerun-if-changed=src/tap.c");
    Build::new()
        .file("src/tap.c")
        .warnings(true)
        .compile("tap");
}