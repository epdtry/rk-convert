#!/bin/bash

make -C deps/astc-encoder/Source VEC=neon BUILD=debug

if [[ src/astcwrap.cpp -nt target/misc/libastcwrap.a ]]; then
    mkdir -p target/misc
    g++ -c src/astcwrap.cpp -o target/misc/astcwrap.o -g \
        -I deps/astc-encoder/Source
    g++ -shared -o target/misc/libastcwrap.so \
        deps/astc-encoder/Source/*.o \
        target/misc/astcwrap.o
fi

