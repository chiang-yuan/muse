#!/bin/bash

TAG=20.15.1
SRC_DIR=$HOME/.local/src
mkdir -p $SRC_DIR

cd $SRC_DIR

wget "https://github.com/m3g/packmol/archive/refs/tags/v$TAG.tar.gz" 
tar -xvf v$TAG.tar.gz

rm v$TAG.tar.gz

cd packmol-$TAG

make

BIN_DIR=$HOME/.local/bin

mv packmol $BIN_DIR