#!/usr/bin/env bash
#
set -eux


funct () {
  PARAM=${1:-1}
  python3 view_glitch.py --id=$PARAM --match\
    --k-path=/home/gautham/stuff/CSAFE/Shoeprints/presentations/2023-aug-23-IAI/images/149_R_00_1.tiff\
    --q-path=/home/gautham/stuff/CSAFE/Shoeprints/presentations/2023-aug-23-IAI/images/149_R_10_2.tiff\
    --eps1=0.5 --alpha=2 --no-flip-k\
    --etor=AKAZE --aligner="kabsch"\
    --output="./plots" --save   
}

funct 1
funct 2
funct 3
