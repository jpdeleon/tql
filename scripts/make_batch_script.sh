#!/usr/bin/sh
cat ../data/toi_fast_rotators.txt | while read toi; do echo ./make_tql -toi=$toi.01 -v -pld -s -o=../rotators --cadence=long --aper_mask=threshold; done > fast_rotators_ffi.batch
