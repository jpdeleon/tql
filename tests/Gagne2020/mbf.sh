#!/usr/bin/env/sh
ifp='muTau_coords'
ofp=$ifp'_sc'
sr=3
cat $ifp.txt | awk -F "," '{print $1" "$2}' | while read x y; do echo tql -coords $x $y -s -v -img -sr $sr -f -o $ofp; done > $ofp.batch
echo 'check: cat '$ofp'.batch'
echo 'run: cat '$ofp'.batch | parallel 2>&1 | tee ' $ofp'.log'
