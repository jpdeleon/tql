#!/usr/bin/env/sh
ifp='zerjal2017'
ofp=$ifp'_sc'
sr=3
cat $ifp.txt | awk -F "," '{print $1" "$2}' | while read x y; do echo tql -coords $x $y -s -v -img -sr $sr -f; done > $ofp.batch
echo 'check: cat '$ofp'.batch'
echo 'test: cat '$ofp'.batch | sed -n 1p | sh'
echo 'run: cat '$ofp'.batch | parallel -j N 2>&1 | tee ' $ofp'.log'
