#!/usr/bin/env/sh
ifp='schneider2019'
ofp=$ifp'_sc'
sr=30
#cat $ifp.txt | while read name; do echo tql -name \'$name\' -v -c long -lc qlp -sr $sr -s -img -f; done > $ofp.batch
cat $ifp.txt | while read name; do echo tql -name \'$name\' -v -sr $sr -s -img -f; done > $ofp.batch
echo 'check: cat '$ofp'.batch'
echo 'test: cat '$ofp'.batch | sed -n 1p | sh'
echo 'run: cat '$ofp'.batch | parallel 2>&1 | tee ' $ofp'.log'
