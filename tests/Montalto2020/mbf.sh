#!/usr/bin/env/sh
ifp='Montalto2020'
awk -F "," '{print $1 }' diamante_catalog_non_toi.csv > $ifp'.txt'
ofp=$ifp'_diamante'
sr=30
cat $ifp.txt | while read tic; do echo tql -tic $tic -v -c long -lc diamante -sr $sr -s -img -f; done > $ofp.batch
#cat $ifp.txt | while read tic; do echo tql -tic $tic -v -sr $sr -s -img -f; done > $ofp.batch
echo 'check: cat '$ofp'.batch'
echo 'test: cat '$ofp'.batch | sed -n 1p | sh'
echo 'run: cat '$ofp'.batch | parallel 2>&1 | tee ' $ofp'.log'
