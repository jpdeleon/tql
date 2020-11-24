#!/usr/bin/env python
#cat binks2020.txt | while read name; do echo tql -name \'$name\' -v -c long -lc qlp -sr 30 -s -img; done > binks_qlp.batch
cat binks2020.txt | while read name; do echo tql -name \'$name\' -v -sr 30 -s -img; done > binks.batch
