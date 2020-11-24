#!/usr/bin/env python
#cat schneider2019.txt | while read name; do echo tql -name \'$name\' -v -c long -lc qlp -sr 30 -s -img; done > schneider2019_qlp.batch
cat schneider2019.txt | while read name; do echo tql -name \'$name\' -v -sr 30 -s -img; done > schneider2019.batch
