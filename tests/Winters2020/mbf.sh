#!/usr/bin/env python
#cat winters2020.txt | while read name; do echo tql -name \'$name\' -v -c long -lc qlp -sr 30 -s; done > winters_qlp.batch
cat winters2020.txt | while read name; do echo tql -name \'$name\' -v -sr 30 -s; done > winters.batch
