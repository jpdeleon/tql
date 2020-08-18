#!/usr/bin/env python
cat muTau_coords.txt | awk -F "," '{print $1" "$2}' | while read x y; do echo tql -coords $x $y -s -v; done > muTau_Gagne2020.batch
