#!/usr/bin/env python
import numpy as np
import chronos as cr

tois = cr.get_tois(clobber=True, remove_FP=True)
#spoc = tois.query("Source=='spoc'")
#qlp = tois.query("Source!='spoc'")

text = []
toiids = []
for k,row in tois.iterrows():
    toi = row.TOI
    toiid = str(toi).split('.')[0]
    t = f"tql -toi {toiid} -s -v "
    if row.Source=='spoc':
        if toiid not in toiids:
            text.append(t)
    else:
        t += "-c long -a square -r 1"
        if toiid not in toiids:
            text.append(t)
    toiids.append(toiid)

fp = 'tql_toi.batch'
np.savetxt(fp, text, fmt='%s')
print("Saved: ", fp)
