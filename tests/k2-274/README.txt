
TQL commands
1. Check if planet b is detected in TESS PDCSAP and SAP lightcurves (see blind_search_planet_b):
#pdcsap
$ tql -name 'k2-274' -v -s -img 
#sap
$ tql -name 'k2-274' -v -s -img -lc sap

2. To search for planet c, mask ephemeris of planet b by adding -em flag (see mask_planet_b_search_planet_c):
#pdcsap
$ tql -name 'k2-274' -v -s -img -em 14.1330314 2457145.111261 3.6 
#sap
$ tql -name 'k2-274' -v -s -img -em 14.1330314 2457145.111261 3.6 -lc sap 
