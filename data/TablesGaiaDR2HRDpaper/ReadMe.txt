I/?  Gaia DR2: Observational Hertzsprung-Russell diagrams (Babusiaux+ 2018)
================================================================================
Gaia Data Release 2: Observational Hertzsprung-Russell diagrams
    Gaia Collaboration, Babusiaux C., van Leeuwen F., Barstow, M. A.,
    Jordi, C., Vallenari, A. + 
    <Astronomy & Astrophysics, 2018>
================================================================================
ADC_Keywords: parallaxes; Hertzsprung-Russell and C-M diagrams; solar 
neighbourhood: Stars: evolution

Description:
We have determined the membership of 46 open clusters. For the nine
clusters within 250 pc we determined optimised parallaxes based on the
combined information extracted from the measured parallax and proper
motion values. These clusters are : in Table A1a: alphaPer,  Blanco1, ComaBer,
Hyades,  IC2391,  IC2602, NGC2451A,  Pleiades,  Praesepe. The
remaining 37 clusters are in Table A1b:  Coll140, IC4651,  IC4665,
 IC4725,  IC4756, NGC0188, NGC0752, NGC0869, NGC0884, NGC1039,
NGC1901, NGC2158, NGC2168, NGC2232, NGC2323, NGC2360, NGC2422,
NGC2423, NGC2437, NGC2447, NGC2516, NGC2547, NGC2548, NGC2682,
NGC3228, NGC3532, NGC6025, NGC6281, NGC6405, NGC6475, NGC6633,
NGC6774, NGC6793, NGC7092,  Stock2, Trump02, Trump10.

File Summary:
--------------------------------------------------------------------------------
 FileName    Lrecl    Records    Explanations
--------------------------------------------------------------------------------
ReadMe          80          .    This file
TableA1a.csv    84       5404    Nine open clusters within 250 pc
TableA1b.csv    66      38926    37 open clusters beyond 250 pc
--------------------------------------------------------------------------------

Byte-by-byte Description of file: TableA1a.csv, TableA1b.csv
Both files have a single header record with column description
--------------------------------------------------------------------------------
   Bytes Format    Units   Label          Explanations
--------------------------------------------------------------------------------
   1- 22  long     ---     DR2 SourceId   Source identifier in Gaia DR2
  24- 32  String   ---     Cluster        Cluster identifier
  36- 44  double   degr.   alpha          Right Ascension (ICRS), epoch 2015.5
  48- 56  double   degr.   delta          Declination (ICRS), epoch 2015.5
  61- 66  float    magn.   Gmag           Gaia G mean magnitude
  70- 75  float    mas     parallax       Optimized parallax (only in TableA1a)
  80- 84  float    mas     parallax_error Uncertainty on optimized parallax
                                          (only in TableA1a)
--------------------------------------------------------------------------------
Note on optimized parallaxes: Obtained for nearby clusters based on
the combined information extracted from the measured parallax and
proper motion values
--------------------------------------------------------------------------------

Author's address:
    Floor van Leeuwen <fvl@ast.cam.ac.uk>
================================================================================
(End)     Floor van Leeuwen [IoA - Cambridge University]             16-May-2018

