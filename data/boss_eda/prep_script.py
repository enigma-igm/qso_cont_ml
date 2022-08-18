'''Script for running the routine in prep.py.'''

from prep import prepRedshiftLuminosityFile

wave_min = 1000.
wave_max = 1970.
SN_min = 30.

prepRedshiftLuminosityFile(wave_min, wave_max, SN_min)