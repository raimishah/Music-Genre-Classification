"""Converts .au to .wav file using the sox tool.
IN: Paths to directory consisting of .au files.
OUT: Complete conversion of files in respective directories passed as inputs.
Run instructions:
python convert-to-wav.py path_dir_1 path_dir_2 ... path_dir_N
Where path_dir_i consists of .au files to be converted
NOTE: 
1. .au files will be DELETED. Make sure you have a backup of it to be safe.
2. Use ONLY absolute paths.
3. Run as per instruction or will lead to disastrous results
"""

import sys
import os

current_genre_path = '.'

for file in os.listdir(current_genre_path):
	# SOX
	os.system("sox " + str(file) + " " + str(file[:-3]) + ".wav")
	
# delete .au from current dir
os.system("rm *.au")

