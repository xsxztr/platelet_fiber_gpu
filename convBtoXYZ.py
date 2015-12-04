#! /usr/bin/python

# This script replicates a given test with varied parameters
# creating unique submit scripts and executing the submission to the CRC SGE queue

# arugments should be passed filename as [1] and total SCEs as [2]
#  run command as
# python convBtoXYZ.py cells.data 10000

# Imports
import os
import shutil
import string
import sys

newDir='xyzFiles'
os.mkdir(newDir)
SCE=int(sys.argv[2])
file3 = 'cells.txt'

fileName= sys.argv[1]
#fileName2= fileName + '.orig'
#shutil.copyfile( fileName , fileName2 )

os.system('od -fv ' + fileName + ' > ' + file3)

numlines = 0
for line in open(file3): numlines+=1

numlines -=1

frames = numlines / SCE

print frames
print numlines

os.system('sed -i \'s/^.........//g \' ' + file3)
os.system('sed -i \'s/............$//g \' ' + file3)

#create the indiviudal xyz files
fin = open(file3)
for i in range(frames):
    xyzfile = newDir + '/pos_' + str(i) + '.xyz'
    fout = open(xyzfile, "w")
    print >> fout, SCE
    print >> fout, "   SCE"
    for j in range(SCE):
        str1 = fin.readline()
        xyzstr = str1.split()
        output = '1 '  + xyzstr[0] + ' ' + xyzstr[1] + ' ' + xyzstr[2] + '\n'
        fout.write(output )
    fout.close()
     

