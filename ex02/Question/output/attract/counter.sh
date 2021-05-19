#! /usr/bin/bash

OUTFILES=$(ls *.out)
OUTCTR=1

for OUTFILE in $OUTFILES
do
	((OUTCTR++))
done


VTKFILES=$(ls *.vtk)
VTKCTR=1

for VTKFILE in $VTKFILES
do
	((VTKCTR++))
done

echo "No. of .out files = $OUTCTR"
echo "No. of .vtk files = $VTKCTR"


