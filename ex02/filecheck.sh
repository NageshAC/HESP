#! /usr/bin/bash -l
comp1="./out/"
comp2='./Question/output/attract/out_5/'
EXT='.out'
op1="./"op1${EXT}
echo $op1
op2=${comp1}op2${EXT}
yourfilenames=$(ls $comp2*$EXT)

for eachfile in $yourfilenames
do
   TMP="${eachfile#$comp2}"
   file1=$comp1$TMP
   file2=$comp2$TMP
   echo $TMP
   
   if [ ! -f $file1 ]
   then
   	echo "$TMP does not exists!"
   else
   	grep -xvFf $file1 $file2 > $op1
   	grep -xvFf $file2 $file1 > $op2
   	
   	if [[ ! -s $op1 ]] && [[ ! -s $op2 ]]
   	then
   		echo "The files $file1 and $file2 are EQUAL!"
   	else
   		echo "The files $file1 and $file2 are NOT EQUAL!"
   		echo "Differences: "
   		cat $op1
   		cat $op2
   	fi
   	rm $op1 $op2
   fi
done
