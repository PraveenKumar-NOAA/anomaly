#
# CADS V3 Test script.
#
# Reima Eresmaa (ECMWF)
# 23/04/20
#----------------------------------------------------------------------------


# Read the sensor name from the command line

if [ $# = 0 ] ; then
echo
echo "   Usage: ./CADS_Test_Script.sh SENSOR"
echo "     where SENSOR is one of the following:"
echo "       AIRS, CRIS, GIIRS, HIRAS, HIRAS2, IASI, IASING, IKFS2, IRS, IRIS"
echo
exit
fi

SENSOR=${1}


# Make links to input and reference output files
rm -f cads_input.dat
gunzip Test_Data/${SENSOR}.input.gz
ln -s Test_Data/${SENSOR}.input cads_input.dat

rm -f reference_output.dat
gunzip Test_Data/${SENSOR}.output.gz
ln -s Test_Data/${SENSOR}.output reference_output.dat


# Check that input and output files exist and are non-zero size
if [ ! -s cads_input.dat ] ; then
echo
echo "Input file not found"
echo "CADS test script FAILED to process "${SENSOR}" data!"
echo
exit
elif [ ! -s reference_output.dat ] ; then
echo
echo "Reference output file not found"
echo "CADS test script FAILED to process "${SENSOR}" data!"
echo
exit
fi


# Clean compile
echo
cd ../src
rm -f *.o *.mod ../CADS
make


# Make links to namelist files in the working directory
cd ../Test_Runs
rm -f *.NL
ln -s ../namelist/*.NL .


# Run the CADS executable
../CADS


echo

# Check that output was produced
if [ ! -s cads_output.dat ] ; then
echo "No output produced."
echo "The test run using the "${SENSOR}" data is UNSUCCESSFUL."

else

# Compare the output with the reference
rm -f result.txt
diff -q cads_output.dat reference_output.dat > result.txt

if [ -s result.txt ] ; then
cat result.txt
echo "The test run using "${SENSOR}" data is UNSUCCESSFUL."
else
echo "Output matches with the reference."
echo "The test run using "${SENSOR}" data is SUCCESSFUL."
rm -f cads_input.dat
rm -f reference_output.dat
gzip Test_Data/${SENSOR}.input
gzip Test_Data/${SENSOR}.output
fi

fi # ... Check that output was produced ...

echo


# Clean the working directory
rm -f *.NL
rm -f result.txt

exit
