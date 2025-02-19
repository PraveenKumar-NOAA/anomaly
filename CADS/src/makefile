# Define directories containing source
SHELL = /bin/sh

# gfortran-specific settings on ECMWF supercomputer
FC = gfortran
CFLAGS = -g

# ifort-specific settings on ECMWF supercomputer
#FC = ifort

# NVIDIA-specific settings on ECMWF supercomputer
#FC = nvfortran

# Allowed SUFFIXES:
.SUFFIXES :
.SUFFIXES :  .F90 .o .mod

SOURCE = CADS_Main.F90

OBJS  = CADS_Module.o \
	CADS_Abort.o \
	CADS_Detect_Aerosol.o \
	CADS_Detect_Cloud.o \
	CADS_Detect_Cloud_Heapsort.o \
	CADS_Detect_Cloud_Imager.o \
	CADS_Detect_Cloud_Scenario.o \
	CADS_Detect_Cloud_Separator.o \
	CADS_Detect_Cloud_Smooth.o \
	CADS_Detect_Land_Sensitivity.o \
	CADS_Detect_Trace_Gas.o \
	CADS_Main.o \
	CADS_Setup_Aerosol.o \
	CADS_Setup_Cloud.o \
	CADS_Setup_Land_Sensitivity.o \
	CADS_Setup_Trace_Gas.o 

MODULES = CADS_Module.o

# Object dependencies:

../CADS : $(OBJS) $(MODULES)
	$(FC) -o ..//CADS $(OBJS)

CADS_Module.o : CADS_Module.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Module.F90

CADS_Abort.o : CADS_Abort.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Abort.F90

CADS_Detect_Aerosol.o : CADS_Module.o CADS_Detect_Aerosol.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Detect_Aerosol.F90

CADS_Detect_Cloud.o : CADS_Module.o CADS_Detect_Cloud.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Detect_Cloud.F90

CADS_Detect_Cloud_Heapsort.o : CADS_Detect_Cloud_Heapsort.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Detect_Cloud_Heapsort.F90

CADS_Detect_Cloud_Imager.o : CADS_Module.o CADS_Detect_Cloud_Imager.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Detect_Cloud_Imager.F90

CADS_Detect_Cloud_Scenario.o : CADS_Module.o CADS_Detect_Cloud_Scenario.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Detect_Cloud_Scenario.F90

CADS_Detect_Cloud_Separator.o : CADS_Module.o CADS_Detect_Cloud_Separator.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Detect_Cloud_Separator.F90

CADS_Detect_Cloud_Smooth.o : CADS_Detect_Cloud_Smooth.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Detect_Cloud_Smooth.F90

CADS_Detect_Land_Sensitivity.o : CADS_Detect_Land_Sensitivity.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Detect_Land_Sensitivity.F90

CADS_Detect_Trace_Gas.o : CADS_Detect_Trace_Gas.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Detect_Trace_Gas.F90

CADS_Main.o : CADS_Main.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Main.F90

CADS_Setup_Aerosol.o : CADS_Module.o CADS_Setup_Aerosol.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Setup_Aerosol.F90

CADS_Setup_Cloud.o : CADS_Module.o CADS_Setup_Cloud.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Setup_Cloud.F90

CADS_Setup_Trace_Gas.o : CADS_Module.o CADS_Setup_Trace_Gas.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Setup_Trace_Gas.F90

CADS_Setup_Land_Sensitivity.o : CADS_Module.o CADS_Setup_Land_Sensitivity.F90
	$(FC) -c $(CFLAGS) $(INCFLAG) CADS_Setup_Land_Sensitivity.F90


#

clean :
	rm -f ${OBJS} *.mod *~
clobber :
	rm -f  ${OBJS} *.mod *~ ../CADS
