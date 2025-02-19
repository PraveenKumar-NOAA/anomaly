SUBROUTINE CADS_Setup_Aerosol

!   This software was developed within the context of the EUMETSAT
!   Satellite Application Facility on Numerical Weather Prediction
!   (NWP SAF), under the Cooperation Agreement dated 7 December 2016,
!   between EUMETSAT and the Met Office, UK, by one or more partners
!   within the NWP SAF. The partners in the NWP SAF are the Met
!   Office, ECMWF, DWD and MeteoFrance.
!
!   Copyright 2020, EUMETSAT, All Rights Reserved.


!   *Aerosol detection setup*
!   J. Letertre-Danczak  ECMWF 06/10/15

!   * PURPOSE *
!   -----------
!   Initialise aerosol detection parameters for advanced infrared sounders.

!   * INTERFACE *
!   -------------
!   CADS_Setup_Aerosol is called from CADS_Main.

!   * METHOD *
!   ----------
!   Default values are assigned to the aerosol detection setup structure.

!   * MODIFICATIONS *
!   -----------------
!   21/12/16   R.Eresmaa   2.3   Import to the NWPSAF CADS V2.3.
!   05/02/19   R.Eresmaa   2.4   Channel-specific flagging.
!                                Explicit KIND specifications.
!   16/04/20   R.Eresmaa   3.0   Include aerosol type recognition, rename.
!   31/01/24   C.Burrows   3.2   Add HIRAS-2 and IRIS

USE CADS_Module, ONLY : JP__Min_Sensor_Index,    &
&                       JP__Max_Sensor_Index,    &
&                       INST_ID_AIRS,            &
&                       INST_ID_CRIS,            &
&                       INST_ID_GIIRS,           &
&                       INST_ID_HIRAS,           &
&                       INST_ID_HIRAS2,          &
&                       INST_ID_IASI,            &
&                       INST_ID_IASING,          &
&                       INST_ID_IKFS2,           &
&                       INST_ID_IRS,             &
&                       INST_ID_IRIS,            &
&                       S__CADS_Setup_Aerosol
IMPLICIT NONE

! Local variables

CHARACTER(LEN=6)  :: CL__InstrumentName
CHARACTER(LEN=20) :: CL__Aerosol_Detection_File

INTEGER(KIND=4) :: J, J__Sensor      ! Loop variables
INTEGER(KIND=4) :: INIU1, IOS

!-----------------------
! Namelist variables
!-----------------------

INTEGER(KIND=4), PARAMETER :: JP__Max_Bands         =   8
INTEGER(KIND=4), PARAMETER :: JP__Max_Aerosol_Chans = 200

INTEGER(KIND=4) :: M__Sensor
INTEGER(KIND=4) :: N__Num_Aerosol_Tests
INTEGER(KIND=4) :: N__Num_Aerosol_Chans(JP__Max_Bands)
INTEGER(KIND=4) :: N__Num_Regression(JP__Max_Bands)
INTEGER(KIND=4) :: N__Aerosol_Chans(JP__Max_Aerosol_Chans,JP__Max_Bands)
REAL(KIND=8)    :: R__Aerosol_TBD(JP__Max_Aerosol_Chans,JP__Max_Bands)
REAL(KIND=8)    :: R__coef_AOD(JP__Max_Aerosol_Chans,JP__Max_Bands)
INTEGER(KIND=4) :: N__Mean_Aerosol_Chans ! number of channels to compute mean
REAL(KIND=8)    :: R__Rank_Thres_Coeff(3)
REAL(KIND=8)    :: R__Unclassified_Thres
REAL(KIND=8)    :: R__Land_Fraction_Thres

! Namelist

NAMELIST / Aerosol_Detect_Coeffs / M__Sensor, N__Num_Aerosol_Tests, &
&        N__Num_Aerosol_Chans, N__Aerosol_Chans, N__Mean_Aerosol_Chans, &
&        R__Aerosol_TBD, N__Num_Regression, R__coef_AOD, &
&        R__Rank_Thres_Coeff, R__Unclassified_Thres, R__Land_Fraction_Thres


INCLUDE 'CADS_Abort.intfb'


!============================================================================

WRITE (*,'(A)') ''
WRITE (*,'(A)') 'Setting up the aerosol detection ...'

!============================================================================
!   Loop through sensors setting up aerosol detection
!============================================================================

SensorLoop : DO J__Sensor = JP__Min_Sensor_Index, JP__Max_Sensor_Index

   SELECT CASE (J__Sensor)

   CASE(INST_ID_AIRS)
      !====================
      ! Set up AIRS
      !====================

      CL__InstrumentName='AIRS'
      CL__Aerosol_Detection_File = 'AIRS_AERDET.NL'

      N__Num_Aerosol_Tests = 3
      N__Num_Aerosol_Chans(:) = 0
      N__Num_Aerosol_Chans(1:N__Num_Aerosol_Tests) = (/ 4, 4, 4 /)
      N__Aerosol_Chans(:,:) = 0
      N__Aerosol_Chans(1,1:N__Num_Aerosol_Chans(1)) = &
&           (/  950, 1285, 1206, 1290 /)
      N__Aerosol_Chans(2,1:N__Num_Aerosol_Chans(2)) = &
&           (/  950, 1285, 1206, 1290 /)
      N__Aerosol_Chans(3,1:N__Num_Aerosol_Chans(3)) = &
&           (/ 1206,  528, 1206, 1290 /)
      N__Mean_Aerosol_Chans = 1
      R__Aerosol_TBD(:,:) = 0.0
      R__Aerosol_TBD(1,1:2) = (/ 1.2, -5.0 /)
      R__Aerosol_TBD(2,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(3,1:2) = (/ -7.5, -5.0 /)
      N__Num_Regression(:) = 0
      N__Num_Regression(1:N__Num_Aerosol_Tests) = (/ 3, 3, 3 /)
      R__coef_AOD(:,:) = 0.0
      R__coef_AOD(1,1:N__Num_Regression(1)) = &
&           (/ -0.06, -0.001, 0.002 /)
      R__Rank_Thres_Coeff(1:3) = (/ -0.01, 2.1, -3.9 /)
      R__Unclassified_Thres = 0.4
      R__Land_Fraction_Thres = 0.5

   CASE(INST_ID_IASI)
      !====================
      ! Set up IASI
      !====================

      CL__InstrumentName='IASI'
      CL__Aerosol_Detection_File = 'IASI_AERDET.NL'

      N__Num_Aerosol_Tests = 3
      N__Num_Aerosol_Chans(:) = 0
      N__Num_Aerosol_Chans(1:N__Num_Aerosol_Tests) = (/ 4, 4, 4 /)
      N__Aerosol_Chans(:,:) = 0
      N__Aerosol_Chans(1,1:N__Num_Aerosol_Chans(1)) = &
&           (/ 1340, 2348, 1782, 2356 /)
      N__Aerosol_Chans(2,1:N__Num_Aerosol_Chans(2)) = &
&           (/ 2093, 2348, 1782, 2356 /)
      N__Aerosol_Chans(3,1:N__Num_Aerosol_Chans(3)) = &
&           (/ 1782, 2356, 1782,  753 /)
      N__Mean_Aerosol_Chans = 11
      R__Aerosol_TBD(:,:) = 0.0
      R__Aerosol_TBD(1,1:2) = (/ 0.2, -1.55 /)
      R__Aerosol_TBD(2,1:2) = (/ -1.5, 999.9 /)
      R__Aerosol_TBD(3,1:2) = (/ -1.8, -2.0 /)
      N__Num_Regression(:) = 0
      N__Num_Regression(1:N__Num_Aerosol_Tests) = (/ 3, 3, 3 /)
      R__coef_AOD(:,:) = 0.0
      R__coef_AOD(1,1:N__Num_Regression(1)) = &
&           (/ -0.18, -0.1, 0.01 /)
      R__Rank_Thres_Coeff(1:3) = (/ -0.01, 2.1, -3.9 /)
      R__Unclassified_Thres = 0.4
      R__Land_Fraction_Thres = 0.5

   CASE(INST_ID_CRIS)
      !====================
      ! Set up CRIS
      !====================

      CL__InstrumentName='CRIS'
      CL__Aerosol_Detection_File = 'CRIS_AERDET.NL'

      N__Num_Aerosol_Tests = 3
      N__Num_Aerosol_Chans(:) = 0
      N__Num_Aerosol_Chans(1:N__Num_Aerosol_Tests) = (/ 4, 4, 4 /)
      N__Aerosol_Chans(:,:) = 0
      N__Aerosol_Chans(1,1:N__Num_Aerosol_Chans(1)) = &
&           (/ 529, 749, 706, 752 /)
      N__Aerosol_Chans(2,1:N__Num_Aerosol_Chans(2)) = &
&           (/ 529, 749, 706, 752 /)
      N__Aerosol_Chans(3,1:N__Num_Aerosol_Chans(3)) = &
&           (/ 706, 752, 706, 294 /)
      N__Mean_Aerosol_Chans = 5
      R__Aerosol_TBD(:,:) = 0.0
      R__Aerosol_TBD(1,1:2) = (/ 0.3, -1.3 /)
      R__Aerosol_TBD(2,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(3,1:2) = (/ -2.0, -0.8 /)
      N__Num_Regression(:) = 0
      N__Num_Regression(1:N__Num_Aerosol_Tests) = (/ 3, 3, 3 /)
      R__coef_AOD(:,:) = 0.0
      R__coef_AOD(1,1:N__Num_Regression(1)) = &
&           (/ -0.09, -0.06, 0.01 /)
      R__Rank_Thres_Coeff(1:3) = (/ -0.01, 2.1, -3.9 /)
      R__Unclassified_Thres = 0.4
      R__Land_Fraction_Thres = 0.5

   CASE(INST_ID_IRS)
      !====================
      ! Set up IRS
      !====================

      CL__InstrumentName='IRS'
      CL__Aerosol_Detection_File = 'IRS_AERDET.NL'

      N__Num_Aerosol_Tests = 3
      N__Num_Aerosol_Chans(:) = 0
      N__Num_Aerosol_Chans(1:N__Num_Aerosol_Tests) = (/ 4, 4, 4 /)
      N__Aerosol_Chans(:,:) = 0
      N__Aerosol_Chans(1,1:N__Num_Aerosol_Chans(1)) = &
&           (/ 449, 811, 626, 811 /)
      N__Aerosol_Chans(2,1:N__Num_Aerosol_Chans(2)) = &
&           (/ 750, 811, 626, 811 /)
      N__Aerosol_Chans(3,1:N__Num_Aerosol_Chans(3)) = &
&           (/ 626, 811, 626, 811 /)
      N__Mean_Aerosol_Chans = 5
      R__Aerosol_TBD(:,:) = 0.0
      R__Aerosol_TBD(1,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(2,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(3,1:2) = (/ -999.9, -999.9 /)
      N__Num_Regression(:) = 0
      N__Num_Regression(1:N__Num_Aerosol_Tests) = (/ 3, 3, 3 /)
      R__coef_AOD(:,:) = 0.0
      R__coef_AOD(1,1:N__Num_Regression(1)) = &
&           (/ 0.0, 0.0, 0.0 /)
      R__Rank_Thres_Coeff(1:3) = (/ -0.01, 2.1, -3.9 /)
      R__Unclassified_Thres = 0.4
      R__Land_Fraction_Thres = 0.5

   CASE(INST_ID_IASING)
      !====================
      ! Set up IASING
      !====================

      CL__InstrumentName='IASING'
      CL__Aerosol_Detection_File = 'IASING_AERDET.NL'

      N__Num_Aerosol_Tests = 3
      N__Num_Aerosol_Chans(:) = 0
      N__Num_Aerosol_Chans(1:N__Num_Aerosol_Tests) = (/ 4, 4, 4 /)
      N__Aerosol_Chans(:,:) = 0
      N__Aerosol_Chans(1,1:N__Num_Aerosol_Chans(1)) = &
&           (/ 2679, 4695, 3563, 4711 /)
      N__Aerosol_Chans(2,1:N__Num_Aerosol_Chans(2)) = &
&           (/ 4185, 4695, 3563, 4711 /)
      N__Aerosol_Chans(3,1:N__Num_Aerosol_Chans(3)) = &
&           (/ 3563, 4711, 3563, 1505 /)
      N__Mean_Aerosol_Chans = 21
      R__Aerosol_TBD(:,:) = 0.0
      R__Aerosol_TBD(1,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(2,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(3,1:2) = (/ -999.9, -999.9 /)
      N__Num_Regression(:) = 0
      N__Num_Regression(1:N__Num_Aerosol_Tests) = (/ 3, 3, 3 /)
      R__coef_AOD(:,:) = 0.0
      R__coef_AOD(1,1:N__Num_Regression(1)) = &
&           (/ 0.0, 0.0, 0.0 /)
      R__Rank_Thres_Coeff(1:3) = (/ -0.01, 2.1, -3.9 /)
      R__Unclassified_Thres = 0.4
      R__Land_Fraction_Thres = 0.5

   CASE(INST_ID_HIRAS)
      !====================
      ! Set up HIRAS
      !====================

      CL__InstrumentName='HIRAS'
      CL__Aerosol_Detection_File = 'HIRAS_AERDET.NL'

      N__Num_Aerosol_Tests = 3
      N__Num_Aerosol_Chans(:) = 0
      N__Num_Aerosol_Chans(1:N__Num_Aerosol_Tests) = (/ 4, 4, 4 /)
      N__Aerosol_Chans(:,:) = 0
      N__Aerosol_Chans(1,1:N__Num_Aerosol_Chans(1)) = &
&           (/ 529, 813, 706, 816 /)
      N__Aerosol_Chans(2,1:N__Num_Aerosol_Chans(2)) = &
&           (/ 529, 813, 706, 816 /)
      N__Aerosol_Chans(3,1:N__Num_Aerosol_Chans(3)) = &
&           (/ 706, 816, 706, 294 /)
      N__Mean_Aerosol_Chans = 5
      R__Aerosol_TBD(:,:) = 0.0
      R__Aerosol_TBD(1,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(2,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(3,1:2) = (/ -999.9, -999.9 /)
      N__Num_Regression(:) = 0
      N__Num_Regression(1:N__Num_Aerosol_Tests) = (/ 3, 3, 3 /)
      R__coef_AOD(:,:) = 0.0
      R__coef_AOD(1,1:N__Num_Regression(1)) = &
&           (/ 0.0, 0.0, 0.0 /)
      R__Rank_Thres_Coeff(1:3) = (/ -0.01, 2.1, -3.9 /)
      R__Unclassified_Thres = 0.4
      R__Land_Fraction_Thres = 0.5

   CASE(INST_ID_HIRAS2)
      !====================
      ! Set up HIRAS-2
      !====================

      CL__InstrumentName='HIRAS2'
      CL__Aerosol_Detection_File = 'HIRAS2_AERDET.NL'

      N__Num_Aerosol_Tests = 3
      N__Num_Aerosol_Chans(:) = 0
      N__Num_Aerosol_Chans(1:N__Num_Aerosol_Tests) = (/ 4, 4, 4 /)
      N__Aerosol_Chans(:,:) = 0
      N__Aerosol_Chans(1,1:N__Num_Aerosol_Chans(1)) = &
&           (/ 529, 932, 706, 935 /)
      N__Aerosol_Chans(2,1:N__Num_Aerosol_Chans(2)) = &
&           (/ 529, 932, 706, 935 /)
      N__Aerosol_Chans(3,1:N__Num_Aerosol_Chans(3)) = &
&           (/ 706, 935, 706, 294 /)
      N__Mean_Aerosol_Chans = 5
      R__Aerosol_TBD(:,:) = 0.0
      R__Aerosol_TBD(1,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(2,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(3,1:2) = (/ -999.9, -999.9 /)
      N__Num_Regression(:) = 0
      N__Num_Regression(1:N__Num_Aerosol_Tests) = (/ 3, 3, 3 /)
      R__coef_AOD(:,:) = 0.0
      R__coef_AOD(1,1:N__Num_Regression(1)) = &
&           (/ 0.0, 0.0, 0.0 /)
      R__Rank_Thres_Coeff(1:3) = (/ -0.01, 2.1, -3.9 /)
      R__Unclassified_Thres = 0.4
      R__Land_Fraction_Thres = 0.5

   CASE(INST_ID_GIIRS)
      !====================
      ! Set up GIIRS
      !====================

      CL__InstrumentName='GIIRS'
      CL__Aerosol_Detection_File = 'GIIRS_AERDET.NL'

      N__Num_Aerosol_Tests = 3
      N__Num_Aerosol_Chans(:) = 0
      N__Num_Aerosol_Chans(1:N__Num_Aerosol_Tests) = (/ 4, 4, 4 /)
      N__Aerosol_Chans(:,:) = 0
      N__Aerosol_Chans(1,1:N__Num_Aerosol_Chans(1)) = &
&           (/ 449, 681, 626, 681 /)
      N__Aerosol_Chans(2,1:N__Num_Aerosol_Chans(2)) = &
&           (/ 449, 681, 626, 681 /)
      N__Aerosol_Chans(3,1:N__Num_Aerosol_Chans(3)) = &
&           (/ 626, 681, 626, 214 /)
      N__Mean_Aerosol_Chans = 5
      R__Aerosol_TBD(:,:) = 0.0
      R__Aerosol_TBD(1,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(2,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(3,1:2) = (/ -999.9, -999.9 /)
      N__Num_Regression(:) = 0
      N__Num_Regression(1:N__Num_Aerosol_Tests) = (/ 3, 3, 3 /)
      R__coef_AOD(:,:) = 0.0
      R__coef_AOD(1,1:N__Num_Regression(1)) = &
&           (/ 0.0, 0.0, 0.0 /)
      R__Rank_Thres_Coeff(1:3) = (/ -0.01, 2.1, -3.9 /)
      R__Unclassified_Thres = 0.4
      R__Land_Fraction_Thres = 0.5

   CASE(INST_ID_IKFS2)
      !====================
      ! Set up IKFS2
      !====================

      CL__InstrumentName='IKFS2'
      CL__Aerosol_Detection_File = 'IKFS2_AERDET.NL'

      N__Num_Aerosol_Tests = 3
      N__Num_Aerosol_Chans(:) = 0
      N__Num_Aerosol_Chans(1:N__Num_Aerosol_Tests) = (/ 4, 4, 4 /)
      N__Aerosol_Chans(:,:) = 0
      N__Aerosol_Chans(1,1:N__Num_Aerosol_Chans(1)) = &
&           (/  915, 1603, 1231, 1606 /)
      N__Aerosol_Chans(2,1:N__Num_Aerosol_Chans(2)) = &
&           (/ 1453, 1603, 1231, 1606 /)
      N__Aerosol_Chans(3,1:N__Num_Aerosol_Chans(3)) = &
&           (/ 1231, 1606, 1231,  495 /)
      N__Mean_Aerosol_Chans = 5
      R__Aerosol_TBD(:,:) = 0.0
      R__Aerosol_TBD(1,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(2,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(3,1:2) = (/ -999.9, -999.9 /)
      N__Num_Regression(:) = 0
      N__Num_Regression(1:N__Num_Aerosol_Tests) = (/ 3, 3, 3 /)
      R__coef_AOD(:,:) = 0.0
      R__coef_AOD(1,1:N__Num_Regression(1)) = &
&           (/ 0.0, 0.0, 0.0 /)
      R__Rank_Thres_Coeff(1:3) = (/ -0.01, 2.1, -3.9 /)
      R__Unclassified_Thres = 0.4
      R__Land_Fraction_Thres = 0.5

   CASE(INST_ID_IRIS)
      !====================
      ! Set up IRIS
      !====================

      CL__InstrumentName='IRIS'
      CL__Aerosol_Detection_File = 'IRIS_AERDET.NL'

      N__Num_Aerosol_Tests = 3
      N__Num_Aerosol_Chans(:) = 0
      N__Num_Aerosol_Chans(1:N__Num_Aerosol_Tests) = (/ 4, 4, 4 /)
      N__Aerosol_Chans(:,:) = 0
      N__Aerosol_Chans(1,1:N__Num_Aerosol_Chans(1)) = &
&           (/ 417, 598, 497, 600 /)
      N__Aerosol_Chans(2,1:N__Num_Aerosol_Chans(2)) = &
&           (/ 417, 598, 497, 600 /)
      N__Aerosol_Chans(3,1:N__Num_Aerosol_Chans(3)) = &
&           (/ 497, 600, 497, 312 /)
      N__Mean_Aerosol_Chans = 5
      R__Aerosol_TBD(:,:) = 0.0
      R__Aerosol_TBD(1,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(2,1:2) = (/ -999.9, -999.9 /)
      R__Aerosol_TBD(3,1:2) = (/ -999.9, -999.9 /)
      N__Num_Regression(:) = 0
      N__Num_Regression(1:N__Num_Aerosol_Tests) = (/ 3, 3, 3 /)
      R__coef_AOD(:,:) = 0.0
      R__coef_AOD(1,1:N__Num_Regression(1)) = &
&           (/ -0.09, -0.06, 0.01 /)
      R__Rank_Thres_Coeff(1:3) = (/ -0.01, 2.1, -3.9 /)
      R__Unclassified_Thres = 0.4
      R__Land_Fraction_Thres = 0.5

   CASE DEFAULT
      CYCLE
   END SELECT

   !------------------------------------------------------------------
   ! Open and read file containing aerosol detection setup for the
   ! current instrument
   !------------------------------------------------------------------
   INIU1=10
   OPEN(INIU1,STATUS='OLD',FORM='FORMATTED', &
        FILE=TRIM(CL__Aerosol_Detection_File), IOSTAT=IOS)
   IF (IOS == 0) THEN
      READ(INIU1,nml=Aerosol_Detect_Coeffs,IOSTAT=IOS)
      IF (IOS == 0) THEN
         WRITE(*,'(3X,A)') TRIM(CL__InstrumentName) // &
&             ' AEROSOL DETECTION FILE READ OK'
      ELSE
         CALL CADS_Abort('PROBLEM READING '//TRIM(CL__InstrumentName)//&
&                 ' AEROSOL DETECTION FILE')
      ENDIF
      CLOSE(INIU1)
   ELSE
      WRITE(*,'(3X,A)') 'NO '//TRIM(CL__InstrumentName) // &
&              ' AEROSOL DETECTION FILE : Using Default Values'
   ENDIF

   M__Sensor = J__Sensor

   !------------------------------------------------------------------
   ! Set up the S__CADS_Setup_Aerosol structure for current sensor
   !------------------------------------------------------------------

   S__CADS_Setup_Aerosol(J__SENSOR) % M__SENSOR = M__Sensor

   S__CADS_Setup_Aerosol(J__SENSOR) % N__Num_Aerosol_Tests = &
&        N__Num_Aerosol_Tests

   ALLOCATE( S__CADS_Setup_Aerosol(J__SENSOR) % &
&        N__Num_Aerosol_Chans(N__Num_Aerosol_Tests) )

   S__CADS_Setup_Aerosol(J__SENSOR) % N__Num_Aerosol_Chans(:) = &
&        N__Num_Aerosol_Chans(1:N__Num_Aerosol_Tests)

   ALLOCATE( S__CADS_Setup_Aerosol(J__SENSOR) % &
&        N__Num_Regression(N__Num_Aerosol_Tests) )

   S__CADS_Setup_Aerosol(J__SENSOR) % N__Num_Regression(:) = &
&        N__Num_Regression(1:N__Num_Aerosol_Tests)

   ALLOCATE(S__CADS_Setup_Aerosol(J__SENSOR) % N__Aerosol_Chans &
&            (N__Num_Aerosol_Tests,MAXVAL(N__Num_Aerosol_Chans(:))))

   S__CADS_Setup_Aerosol(J__SENSOR) % N__Aerosol_Chans(:,:) = 0
   DO J = 1, N__Num_Aerosol_Tests
      S__CADS_Setup_Aerosol(J__SENSOR) % &
&          N__Aerosol_Chans(J,1:N__Num_Aerosol_Chans(J)) = &
&          N__Aerosol_Chans(J,1:N__Num_Aerosol_Chans(J))
   ENDDO

   S__CADS_Setup_Aerosol(J__SENSOR) % N__Mean_Aerosol_Chans = &
&          N__Mean_Aerosol_Chans

   ALLOCATE(S__CADS_Setup_Aerosol(J__SENSOR) % R__Aerosol_TBD &
&            (N__Num_Aerosol_Tests,MAXVAL(N__Num_Aerosol_Chans(:))))

   S__CADS_Setup_Aerosol(J__SENSOR) % R__Aerosol_TBD(:,:) = 0.0
   DO J = 1, N__Num_Aerosol_Tests
      S__CADS_Setup_Aerosol(J__SENSOR) % &
&          R__Aerosol_TBD(J,1:N__Num_Aerosol_Chans(J)) = &
&          R__Aerosol_TBD(J,1:N__Num_Aerosol_Chans(J))
   ENDDO

   ALLOCATE(S__CADS_Setup_Aerosol(J__SENSOR) % R__coef_AOD &
&             (N__Num_Aerosol_Tests,MAXVAL(N__Num_Regression(:))))

   S__CADS_Setup_Aerosol(J__SENSOR) % R__coef_AOD(:,:) = 0.0
   DO J = 1, N__Num_Aerosol_Tests
      S__CADS_Setup_Aerosol(J__SENSOR) % &
&          R__coef_AOD(J,1:N__Num_Regression(J)) = &
&          R__coef_AOD(J,1:N__Num_Regression(J))
   ENDDO

   S__CADS_Setup_Aerosol(J__SENSOR) % R__Rank_Thres_Coeff(:) = 0.0
   DO J = 1, 3
      S__CADS_Setup_Aerosol(J__SENSOR) % R__Rank_Thres_Coeff(J) = &
&        R__Rank_Thres_Coeff(J)
   ENDDO

   S__CADS_Setup_Aerosol(J__SENSOR) % R__Unclassified_Thres = &
&          R__Unclassified_Thres

   S__CADS_Setup_Aerosol(J__SENSOR) % R__Land_Fraction_Thres = &
&          R__Land_Fraction_Thres

ENDDO SensorLoop

END SUBROUTINE CADS_Setup_Aerosol
