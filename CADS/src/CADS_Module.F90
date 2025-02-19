MODULE CADS_Module

!   This software was developed within the context of the EUMETSAT
!   Satellite Application Facility on Numerical Weather Prediction
!   (NWP SAF), under the Cooperation Agreement dated 7 December 2016,
!   between EUMETSAT and the Met Office, UK, by one or more partners
!   within the NWP SAF. The partners in the NWP SAF are the Met
!   Office, ECMWF, DWD and MeteoFrance.
!
!   Copyright 2020, EUMETSAT, All Rights Reserved.

!   * CADS_Module *
!   A. Collard  ECMWF 01/02/06

!   * PURPOSE *
!   -----------
!   Sets up structures to be used in processing of advanced IR sounders.

!   * MODIFICATIONS *
!   -----------------
!   01/02/06   A.Collard   1.0   Original export version.
!   17/11/09   R.Eresmaa   1.1   Include parameters of the Quick Exit /
!                                long-wave window gradient check.
!   11/11/11   R.Eresmaa   1.2   Add processing capability for CrIS.
!   03/12/13   R.Eresmaa   2.0   Add imager-assisted cloud detection.
!   10/11/15   R.Eresmaa   2.2   Changed instrument ID naming convention.
!                                Changed aerosol detection parameters.
!   20/12/16   R.Eresmaa   2.3   Remove aerosol detection parameters.
!   05/02/19   R.Eresmaa   2.4   Explicit KIND specifications.
!   16/04/20   R.Eresmaa   3.0   Combine cloud and aerosol detection, rename.
!                                Include aerosol type recognition.
!                                Include land sensitivity parameters.
!                                Include trace gas detection. Rename.
!   31/01/24   C.Burrows   3.2   Include HIRAS-2 and IRIS

IMPLICIT NONE

SAVE

INTEGER(KIND=4), PARAMETER :: INST_ID_AIRS = 11
INTEGER(KIND=4), PARAMETER :: INST_ID_IASI = 16
INTEGER(KIND=4), PARAMETER :: INST_ID_CRIS = 27
INTEGER(KIND=4), PARAMETER :: INST_ID_IRS = 57
INTEGER(KIND=4), PARAMETER :: INST_ID_IASING = 59
INTEGER(KIND=4), PARAMETER :: INST_ID_IKFS2 = 94
INTEGER(KIND=4), PARAMETER :: INST_ID_HIRAS = 97
INTEGER(KIND=4), PARAMETER :: INST_ID_HIRAS2 = 133
INTEGER(KIND=4), PARAMETER :: INST_ID_GIIRS = 98
INTEGER(KIND=4), PARAMETER :: INST_ID_IRIS = 69

INTEGER(KIND=4), PARAMETER :: JP__MIN_SENSOR_INDEX = INST_ID_AIRS
INTEGER(KIND=4), PARAMETER :: JP__MAX_SENSOR_INDEX = INST_ID_HIRAS2

TYPE Aerosol_Detect_Type
INTEGER(KIND=4) :: M__Sensor                         ! Unique ID for sensor
INTEGER(KIND=4) :: N__Num_Aerosol_Tests              ! Number of aerosol
                                                     ! detection tests
INTEGER(KIND=4), POINTER :: N__Num_Regression(:)     ! Number of conversion
                                                     ! coefficients for AOD
INTEGER(KIND=4), POINTER :: N__Num_Aerosol_Chans(:)  ! Number of aerosol
                                                     ! detection channels
INTEGER(KIND=4), POINTER :: N__Aerosol_Chans(:,:)    ! List of aerosol
                                                     ! detection channels
INTEGER(KIND=4)          :: N__Mean_Aerosol_Chans    ! Boxcar averaging window
                                                     ! width
REAL(KIND=8), POINTER    :: R__Aerosol_TBD(:,:)      ! Aerosol detection
                                                     ! thresholds
REAL(KIND=8), POINTER    :: R__coef_AOD(:,:)         ! Coefficients for
                                                     ! conversion to AOD
REAL(KIND=8)             :: R__Rank_Thres_Coeff(3)   ! Coefficients to
                                                     ! restrict rejections
                                                     ! to affected channels
REAL(KIND=8)             :: R__Unclassified_Thres    ! Rejection threshold for
                                                     ! unclassified aerosol
REAL(KIND=8)             :: R__Land_Fraction_Thres   ! Threshold for land
                                                     ! fraction in FOV
END TYPE Aerosol_Detect_Type

TYPE Cloud_Detect_Type
INTEGER(KIND=4)         :: M__Sensor                 ! Unique ID for sensor
INTEGER(KIND=4)         :: N__Num_Bands              ! Number of channel bands
INTEGER(KIND=4), POINTER :: N__GradChkInterval(:)    ! Window width used in
                                                     ! gradient calculation
INTEGER(KIND=4), POINTER :: N__Band_Size(:)          ! Number of channels in
                                                     ! each band
INTEGER(KIND=4), POINTER :: N__Bands(:,:)            ! Channel lists
INTEGER(KIND=4), POINTER :: N__Window_Width(:)       ! Smoothing filter window
                                                     ! widths per band
INTEGER(KIND=4), POINTER :: N__Window_Bounds(:,:)    ! Channels in the spectral
                                                     ! window gradient check
INTEGER(KIND=4), POINTER :: N__BandToUse(:)          ! Band number assignment
                                                     ! for each channel
LOGICAL  :: L__Do_Quick_Exit                         ! On/off switch for the
                                                     ! Quick Exit scenario
LOGICAL  :: L__Do_CrossBand                          ! On/off switch for the
                                                     ! cross-band method
REAL(KIND=8), POINTER :: R__BT_Threshold(:)          ! BT threshold for cloud
                                                     ! contamination
REAL(KIND=8), POINTER :: R__Grad_Threshold(:)        ! Gradient threshold for
                                                     ! cloud contamination
REAL(KIND=8), POINTER :: R__Window_Grad_Threshold(:) ! Threshold for window
                                                     ! gradient check in QE

LOGICAL  :: L__Do_Imager_Cloud_Detection             ! On/off switch for the
                                                     ! imager cloud detection
INTEGER(KIND=4)         :: N__Num_Imager_Chans       ! No. of imager channels
INTEGER(KIND=4)         :: N__Num_Imager_Clusters    ! No. of clusters to be
                                                     ! expected
INTEGER(KIND=4),POINTER :: N__Imager_Chans(:)        ! List of imager channels
REAL(KIND=8),POINTER    :: R__Stddev_Threshold(:)    ! St. Dev. threshold, one
                                                     ! for each imager channel
REAL(KIND=8)            :: R__Coverage_Threshold     ! Threshold for
                                                     ! fractional coverage
                                                     ! of a cluster
REAL(KIND=8)            :: R__FG_Departure_Threshold ! Threshold for imager
                                                     ! FG departure

END TYPE Cloud_Detect_Type

TYPE Land_Sensitivity_Type
INTEGER(KIND=4)         :: M__Sensor                 ! Unique ID for sensor
REAL(KIND=8)            :: R__Land_Fraction_Thres    ! Threshold on land
                                                     ! fraction
REAl(KIND=8)            :: R__Level_Thres            ! Threshold on normalized
                                                     ! channel height assignment
END TYPE Land_Sensitivity_Type

TYPE Trace_Gas_Detect_Type
INTEGER(KIND=4)         :: M__Sensor                  ! Unique ID for sensor
INTEGER(KIND=4)         :: N__Num_Trace_Gas_Checks    ! Number of trace gases
                                                      ! to be checked
INTEGER(KIND=4),POINTER :: N__Num_Tracer_Channels(:)  ! Number of gas-sensitive
                                                      ! channels
INTEGER(KIND=4),POINTER :: N__Tracer_Channels(:,:)    ! Gas-sensitive channels
INTEGER(KIND=4),POINTER :: N__Num_Control_Channels(:) ! Number of control
                                                      ! channels
INTEGER(KIND=4),POINTER :: N__Control_Channels(:,:)   ! Control channels
INTEGER(KIND=4),POINTER :: N__Num_Flagged_Channels(:) ! Number of affected
                                                      ! channels
INTEGER(KIND=4),POINTER :: N__Flagged_Channels(:,:)   ! Affected channels

REAL(KIND=8),POINTER    :: R__D_Obs_Threshold(:)      ! Observed Tb difference
                                                      ! threshold
REAL(KIND=8),POINTER    :: R__D_Dep_Threshold(:)      ! Departure difference
                                                      ! threshold
END TYPE Trace_Gas_Detect_Type


TYPE(Aerosol_Detect_Type) :: &
&  S__CADS_Setup_Aerosol(JP__Min_Sensor_Index:JP__Max_Sensor_Index)

TYPE(Cloud_Detect_Type) :: &
&  S__CADS_Setup_Cloud(JP__Min_Sensor_Index:JP__Max_Sensor_Index)

TYPE(Land_Sensitivity_Type) :: &
&  S__CADS_Setup_Land(JP__Min_Sensor_Index:JP__Max_Sensor_Index)

TYPE(Trace_Gas_Detect_Type) :: &
&  S__CADS_Setup_Trace_Gas(JP__Min_Sensor_Index:JP__Max_Sensor_Index)


END MODULE CADS_Module
