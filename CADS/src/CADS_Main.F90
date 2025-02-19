PROGRAM CADS_Main

!   This software was developed within the context of the EUMETSAT
!   Satellite Application Facility on Numerical Weather Prediction
!   (NWP SAF), under the Cooperation Agreement dated 7 December 2016,
!   between EUMETSAT and the Met Office, UK, by one or more partners
!   within the NWP SAF. The partners in the NWP SAF are the Met
!   Office, ECMWF, DWD and MeteoFrance.
!
!   Copyright 2020, EUMETSAT, All Rights Reserved.

!   * CADS_Main *
!   A. Collard   ECMWF   22/05/06

!   PURPOSE
!   -------
!   Main program for the NWPSAF Cloud and Aerosol Detection Software
!   (CADS).

!   MODIFICATIONS
!   -------------
!   22/05/06   A.Collard   1.0   Original version.
!   23/11/09   R.Eresmaa   1.1   Print out the elapsed CPU time.
!   03/12/13   R.Eresmaa   2.0   Modify to include the imager-assisted scheme.
!   19/01/15   R.Eresmaa   2.1   Add imager channel IDs to the call to the
!                                imager cloud detection.
!                                Specify I__Min_Level & I__Max_Level separately
!                                at each FOV.
!   10/11/15   R.Eresmaa   2.2   Add aerosol flag separately from cloud flags.
!   21/12/16   R.Eresmaa   2.3   Separate calls to cloud and aerosol detection.
!   05/02/19   R.Eresmaa   2.4   Rename (previously cloud_detect_wrapper).
!                                Explicit KIND specifications.
!   27/04/20   R.Eresmaa   3.0   Renaming and tidying up for CADS V3. Include
!                                trace-gas and land-sensitivity detection.
!                                Extend diagnostic output.

IMPLICIT NONE

! Instrument and channel configuration: these variables are unchanged
! during the processing of multiple observations
INTEGER(KIND=4) :: I__Sensor_ID                      ! Sensor ID
INTEGER(KIND=4) :: I__Num_Chans                      ! Number of sounder
                                                     ! channels in processing
INTEGER(KIND=4), ALLOCATABLE :: I__Chan_ID(:)        ! Sounder channel IDs
INTEGER(KIND=4) :: I__Num_Imager_Chans               ! Number of collocated
                                                     ! imager channels
INTEGER(KIND=4) :: I__Num_Imager_Clusters            ! Number of collocated
                                                     ! imager clusters
INTEGER(KIND=4), ALLOCATABLE :: I__Chan_ID_Imager(:) ! Imager channel IDs


! Input data: these variables are specific to each observation (FOV)
INTEGER(KIND=4) :: I__Min_Level                      ! Height assignment at the
                                                     ! tropopause level
INTEGER(KIND=4) :: I__Max_Level                      ! Height assignment at the
                                                     ! top of the boundary layer
REAL(KIND=8)    :: Z__Latitude, Z__Longitude         ! Observation coordinates
REAL(KIND=8)    :: Z__Land_Fraction                  ! Fraction of land in FOV
REAL(KIND=8), ALLOCATABLE :: Z__BT_Obser(:)          ! Observed brightness
                                                     ! temperatures (BTs)
REAL(KIND=8), ALLOCATABLE :: Z__BT_Model(:)          ! NWP forecast BTs
REAL(KIND=8), ALLOCATABLE :: Z__Chan_Height(:)       ! Channel height
                                                     ! assignments
REAL(KIND=8), ALLOCATABLE :: Z__Cluster_Fraction(:)  ! Fractional coverages of
                                                     ! imager clusters
REAL(KIND=8), ALLOCATABLE :: Z__BT_in_Cluster(:,:)   ! Cluster mean BTs
REAL(KIND=8), ALLOCATABLE :: Z__BT_Overall_SDev(:)   ! Imager BT standard
                                                     ! deviations in the FOV
REAL(KIND=8), ALLOCATABLE :: Z__BT_Model_Imager(:)   ! NWP forecast BTs in
                                                     ! imager channels


! Output flags: produced separately for each observation
INTEGER(KIND=4), ALLOCATABLE :: I__Flag_Cloud(:)     ! Cloud flags
INTEGER(KIND=4), ALLOCATABLE :: I__Flag_Aerosol(:)   ! Aerosol flags
INTEGER(KIND=4), ALLOCATABLE :: I__Flag_Trace_Gas(:) ! Trace-gas flags
INTEGER(KIND=4), ALLOCATABLE :: I__Flag_Land_Sens(:) ! Land-sensitivity flags
INTEGER(KIND=4) :: I__Aerosol_Type                   ! Aerosol type index

! Interim products
CHARACTER(LEN=200) :: C__Error_Message               ! Message output in abort
INTEGER(KIND=4) :: I__Obs                            ! Observation counter
INTEGER(KIND=4) :: I__Num_Observations               ! Count of observations
                                                     ! processed at the end

! Diagnostics: percentages of positive detections
INTEGER(KIND=4) :: I__Chan                           ! Channel index
REAL(KIND=8) :: Z__Per_Cloud                         ! Cloud
REAL(KIND=8) :: Z__Per_Aerosol                       ! Aerosol
REAL(KIND=8) :: Z__Per_Tracegas                      ! Trace gas
REAL(KIND=8) :: Z__Per_Land                          ! Land sensitivity

! Input/Output file management
CHARACTER(LEN=100) :: C__Input_Filename
CHARACTER(LEN=100) :: C__Output_Filename
CHARACTER(LEN=50)  :: C__Record
INTEGER(KIND=4)    :: I__In_Unit, I__Out_Unit, I__IO_Status
LOGICAL            :: L__Input_File_Found
LOGICAL            :: L__Imager_Found

! Computing time monitoring
REAL(KIND=8)       :: Z__Start_Time, Z__End_Time


INCLUDE 'CADS_Abort.intfb'
INCLUDE 'CADS_Setup_Aerosol.intfb'
INCLUDE 'CADS_Setup_Cloud.intfb'
INCLUDE 'CADS_Setup_Land_Sensitivity.intfb'
INCLUDE 'CADS_Setup_Trace_Gas.intfb'
INCLUDE 'CADS_Detect_Aerosol.intfb'
INCLUDE 'CADS_Detect_Cloud.intfb'
INCLUDE 'CADS_Detect_Land_Sensitivity.intfb'
INCLUDE 'CADS_Detect_Trace_Gas.intfb'


!---------------------------------------------------------------------

CALL CPU_TIME(Z__Start_Time)


!---------------------------------------------------------------------
! 1 Initialize the diagnostic arrays
Z__Per_Cloud    = 0.0
Z__Per_Aerosol  = 0.0
Z__Per_Tracegas = 0.0
Z__Per_Land     = 0.0


!---------------------------------------------------------------------
! 2 Setup

CALL CADS_Setup_Cloud
CALL CADS_Setup_Aerosol
CALL CADS_Setup_Trace_Gas
CALL CADS_Setup_Land_Sensitivity


!---------------------------------------------------------------------
! 3 Open the input and output data files, read header information and
!   allocate arrays as necessary

C__Input_Filename='cads_input.dat'
C__Output_Filename='cads_output.dat'

L__Input_File_Found=.TRUE.
INQUIRE ( FILE=TRIM(C__Input_Filename), EXIST=L__Input_File_Found )
IF (.NOT. L__Input_File_Found) THEN
  C__Error_Message='Input file not found: ' // TRIM(C__Input_Filename)
  CALL CADS_Abort(TRIM(C__Error_Message))
ENDIF

I__In_Unit=81
I__Out_Unit=82

OPEN ( I__In_Unit, FILE=TRIM(C__Input_Filename), STATUS='OLD' )
OPEN ( I__Out_Unit, FILE=TRIM(C__Output_Filename), STATUS='UNKNOWN' )

READ(I__In_Unit,*) I__Sensor_ID
READ(I__In_Unit,*) I__Num_Chans

ALLOCATE ( I__Chan_ID(I__Num_Chans) )
ALLOCATE ( Z__BT_Obser(I__Num_Chans) )
ALLOCATE ( Z__BT_Model(I__Num_Chans) )
ALLOCATE ( Z__Chan_Height(I__Num_Chans) )

ALLOCATE ( I__Flag_Aerosol(I__Num_Chans) )
ALLOCATE ( I__Flag_Cloud(I__Num_Chans) )
ALLOCATE ( I__Flag_Trace_Gas(I__Num_Chans) )
ALLOCATE ( I__Flag_Land_Sens(I__Num_Chans) )

READ(I__In_Unit,*) I__Chan_ID(:)
READ(I__In_Unit,*) I__Num_Observations


C__Record='.'
! C__Record, to be read in the following, is to contain either header
! information for the first observation or, if collocated imager data
! are included, number of provided imager channels.

READ(I__In_Unit,'(A)',IOSTAT=I__IO_Status) C__Record

L__Imager_Found=.FALSE. ! Don't expect to have input imager data by default
IF (INDEX(C__Record,'.')==0) THEN
   L__Imager_Found=.TRUE.
   READ(C__Record,*) I__Num_Imager_Chans
   ALLOCATE ( I__Chan_ID_Imager(I__Num_Imager_Chans) )
   READ(I__In_Unit,*) I__Chan_ID_Imager(1:I__Num_Imager_Chans)
   READ(I__In_Unit,*) I__Num_Imager_Clusters
   ALLOCATE ( Z__Cluster_Fraction(I__Num_Imager_Clusters) )
   ALLOCATE ( Z__BT_in_Cluster(I__Num_Imager_Chans,I__Num_Imager_Clusters) )
   ALLOCATE ( Z__BT_Overall_SDev(I__Num_Imager_Chans) )
   ALLOCATE ( Z__BT_Model_Imager(I__Num_Imager_Chans) )
   READ(I__In_Unit,'(A)',IOSTAT=I__IO_Status) C__Record
END IF

IF (I__IO_Status<0) THEN
  C__Error_Message='Input file does not seem to contain any observations!'
  CALL CADS_Abort(TRIM(C__Error_Message))
ENDIF

READ(C__Record,*) &
     Z__Longitude, Z__Latitude, Z__Land_Fraction, I__Min_Level, I__Max_Level


!---------------------------------------------------------------------
! 4 Processing one observation at a time, go through all input data.
!   Report the outcome inside the loop.

ObservationLoop: DO I__Obs = 1, I__Num_Observations

  ! 4.1 Read input
  IF (I__Obs>1) READ(I__In_Unit,*)  &
       Z__Longitude, Z__Latitude, Z__Land_Fraction, I__Min_Level, I__Max_Level
  READ(I__In_Unit,*)  Z__BT_Obser(:)
  READ(I__In_Unit,*)  Z__BT_Model(:)
  READ(I__In_Unit,*)  Z__Chan_Height(:)

  IF (L__Imager_Found) THEN
    READ (I__In_Unit,*) Z__Cluster_Fraction(:), &
                        Z__BT_in_Cluster(:,:), &
                        Z__BT_Overall_SDev(:), &
                        Z__BT_Model_Imager(:)
  ENDIF

  ! 4.2 Cloud detection
  CALL CADS_Detect_Cloud( &
       I__Sensor_ID,           & ! in
       I__Num_Chans,           & ! in
       I__Chan_ID,             & ! in
       I__Min_Level,           & ! in
       I__Max_Level,           & ! in
       I__Num_Imager_Chans,    & ! in new
       I__Chan_ID_Imager,      & ! in new
       I__Num_Imager_Clusters, & ! in new
       I__Flag_Cloud,          & ! out
       Z__BT_Obser,            & ! in
       Z__BT_Model,            & ! in
       Z__Chan_Height,         & ! in
       Z__Cluster_Fraction,    & ! in new
       Z__BT_in_Cluster,       & ! in new
       Z__BT_Overall_SDev,     & ! in new
       Z__BT_Model_Imager )      ! in new

  ! 4.3 Aerosol detection
  CALL CADS_Detect_Aerosol( &
       I__Sensor_ID,        & ! in
       I__Num_Chans,        & ! in
       I__Chan_ID,          & ! in
       I__Aerosol_Type,     & ! out
       I__Flag_Aerosol,     & ! out
       Z__Land_Fraction,    & ! in
       Z__BT_Obser,         & ! in
       Z__Chan_Height )       ! in

  ! 4.4 Trace-gas detection
  CALL CADS_Detect_Trace_Gas( &
       I__Sensor_ID,      & ! in
       I__Num_Chans,      & ! in
       I__Chan_ID,        & ! in
       Z__BT_Obser,       & ! in
       Z__BT_Model,       & ! in
       I__Flag_Trace_Gas )  ! out

  ! 4.5 Land-sensitivity identification
  CALL CADS_Detect_Land_Sensitivity( &
       I__Sensor_ID,      & ! in
       I__Num_Chans,      & ! in
       Z__Land_Fraction,  & ! in
       Z__Chan_Height,    & ! in
       I__Flag_Land_Sens )  ! out

  ! 4.6 Report the outcome
  WRITE (I__Out_Unit,'(A)') '-----'
  WRITE (I__Out_Unit,'(A)') 'Longitude Latitude ObNumber'
  WRITE (I__Out_Unit,'(2(1X,F9.4),1X,I8)') Z__Longitude, Z__Latitude, I__Obs
  WRITE (I__Out_Unit,'(A)') 'Cloud flags'
  WRITE (I__Out_Unit,'(20(1X,I1))') I__Flag_Cloud(:)
  WRITE (I__Out_Unit,'(A,I1)') 'Aerosol type index: ', I__Aerosol_Type
  WRITE (I__Out_Unit,'(A)') 'Aerosol flags'
  WRITE (I__Out_Unit,'(20(1X,I1))') I__Flag_Aerosol(:)
  WRITE (I__Out_Unit,'(A)') 'Trace gas flags'
  WRITE (I__Out_Unit,'(20(1X,I1))') I__Flag_Trace_Gas(:)
  WRITE (I__Out_Unit,'(A)') 'Land sensitivity flags'
  WRITE (I__Out_Unit,'(20(1X,I1))') I__Flag_Land_Sens(:)

  DO I__Chan=1,I__Num_Chans
     IF (I__Flag_Cloud(I__Chan)==1)     Z__Per_Cloud    = Z__Per_Cloud+1.0
     IF (I__Flag_Aerosol(I__Chan)==1)   Z__Per_Aerosol  = Z__Per_Aerosol+1.0
     IF (I__Flag_Trace_Gas(I__Chan)==1) Z__Per_Tracegas = Z__Per_Tracegas+1.0
     IF (I__Flag_Land_Sens(I__Chan)==1) Z__Per_Land     = Z__Per_Land+1.0
  ENDDO

ENDDO ObservationLoop


!---------------------------------------------------------------------
! 5 Close the input/output files, release memory allocations, report
! on the computing time and other diagnostics, and close the program

CLOSE(I__In_Unit)
CLOSE(I__Out_Unit)

IF ( ALLOCATED(I__Chan_ID) )          DEALLOCATE (I__Chan_ID)
IF ( ALLOCATED(I__Chan_ID_Imager) )   DEALLOCATE (I__Chan_ID_Imager)
IF ( ALLOCATED(I__Flag_Aerosol) )     DEALLOCATE (I__Flag_Aerosol)
IF ( ALLOCATED(I__Flag_Cloud) )       DEALLOCATE (I__Flag_Cloud)
IF ( ALLOCATED(I__Flag_Trace_Gas) )   DEALLOCATE (I__Flag_Trace_Gas)
IF ( ALLOCATED(I__Flag_Land_Sens) )   DEALLOCATE (I__Flag_Land_Sens)
IF ( ALLOCATED(Z__BT_Obser) )         DEALLOCATE (Z__BT_Obser)
IF ( ALLOCATED(Z__BT_Model) )         DEALLOCATE (Z__BT_Model)
IF ( ALLOCATED(Z__Chan_Height) )      DEALLOCATE (Z__Chan_Height)
IF ( ALLOCATED(Z__Cluster_Fraction) ) DEALLOCATE (Z__Cluster_Fraction)
IF ( ALLOCATED(Z__BT_in_Cluster) )    DEALLOCATE (Z__BT_in_Cluster)
IF ( ALLOCATED(Z__BT_Overall_SDev) )  DEALLOCATE (Z__BT_Overall_SDev)
IF ( ALLOCATED(Z__BT_Model_Imager) )  DEALLOCATE (Z__BT_Model_Imager)

WRITE (*,'(A)') ''
WRITE (*,'(A)') 'Percentages of positive detections:'
WRITE (*,'(A)') &
     '             Cloud' // &
     '           Aerosol' // &
     '         Trace Gas' // &
     '  Land sensitivity'
WRITE (*,'(4(8x,F10.5))') &
     100*Z__Per_Cloud/(I__Num_Observations*I__Num_Chans), &
     100*Z__Per_Aerosol/(I__Num_Observations*I__Num_Chans), &
     100*Z__Per_Tracegas/(I__Num_Observations*I__Num_Chans), &
     100*Z__Per_Land/(I__Num_Observations*I__Num_Chans)

CALL CPU_TIME(Z__End_Time)

WRITE (*,'(A)') ''
WRITE (*,'(A)') 'Processing completed'
WRITE (*,'(I6,A,F8.2,A)') &
     I__Num_Observations, ' observations were processed in ', &
     Z__End_Time-Z__Start_Time, ' seconds'

END PROGRAM CADS_Main
