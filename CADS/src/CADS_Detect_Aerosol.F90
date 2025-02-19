SUBROUTINE CADS_Detect_Aerosol ( &
&    K__Sensor,                  &
&    K__NChans,                  &
&    K__ChanID,                  &
&    K__Aerosol_Type,            &
&    K__Aerosol_Flag,            &
&    P__Land_Fraction,           &
&    P__ObsBTs,                  &
&    P__CldLev)

!   This software was developed within the context of the EUMETSAT
!   Satellite Application Facility on Numerical Weather Prediction
!   (NWP SAF), under the Cooperation Agreement dated 7 December 2016,
!   between EUMETSAT and the Met Office, UK, by one or more partners
!   within the NWP SAF. The partners in the NWP SAF are the Met
!   Office, ECMWF, DWD and MeteoFrance.
!
!   Copyright 2020, EUMETSAT, All Rights Reserved.

!   * CADS_Detect_Aerosol *
!   A. Collard  ECMWF 17/05/06

!   * PURPOSE *
!   -----------
!   Identify IR radiances that are affected by aerosol in the FOV.

!   * INTERFACE *
!   -------------
!   *CALL* * CADS_Detect_Aerosol()* from CADS_Main.

!   * METHOD *
!   ----------
!   A unique theoretically derived aerosol signal is sought through
!   observed brightness temperature differences in two spectral
!   locations within long-wave window region. Two additional spectral
!   locations are made use of to identify the aerosol type. Affected
!   channels are flagged, and what is considered affected depends on
!   the aerosol type.

!   * MODIFICATIONS *
!   -----------------
!   17/05/06  1.0 Original code.                                   A. Collard
!   19/10/06  1.1 Modification to present channels test.           A. Collard
!   13/01/15  2.1 Make array size specifications implicit.         R. Eresmaa
!   10/11/15  2.2 New algorithm.             J. Letertre-Danczak & R. Eresmaa
!   29/12/16  2.3 Make sensor-independent.   J. Letertre-Danczak & R. Eresmaa
!   05/02/19  2.4 Channel-specific flagging.                       R. Eresmaa
!                 Explicit KIND specifications.                    R. Eresmaa
!   16/04/20  3.0 Aerosol type recognition.                        R. Eresmaa

USE CADS_Module, ONLY : S__CADS_Setup_Aerosol

IMPLICIT NONE

! Subroutine arguments
INTEGER(KIND=4), INTENT(IN)  :: K__Sensor          ! Sensor ID
INTEGER(KIND=4), INTENT(IN)  :: K__NChans          ! Number of channels
INTEGER(KIND=4), INTENT(IN)  :: K__ChanID(:)       ! Channel index list
INTEGER(KIND=4), INTENT(OUT) :: K__Aerosol_Type    ! Aerosol type (output)
INTEGER(KIND=4), INTENT(OUT) :: K__Aerosol_Flag(:) ! Aerosol flags (output)
REAL(KIND=8),    INTENT(IN)  :: P__Land_Fraction   ! Land fraction in FOV
REAL(KIND=8),    INTENT(IN)  :: P__ObsBTs(:)       ! Brightness temperature
                                                   ! observations
REAL(KIND=8),    INTENT(IN)  :: P__CldLev(:)       ! Channel height assignments

! Local variables
INTEGER(KIND=4)              :: J, I__TB, I__K, I__Test, I__R
INTEGER(KIND=4)              :: I__MaxChans
INTEGER(KIND=4)              :: I__Num_Aerosol_Chans
INTEGER(KIND=4)              :: I__Num_Regression
INTEGER(KIND=4)              :: I__Aerosol
INTEGER(KIND=4), POINTER     :: I__Aerosol_Chans(:)
INTEGER(KIND=4)              :: I__M
INTEGER(KIND=4)              :: I__Mean_Aerosol_Chans
INTEGER(KIND=4), ALLOCATABLE :: I__NumFoundChans_1(:)
INTEGER(KIND=4)              :: I__Mean2_Aerosol_Chans

REAL(KIND=8)                 :: Z__DIFF_Aerosol
REAL(KIND=8)                 :: Z__Rank_Threshold
REAL(KIND=8)                 :: Z__Minposition
REAL(KIND=8)                 :: Z__Rank_Normalized(K__NChans)
REAL(KIND=8)                 :: Z__AOD, Z__Dust_AOD, Z__Ash_AOD, Z__Other_AOD
REAL(KIND=8), POINTER        :: Z__TBD(:)
REAL(KIND=8), POINTER        :: Z__coef_AOD(:)
REAL(KIND=8), ALLOCATABLE    :: Z_TBM(:)
REAL(KIND=8)                 :: Z__Rank_Thres_Coeff(3)
REAL(KIND=8)                 :: Z__Unclassified_Threshold
REAL(KIND=8)                 :: Z__LSM_Threshold
REAL(KIND=8)                 :: Z__Min_Cldlev, Z__Max_Cldlev
REAL(KIND=8)                 :: Z__SuperBTDiff(6)


!-----------------------------------
! Initialise
!-----------------------------------

Z__SuperBTDiff(:)=0.0

K__Aerosol_Type = 0 ! Initialize to 0: no aerosol
                    ! Other possibilities:
                    ! 1: Desert dust
                    ! 2: Volcanic ash
                    ! 3: Other
                    ! 4: Any type over land

I__MaxChans = &
&      MAXVAL(S__CADS_Setup_Aerosol(K__Sensor) % N__Num_Aerosol_Chans(:))
Z__Rank_Thres_Coeff(1:3) = &
&      S__CADS_Setup_Aerosol(K__Sensor) % R__Rank_Thres_Coeff(1:3)
Z__Unclassified_Threshold = &
&      S__CADS_Setup_Aerosol(K__Sensor) % R__Unclassified_Thres
Z__LSM_Threshold = &
&      S__CADS_Setup_Aerosol(K__Sensor) % R__Land_Fraction_Thres

!-----------------------------------
! Loop through tests
!-----------------------------------

ALLOCATE(Z_TBM(I__MaxChans))
ALLOCATE(I__NumFoundChans_1(I__MaxChans))

I__Aerosol=0
Z__AOD=0.0

TestLoop : DO I__Test = &
&                1, S__CADS_Setup_Aerosol(K__Sensor) % N__Num_Aerosol_Tests

   I__Num_Aerosol_Chans = &
&        S__CADS_Setup_Aerosol(K__Sensor) % N__Num_Aerosol_Chans(I__Test)
   I__Num_Regression = &
&        S__CADS_Setup_Aerosol(K__Sensor) % N__Num_Regression(I__Test)
   I__Aerosol_Chans => S__CADS_Setup_Aerosol(K__Sensor) % &
&        N__Aerosol_Chans(I__Test,1:I__Num_Aerosol_Chans)
   Z__TBD => S__CADS_Setup_Aerosol(K__Sensor) % &
&        R__Aerosol_TBD(I__Test,1:I__Num_Aerosol_Chans)
   Z__coef_AOD => S__CADS_Setup_Aerosol(K__Sensor) % &
&        R__coef_AOD(I__Test,1:I__Num_Regression)
   I__Mean_Aerosol_Chans = S__CADS_Setup_Aerosol(K__Sensor) % &
&        N__Mean_Aerosol_Chans
   I__Mean2_Aerosol_Chans = int(I__Mean_Aerosol_Chans/2)+1

   DO I__TB=1, I__Num_Aerosol_Chans
      I__NumFoundChans_1(I__TB)=0
      Z_TBM(I__TB)=0
   ENDDO

   Z__Min_Cldlev=P__CldLev(1)
   Z__Max_Cldlev=P__CldLev(1)

   DO I__K=1,K__NChans
      IF (P__ObsBTs(I__K) <= 0.) CYCLE
      DO I__TB=1, I__Num_Aerosol_Chans
         DO J=1, I__Mean_Aerosol_Chans
            I__M = I__Aerosol_Chans(I__TB)-I__Mean2_Aerosol_Chans+J
            IF (I__M == K__ChanID(I__K)) THEN
               Z_TBM(I__TB) = Z_TBM(I__TB)+P__ObsBTs(I__K)
               I__NumFoundChans_1(I__TB) = I__NumFoundChans_1(I__TB)+1
            ENDIF
         ENDDO
      ENDDO

      IF (P__CldLev(I__K) <= 0. .OR. I__K==1) CYCLE
      IF (P__CldLev(I__K)<Z__Min_Cldlev) Z__Min_Cldlev=P__CldLev(I__K)
      IF (P__CldLev(I__K)>Z__Max_Cldlev) Z__Max_Cldlev=P__CldLev(I__K)

   ENDDO

   DO I__TB=1, I__Num_Aerosol_Chans
      IF(I__NumFoundChans_1(I__TB)==0) CYCLE TestLoop
      Z_TBM(I__TB) = Z_TBM(I__TB)/I__NumFoundChans_1(I__TB)
   ENDDO

   IF (I__Test==1) THEN
      Z__SUPERBTDIFF(1)=Z_TBM(1)-Z_TBM(2)
      Z__SUPERBTDIFF(2)=Z_TBM(3)-Z_TBM(4)
   ELSEIF (I__Test==2) THEN
      Z__SUPERBTDIFF(3)=Z_TBM(1)-Z_TBM(2)
      Z__SUPERBTDIFF(4)=Z_TBM(3)-Z_TBM(4)
   ELSE
      Z__SUPERBTDIFF(5)=Z_TBM(1)-Z_TBM(2)
      Z__SUPERBTDIFF(6)=Z_TBM(3)-Z_TBM(4)
   ENDIF


   Z__DIFF_Aerosol=Z_TBM(1)-Z_TBM(2)
   IF (Z__DIFF_Aerosol <= Z__TBD(1)) I__Aerosol=I__Aerosol+1
   Z__DIFF_Aerosol=Z_TBM(3)-Z_TBM(4)
   IF (Z__DIFF_Aerosol <= Z__TBD(2)) I__Aerosol=I__Aerosol+1

   Z__AOD=0
   DO I__R=1, I__Num_Regression
      Z__AOD=Z__AOD+Z__coef_AOD(I__R)*(Z__DIFF_Aerosol**(I__R-1))
   ENDDO
   IF (Z__AOD < 0.0) Z__AOD=0.0

   IF (I__Test==1) THEN
      Z__Dust_AOD=Z__AOD
      IF (I__Aerosol==2) THEN
         K__Aerosol_Type=1
      ELSE
         K__Aerosol_Type=0
         EXIT TestLoop ! No aerosol, thus no need to test further.
      ENDIF

   ELSEIF (I__Test==2) THEN
      Z__Ash_AOD=Z__AOD
      IF (I__Aerosol==4) THEN
         K__Aerosol_Type=2
         EXIT TestLoop ! Aerosol present and identified as volcanic ash.
      ELSE
         I__Aerosol=2
      ENDIF

   ELSEIF (I__Test==3) THEN
      Z__Other_AOD=Z__AOD
      IF (I__Aerosol==4) THEN
         K__Aerosol_Type=1
      ELSE
         K__Aerosol_Type=3
      ENDIF

   ENDIF

ENDDO TestLoop

K__Aerosol_Flag(:)=0
IF (K__Aerosol_Type==1) THEN
! For desert dust aerosol, rejection threshold depends on the AOD
! estimate
   Z__AOD=Z__Dust_AOD

   Z__Rank_Threshold = 1.0
   IF (Z__Dust_AOD>1.0E-6) THEN
      Z__Rank_Threshold = ( &
&      ( Z__Rank_Thres_Coeff(1)/Z__Dust_AOD) - Z__Rank_Thres_Coeff(2) ) / &
&        Z__Rank_Thres_Coeff(3)
   ENDIF

ELSEIF (K__Aerosol_Type==2) THEN
! For volcanic ash, set rejection threshold such that all channels get
! rejected
  Z__AOD=Z__Ash_AOD
  Z__Rank_Threshold=-1.0

ELSEIF (K__Aerosol_Type==3) THEN
! For other aerosol detections, reject according to the unclassified
! threshold (0.0 or less rejects all, 1.0 or more rejects none)
  Z__AOD=Z__Other_AOD
  Z__Rank_Threshold=Z__Unclassified_Threshold

ENDIF

! If a meaningful land-fraction threshold is set, reject full spectra
! in case of any aerosol over land.
IF (P__Land_Fraction>Z__LSM_Threshold .AND. K__Aerosol_Type/=0) THEN
  K__Aerosol_Type=4
  Z__Rank_Threshold=-1.0
ENDIF

IF (K__Aerosol_Type>0) THEN
  Z__Rank_Normalized(:) = (P__CldLev(:) - Z__Min_Cldlev) / &
&      ( Z__Max_Cldlev-Z__Min_Cldlev )
  WHERE (Z__Rank_Normalized(:)>Z__Rank_Threshold) K__Aerosol_Flag(:)=1
ENDIF

IF (ALLOCATED(Z_TBM))              DEALLOCATE(Z_TBM)
IF (ALLOCATED(I__NumFoundChans_1)) DEALLOCATE(I__NumFoundChans_1)

NULLIFY(I__Aerosol_Chans)
NULLIFY(Z__TBD,Z__coef_AOD)

END SUBROUTINE CADS_Detect_Aerosol
