SUBROUTINE CADS_Abort(String)

!   This software was developed within the context of the EUMETSAT
!   Satellite Application Facility on Numerical Weather Prediction
!   (NWP SAF), under the Cooperation Agreement dated 7 December 2016,
!   between EUMETSAT and the Met Office, UK, by one or more partners
!   within the NWP SAF. The partners in the NWP SAF are the Met
!   Office, ECMWF, DWD and MeteoFrance.
!
!   Copyright 2020, EUMETSAT, All Rights Reserved.

!   *CADS_Abort*
!   R. Eresmaa   ECMWF   16/04/20

!   * PURPOSE *
!   -----------
!   Controlled abortion of running CADS when facing exceptions such as
!   necessary input files missing or they are corrupt.

!   * INTERFACE *
!   -------------
!   *CALL* * CADS_Abort()* from
!      CADS_Main, CADS_Setup_Aerosol, CADS_Setup_Cloud,
!      CADS_Setup_Land_Sensitivity, or CADS_Setup_Trace_Gas.

  IMPLICIT NONE
  CHARACTER(LEN=*) :: String

  WRITE(*,*) String
  STOP

END SUBROUTINE CADS_Abort
