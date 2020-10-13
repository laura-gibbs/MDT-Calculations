      program main

      implicit none

!     Variable declarations
!===========================================================
!     General
!-------------------------------------------------
      integer  :: i,j,n,m,II,JJ
      integer  :: NN_mss,NN_egm,NN,rr
      character(len=4) :: nn_str,rr_str
      character(len=4)  :: tide_in ! Given tide system
      character(len=20) :: name_mss,name_egm
!-------------------------------------------------

!     For data
!-------------------------------------------------
      real,allocatable    :: mss(:,:)
      real,allocatable    :: geoid(:,:)
      real,allocatable    :: msk(:,:)
      real,allocatable    :: mdt(:,:)
      real,allocatable    :: lon(:)
      real,allocatable    :: lat(:)
!-------------------------------------------------

!     Required output grid
!-------------------------------------------------
      real*8 :: lon_stp       ! Longitude interval
      real*8 :: ltgd_stp      ! Latitude interval

      real*8 :: lon_min       ! Min longitude
      real*8 :: lon_max       ! Max longitude

      real*8 :: ltgd_min      ! Min latitude
      real*8 :: ltgd_max      ! Max latitude
!-------------------------------------------------

!     Misc
!-------------------------------------------------
      logical :: msk_exists
      character(len=128) :: path0,pin1,pin2,pout,fin,fout
!-------------------------------------------------
!===========================================================    

!     Start of proceedure
!===========================================================      
!-------------------------------------------------
      pin1='./data/src/'
      pin2='./data/res/'
      pout='./data/res/'
!-------------------------------------------------

!     Read in the mdt parameter file
!--------------------------------------------------
      open(21,file='./parameters.txt',form='formatted')
      read(21,'(A20)')name_mss
      read(21,'(I4)')NN_mss
      read(21,'(A20)')name_egm
      read(21,'(I4)')NN_egm
      read(21,'(A4)')tide_in
      read(21,'(I4)')rr
      read(21,'(I4)')NN
      close(21)
!-------------------------------------------------

!     Required output grid
!-------------------------------------------------
      lon_stp  = 1.0d0/rr                ! Longitude interval
      ltgd_stp = 1.0d0/rr                ! Latitude interval

      lon_min = 0.5d0*lon_stp            ! Min longitude
      lon_max = 360.0d0-0.5d0*lon_stp    ! Max longitude

      ltgd_min = -90.00d0+0.5d0*ltgd_stp   ! Min latitude
      ltgd_max =  90.00d0-0.5d0*ltgd_stp   ! Max latitude
!-------------------------------------------------

!     Compute grid dimensions
!     and make memory allocations 
!-------------------------------------------------
      II = nint((lon_max-lon_min)/lon_stp)+1
      JJ = nint((ltgd_max-ltgd_min)/ltgd_stp)+1

      write(*,*)II,JJ
      !stop

      allocate(mss(II,JJ))
      allocate(geoid(II,JJ))
      allocate(msk(II,JJ))
      allocate(mdt(II,JJ))
      allocate(lon(II))
      allocate(lat(JJ))
!-------------------------------------------------

!     Calculate longitude and geodetic latitude 
!     arrays for points on output grid
!-------------------------------------------------
      do i=1,II
         lon(i)=lon_stp*(i-1)+lon_min
      end do

      do j=1,JJ
         lat(j)=ltgd_stp*(j-1)+ltgd_min
      end do
!-------------------------------------------------

!-------------------------------------------------
      write(nn_str,'(I4)')NN
      write(rr_str,'(I4)')rr
      do i=1,4
         if(nn_str(i:i).eq.' ')nn_str(i:i)='0'
         if(rr_str(i:i).eq.' ')rr_str(i:i)='0'
      end do
!-------------------------------------------------

!     Read in surfaces
!-------------------------------------------------
      fin=trim(name_mss)//'_do'//nn_str//'_rr'//rr_str//'.dat'
      open(20,file=trim(pin2)//trim(fin),&
      &form='unformatted')
      read(20)mss
      close(20)

      fin=trim(name_egm)//'_do'//nn_str//'_rr'//rr_str//'.dat'
      open(20,file=trim(pin2)//trim(fin),&
      &form='unformatted')
      read(20)geoid
      close(20)

      fin='mask_rr'//rr_str//'.dat'
      inquire(file=trim(pin1)//trim(fin),exist=msk_exists)
      if(msk_exists)then
         open(20,file=trim(pin1)//trim(fin),&
         &form='unformatted')
         read(20)msk
         close(20)
      else
         write(*,*)'mask does not exist'
         msk=0.0
      end if
!-------------------------------------------------

!     Compute the MDT
!-------------------------------------------------
      mdt=mss-geoid
!-------------------------------------------------

!     Appy mask
!-------------------------------------------------
      do i=1,II
         do j=1,JJ
            if(msk(i,j) < -1.7e7)mdt(i,j)=msk(i,j)
         end do
      end do
!-------------------------------------------------

!     Write MDT to file
!-------------------------------------------------
      fout=trim(name_mss)//'_'//trim(name_egm)//&
                  &'_do'//nn_str//'_rr'//rr_str//'.dat'

      open(20,file=trim(pout)//trim(fout),&
      &form='unformatted')
      write(20)mdt
      close(20)
!-------------------------------------------------

!-------------------------------------------------
     deallocate(mss,geoid,msk,mdt,lon,lat)
!-------------------------------------------------

      stop

!===========================================================  
      end program main
