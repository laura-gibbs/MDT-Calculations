      implicit none
!     Variable declarations
!===========================================================
!     General
!-------------------------------------------------  
      integer :: i,j,II,JJ
      integer :: rr
      character*128 :: pin0,pin1,pout,fout
      character(len=40) :: name_mdt
      character(len=4) :: rr_str

      real,allocatable :: gmdt(:,:),mask(:,:),gcs(:,:)
      real,allocatable :: glon(:),glat(:),lat(:)
      real,allocatable :: ds(:)
      real,allocatable :: mdt(:,:),cs(:,:)

      real :: r,g,omega,torad
      real :: lat0,lats
      real :: sm,area,mn
!     Required output grid
!-------------------------------------------------
      real*8 :: lon_stp       ! Longitude interval
      real*8 :: ltgd_stp      ! Latitude interval

      real*8 :: lon_min       ! Min longitude
      real*8 :: lon_max       ! Max longitude

      real*8 :: ltgd_min      ! Min latitude
      real*8 :: ltgd_max      ! Max latitude


      pin0='./../a_mdt_data/computations/masks/'
      pin1='./../a_mdt_data/computations/mdts/'
      pout='./../a_mdt_data/computations/currents/'

!     Read in the mdt parameter file
!--------------------------------------------------
      open(21,file='./cs_params.txt',form='formatted')
      read(21,'(A40)')name_mdt
      read(21,'(I4)')rr
      close(21)

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

      allocate(mask(II,JJ))
      allocate(glon(II))
      allocate(glat(JJ))
      allocate(ds(JJ))
      allocate(lat(JJ))
      allocate(gmdt(II,JJ))
      allocate(gcs(II,JJ))
      allocate(cs(II,JJ))
      allocate(mdt(II,JJ))
      
!-------------------------------------------------


!    Define global lon and lat
!------------------------------------------------
     do i=1,II
        glon(i)=lon_stp*(i-0.5)
     end do

     do j=1,JJ
        glat(j)=ltgd_stp*(j-0.5)-90.0
     end do
!------------------------------------------------

      write(rr_str,'(I4)')rr
      do i=1,4
         if(rr_str(i:i).eq.' ')rr_str(i:i)='0'
      end do
!-------------------------------------------------
      ! open(21,file=trim(pin1)//'GTIM5/qrt/sh_mdt_GTIM5_L240.dat',form='unformatted')
      open(21,file=trim(pin1)//trim(name_mdt)//'.dat',form='unformatted')
      !open(21,file=trim(pin1)//'EGM2008/qrt/sh_mdt_EGM2008_L300.dat',&&form='unformatted')
      read(21)gmdt
      close(21)
      call mdt_cs(II,JJ,glat,gmdt,gcs)
      

      ! open(21,file=trim(pin0)//'masks/mask_glbl_qrtd.dat',form='unformatted')
      open(21,file=trim(pin0)//'mask_rr'//rr_str//'.dat',form='unformatted')
      read(21)mask
      close(21)
  
      gmdt(:,:)=gmdt(:,:)!+mask(:,:)
      gcs(:,:)=gcs(:,:)!+mask(:,:)
!-------------------------------------------------

!-------------------------------------------------
!       tmp(1:1440,:)=gmdt(721:IIin,:)
!       tmp(721:IIin,:)=gmdt(1:1440,:)
!       gmdt(:,:)=tmp(:,:)
      mdt(:,:)=gmdt(1:II,1:JJ)
! !-------------------------------------------------

! !-------------------------------------------------
!       tmp(1:1440,:)=gcs(721:IIin,:)
!       tmp(721:IIin,:)=gcs(1:1440,:)
!       gcs(:,:)=tmp(:,:)
      cs(:,:)=gcs(1:II,1:JJ)
!-------------------------------------------------

!     Area elements for each latitude
!-------------------------------------------------
      r = 6371229.0
      omega = 7.29e-5 
      g = 9.80665  

      torad = atan(1.0)/45.0

      lat0=-90.00d0+0.5d0*ltgd_stp
      lats=ltgd_stp*torad

      do j=1,JJ
         lat(j) = lat0+(j-1)*lats
         ds(j) = dble(0.50*(r*lats)**2 & 
            & *(cos(lat(j)+0.5*lats)+cos(lat(j)-0.5*lats)))
      end do
!-------------------------------------------------

!
!-------------------------------------------------
      sm=0.0
      area=0.0
      do i=1,II
         do j=1,JJ
            if(mdt(i,j).gt.-1.7e7)then
               sm=sm+mdt(i,j)*ds(j)
               area=area+ds(j)
            end if
         end do
      end do
      mn=sm/area
      do i=1,II
         do j=1,JJ
            if(mdt(i,j).gt.-1.7e7)then
               mdt(i,j)=mdt(i,j)-mn
            end if
         end do
      end do
!-------------------------------------------------

!-------------------------------------------------
      open(21,file=trim(pout)//'tmp.dat',form='unformatted')
      write(21)gmdt
      close(21)

 
      open(21,file=trim(pout)//'tmp2.dat',form='unformatted')
      write(21)mdt
      close(21)
      
      fout=trim(name_mdt)//'_cs.dat'
      open(21,file=trim(pout)//trim(fout),form='unformatted')
      write(21)cs
      close(21)
!-------------------------------------------------

      stop
      
      end

      subroutine mdt_cs(II,JJ,lat,mdt,cs)
!=====================================================================

      implicit none

!------------------------------------------------
!     Input
!---------------------------------------
      integer :: II,JJ
      real    :: lat(JJ),mdt(II,JJ)
!---------------------------------------

!     Output
!---------------------------------------
      real    :: cs(II,JJ)
!---------------------------------------

!     Local
!---------------------------------------       
      real, parameter :: m2cm=100.0

      integer :: i,j
      real    :: r,g,omega,torad
      real    :: lats_r
      real    :: dx(JJ),dy,f0(JJ)
      real    :: u(II,JJ),v(II,JJ)
!---------------------------------------       
!------------------------------------------------

!     Define parameters
!------------------------------------------------
      r = 6371229.0
      omega = 7.29e-5 
      g = 9.80665  
      torad = atan(1.0)/45.0
      lats_r = torad*(lat(2)-lat(1))
!------------------------------------------------

!     Calculate zonal width of a grid cell (m) (depends on Latitude)
!------------------------------------------------
      do j=1,JJ
         dx(j) = r*lats_r*cos(torad*lat(j))
      end do
!------------------------------------------------

!     Calculate meridional width of a grid cell (m) (does not depend on Latitude)
!------------------------------------------------
      dy = r*lats_r
!------------------------------------------------

!     Calculate the coriolis parameter 
!------------------------------------------------
      do j=1,JJ
            if(lat(j)<20 .AND. lat(j)>0)then
                  lat(j)=20
            else if (lat(j)>-20 .AND. lat(j)<0)then
                  lat(j)=-20
            end if
         f0(j) = 2.0*omega*sin(torad*lat(j))
      end do
!------------------------------------------------
      
!     Compute currents
!-------------------------------------------------
      do j=2,JJ-1

         if((mdt(1,j).gt.-1.9e9).and.(mdt(1,j-1).gt.-1.9e9))then
            u(1,j)=-(g/f0(j))*(mdt(1,j)-mdt(1,j-1))/(dy)
         end if

         if((mdt(1,j).gt.-1.9e9).and.(mdt(II,j).gt.-1.9e9))then
            v(1,j)=(g/f0(j))*(mdt(1,j)-mdt(II,j))/(dx(j))
         end if        

         cs(1,j)=sqrt(u(1,j)**2+v(1,j)**2)

         do i=2,II

            if((mdt(i,j).gt.-1.9e9).and.(mdt(i,j-1).gt.-1.9e9))then
               u(i,j)=-(g/f0(j))*(mdt(i,j)-mdt(i,j-1))/(dy)
            end if

            if((mdt(i,j).gt.-1.9e9).and.(mdt(i-1,j).gt.-1.9e9))then
               v(i,j)=(g/f0(j))*(mdt(i,j)-mdt(i-1,j))/(dx(j))
            end if        

            cs(i,j)=sqrt(u(i,j)**2+v(i,j)**2)

         end do

      end do
!-------------------------------------------------

!-------------------------------------------------
!     End of proceedure

      return

!=====================================================================
      end subroutine mdt_cs
