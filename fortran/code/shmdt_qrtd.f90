      implicit none
      
      ! integer, parameter :: i1=680,i2=801,j1=628,j2=657
      integer, parameter :: i1=1,i2=1440,j1=1,j2=720
      integer, parameter :: II=i2-i1+1 !=122
      integer, parameter :: JJ=j2-j1+1 !=30
      integer, parameter :: IIin=1440,JJin=720

      integer :: i,j,n
      real    :: gmdt(IIin,JJin),tmp(IIin,JJin),mask(IIin,JJin)
      real    :: glon(IIin),glat(JJin),gcs(IIin,JJin)
      real    :: r,g,omega,torad
      real    :: lat0,lats,ln,lt
      real    :: lon(II),lat(JJ),dx(JJ),dy,f0(JJ)
      real    :: ds(JJ)
      real :: mdt(II,JJ),sm,area,mn
      real :: cs(II,JJ)
      character*24 :: hdr
      character*128 :: path0,path1,path2,path_gmt,fn

      ! path0='~/home/data/RDSF/sync/data/analysis/mdt/'
      ! path1='~/home/data/RDSF/sync/data/analysis/mdt/mdts_by_deg/geodetic/DTU13/'
      ! path2='./data/'

      path0='./data/src/'
      path1='./data/res/'
      path2='./data/'

!    Define global lon and lat
!------------------------------------------------
     do i=1,IIin
        glon(i)=0.25*(i-0.5)
     end do

     do j=1,JJin
        glat(j)=0.25*(j-0.5)-90.0
     end do
!------------------------------------------------

!-------------------------------------------------
      ! open(21,file=trim(path1)//'GTIM5/qrt/sh_mdt_GTIM5_L240.dat',form='unformatted')
      open(21,file=trim(path1)//'shmdtfile.dat',form='unformatted')
      !open(21,file=trim(path1)//'EGM2008/qrt/sh_mdt_EGM2008_L300.dat',&&form='unformatted')
      read(21)gmdt
      close(21)
      call mdt_cs(IIin,JJin,glat,gmdt,gcs)
      

      ! open(21,file=trim(path0)//'masks/mask_glbl_qrtd.dat',form='unformatted')
      open(21,file=trim(path0)//'mask_rr0004.dat',form='unformatted')
      read(21)mask
      close(21)
  
      gmdt(:,:)=gmdt(:,:)+mask(:,:)
      gcs(:,:)=gcs(:,:)+mask(:,:)
!-------------------------------------------------

!-------------------------------------------------
!       tmp(1:720,:)=gmdt(721:IIin,:)
!       tmp(721:IIin,:)=gmdt(1:720,:)
!       gmdt(:,:)=tmp(:,:)
      mdt(:,:)=gmdt(i1:i2,j1:j2)
! !-------------------------------------------------

! !-------------------------------------------------
!       tmp(1:720,:)=gcs(721:IIin,:)
!       tmp(721:IIin,:)=gcs(1:720,:)
!       gcs(:,:)=tmp(:,:)
      cs(:,:)=gcs(i1:i2,j1:j2)
!-------------------------------------------------

!     Area elements for each latitude
!-------------------------------------------------
      r = 6371229.0
      omega = 7.29e-5 
      g = 9.80665  

      torad = atan(1.0)/45.0

      lat0=-89.875*torad
      lats=0.25*torad

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
      open(21,file=trim(path2)//'tmp.dat',form='unformatted')
      write(21)gmdt
      close(21)

 
      open(21,file=trim(path2)//'tmp2.dat',form='unformatted')
      write(21)mdt
      close(21)
 
      open(21,file=trim(path2)//'shmdtout.dat',form='unformatted')
      write(21)cs
      close(21)
!-------------------------------------------------




      stop

      !open(30,file=trim(path_gmt)//'/shmdt_cm.ascii',form='formatted') 
      !do j=1,JJin
      !   lt=0.25*(j-1)-89.875
      !   do i=1,IIin
      !      ln=0.25*(i-1)+0.125
      !      write(30,*)ln,lt,data(i,j)*100.0 !m->cm
      !   end do
      !end do
      !close(30)
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

      integer :: i,j,k,l,m,n,ix,jx
      real    :: r,g,omega,torad
      real    :: lats_r
      real    :: dx(JJ),dy,f0(JJ)
      real    :: u(II,JJ),v(II,JJ)
      real    :: dir(II,JJ)
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
