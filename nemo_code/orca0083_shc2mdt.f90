!     Compile with:
!     mpif90 orca0083_shc2mdt.f90 -o orca0083_shc2mdt -O3 -fbounds-check -I/newhome/rb13801/local/include -L/newhome/rb13801/local/lib -lnetcdff

      program main

      implicit none

     !--------------------------------------------
      include 'mpif.h'
      integer :: rank,nprocs
      integer :: tag
      integer :: err
      integer :: status(MPI_STATUS_SIZE)
     !--------------------------------------------

     !Dimensions of input grid
     !--------------------------------------------
      integer, parameter :: II=4320,JJ=2160
     !--------------------------------------------

     !Maximum and required degree for expansion
     !--------------------------------------------
      integer,parameter :: NNmax = 2190
      integer,parameter :: NN = 2190
     !--------------------------------------------
      
     !Inputs
     !--------------------------------------------
      real*8 :: c_h(0:NNmax,0:NNmax)          ! Sh coeffs of MSS
      real*8 :: s_h(0:NNmax,0:NNmax)          ! Sh coeffs of MSS
     !--------------------------------------------

     !Outputs
     !--------------------------------------------
      real   :: h(II,JJ)
     !--------------------------------------------

     !Local
     !--------------------------------------------
      integer :: i,j,m,n
      integer :: nrows
      integer :: j1,j2,JJstrip
      real    :: lon(II),lat(JJ)
      real*8  :: h_lat(II)
      real,allocatable  :: h_proc(:,:) 
      character(len=256) :: fn,ffn
      character(len=256) :: path0,pin1,pin2,pout
     !--------------------------------------------

!----------------------------------------------------------
!----------------------------------------------------------
!----------------------------------------------------------

!     get the number of processes, and the id of this process
!----------------------------------------------------------
      call MPI_Init(err)

      call MPI_Comm_rank(MPI_COMM_WORLD, rank, err)
      call MPI_Comm_size(MPI_COMM_WORLD, nprocs, err)
!----------------------------------------------------------

!     Determine nominal number of latitudes (rows) per process
!----------------------------------------------------------
      nrows = ceiling(1.0*JJ/nprocs)
      if (rank == 0)then
         write(*,*)'nominal latitudes per process = ',nrows
      end if
!----------------------------------------------------------

!     Use rank of process to work out the latitude limits for each process
!----------------------------------------------------------
      j1 = nrows*rank + 1
      j2 = j1 + nrows - 1
      j2 = min(j2,JJ)
      JJstrip = j2 - j1 + 1
      write(*,*)'limits for process',rank,':'
      write(*,*)j1,j2,JJstrip
!----------------------------------------------------------

!     Specify paths
!----------------------------------------------------------
      path0='/newhome/rb13801/rdsf/data/'

      !pin1=trim(path0)//&
      !      &'projects/cage/mdt/'
      pin1='./data/out/'

      !pout=trim(path0)//&
      !      &'projects/cage/mdt/'
      pout='./data/out/'
!----------------------------------------------------------

!     Read in MDT coefficients
!----------------------------------------------------------
      fn='orca0083_mdt_sh_c.dat'
      open(22,file=trim(pout)//trim(fn),form='unformatted')
      read(22)c_h
      close(22)
      c_h(NN,0)=0.0d0

      fn='orca0083_mdt_sh_s.dat'
      open(22,file=trim(pout)//trim(fn),form='unformatted')
      read(22)s_h
      close(22)
!----------------------------------------------------------

!     Calculate lon/lat
!----------------------------------------------------------
      do i=1,II
         lon(i)=(i-0.5)/12.0
      end do

      do j=1,JJ
         lat(j)=(j-0.5)/12.0-90.0
      end do
!----------------------------------------------------------

!     Initialise variables
!----------------------------------------------------------
      if(rank == 0)then
         h = 0.0
      end if

      allocate(h_proc(II,JJstrip))
      h_proc = 0.0
!----------------------------------------------------------

!     Calculate the weights for set profiles
!     and save result to file
!----------------------------------------------------------
      write(*,*)'beginning computation by proc = ',rank
      do j=j1,j2

         write(*,*)'computing mdt for row',j

         call sh2grid_mfc(NNmax,NN,c_h,s_h,II,&
                           &dble(lon),dble(lat(j)),h_lat)
     
         h_proc(:,j-j1+1) = sngl(h_lat)

      end do
      write(*,*)'computation by proc = ',rank,' completed'
!----------------------------------------------------------

!     Combine the results from individual processes
!----------------------------------------------------------
      if(rank.ne.0)then

        ! Send j index of first latitude computed by process
        !-----------------------------------
         tag = 0
         call MPI_Send(j1,1,MPI_INTEGER,0,tag,MPI_COMM_WORLD,err)
        !-----------------------------------

        ! Send j index of last latitude computed by process
        !-----------------------------------
         tag = tag + 1
         call MPI_Send(j2,1,MPI_INTEGER,0,tag,MPI_COMM_WORLD,err)
        !-----------------------------------

        ! Send the regridded rows computed by proccess
        !-----------------------------------
         do j=1,JJstrip
            tag = tag + 1
            call MPI_Send(h_proc(:,j),II,&
                                  &MPI_FLOAT,0,tag,MPI_COMM_WORLD,err)
         end do
        !-----------------------------------

      else

        !-----------------------------------
         h(:,j1:j2) = h_proc
        !-----------------------------------

        !-----------------------------------
         do n=1,nprocs-1

           ! Receive j index of first latitude computed by process
           !-----------------------------------
            write(*,*)&
               &'receiving index of first lat computed by proc = ',n
            tag = 0
            call MPI_Recv(j1,1,MPI_INTEGER,n,tag,MPI_COMM_WORLD,status,err)
            write(*,*)'index of 1st lat computed by proc       = ',j1
           !-----------------------------------

           ! Receive j index of last latitude computed by process
           !-----------------------------------
            write(*,*)&
               &'receiving index of last lat computed by proc = ',n
            tag = tag + 1
            call MPI_Recv(j2,1,MPI_INTEGER,n,tag,MPI_COMM_WORLD,status,err)
            write(*,*)'index of last lat computed by proc       = ',j2
           !-----------------------------------

           ! Receive regridded rows computed by process
           !-----------------------------------
            write(*,*)&
               &'receiving results from proc = ',n
            do j=j1,j2
               tag = tag + 1
               call MPI_Recv(h(:,j),II,&
                                    &MPI_FLOAT,n,tag,MPI_COMM_WORLD,status,err)
            end do
            write(*,*)&
               &'received results from proc = ',n
           !-----------------------------------

         end do
        !-----------------------------------

        !Write results to file 
        !-----------------------------------
         write(*,*)'-----------------------------------------'
         write(*,*)'writing results to file'

         open(21,file=trim(pout)//'orca0083_mdt12th_do2190.dat',&
         &form='unformatted')
         write(21)h
         close(21)

         write(*,*)'results written to file'
         write(*,*)'-----------------------------------------'
        !-----------------------------------

      end if
!----------------------------------------------------------

!-------------------------------------------------
      call MPI_Finalize()
!-------------------------------------------------

      stop

!===========================================================  
      end program main


      subroutine sh2grid_mfc(NNmax,NN,c_n,s_n,II,lon_d,lat_d,gh) 
!=====================================================================
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
!     Description:
!        Regrids a field from its sh coefficients along a parallel using the 
!        modified forward column method
!    
!     Created by:    
!        Rory Bingham
!        25/02/17
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

!     Start of declarations
!-----------------------------------------------------------

      implicit none

!     Input variables
!-------------------------------------------------
      integer,intent(in) :: NNmax             ! Max degree and order 
      integer,intent(in) :: NN                ! Require degree and order 
      integer,intent(in) :: II                ! Number of longitude points 

      real*8,intent(in)  :: c_n(0:NNmax,0:NNmax)    ! Gravity model sh coeffs
      real*8,intent(in)  :: s_n(0:NNmax,0:NNmax)

      real*8,intent(in)  :: lon_d(II)         ! Longitude in degrees
      real*8,intent(in)  :: lat_d             ! Geodetic latitude in degrees
!-------------------------------------------------

!     Output variables
!-------------------------------------------------
      real*8,intent(out) :: gh(II)   !geoid height along parallel
!-------------------------------------------------

!     Local variables
!-------------------------------------------------
      real*8,parameter :: pi=4.0d0*datan(1.0d0)

     !These values correspond to the GRS80 reference ellipsoid
      real*8,parameter :: a=6378137.3d0            ! Equatorial raidus
      real*8,parameter :: f=1.0d0/298.257222101d0  ! Recipricol of flattenning

      real*8   :: alf(0:NN,0:NN)   ! Associated Legendre functions
      real*8   :: w(0:NN)          ! Gaussian filter weights
      real*8   :: gamma            ! Ellipsoid normal gravity
      real*8   :: lon(II)          ! Lon points (rads)
      real*8   :: cltgd            ! Geodetic latitudes (rads)
      real*8   :: cltgc            ! Geocentric latitudes
      real*8   :: b                ! Semi-minor axis
      real*8   :: rr               ! Ellipsoidal radius
      real*8   :: a2,b2                            
      real*8   :: a2sin2,b2cos2
      real*8   :: rho,rho2
      real*8   :: sindefl,cosdefl
      real*8   :: dcoslat
      real*8   :: rpwr(0:NN),cnst

      integer  :: i,j,n,m
      real*8   :: lons,u,sf
      real*8   :: x1(0:NN),x2(0:NN)
      real*8   :: x(0:NN,2),x_new(0:NN,2)
!-------------------------------------------------

!-----------------------------------------------------------
!     End of declarations


!     Start of proceedure
!-----------------------------------------------------------

!     Initialise arrays
!-------------------------------------------------
      lon(:)=0.0d0
      w(:)=0.0d0
      rpwr(:)=0.0d0
      alf(:,:)=0.0d0
      gh(:)=0.0d0
!-------------------------------------------------

!     Calculate longitude, geodetic colatiude, and
!     goecentric colatitude arrays for points on grid
!-------------------------------------------------
      lon(:)=dble(lon_d(:))*pi/180.0d0
      lons=lon(2)-lon(1)
 
      b=a*(1.0d0-f)
      a2=a*a
      b2=b*b
      cltgd=dble(90.0-lat_d)*pi/180.0d0
      a2sin2=a2*dsin(cltgd)**2
      b2cos2=b2*dcos(cltgd)**2
      rho2=a2sin2+b2cos2
      rho=dsqrt(rho2)
      rr=dsqrt(a2*a2sin2+b2*b2cos2)/rho
      cosdefl=rho/rr
      sindefl=(a2-b2)*dsin(cltgd)*dcos(cltgd)/(rho*rr)
      dcoslat=dcos(cltgd)*cosdefl-dsin(cltgd)*sindefl
      cltgc=dacos(dcoslat)
!-------------------------------------------------

!     Compute the associated legendre functions
!     at required goedetic colatitude for
!     each required degree and order.
!-------------------------------------------------
      call alf_mfc(NN,cltgc,alf)         

      u=dsin(cltgc)    
      sf=1.0d280
!-------------------------------------------------

!-------------------------------------------------
      x(:,:)=0.0d0
      do i=1,2

         x1(:)=0.0d0
         x2(:)=0.0d0
         do m=0,NN
            x1(m)=sum(alf(m:NN,m)*c_n(m:NN,m))
            x2(m)=sum(alf(m:NN,m)*s_n(m:NN,m))
            x(m,i)=x1(m)*dcos(m*lon(i))+x2(m)*dsin(m*lon(i))
         end do

         gh(i)=x(NN,i)*u
         do m=NN-1,1,-1
            gh(i)=(gh(i)+x(m,i))*u
         end do
         gh(i)=gh(i)+x(0,i)

      end do

      do i=3,II

         x_new(:,1)=x(:,2)
         do m=0,NN
            x_new(m,2)=2.0d0*dcos(m*lons)*x(m,2)-x(m,1)
         end do
         x(:,:)=x_new(:,:)

         gh(i)=x(NN,2)*u
         do m=NN-1,1,-1
            gh(i)=(gh(i)+x(m,2))*u
         end do
         gh(i)=gh(i)+x(0,2)

      end do

      gh(:)=gh(:)*sf 
!-------------------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine sh2grid_mfc

      subroutine alf_mfc(nn,cltgc,p)
!=====================================================================
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
!     Description:
!     Calculates scaled versions of the fully normalised 
!     associated legendre functions at a particular latitude
!     according to the 1st modified forward column (MFC) method 
!     as decribed in Holmes and Featherstone 2002
!     
!     Created by:    
!        Rory Bingham
!        09/11/2016   
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


!     Start of declarations
!-----------------------------------------------------------

      implicit none

!     Input variables
!---------------------------------------
      integer,intent(in)      :: NN          ! Max degree and order
      real*8,intent(in)       :: cltgc       ! Goecentric colatitude
!---------------------------------------

!     Output variables
!---------------------------------------
      real*8,intent(out)     :: p(0:NN,0:NN)
!---------------------------------------

!     Local variables
!---------------------------------------
      !additional scale factor to prevent overflow
      real*8, parameter :: sf=1.0d-280 

      integer :: n,m
      real*8  :: a,b
!---------------------------------------

!-----------------------------------------------------------
!     End of declarations


!     Start of proceedure
!-----------------------------------------------------------

!     Initialise arrays
!---------------------------------------
      p(:,:)=0.0d0
!---------------------------------------

!     
!---------------------------------------
      p(0,0)=sf*1.0d0
      
      p(1,1)=sf*dsqrt(3.0d0)    
      
      do n=2,NN
         p(n,n)=dsqrt((2.0d0*n+1.0d0)/(2.0d0*n))*p(n-1,n-1)
      end do

      !Do m=0 separately to aviod out of bounds referencing on p(n-2,m)
      !for n=1 b=0 in anycase      
      m=0
      n=1
      a=dsqrt(((2.0d0*n-1.0d0)*(2.0d0*n+1.0d0))/((n-m)*(n+m)))
      p(n,m)=a*dcos(cltgc)*p(n-1,m)
      do n=2,NN
         a=dsqrt(((2.0d0*n-1.0d0)*(2.0d0*n+1.0d0))/((n-m)*(n+m)))
         b=dsqrt(((2.0d0*n+1.0d0)*(n+m-1.0d0)*(n-m-1.0d0))/&
                                &((n-m)*(n+m)*(2.0d0*n-3.0d0)))
         p(n,m)=a*dcos(cltgc)*p(n-1,m)-b*p(n-2,m)
      end do
      do m=1,NN
         do n=m+1,NN
            a=dsqrt(((2.0d0*n-1.0d0)*(2.0d0*n+1.0d0))/((n-m)*(n+m)))
            b=dsqrt(((2.0d0*n+1.0d0)*(n+m-1.0d0)*(n-m-1.0d0))/&
                                   &((n-m)*(n+m)*(2.0d0*n-3.0d0)))
            p(n,m)=a*dcos(cltgc)*p(n-1,m)-b*p(n-2,m)
         end do
      end do
!---------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine alf_mfc

      subroutine legendre(cltgc,nn,mm,p)
!=====================================================================
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
!     Description:
!     Calculates the fully normalised associated legendre functions
!     at a particular latitude based on the definitions given by
!     Holmes and Featherstone 2002 sec. 2.1
!     
!     Created by:    
!        Rory Bingham
!
!     Created on:
!        14/11/06   
!
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


!     Start of declarations
!-----------------------------------------------------------

      implicit none

!     Input variables
!---------------------------------------
      integer,intent(in)      :: NN,MM       ! Max degree and order
      real*8,intent(in)       :: cltgc       ! Goecentric colatitude
!---------------------------------------

!     Output variables
!---------------------------------------
      real*8,intent(out)     :: p(0:NN,0:MM)
!---------------------------------------

!     Local variables
!---------------------------------------
      integer :: n,m
      real*8  :: a,b
!---------------------------------------

!-----------------------------------------------------------
!     End of declarations


!     Start of proceedure
!-----------------------------------------------------------

!     Initialise arrays
!---------------------------------------
      p(:,:)=0.0d0
!---------------------------------------

!     
!---------------------------------------
      p(0,0)=1.0d0  

      p(1,1)=dsqrt(3.0d0)*dsin(cltgc)    
      
      do n=2,NN
         p(n,n)=dsin(cltgc)*dsqrt((2.0d0*n+1.0d0)/(2.0d0*n))*p(n-1,n-1)
      end do
      
      m=0
      n=1
      a=dsqrt(((2.0d0*n-1.0d0)*(2.0d0*n+1.0d0))/((n-m)*(n+m)))
      p(n,m)=a*dcos(cltgc)*p(n-1,m)
      do n=2,NN
         a=dsqrt(((2.0d0*n-1.0d0)*(2.0d0*n+1.0d0))/((n-m)*(n+m)))
         b=dsqrt(((2.0d0*n+1.0d0)*(n+m-1.0d0)*(n-m-1.0d0))/&
                                &((n-m)*(n+m)*(2.0d0*n-3.0d0)))
         p(n,m)=a*dcos(cltgc)*p(n-1,m)-b*p(n-2,m)
      end do
      do m=1,NN
         do n=m+1,NN
            a=dsqrt(((2.0d0*n-1.0d0)*(2.0d0*n+1.0d0))/((n-m)*(n+m)))
            b=dsqrt(((2.0d0*n+1.0d0)*(n+m-1.0d0)*(n-m-1.0d0))/&
                                   &((n-m)*(n+m)*(2.0d0*n-3.0d0)))
            p(n,m)=a*dcos(cltgc)*p(n-1,m)-b*p(n-2,m)
         end do
      end do
!---------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine legendre


      subroutine gd2gc_clat(a,f,II,JJ,lon,cltgd,cltgc,ds)
!=====================================================================
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
!     Description:
!        Converts geodetic (map) colatitude to 
!        geocentric (math) colatitude.
!     
!     Created by:    
!        Rory Bingham
!
!     Created on:
!        14/11/06    
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


!     Start of declarations
!-----------------------------------------------------------

      implicit none

!     Input variables
!---------------------------------------
      integer              :: II,JJ       ! No at lon/lat points
      real*8,intent(in)    :: a           ! Equatorial radius
      real*8,intent(in)    :: f           ! Recipricol of flattening
      real*8,intent(in)    :: cltgd(JJ)   ! Geodetic colatitude
      real*8,intent(in)    :: lon(II)     ! Longitude array
!---------------------------------------

!     Output variables
!---------------------------------------
      real*8,intent(out)   :: cltgc(JJ)   ! Geodcentric colatitude
      real*8,intent(out)   :: ds(JJ)      ! Area elements
!---------------------------------------

!     Local variables
!---------------------------------------
      integer  :: j
      real*8   :: pi
      real*8   :: b              ! Semi-minor axis
      real*8   :: rr             ! Ellipsoidal radius
      real*8   :: a2,b2          !  at a particular latitude
      real*8   :: a2sin2,b2cos2
      real*8   :: rho,rho2
      real*8   :: sindefl,cosdefl
      real*8   :: dcoslat
      real*8   :: ltlm(JJ,2)
!---------------------------------------

!-----------------------------------------------------------
!     End of declarations

!     Start of proceedure
!-----------------------------------------------------------

!     Initialise arrays
!---------------------------------------
      cltgc(:)=0.0d0
      ltlm(:,:)=0.0d0
!---------------------------------------

!     Convenience parameter
!---------------------------------------
      pi=4.0d0*datan(1.0d0)
!---------------------------------------

!     Compute goecentric colatitude     
!---------------------------------------
      b=a*(1.0d0-f)
      a2=a*a
      b2=b*b
      do j=1,JJ
         a2sin2=a2*dsin(cltgd(j))**2
         b2cos2=b2*dcos(cltgd(j))**2
         rho2=a2sin2+b2cos2
         rho=dsqrt(rho2)
         rr=dsqrt(a2*a2sin2+b2*b2cos2)/rho
         cosdefl=rho/rr
         sindefl=(a2-b2)*dsin(cltgd(j))*dcos(cltgd(j))/(rho*rr)
         dcoslat=dcos(cltgd(j))*cosdefl-dsin(cltgd(j))*sindefl
         cltgc(j)=dacos(dcoslat)
      end do
!---------------------------------------

!     Calculate area elements for each latitude interval.
!     Takes into account the fact that we are now working
!     with an latitudionally irregularly spaced grid. Used
!     for when integrating to compute spherical harm coeffs.
!----------------------------------------------------------------
      ltlm(1,1)=pi-cltgc(1)
      ltlm(1,2)=0.5d0*(cltgc(1)-cltgc(2))
      do j=2,JJ-1
         ltlm(j,1)=ltlm(j-1,2)
         ltlm(j,2)=0.5d0*(cltgc(j)-cltgc(j+1))
      end do
      ltlm(JJ,1)=ltlm(JJ-1,2)
      ltlm(JJ,2)=cltgc(JJ)

      do j=1,JJ
         ds(j)=0.5d0*(lon(2)-lon(1))*sum(ltlm(j,:))&
               &*(dsin(cltgc(j)-ltlm(j,1))+dsin(cltgc(j)+ltlm(j,2)))
      end do
!---------------------------------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine gd2gc_clat
