!     Compile with:
!     mpif90 orca0083_mdt2shc.f90 -o orca0083_mdt2shc -O3 -fbounds-check

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

     !Maximum degree for expansion
     !--------------------------------------------
      integer,parameter :: NN=2190
     !--------------------------------------------
      
     !Maximum number of rows for a calculation
     !--------------------------------------------
      integer,parameter :: max_rows=45
     !--------------------------------------------

     !Inputs
     !--------------------------------------------
      real   :: h(II,JJ)
     !--------------------------------------------

     !Outputs
     !--------------------------------------------
      real*8 :: c_h(0:NN,0:NN)          ! Sh coeffs of MSS
      real*8 :: s_h(0:NN,0:NN)          ! Sh coeffs of MSS
     !--------------------------------------------

     !Local
     !--------------------------------------------
      integer :: i,j,m,n
      integer :: nrows
      integer :: j1,j2,JJstrip
      integer :: nrows_sub
      integer :: j1_sub,j2_sub,JJstrip_sub
      integer :: l,LL
      real    :: lon(II),lat(JJ)
      real*8  :: c_h_tmp(0:NN,0:NN) 
      real*8  :: s_h_tmp(0:NN,0:NN)
      real*8  :: c_h_proc(0:NN,0:NN) 
      real*8  :: s_h_proc(0:NN,0:NN)
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
      write(*,*)'limits for process',rank,':'
      write(*,*)j1,j2
!----------------------------------------------------------

!     Determine how many sub-strips the process will need
!     to compute given the maximum number of latitudes that
!     can be held in memory
!----------------------------------------------------------
      JJstrip = j2 - j1 + 1
      if(JJstrip.gt.max_rows)then
         LL = ceiling(1.0*JJstrip/max_rows)
         nrows_sub = max_rows
      else
        LL = 1
        nrows_sub = JJstrip 
      end if
      write(*,*)'number of sub-strips for process',rank,':'
      write(*,*)LL
      write(*,*)'number of rows per sub-strips for process',rank,':'
      write(*,*)nrows_sub
!----------------------------------------------------------

!     Specify paths
!----------------------------------------------------------
      path0='/newhome/rb13801/rdsf/data/'

      !pin1=trim(path0)//&
      !         &'bin/nemo/orca0083/n06/mnthly/ssh/'
      pin1='./data/in/'

      !pout=trim(path0)//&
      !      &'projects/cage/mdt/'
      pout='./data/out/'
!----------------------------------------------------------

!     Read in MDT
!----------------------------------------------------------
      open(21,file=trim(pin1)//'orca0083_mdt_12th_filled_no_mean.dat',&
      &form='unformatted')
      read(21)h
      close(21)
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
         c_h = 0.0d0
         s_h = 0.0d0
      end if

      c_h_proc = 0.0d0
      s_h_proc = 0.0d0
!----------------------------------------------------------

!----------------------------------------------------------
      write(*,*)'beginning computation by proc = ',rank
      do l=1,LL

         write(*,*)'sub-strip =',l

        !Determine j limits for sub-strip calculation
         j1_sub = nrows_sub*(l-1) + j1
         j2_sub = j1_sub + nrows_sub - 1
         j2_sub = min(j2_sub, j2) 
         JJstrip_sub = j2_sub - j1_sub + 1 

         write(*,*)'sub-strip limits =',j1_sub,j2_sub,JJstrip_sub

        !Calculate contribution from range
         c_h_tmp(:,:)=0.0d0
         s_h_tmp(:,:)=0.0d0
         call grid2sh_lat_rng(II,JJ,j1_sub,j2_sub,&
                             &h(:,j1_sub:j2_sub),&
                             &lon,lat,NN,c_h_tmp,s_h_tmp)

        !Add contribution from sub-strip
         c_h_proc = c_h_proc + c_h_tmp
         s_h_proc = s_h_proc + s_h_tmp

      end do
      write(*,*)'computation by proc = ',rank,' completed'
!----------------------------------------------------------

!     Combine the results from individual processes
!----------------------------------------------------------
      if(rank.ne.0)then


        ! Send the C coeffs found by the process
        !-----------------------------------
         tag = 0
         do m=0,NN
            tag = tag + 1
            call MPI_Send(c_h_proc(:,m),NN,&
                                  &MPI_DOUBLE,0,tag,MPI_COMM_WORLD,err)
         end do
        !-----------------------------------

        ! Send the S coeffs found by the process
        !-----------------------------------
         do m=0,NN
            tag = tag + 1
            call MPI_Send(s_h_proc(:,m),NN,&
                                  &MPI_DOUBLE,0,tag,MPI_COMM_WORLD,err)
         end do
        !-----------------------------------

      else

        !-----------------------------------
         c_h = c_h + c_h_proc
         s_h = s_h + s_h_proc
        !-----------------------------------

        !-----------------------------------
         do n=1,nprocs-1

           ! Receive C coeffs found by process
           ! and add them to the overall C calc
           !-----------------------------------
           write(*,*)&
               &'receiving C coeffs from proc = ',n
            tag = 0
            do m=0,NN
               tag = tag + 1
               call MPI_Recv(c_h_tmp(:,m),NN,&
                                    &MPI_DOUBLE,n,tag,MPI_COMM_WORLD,status,err)
               c_h(:,m) = c_h(:,m) + c_h_tmp(:,m)
            end do
            write(*,*)&
               &'received C coeffs from proc = ',n
            write(*,*)'-----------------------------------------'
           !-----------------------------------

           ! Receive S coeffs found by process
           ! and add them to the overall S calc
           !-----------------------------------
           write(*,*)&
               &'receiving S coeffs from proc = ',n
            do m=0,NN
               tag = tag + 1
               call MPI_Recv(s_h_tmp(:,m),NN,&
                                    &MPI_DOUBLE,n,tag,MPI_COMM_WORLD,status,err)
               s_h(:,m) = s_h(:,m) + s_h_tmp(:,m)
            end do
            write(*,*)&
               &'received S coeffs from proc = ',n
            write(*,*)'-----------------------------------------'
           !-----------------------------------

         end do
        !-----------------------------------

        !Write results to file 
        !-----------------------------------
         write(*,*)'-----------------------------------------'
         write(*,*)'writing results to file'

         fn='orca0083_mdt_sh_c.dat'
         open(22,file=trim(pout)//trim(fn),form='unformatted')
         write(22)c_h
         close(22)

         fn='orca0083_mdt_sh_s.dat'
         open(22,file=trim(pout)//trim(fn),form='unformatted')
         write(22)s_h
         close(22)

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

      subroutine grid2sh_lat_rng(II,JJ,j1,j2,h,lon,lat,NN,c,s)
!=====================================================================
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
!     Description:
!        Calculates the spherical harmonic coefficient 
!        contribution from a specified lat range of gridded field.
!
!     Warning: Since this routine pre-computes all the Legendre polynomials
!     for the latitude range this may limit the range over which the routine
!     will work otherwise the array storing the polynomials may require too
!     much memory. For d/o=2500 the array for a single latitude is ~25MB.
!     So 40 lines of latitude requires ~1GB of memory. 
!
!     Note: The inputs to this version are given as real*4 
!
!     Created by:    
!        Rory Bingham
!
!     Created on:
!        23/02/2017
!
!     Update 25/08/2020:
!        To save memory, only the required strip of the input field
!        is passed to the subroutine. Note, the full latitude is still
!        passed to prevent issues with the geocentric calculation in
!        cased where only one row is to be calculated (ie j1=j2) 
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      implicit none

!     Start of declarations
!-----------------------------------------------------------

!     Input variables
!---------------------------------------
      integer,intent(in)   :: II,JJ                ! Lon/lat dimensions
      integer,intent(in)   :: j1,j2                ! Lat limits for calculation
      real,intent(in)      :: lon(II),lat(JJ)      ! Lon/lat of grid points in degrees
      real,intent(in)      :: h(II,j2-j1+1)        ! Gridded field
      integer,intent(in)   :: NN                   ! Max degree required
!---------------------------------------

!     Output variables
!---------------------------------------
      real*8,intent(out)       :: c(0:NN,0:NN)     ! SH coeffs
      real*8,intent(out)       :: s(0:NN,0:NN) 
!---------------------------------------

!     Local variables
!---------------------------------------
      real*8,parameter :: pi=4.0d0*datan(1.0d0)
      real*8,parameter :: torad=pi/180.0d0

     !These values correspond to the GRS80 reference ellipsoid
      real*8,parameter :: a=6378137.3d0            ! Equatorial raidus
      real*8,parameter :: f=1.0d0/298.257222101d0  ! Recipricol of flattenning

      real*8,parameter :: sf=1.0d280  !Scale factor for ALF computation

      integer  :: i,j,k,n,m
      real*8   :: lon_r(II)                             ! Lon points (rads)
      real*8   :: cltgd(JJ)                             ! Geodetic latitudes (rads)
      real*8   :: cltgc(JJ)                             ! Geocentric latitudes
      real*8   :: ds(JJ)                                ! Grid box areas
      real*8   :: np(0:NN,0:NN,j2-j1+1)                 ! Associated Legendre polys
      real*8   :: cos_eval(0:NN,II),sin_eval(0:NN,II),aa(II)
      real*8   :: haw(II,j2-j1+1)
!---------------------------------------

!-----------------------------------------------------------
!     End of declarations


!     Start of proceedure
!-----------------------------------------------------------

!     Initialise arrays
!---------------------------------------
      lon_r(:)=0.0d0
      cltgd(:)=0.0d0
      cltgc(:)=0.0d0
      ds(:)=0.0d0
      np(:,:,:)=0.0d0
!---------------------------------------

!     Calculate longitude, geodetic colatiude, and
!     goecentric colatitude arrays for points on grid
!---------------------------------------
      lon_r(:)=dble(lon(:))*pi/180.0d0

      do j=1,JJ
         cltgd(j)=dble(90.0d0-lat(j))*pi/180.0d0
      end do

      call gd2gc_clat(a,f,II,JJ,lon_r,cltgd,cltgc,ds)
!---------------------------------------

!     Calculate fully normalised ALFs at each latitude 
!---------------------------------------
      do j=j1,j2
         call legendre17(cltgc(j),nn,nn,np(:,:,j-j1+1),sf) 
      end do      
!---------------------------------------

!     Evaluate the latitude independent quantites 
!---------------------------------------
      do i=1,II
         do m=0,NN
            cos_eval(m,i)=dcos(m*lon_r(i))                        
            sin_eval(m,i)=dsin(m*lon_r(i))
         end do
      end do
      
      ds(:)=ds(:)/(4.0d0*pi)
!---------------------------------------

!     Appy area weighting
!---------------------------------------
      do j=j1,j2
         haw(:,j-j1+1)=dble(h(:,j-j1+1))*ds(j)
      end do
!---------------------------------------

!     Calculate the spherical harmonic coefficients
!---------------------------------------
      c(:,:)=0.0d0
      s(:,:)=0.0d0
      do n=0,NN
         !write(*,*)n
         do m=0,n
            do i=1,II
               aa(i)=sum(haw(i,:)*np(n,m,:))
            end do
            c(n,m)=sum(cos_eval(m,:)*aa(:))
            s(n,m)=sum(sin_eval(m,:)*aa(:))
         end do
      end do
!---------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine grid2sh_lat_rng

      subroutine legendre17(cltgc,nn,mm,p,sf)
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
!     Modified 02/03/2017 to make more stable for high=d/o
!     Achieved primarily by introducing a scale factor 
!     (of 1.0d280)
!
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


!     Start of declarations
!-----------------------------------------------------------

      implicit none

!     Input variables
!---------------------------------------
      integer,intent(in)      :: NN,MM       ! Max degree and order
      real*8,intent(in)       :: cltgc       ! Goecentric colatitude
      real*8, intent(in)      :: sf          ! Scale factor
!---------------------------------------

!     Output variables
!---------------------------------------
      real*8,intent(out)     :: p(0:NN,0:MM)
!---------------------------------------

!     Local variables
!---------------------------------------
      integer :: n,m
      real*8  :: a,b,tmp
      real*8  :: u,t
      real*8  :: nd(0:2*NN+1)
      real*8  :: ns(0:2*NN+1)
      real*8  :: nsi(0:2*NN+1)
!---------------------------------------

!-----------------------------------------------------------
!     End of declarations


!     Start of proceedure
!-----------------------------------------------------------

!     Fixed paramters
!---------------------------------------

      u=dsin(cltgc)
      t=dcos(cltgc)
!---------------------------------------

!     Convenience values
!---------------------------------------
      nd(0)=0.0d0
      ns(0)=0.0d0
      nsi(0)=0.0d0
      do n=1,2*NN+1
         nd(n)=dble(n*1.0d0)
         ns(n)=dsqrt(nd(n))
         nsi(n)=1.0d0/ns(n)
      end do
!---------------------------------------

!     Initialise arrays
!---------------------------------------
      p(:,:)=0.0d0
!---------------------------------------

!     Compute sectorial ALFs
!---------------------------------------
      p(0,0)=1.0d0*sf  

      p(1,1)=ns(3)*u*sf   
      
      do n=2,NN
         p(n,n)=u*ns(2*n+1)*nsi(2*n)*p(n-1,n-1)
      end do
!---------------------------------------

!     Compute non-sectorial ALFs 
!---------------------------------------
      m=0
      n=1
      a=ns(2*n-1)*ns(2*n+1)*nsi(n-m)*nsi(n+m)
      p(n,m)=a*dcos(cltgc)*p(n-1,m)
      do n=2,NN
         tmp=ns(2*n+1)*nsi(n-m)*nsi(n+m)
         a=tmp*ns(2*n-1)
         b=tmp*ns(n+m-1)*ns(n-m-1)*nsi(2*n-3)
         p(n,m)=a*t*p(n-1,m)-b*p(n-2,m)
      end do

      do m=1,NN
         do n=m+1,NN
            tmp=ns(2*n+1)*nsi(n-m)*nsi(n+m)
            a=tmp*ns(2*n-1)
            b=tmp*ns(n+m-1)*ns(n-m-1)*nsi(2*n-3)
            p(n,m)=a*t*p(n-1,m)-b*p(n-2,m)
         end do
      end do
!---------------------------------------

!     Re-scale
!---------------------------------------
      p(:,:)=p(:,:)/sf
!---------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine legendre17

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
         !write(*,*)j,ds(j)
      end do
      !stop
!---------------------------------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine gd2gc_clat
