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

!     For geoid data
!-------------------------------------------------
      real*8,allocatable  :: c_n(:,:)     ! Sh coeffs for MSS
      real*8,allocatable  :: s_n(:,:)           
      real*8,allocatable  :: mss0(:),lat(:),lon(:)
      real,allocatable    :: mss(:,:)
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
      character(len=128) :: path0,pin,pout,fout
!-------------------------------------------------
!===========================================================    

!     Start of proceedure
!===========================================================      
!-------------------------------------------------
      pin='./data/src/'
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

      allocate(mss0(II))
      allocate(mss(II,JJ))
      allocate(lon(II))
      allocate(lat(JJ))
      allocate(c_n(0:NN_mss,0:NN_mss))
      allocate(s_n(0:NN_mss,0:NN_mss))
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

!     Read in gravity coefficients
!-------------------------------------------------
      write(*,*)'Reading in sh coefficients...'

      open(11,file=trim(pin)//trim(name_mss)//'.dat',form='unformatted')
      read(11)c_n
      read(11)s_n
      close(11)
!-------------------------------------------------

!     Compute field (mss/geoid) on grid 
!-------------------------------------------------
      do j=1,JJ

         write(*,*)'computing mss for row',j

         call sh2grid_mfc(NN,c_n(0:NN,0:NN),s_n(0:NN,0:NN)&
                                             &,II,lon,lat(j),mss0)
     
         mss(:,j)=mss0(:)

      end do
!-------------------------------------------------

!-------------------------------------------------
      write(nn_str,'(I4)')NN
      write(rr_str,'(I4)')rr
      do i=1,4
         if(nn_str(i:i).eq.' ')nn_str(i:i)='0'
         if(rr_str(i:i).eq.' ')rr_str(i:i)='0'
      end do

      fout=trim(name_mss)//'_do'//nn_str//'_rr'//rr_str//'.dat'

      open(20,file=trim(pout)//trim(fout),&
      &form='unformatted')
      write(20)mss
      close(20)
!-------------------------------------------------

      stop

!===========================================================  
      end program main

      subroutine sh2grid_mfc(NN,c_n,s_n,II,lon_d,lat_d,gh) 
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
      integer,intent(in) :: NN                ! Max degree and order 
      integer,intent(in) :: II                ! Number of longitude points 

      real*8,intent(in)  :: c_n(0:NN,0:NN)    ! Gravity model sh coeffs
      real*8,intent(in)  :: s_n(0:NN,0:NN)

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
         write(*,*)j,cltgd(j),cltgc(j),'chk'
      end do
      !stop
!---------------------------------------

!     Calculate area elements for each latitude interval.
!     Takes into account the fact that we are now working
!     with an latitudionally irregularly spaced grid. Used
!     for when integrating to compute spherical harm coeffs.
!----------------------------------------------------------------
      ltlm(1,1)=pi-cltgc(1)
      ltlm(1,2)=0.5*(cltgc(1)-cltgc(2))
      do j=2,JJ-1
         ltlm(j,1)=ltlm(j-1,2)
         ltlm(j,2)=0.5*(cltgc(j)-cltgc(j+1))
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

      subroutine ref_pot(a,f,gm,omega,NN,c_ref)
!=====================================================================
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
!     Description: 
!        Calculates the spherical harmonic coeffs
!        for the gravitational potential a specified 
!        reference ellipsiod.
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
      integer,intent(in)     :: NN                 ! Max degree required 
      real*8,intent(in)      :: a                  ! Ellipsoid semi-major axis
      real*8,intent(in)      :: f                  ! Recipricol of flattening
      real*8,intent(in)      :: gm                 ! Ellipsoid gravity mass constant
      real*8,intent(in)      :: omega              ! Rotation - angular velocity
!---------------------------------------

!     Output variables
!---------------------------------------
      real*8,intent(out) :: c_ref(0:NN,0:NN)       !Coefficients of refernce potential
!---------------------------------------

!     Local variables
!---------------------------------------
      integer :: n
      real*8  :: b                                 ! Ellipsoid semi-minor axis
      real*8  :: e0    	                           ! Linear eccentricity
      real*8  :: e1    	                           ! First eccentricity
      real*8  :: e2    	                           ! Second eccentricity      
      real*8  :: ratio
      real*8  :: q0
      real*8  :: j2                                ! Dynamic form factor
!---------------------------------------

!-----------------------------------------------------------
!     End of declarations


!     Start of proceedure
!-----------------------------------------------------------

!     Initialise arrays
!---------------------------------------
      c_ref(:,:)=0.0d0
!---------------------------------------

!     Derived paramters
!---------------------------------------
      b=a*(1.0d0-f)
      
      e0=dsqrt(a**2-b**2)
      
      e1=e0/a
      
      e2=e0/b
      
      ratio=((omega*a)**2)*b/gm
      
      q0=0.5d0*(1.0d0+(3.0d0/(e2**2)))*datan(e2)-3.0d0/(2.0d0*e2) 
            
!      j2=(e1**2)*(1.0d0-2.0d0*ratio*e2/(15.0d0*q0))/3.0d0      
     
      j2=dsqrt(5.0d0)*4.8416942378999d-4
!---------------------------------------

!
!---------------------------------------
      c_ref(2,0)=-j2/dsqrt(5.0d0)
      
      do n=2,NN/2
	 		if(abs(c_ref(2*(n-1),0)).lt.1.0d-100)then
	    		c_ref(2*n,0)=0.0
	 		else
	    		c_ref(2*n,0)=-((-1.0d0)**(n+1))*((3.0d0*e1**(2*n))/((2*n+1)*(2*n+3)))*&
				               &(1.0d0-n+5.0d0*n*j2*a**2/e0**2)/dsqrt(1.0d0+4.0d0*n)
	 		end if
      end do
!---------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine ref_pot


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
      
      do m=0,MM
         do n=m+1,NN
	    		a=dsqrt(((2.0d0*n-1.0d0)*(2.0d0*n+1.0d0))/((n-m)*(n+m)))
	    		b=dsqrt(((2.0d0*n+1.0d0)*(n+m-1.0d0)*(n-m-1.0d0))/((n-m)*(n+m)*(2.0d0*n-3.0d0)))
	    		p(n,m)=a*dcos(cltgc)*p(n-1,m)-b*p(n-2,m)
	 		end do
      end do
!---------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine legendre


      subroutine norm_grav(a,f,gm,omega,cltgd,gamma)
!=====================================================================
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
!     Description:
!        Computes normal gravity on the ellipsoidal surface
!        using formula of Somigliana (Moritz eq 2-16) at 
!        required latitude. Note: Geodetic rather than 
!        geocentric colatitude in required as input. And 
!        colatitude is converted to latitude for the 
!        formula.
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
      real*8,intent(in)    :: a           ! Equatorial radius
      real*8,intent(in)    :: f           ! Recipricol of flattening
      real*8,intent(in)    :: gm          ! Ellipsoid gravity mass constant
      real*8,intent(in)    :: omega       ! Rotation - angular velocity
      real*8,intent(in)    :: cltgd       ! Geodetic colatitudes    
!---------------------------------------

!     Output variables
!---------------------------------------
      real*8,intent(out)     :: gamma   ! Normal gravity
!---------------------------------------

!     Local variables
!---------------------------------------
      integer :: j
      real*8  :: pi
      real*8  :: b         ! Semi-minor axis
      real*8  :: e0    	   ! Linear eccentricity
      real*8  :: e1    	   ! First eccentricity
      real*8  :: e2    	   ! Second eccentricity      
      real*8  :: ratio
      real*8  :: q0,q00
      real*8  :: gamma_a   ! Normal gravity at the equator
      real*8  :: gamma_b   ! Normal gravity at the poles
      real*8  :: phi       ! Geodetic latitude
!---------------------------------------

!-----------------------------------------------------------
!     End of declarations


!     Start of proceedure
!-----------------------------------------------------------

!     Convenience parameter
!---------------------------------------
      pi=4.0d0*datan(1.0d0)
!---------------------------------------

!     Derived parameters
!---------------------------------------
      b=a*(1.0d0-f)
      
      e0=dsqrt(a**2-b**2)
      
      e1=e0/a
      
      e2=e0/b
      
      ratio=((omega*a)**2)*b/gm
      
      q0=0.5d0*(1.0d0+(3.0d0/(e2**2)))*datan(e2)-3.0d0/(2.0d0*e2) 
      
      q00=3.0d0*(1.0d0+1.0d0/e2**2)*(1.0d0-datan(e2)/e2)-1.0d0
!---------------------------------------

!     Compute normal gravity on the ellipsoidal surface
!     using formula of Somigliana (Moritz eq 2-16)
!---------------------------------------
      gamma_a = gm*(1.0d0-ratio-ratio*e2*q00/(6.0d0*q0))/(a*b)   

      gamma_b = gm*(1.0d0+ratio*e2*q00/(3.0d0*q0))/(a*a)  

      phi = pi/2.0d0-cltgd   ! Convert to geodetic lat

		gamma = (a*gamma_a*dcos(phi)**2+b*gamma_b*dsin(phi)**2)/&
                        &dsqrt((a*dcos(phi))**2+(b*dsin(phi))**2)
!---------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine norm_grav




      subroutine gauss_av(NN,a,r,w)
!=====================================================================
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
!     Description:
!        Calulates the weights required for Gaussian
!        filtering in the spectral domain for a specified
!        half-weight radius. If no filtering is 
!        required set r to zero.
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
      integer,intent(in)   :: NN       ! Max degree
      real*8,intent(in)    :: a        ! Equatorial radius 
      real*8,intent(in)    :: r        ! Half weight radius of filter
!---------------------------------------

!     Output variables
!---------------------------------------
      real*8,intent(out)      :: w(0:NN)  !Filter weights
!---------------------------------------

!     Local variables
!---------------------------------------
      integer  :: n
      real*8   :: pi
      real*8   :: bb
!---------------------------------------

!-----------------------------------------------------------
!     End of declarations


!     Start of proceedure
!-----------------------------------------------------------


!     Define convienence parameter      
!---------------------------------------
      pi = 4.0d0*datan(1.0d0)
!---------------------------------------

!     Initialise arrays
!---------------------------------------
      w(:)=0.0d0
!---------------------------------------

!     Compute weights
!---------------------------------------
      if(r.eq.0.0d0)then
         w(:) = 1.0d0/(2.0d0*pi)
      else
         bb = dlog(2.0d0)/(1.0d0-dcos(r/a))
         w(0) = 1.0d0/(2.0d0*pi)
         w(1) = (1.0d0/(2.0d0*pi))*((1.0d0+dexp(-2.0d0*bb))/&
                  &(1.0d0-dexp(-2.0d0*bb))-1.0d0/bb)
         do n=2,NN
	 		   w(n) = w(n-2)-((2.0d0*dble(real((n-1)))+1.0d0)/bb)*w(n-1)
            if((w(n).gt.w(n-1)).or.(w(n-1).eq.0.0).or.(w(n).lt.0.0))then
	    	   	w(n)=0.0
            end if
         end do
      end if
!---------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine gauss_av


      subroutine convert_tide(c20,tide_in,tide_out)
!=====================================================================
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
!     Description:
!        Converts between tide systems
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
      real*8               :: c20         ! 
      character*4          :: tide_in     ! Given tide system
      character*4          :: tide_out    ! Required tide system
!---------------------------------------


!-----------------------------------------------------------
!     End of declarations


!     Start of proceedure
!-----------------------------------------------------------

!     Convert to required tide system
!---------------------------------------
      select case (tide_in)

         case ('free')
            select case (tide_out)
               case('free')
                  c20=c20+0.0d0
               case('mean')
                  c20=c20-1.8157d-08
               case('zero')
                  c20=c20-4.173d-9
            end select

         case('mean')
            select case (tide_out)
               case('free')
                  c20=c20+1.8157d-08
               case('mean')
                  c20=c20+0.0d0
               case('zero')
                  c20=c20+1.39844d-8
            end select

        case('zero')
            select case (tide_out)
               case('free')
                  c20=c20+4.173d-9
               case('mean')
                  c20=c20-1.39844d-8
               case('zero')
                  c20=c20+0.0d0
            end select

      end select
!---------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine convert_tide

