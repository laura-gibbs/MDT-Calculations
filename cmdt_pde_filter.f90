      program main

      implicit none

!     User control parameters
!-------------------------------------------------
      character*5, parameter :: mss = 'dtu18_do0280_rr0008' 

      character*64, parameter :: mdt_ref='cls18_mdt'
!-------------------------------------------------

!-------------------------------------------------
      integer, parameter :: NNmax=300
      integer, parameter :: II=1440,JJ=720
      integer, parameter :: TT=500 !total iterations
      !integer, parameter :: MM=4   !total no. of regions (inc. global) 
      integer, parameter :: MM=1   !total no. of regions 
      integer, parameter :: KK=9 !number of trucations

      integer :: i,j,k,l,n,m,t,t_min,rgns(MM,4),NN
      integer :: n_all(KK)
      real    :: msk1(II,JJ),msk2(II,JJ),msk(II,JJ)
      real    :: mdt(II,JJ),mdts(II,JJ,30)
      real    :: rmdt(II,JJ),ref_cs(II,JJ)
      real    :: lat(JJ),lon(II)
      real    :: mrmsd(II,JJ)
      real    :: rmsd_rgns(MM,4,0:TT)
      real    :: diff_gm(0:NNmax,0:TT)
      real    :: rmsd0(MM,0:NNmax),rmsd_min(MM,0:NNmax)
      real    :: rmsd_min_mdt(II,JJ,MM)
      integer :: rmsd_min_nits(MM,0:NNmax)
      character*3 :: trnc(KK)
      character*3   :: dmax
      character*128 :: fn,mdt_name
      character*128 :: path0,path_ref_bd,path_mss
      character*128 :: pin,pout
!-------------------------------------------------
!-------------------------------------------------

!     Set the trucations we want
!-------------------------------------------------
      data n_all/50,100,150,200,220,240,260,280,300/

      data trnc/'050','100','150','200',&
               &'220','240','260','280','300'/
!-------------------------------------------------

!     Set paths
!-------------------------------------------------
      path0='/../a_mdt_data/computations/'

      path_ref_bd=trim(path0)//'mdts/'//trim(mdt_ref)//'/'

      path_mss=trim(path0)//'mss/dtu18_do0280_rr0008/'

      pin = '/local-scratch/rb13801/scratch/goce_err/'

      pout = '/test/'
!------------------------------------------------- 
 
!     Read in land mask
!-------------------------------------------------
     !For geodetic MDT (use common MSS mask)
     ! This is also used when calculating the area means
     ! of the moving window RMSD to ensure consistency 
     ! across different reference MDTs. Since the RMSD 
     ! is calculated in a window the will be values
     ! even where the MDT is undefined, except where the
     ! the window lies entirely over land. 
     !----------------------------------
      open(20,file=trim(path0)//'mask_rr0008.dat',form='unformatted')
      read(20)msk1
      close(20)
     !----------------------------------

     !For reference MDT
     !----------------------------------
      open(20,file=trim(path0)//'mask_rr0008.dat',form='unformatted')
      read(20)msk2
      close(20)
     !----------------------------------

     !Merge masks
     !----------------------------------
      do i=1,II
         do j=1,JJ
            if((msk1(i,j).eq.0).and.(msk2(i,j).eq.0))then
               msk(i,j)=0.0
            else
               msk(i,j)=-1.9e19
            end if
         end do
      end do
     !----------------------------------

     !Mask ITF region
     !----------------------------------
      do i=478,518
         do j=318,372
            msk1(i,j)=-1.9e19
            msk2(i,j)=-1.9e19
            msk(i,j)=-1.9e19
         end do
      end do
     !----------------------------------

     !mask equator (10S-10N) 
     !----------------------------------
      !do j=320,400
      !   msk(:,j)=-1.9e19
      !end do
     !----------------------------------
!------------------------------------------------

!    Define lon and lat
!------------------------------------------------
     do i=1,II
        lon(i)=0.25*(i-0.5)
     end do

     do j=1,JJ
        lat(j)=0.25*(j-0.5)-90.0
     end do
!------------------------------------------------


!     Define regions for statistics
!     For mrmsd routine global domain defined
!     in row 1 by default.
!     Limits for QUARTER degree grid.
!------------------------------------------------
      data rgns(1,1:4)/1100,1400,440,640/ !North Atlantic
      !data rgns(2,1:4)/   1,  II,121,640/ !Global exc. poles
      !data rgns(3,1:4)/1100,1400,440,640/ !North Atlantic
      !data rgns(4,1:4)/ 750,1100,162,320/ !South Pacific
!------------------------------------------------

!-------------------------------------------------
      do k=1,KK

         n = n_all(k)

         write(*,*)'working on degree:',n
      
        !Convert degree into text string 
        !for filenames
        !------------------------------   
         write(dmax,'(I3)')n
         if(dmax(1:1).eq.' ')dmax(1:1)='0'
         if(dmax(2:2).eq.' ')dmax(2:2)='0'
        !------------------------------   

        !Read in the reference MDT
        !and compute current mag field
        !-------------------------------
         open(20,file=trim(path_ref_bd)//&
            &'cls18_mdt.dat',form='unformatted')
         read(20)rmdt
         close(20)
        !-------------------------------

        !Read in the geodetic MDT and prepare for diffusive filtering
        !Note: no need to prep for sh mdts since land has already 
        !been filled. For other mdts we fill land by linear interpolation.
        !------------------------------   
         open(21,file=trim(pin)//'mdts_L'//dmax//'.dat',&
                   &form='unformatted')
         read(21)mdts
         close(21)
        !------------------------------   

        !-------------------------------
         do m=1,19
            mdt(:,:)=mdts(:,:,m)
            if(mdt(1,1)>-1.7e7)then
               write(fn,'(A,I2,A)')'fmdt_',m,'_L'//dmax
               if(fn(6:6).eq.' ')fn(6:6)='0'
               open(20,file=trim(pout)//trim(fn)//'.dat',&
                                             &form='unformatted')
               open(21,file=trim(pout)//trim(fn)//'_rmsd.dat',&
                                             &form='unformatted')
               call rmsd_lite_2(II,JJ,lat,mdt,rmdt,msk,5.0,MM,rgns,.TRUE.,&
                        & mrmsd)
               write(20)mdt
               write(21)mrmsd
               do t=1,TT
                  write(*,*)m,t
                  call pde_fltr(II,60,JJ,121,640,-89.875,0.25,mdt)
                  call rmsd_lite_2(II,JJ,lat,mdt,rmdt,msk,5.0,MM,rgns,.TRUE.,&
                        & mrmsd)
                  write(20)mdt
                  write(21)mrmsd
               end do
               close(20)
               close(21)
            end if
         end do
        !-------------------------------
              
      end do
!-------------------------------------------------

      stop

!===========================================================  
      end program main

      subroutine rmsd_lite_2(II,JJ,lat,data,ref,msk,hw0,MM,rgns,rm_gm,&
                        &mrmsd)
!=====================================================================
      implicit none

!     Input
!---------------------------------------
      integer :: II,JJ
      real    :: lat(JJ)
      real    :: data(II,JJ)
      real    :: ref(II,JJ)
      real    :: msk(II,JJ)
      real    :: hw0
      integer :: MM !must be at least 1
      integer :: rgns(MM,4)
      logical :: rm_gm
!---------------------------------------

!     Output
!---------------------------------------
      real    :: mrmsd(II,JJ)
      real    :: rmsd_rgns(MM,4)
      real    :: diff_gm
!---------------------------------------

!     Local
!---------------------------------------       
      integer :: i,j,k,l,m,n
      integer :: j0,LL,ov,LLx(JJ)
      real    :: r,torad,latr,lats,dx(JJ),ds(JJ)
      real,allocatable :: diff(:,:)
      real    :: sm,cnt,mn
!---------------------------------------       
!-------------------------------------------------

!     Insert global domain into region definitions
!-------------------------------------------------
      !rgns(1,1)=1
      !rgns(1,2)=II
      !rgns(1,3)=1
      !rgns(1,4)=JJ
!-------------------------------------------------

!    Define geographical meta data 
!-------------------------------------------------
      r = 6371229.0
      torad = atan(1.0)/45.0

      lats=torad*(lat(2)-lat(1))

      do j=1,JJ

         latr=torad*lat(j)

         dx(j) = r*lats*cos(latr)

         ds(j) = dble(0.50*(r*lats)**2 & 
               & *(cos(latr+0.5*lats)+cos(latr-0.5*lats)))

      end do
!-------------------------------------------------

!     Calculate zonal half width of filter in 
!     grid points as function of latitude.
!     Input specifies window half-width in degrees at 45N
!-------------------------------------------------
      j0=nint(torad*(45.0-lat(1))/lats)+1

      LL=nint(hw0/(lats/torad))

      ov=3*LL !set max window width (arbitrarily)
              !at 3 times width at 45N.
              !Needed for wrapping.

      do j=1,JJ
         LLx(j) = nint(LL*dx(j0)/dx(j))
         if(LLx(j).gt.ov)LLx(j)=ov
         !write(*,*)j,LLx(j) 
      end do
!-------------------------------------------------

!     
!-------------------------------------------------
     allocate(diff(II+2*ov,JJ))

     !Compute the difference between the fields
     !taking common mask into account 
     !----------------------------------
      do i=1,II
         do j=1,JJ
            if(msk(i,j).eq.0.0)then
               diff(i+ov,j)=data(i,j)-ref(i,j)
            else
               diff(i+ov,j)=-1.9e19
            end if
         end do
      end do
     !----------------------------------

     !Calculate and remove global offset
     !----------------------------------
     !Deleted in lite version since local
     !offset is removed in anycase.
     !----------------------------------

     !Wrap the difference 
     !----------------------------------
      do j=1,JJ
         diff(1:ov,j)=diff(II+1:II+ov,j)
         diff(II+ov+1:II+2*ov,j)=diff(ov+1:2*ov,j)
      end do
     !----------------------------------
!-------------------------------------------------

!     Compute the (1) RMSD and (2) RMSD less offset
!     over arbitrary regions defined as inputs.
!     Global domain values are calculated by default.
!-------------------------------------------------
      !Deleted in lite version
!-------------------------------------------------

!     Compute the RMSD globally in a moving window 
!     whose width is set by the parameter LL
!-------------------------------------------------
      do i=1+ov,II+ov
         do j=1+LL,JJ-LL

            if(msk(i-ov,j).gt.-1.e7)then

           !Compute mean difference in moving window
           !----------------------------
            cnt=0.0
            sm=0.0
            do k=i-LLx(j),i+LLx(j)
               do l=j-LL,j+LL
                  if(diff(k,l).gt.-1.7e7)then
                     sm=sm+diff(k,l)*ds(j)
                     cnt=cnt+ds(j)
                  end if
               end do
            end do
            if(cnt.gt.0.0)then
               mn=sm/cnt
            else
               mn=-1.9e19
            end if         
           !----------------------------
            
           !Compute RMSD in window with offset
           !----------------------------
           !Deleted in lite version
           !----------------------------

           !Compute RMSD in window without offset 
           !----------------------------
            cnt=0.0
            sm=0.0
            if(mn.gt.-1.7e7)then
               do k=i-LLx(j),i+LLx(j)
                  do l=j-LL,j+LL
                     if(diff(k,l).gt.-1.7e7)then
                        sm=sm+(diff(k,l)-mn)**2*ds(j)
                        cnt=cnt+ds(j)
                     end if
                  end do
               end do
            end if
            if(cnt.gt.0.0)then
               mrmsd(i-ov,j)=sqrt(sm/cnt)
            else
               mrmsd(i-ov,j)=-1.9e19
            end if
           !----------------------------

            end if           

         end do
      end do
!-------------------------------------------------

!-------------------------------------------------
      deallocate(diff)
!-------------------------------------------------

!-------------------------------------------------
!     End of proceedure

      return

!=====================================================================
      end subroutine rmsd_lite_2

      subroutine rmsd_lite(II,JJ,lat,data,ref,msk,hw0,MM,rgns,rm_gm,&
                        &mrmsd,rmsd_rgns,diff_gm)
!=====================================================================
      implicit none

!     Input
!---------------------------------------
      integer :: II,JJ
      real    :: lat(JJ)
      real    :: data(II,JJ)
      real    :: ref(II,JJ)
      real    :: msk(II,JJ)
      real    :: hw0
      integer :: MM !must be at least 1
      integer :: rgns(MM,4)
      logical :: rm_gm
!---------------------------------------

!     Output
!---------------------------------------
      real    :: mrmsd(II,JJ,2)
      real    :: rmsd_rgns(MM,4)
      real    :: diff_gm
!---------------------------------------

!     Local
!---------------------------------------       
      integer :: i,j,k,l,m,n
      integer :: j0,LL,ov,LLx(JJ)
      real    :: r,torad,latr,lats,dx(JJ),ds(JJ)
      real,allocatable :: diff(:,:)
      real    :: sm,cnt,mn
!---------------------------------------       
!-------------------------------------------------

!     Insert global domain into region definitions
!-------------------------------------------------
      !rgns(1,1)=1
      !rgns(1,2)=II
      !rgns(1,3)=1
      !rgns(1,4)=JJ
!-------------------------------------------------

!    Define geographical meta data 
!-------------------------------------------------
      r = 6371229.0
      torad = atan(1.0)/45.0

      lats=torad*(lat(2)-lat(1))

      do j=1,JJ

         latr=torad*lat(j)

         dx(j) = r*lats*cos(latr)

         ds(j) = dble(0.50*(r*lats)**2 & 
               & *(cos(latr+0.5*lats)+cos(latr-0.5*lats)))

      end do
!-------------------------------------------------

!     Calculate zonal half width of filter in 
!     grid points as function of latitude.
!     Input specifies window half-width in degrees at 45N
!-------------------------------------------------
      j0=nint(torad*(45.0-lat(1))/lats)+1

      LL=nint(hw0/(lats/torad))

      ov=3*LL !set max window width (arbitrarily)
              !at 3 times width at 45N.
              !Needed for wrapping.

      do j=1,JJ
         LLx(j) = nint(LL*dx(j0)/dx(j))
         if(LLx(j).gt.ov)LLx(j)=ov
         !write(*,*)j,LLx(j) 
      end do
!-------------------------------------------------

!     
!-------------------------------------------------
     allocate(diff(II+2*ov,JJ))

     !Compute the difference between the fields
     !taking common mask into account 
     !----------------------------------
      do i=1,II
         do j=1,JJ
            if(msk(i,j).eq.0.0)then
               diff(i+ov,j)=data(i,j)-ref(i,j)
            else
               diff(i+ov,j)=-1.9e19
            end if
         end do
      end do
     !----------------------------------

     !Calculate and remove global offset
     !----------------------------------
     !Deleted in lite version since local
     !offset is removed in anycase.
     !----------------------------------

     !Wrap the difference 
     !----------------------------------
      do j=1,JJ
         diff(1:ov,j)=diff(II+1:II+ov,j)
         diff(II+ov+1:II+2*ov,j)=diff(ov+1:2*ov,j)
      end do
     !----------------------------------
!-------------------------------------------------

!     Compute the (1) RMSD and (2) RMSD less offset
!     over arbitrary regions defined as inputs.
!     Global domain values are calculated by default.
!-------------------------------------------------
      !Deleted in lite version
!-------------------------------------------------

!     Compute the RMSD globally in a moving window 
!     whose width is set by the parameter LL
!-------------------------------------------------
      do i=1+ov,II+ov
         do j=1+LL,JJ-LL

            if(msk(i-ov,j).gt.-1.e7)then

           !Compute mean difference in moving window
           !----------------------------
            cnt=0.0
            sm=0.0
            do k=i-LLx(j),i+LLx(j)
               do l=j-LL,j+LL
                  if(diff(k,l).gt.-1.7e7)then
                     sm=sm+diff(k,l)*ds(j)
                     cnt=cnt+ds(j)
                  end if
               end do
            end do
            if(cnt.gt.0.0)then
               mn=sm/cnt
            else
               mn=-1.9e19
            end if         
           !----------------------------
            
           !Compute RMSD in window with offset
           !----------------------------
           !Deleted in lite version
           !----------------------------

           !Compute RMSD in window wwithout offset 
           !----------------------------
            cnt=0.0
            sm=0.0
            if(mn.gt.-1.7e7)then
               do k=i-LLx(j),i+LLx(j)
                  do l=j-LL,j+LL
                     if(diff(k,l).gt.-1.7e7)then
                        sm=sm+(diff(k,l)-mn)**2*ds(j)
                        cnt=cnt+ds(j)
                     end if
                  end do
               end do
            end if
            if(cnt.gt.0.0)then
               mrmsd(i-ov,j,2)=sqrt(sm/cnt)
            else
               mrmsd(i-ov,j,2)=-1.9e19
            end if
           !----------------------------

            end if           

         end do
      end do
!-------------------------------------------------

!     Compute the area means of the mapped RMSD 
!     (with and without offsets) over arbitrary regions
!     defined as inputs
!-------------------------------------------------
      do m=1,MM

        !Area means with offset
        !-------------------------------
           !Deleted in lite version
        !-------------------------------

        !Area means without offset
        !-------------------------------
         sm=0.0
         cnt=0.0
         do i=rgns(m,1),rgns(m,2)
            do j=rgns(m,3),rgns(m,4)
               !write(*,*)i,j
               if(msk(i,j).gt.-1.7e7)then
                  sm=sm+mrmsd(i,j,2)*ds(j)
                  cnt=cnt+ds(j)
               end if
            end do
         end do
         if(cnt.gt.0.0)then
            rmsd_rgns(m,4)=sm/cnt
         else
            rmsd_rgns(m,4)=0.0
         end if
        !-------------------------------
 
      end do !end loop over regions
!-------------------------------------------------

!-------------------------------------------------
      deallocate(diff)
!-------------------------------------------------

!-------------------------------------------------
!     End of proceedure

      return

!=====================================================================
      end subroutine rmsd_lite

      subroutine rmsd(II,JJ,lat,data,ref,msk,hw0,MM,rgns,rm_gm,&
                        &mrmsd,rmsd_rgns,diff_gm)
!=====================================================================
      implicit none

!     Input
!---------------------------------------
      integer :: II,JJ
      real    :: lat(JJ)
      real    :: data(II,JJ)
      real    :: ref(II,JJ)
      real    :: msk(II,JJ)
      real    :: hw0
      integer :: MM !must be at least 1
      integer :: rgns(MM,4)
      logical :: rm_gm
!---------------------------------------

!     Output
!---------------------------------------
      real    :: mrmsd(II,JJ,2)
      real    :: rmsd_rgns(MM,4)
      real    :: diff_gm
!---------------------------------------

!     Local
!---------------------------------------       
      integer :: i,j,k,l,m,n
      integer :: j0,LL,ov,LLx(JJ)
      real    :: r,torad,latr,lats,dx(JJ),ds(JJ)
      real,allocatable :: diff(:,:)
      real    :: sm,cnt,mn
!---------------------------------------       
!-------------------------------------------------

!     Insert global domain into region definitions
!-------------------------------------------------
      !rgns(1,1)=1
      !rgns(1,2)=II
      !rgns(1,3)=1
      !rgns(1,4)=JJ
!-------------------------------------------------

!    Define geographical meta data 
!-------------------------------------------------
      r = 6371229.0
      torad = atan(1.0)/45.0

      lats=torad*(lat(2)-lat(1))

      do j=1,JJ

         latr=torad*lat(j)

         dx(j) = r*lats*cos(latr)

         ds(j) = dble(0.50*(r*lats)**2 & 
               & *(cos(latr+0.5*lats)+cos(latr-0.5*lats)))

      end do
!-------------------------------------------------

!     Calculate zonal half width of filter in 
!     grid points as function of latitude.
!     Input specifies window half-width in degrees at 45N
!-------------------------------------------------
      j0=nint(torad*(45.0-lat(1))/lats)+1

      LL=nint(hw0/(lats/torad))

      ov=3*LL !set max window width (arbitrarily)
              !at 3 times width at 45N.
              !Needed for wrapping.

      do j=1,JJ
         LLx(j) = nint(LL*dx(j0)/dx(j))
         if(LLx(j).gt.ov)LLx(j)=ov
         !write(*,*)j,LLx(j) 
      end do
!-------------------------------------------------

!     
!-------------------------------------------------
     allocate(diff(II+2*ov,JJ))

     !Compute the difference between the fields
     !taking common mask into account 
     !----------------------------------
      do i=1,II
         do j=1,JJ
            if(msk(i,j).eq.0.0)then
               diff(i+ov,j)=data(i,j)-ref(i,j)
            else
               diff(i+ov,j)=-1.9e19
            end if
         end do
      end do
     !----------------------------------

     !Calculate and remove global offset
     !----------------------------------
      sm=0.0
      cnt=0.0
      do i=1,II
         do j=1,JJ
            if(diff(i+ov,j).gt.-1.7e7)then
               sm=sm+diff(i+ov,j)*ds(j)
               cnt=cnt+ds(j)
            end if
         end do
      end do
      diff_gm=sm/cnt
      if((rm_gm).and.(diff_gm.ne.0.0))then
         do i=1,II
            do j=1,JJ
               if(diff(i+ov,j).gt.-1.7e7)then
                  diff(i+ov,j)=diff(i+ov,j)-diff_gm
               end if
            end do
         end do
      end if
     !----------------------------------

     !Wrap the difference 
     !----------------------------------
      do j=1,JJ
         diff(1:ov,j)=diff(II+1:II+ov,j)
         diff(II+ov+1:II+2*ov,j)=diff(ov+1:2*ov,j)
      end do
     !----------------------------------
!-------------------------------------------------

!     Compute the (1) RMSD and (2) RMSD less offset
!     over arbitrary regions defined as inputs.
!     Global domain values are calculated by default.
!-------------------------------------------------
      do m=1,MM

        !Compute mean difference (offset)
        !----------------------------
         sm=0.0
         cnt=0.0
         do i=rgns(m,1),rgns(m,2)
            do j=rgns(m,3),rgns(m,4)
               if(diff(i+ov,j).gt.-1.7e7)then
                  sm=sm+diff(i+ov,j)*ds(j)
                  cnt=cnt+ds(j)
               end if
            end do
         end do
         if(cnt.gt.0.0)then
            mn=sm/cnt
         else
            mn=0.0
         end if
        !----------------------------
         
        !Compute RMSD with offset         
        !----------------------------
         sm=0.0
         cnt=0.0
         do i=rgns(m,1),rgns(m,2)
            do j=rgns(m,3),rgns(m,4)
               if(diff(i+ov,j).gt.-1.7e7)then
                  sm=sm+diff(i+ov,j)**2*ds(j)
                  cnt=cnt+ds(j)
               end if
            end do
         end do
         if(cnt.gt.0.0)then
            rmsd_rgns(m,1)=sqrt(sm/cnt)
         else
            rmsd_rgns(m,1)=0.0
         end if
         !write(*,*)rmsd_rgns(m,1)
        !----------------------------

        !Compute RMSD without offset         
        !----------------------------
         sm=0.0
         cnt=0.0
         do i=rgns(m,1),rgns(m,2)
            do j=rgns(m,3),rgns(m,4)
               if(diff(i+ov,j).gt.-1.7e7)then
                  sm=sm+(diff(i+ov,j)-mn)**2*ds(j)
                  cnt=cnt+ds(j)
               end if
            end do
         end do
         if(cnt.gt.0.0)then
            rmsd_rgns(m,2)=sqrt(sm/cnt)
         else
            rmsd_rgns(m,2)=0.0
         end if
         !write(*,*)rmsd_rgns(m,2)
        !----------------------------
 
      end do !end loop over regions
!-------------------------------------------------

!     Compute the RMSD globally in a moving window 
!     whose width is set by the parameter LL
!-------------------------------------------------
      do i=1+ov,II+ov
         do j=1+LL,JJ-LL

           !Compute mean difference in moving window
           !----------------------------
            cnt=0.0
            sm=0.0
            do k=i-LLx(j),i+LLx(j)
               do l=j-LL,j+LL
                  if(diff(k,l).gt.-1.7e7)then
                     sm=sm+diff(k,l)*ds(j)
                     cnt=cnt+ds(j)
                  end if
               end do
            end do
            if(cnt.gt.0.0)then
               mn=sm/cnt
            else
               mn=-1.9e19
            end if         
           !----------------------------
            
           !Compute RMSD in window with offset
           !----------------------------
            cnt=0.0
            sm=0.0
            if(mn.gt.-1.7e7)then
               do k=i-LLx(j),i+LLx(j)
                  do l=j-LL,j+LL
                     if(diff(k,l).gt.-1.7e7)then
                        sm=sm+diff(k,l)**2*ds(j)
                        cnt=cnt+ds(j)
                     end if
                  end do
               end do
            end if
            if(cnt.gt.0.0)then
               mrmsd(i-ov,j,1)=sqrt(sm/cnt)
            else
               mrmsd(i-ov,j,1)=-1.9e19
            end if
           !----------------------------

           !Compute RMSD in window less mean diff
           !----------------------------
            cnt=0.0
            sm=0.0
            if(mn.gt.-1.7e7)then
               do k=i-LLx(j),i+LLx(j)
                  do l=j-LL,j+LL
                     if(diff(k,l).gt.-1.7e7)then
                        sm=sm+(diff(k,l)-mn)**2*ds(j)
                        cnt=cnt+ds(j)
                     end if
                  end do
               end do
            end if
            if(cnt.gt.0.0)then
               mrmsd(i-ov,j,2)=sqrt(sm/cnt)
            else
               mrmsd(i-ov,j,2)=-1.9e19
            end if
           !----------------------------
           
         end do
      end do
!-------------------------------------------------

!     Compute the area means of the mapped RMSD 
!     (with and without offsets) over arbitrary regions
!     defined as inputs
!-------------------------------------------------
      do m=1,MM

        !Area means with offset
        !-------------------------------
         sm=0.0
         cnt=0.0
         do i=rgns(m,1),rgns(m,2)
            do j=rgns(m,3),rgns(m,4)
               if(msk(i,j).gt.-1.7e7)then
                  sm=sm+mrmsd(i,j,1)*ds(j)
                  cnt=cnt+ds(j)
               end if
            end do
         end do
         if(cnt.gt.0.0)then
            rmsd_rgns(m,3)=sm/cnt
         else
            rmsd_rgns(m,3)=0.0
         end if
         !write(*,*)rmsd_rgns(m,3)
        !-------------------------------

        !Area means without offset
        !-------------------------------
         sm=0.0
         cnt=0.0
         do i=rgns(m,1),rgns(m,2)
            do j=rgns(m,3),rgns(m,4)
               if(msk(i,j).gt.-1.7e7)then
                  sm=sm+mrmsd(i,j,2)*ds(j)
                  cnt=cnt+ds(j)
               end if
            end do
         end do
         if(cnt.gt.0.0)then
            rmsd_rgns(m,4)=sm/cnt
         else
            rmsd_rgns(m,4)=0.0
         end if
         !write(*,*)rmsd_rgns(m,3)
        !-------------------------------
 
      end do !end loop over regions
!-------------------------------------------------

!-------------------------------------------------
      deallocate(diff)
!-------------------------------------------------

!-------------------------------------------------
!     End of proceedure

      return

!=====================================================================
      end subroutine rmsd

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

      subroutine pde_fltr(IIin,ov,JJ,j1,j2,lat0_d,lats_d,mdt)
!=====================================================================
      implicit none

!     Input/Output
!-------------------------------------------------       
      integer :: IIin,ov,JJ,j1,j2
      real    :: lat0_d,lats_d 
      real    :: mdt(IIin,JJ)
!-------------------------------------------------       

!     Local
!-------------------------------------------------       
      real*8, parameter :: dt=0.1d0    !Pseudo timestep
      real*8, parameter :: K=0.22d0     !gradient sensitivity parameter
      integer, parameter :: diff_type=2 !sets the diffusion specifier

      integer :: i,j,l,n,m,II
      real*8 :: torad,r,lats,lat(JJ),dx(JJ),dy
      real*8,allocatable :: f(:,:),inc(:,:)
      real*8,allocatable :: dfdx(:,:),dfdy(:,:)
      real*8,allocatable :: dfdx2(:,:),dfdy2(:,:)
      real*8,allocatable :: mag_grd_f(:,:)
      real*8,allocatable :: d(:,:),dddx(:,:),dddy(:,:)
      real*8,allocatable :: h1(:,:),h2(:,:)
      real*8 :: mn,rms1,rms2
!-------------------------------------------------       

!     Start of proceedure
!=================================================


!     Allocate arrays
!-------------------------------------------------
      II=IIin+2*ov

      allocate(f(II,JJ))
      allocate(dfdx(II,JJ))
      allocate(dfdy(II,JJ))
      allocate(dfdx2(II,JJ))
      allocate(dfdy2(II,JJ))
      allocate(mag_grd_f(II,JJ))
      allocate(d(II,JJ))
      allocate(dddx(II,JJ))
      allocate(dddy(II,JJ))
      allocate(inc(II,JJ))
      allocate(h1(0:II+1,JJ))
      allocate(h2(0:II+1,JJ))
!-------------------------------------------------

!     Calculate grid spacing. (Note normalisation by dy)
!-------------------------------------------------
      r = 6371229.0d0

      torad = datan(1.0d0)/45.0d0
      lats = torad*lats_d

      do j=1,JJ
         lat(j)=torad*(lats_d*(j-1)+lat0_d)
      end do

      dy=r*lats
      
      do j=1,JJ
         dx(j)= r*lats*dcos(lat(j))/dy
      end do

      dy=1.0d0
!-------------------------------------------------

!     Convert field to double and wrap if required
!-------------------------------------------------
      f(ov+1:ov+IIin,:)=dble(mdt(:,:))
      f(1:ov,:)=dble(mdt(IIin-ov+1:IIin,:))
      f(ov+IIin+1:IIin+2*ov,:)=dble(mdt(1:ov,:))
!-------------------------------------------------

!     Calculate increment to field
!-------------------------------------------------
     !Compute central x-differences of field
     !---------------------------------
      do i=2,II-1
         do j=1,JJ
            dfdx(i,j)=(f(i+1,j)-f(i-1,j))/(2.0d0*dx(j))
            dfdx2(i,j)=(f(i+1,j)-2.0d0*f(i,j)+f(i-1,j))/(dx(j)**2)
         end do
      end do
     !---------------------------------

     !Compute central y-differences of field
     !---------------------------------
      do i=1,II
         do j=2,JJ-1
            dfdy(i,j)=(f(i,j+1)-f(i,j-1))/(2.0d0*dy)
            dfdy2(i,j)=(f(i,j+1)-2.0d0*f(i,j)+f(i,j-1))/(dy**2)
         end do
      end do
     !---------------------------------

     !Compute magnitude of field gradient
     !---------------------------------
      do i=1,II
         do j=2,JJ-1
            mag_grd_f(i,j)=dsqrt(dfdx(i,j)**2+dfdy(i,j)**2)
         end do
      end do
     !---------------------------------

     !Compute diffusion constant
     !---------------------------------
      select case (diff_type)
         case (1)
            do i=1,II
               do j=2,JJ-1
                  d(i,j)=dexp(-1.0d0*(mag_grd_f(i,j)/K)**2)
               end do
            end do
         case (2)
            do i=1,II
               do j=2,JJ-1
                  d(i,j)=1.0d0/(1.0d0+(mag_grd_f(i,j)/K)**2)
               end do
            end do
         case (3)
            do i=1,II
               do j=2,JJ-1
                  d(i,j)=1.0d0/dsqrt(1.0d0+mag_grd_f(i,j)**2)
               end do
            end do
      end select
     !---------------------------------

     !Compute central x-differences of diffusion constant
     !---------------------------------
      do i=2,II-1
         do j=1,JJ
            dddx(i,j)=(d(i+1,j)-d(i-1,j))/(2.0d0*dx(j))
         end do
      end do
     !---------------------------------

     !Compute central y-differences of diffusion constant
     !---------------------------------
      do i=1,II
         do j=2,JJ-1
            dddy(i,j)=(d(i,j+1)-d(i,j-1))/(2.0d0*dy)
         end do
      end do
     !---------------------------------

     !Compute increment to height field 
     !---------------------------------
      do i=2,II-1
         do j=2,JJ-1
            inc(i,j)=dt*((d(i,j)*dfdx2(i,j)+dddx(i,j)*dfdx(i,j))&
                         &+(d(i,j)*dfdy2(i,j)+dddy(i,j)*dfdy(i,j)))
         end do
      end do
      inc(:,1)=inc(:,2)
      inc(:,JJ)=inc(:,JJ-1)
     !---------------------------------

     !Supress gradient sharpening 
     !---------------------------------
      h1(1:II,:)=f(:,:)
      h1(0,:)=f(II,:)
      h1(II+1,:)=f(1,:)

      h2(1:II,:)=f(:,:)+inc(:,:)
      h2(0,:)=f(II,:)+inc(II,:)
      h2(II+1,:)=f(1,:)+inc(1,:)

      do i=1,II
         do j=2,JJ-1

            mn=0.0d0
            do l=-1,1
               mn=mn+(h1(i,j)-h1(i+l,j+1))**2
               mn=mn+(h1(i,j)-h1(i+l,j-1))**2
            end do
            mn=mn+(h1(i,j)-h1(i-1,j))**2
            mn=mn+(h1(i,j)-h1(i+1,j))**2
            rms1=sqrt(mn/9.0d0)

            mn=0.0d0
            do l=-1,1
               mn=mn+(h2(i,j)-h2(i+l,j+1))**2
               mn=mn+(h2(i,j)-h2(i+l,j-1))**2
            end do
            mn=mn+(h2(i,j)-h2(i-1,j))**2
            mn=mn+(h2(i,j)-h2(i+1,j))**2
            rms2=sqrt(mn/9.0d0)

            if(rms2.gt.rms1)then
               inc(i,j)=0.0 !Set increment to zero
            end if          !if it leads to gradient sharpenning

         end do
      end do
     !---------------------------------
!-------------------------------------------------

!-------------------------------------------------
      mdt(:,j1:j2)=mdt(:,j1:j2)+sngl(inc(ov+1:ov+IIin,j1:j2))
!-------------------------------------------------

!     Deallocate arrays
!-------------------------------------------------
      deallocate(f)
      deallocate(dfdx)
      deallocate(dfdy)
      deallocate(dfdx2)
      deallocate(dfdy2)
      deallocate(mag_grd_f)
      deallocate(d)
      deallocate(dddx)
      deallocate(dddy)
      deallocate(h1)
      deallocate(h2)
!-------------------------------------------------

!=================================================
!     End of proceedure

      return

!=====================================================================
      end subroutine pde_fltr




