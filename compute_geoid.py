import numpy as np

tide_out = 'mean'
pin='./data/src/'
pout='./data/res/'

a = 6378137.0               # Equatorial raidus (m)
f = 1.0 / 298.257202101     # Recipricol of flattening
gm = 3.986005e14              # Gravity mass constant
omega = 7.292115e-5          # Rotation rate

# Following should be replaced with create_coords and define_dims?
lon_stp  = 1.0 / rr                # Longitude interval
ltgd_stp = 1.0 / rr                # Latitude interval
lon_min = 0.5 * lon_stp            # Min longitude
lon_max = 360.0 - 0.5 * lon_stp    # Max longitude
ltgd_min = -90.00 + 0.5 * ltgd_stp   # Min latitude
ltgd_max = 90.00 - 0.5 * ltgd_stp   # Max latitude

# open(21,file='./parameters.txt',form='formatted')
# read(21,'(A20)')name_mss
# read(21,'(I4)')NN_mss
# read(21,'(A20)')name_egm
# read(21,'(I4)')NN_egm
# read(21,'(A4)')tide_in
# read(21,'(I4)')rr
# read(21,'(I4)')NN
# close(21)

II = nint((lon_max-lon_min)/lon_stp)+1
JJ = nint((ltgd_max-ltgd_min)/ltgd_stp)+1

# Calculate longitude and geodetic latitude arrays for points on output grid
for i in range(II):
    lon[i] = lon_stp * (i-1) + lon_min

for j in range(JJ):
    lat[j] = ltgd_stp * (j-1) + ltgd_min


def convert_tide(c20, tide_in, tide_out):
    r"""
    Converts between tide systems
    """
    tide_systems = {
        'free': {
            'free': 0, 'mean': -1.8157e-08, 'zero': -4.173e-9
        },
        'mean': {
            'free': 1.8157e-08, 'mean': 0, 'zero': 1.39844e-8
        },
        'zero': {
            'free': 4.173e-9, 'mean': -1.39844e-8, 'zero': 0
        }}

    return c20 + tide_systems[tide_in][tide_out]



def legendre(cltgc, nn, mm, p):
    r"""
    Calculates the fully normalised associated legendre functions
    at a particular latitude based on the definitions given by
    Holmes and Featherstone 2002 sec. 2.1
    """
    # Max degree and order
    NN = ?

    p[0,0] = sf*1.0
    p[1,1] = sf*math.sqrt(3.0)

    for n in range(2, NN)



# Calculate the potential of the reference ellipsoid
#       call ref_pot(a,f,gm,omega,NN_egm,c_ref)


def ref_pot(a, f, gm, omega, NN, c_ref):
    r"""
    Calculates the spherical harmonic coeffs for the gravitational potential a
    specified reference ellipsiod

    Args:
    NN = Max degree required
    a = Ellipsoid semi-major axis
    f = Recipricol of flattening
    gm = Ellipsoid gravity mass constant
    omega = Rotation - angular velocity
    """


def main():


if __main__ == 'main':
    main()

