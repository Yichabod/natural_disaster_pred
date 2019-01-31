import math

# Distances are measured in miles.
# Longitudes and latitudes are measured in degrees.
# Earth is assumed to be perfectly spherical.

earth_radius = 6271.0
degrees_to_radians = math.pi/180.0
radians_to_degrees = 180.0/math.pi

def change_in_latitude(kms):
    "Given a distance north, return the change in latitude."
    return (kms/earth_radius)*radians_to_degrees

def change_in_longitude(latitude, kms):
    "Given a latitude and a distance west, return the change in longitude."
    # Find the radius of a circle around the earth at given latitude.
    r = earth_radius*math.cos(latitude*degrees_to_radians)
    return (kms/r)*radians_to_degrees

def ten_km_square(latitude, longitude):
    slat, nlat = latitude+change_in_latitude(-5), latitude+change_in_latitude(5)
    wlon = longitude+change_in_longitude(latitude,-5)
    elon = longitude+change_in_longitude(latitude, 5)
    return(nlat, wlon, slat, elon)

def main(lon, lat):
    '''First argument degrees longitude (E is positive, W negative)
        of the landslide location,
        second argument latitude (N positive, S negative),
        in decimal format(not minutes etc.)'''
    nlat, wlon, slat, elon = ten_km_square(lat,lon)
    print("(NLat:{:.4f},WLon:{:.4f}),(SLat:{:.4f},ELon:{:.4f})".format(nlat, wlon, slat, elon))


main(-121.4323838,35.86562803)
