'''
 Extract tiles of 50x50m from the stacked Swisstopo 
 irgb raster and DEM raster.
 The center of the tile matches the NOLU point 
 from the land cover and land use classification.
'''
import gdal,os, glob
import rasterio as rio
# Extract tiles from Swissimage
src  = "/home/valerie/data/tile5D/"
dest =  "/home/valerie/data/refsurface/"
# Open raster 
os.chdir(src)
lst = glob.glob('*.tif')
N = len(lst)
count =0

for raster in range(N) : 
    # include in a loop later over the 56 tile5D
    print('\nStart of new raster :', lst[raster] ,raster+1, 'over 56.' )

    # Define boundaries of the raster and interval
    img = rio.open(src+lst[raster])
    e_min  = int (img.bounds[0]+25 )      # left 
    e_max  = int (img.bounds[2]    )      # right
    n_min  = int (img.bounds[1]    )      # bottom
    n_max  = int (img.bounds[3]-25 )      # top
    e_step = 100
    n_step = -100
    
    # Loop on map coordinate 
    # e and n are the coordinate of the upper left corner of each the refsurface
    for  e in range( e_min, e_max , e_step):
        for n in range ( n_max, n_min , n_step):
            out_name = str(int((e+25)/100)-20000)+ str(int((n-25)/100)-10000) +'.tif'
            # Write the gdal command as a string 
            com_str = ("gdal_translate -ot UInt16 -of GTIFF "
                        + " -projwin "  # Selects a subwindow from the source image
                                        # with the corners given in georeferenced coordinates 
                                        # <ulx> <uly> <lrx> <lry>
                        + str(e) + " " + str(n) + " " + str(e+e_step/2) + " " + str(n+n_step/2) + " " 
                        + str ("-a_nodata 0 ")
                        + str(src + lst[raster] ) + " "  # source dataset
                        + (dest+ out_name )         #destination dataset    
                        )          
            os.system(com_str)
            count +=1
            if int(count%100) == 0 :
                print('Progress:' , count, '/59976' )

print('All good')
