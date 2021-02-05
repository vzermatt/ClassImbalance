'''
Produce equal tiles of 2x5 km (2.1x5.1km with 50m overlapping)
with a resolution of 25cm on the Area of interest
from Swisstopo raster IR-R-G-B tiles and DEM
see QGIS from more info about the tile production
'''
# import modules
import os, glob, gdal
from numpy import unique
from osgeo import ogr


# 1. From IRDGB raster to 2x5km tiles
#------------------------------------
# Read all the IRDGB raster files stored in the data folder
path = '/home/valerie/data/'
os.chdir(path + '2020_LUBIS_ADS_Line_RS/')
irgb_list = glob.glob( '*.tif' )        # list with all the IRGB .tif 
N  = len (irgb_list)                    # 585 tif 

# Select on the tiles with a 25cm resolution (contains '12504' in the file name)
# We will also extract row id for the selected tiles
irgb_select =[]
row_id =[]
for k in range(N):
    if irgb_list[k].find('12504')>=0:
        irgb_select += [irgb_list[k]]
        row_id += [irgb_list[k][9:13]]

N = len(irgb_select)            # we have left 140 tif with 25 cm resolution
all_row_id = unique(row_id)     # store unique row labels

# See documentation : our Area of Interest 
# measures 35km E x 16km N. It will be cut in 40 tiles of
# 5 x 2 km : so 7 tiles in E direction and 8 in N direction
# we will loop over the 8 horizontal row to follow swisstopo image aquisition

#Select the tiles corresponding to each row
def select_tif(lst):
    out =''
    for row in lst:
        for k in range(N):
            if irgb_select[k].find(row)>=0:
                out += (path + '2020_LUBIS_ADS_Line_RS/' + irgb_select[k]+' ')
    return out  # contain the paths to several tiff 

# we give to each row the path to the tif fileS covering this area
row=[]
row += [select_tif(['1054'])]     # for first line : 0
row += [select_tif(['1033','1013'])]# for second line :1
row += [select_tif(['0952'])]     # 2
row += [select_tif(['1235'])]     # 3
row += [select_tif(['1123'])]     # 4
row += [select_tif(['1105'])]     # 5
row += [select_tif(['1046'])]     # 6 
row += [select_tif(['1007'])]     # 7

# now we can start to produce tif with gdal 
# we will merge several tiff to obtain larger overlapping tiles
# for each tile :
# n upper left corner coordinates towards North
# e upper left corner coordinates towards East
n_min  = 1134050
n_max  = n_min -16000 -100 
n_step = -2100
e_min  = 2584950
e_max  = e_min + 35000 +100
e_step = 5100

# Loop over e,n coordinate to extract the 2x5km tiles
count =0
for n in range(n_min,n_max,n_step):
    tif_list =row[count]
    count += 1
    for e in range (e_min,e_max,e_step):
        out_name = str('/home/valerie/data/tile/irgb_'
                    + str( int( ( e+50 )/100 ) )  +'_'
                    + str( int( ( n-50 )/100 ) ) +'.tif'
                    )
        ul_lr = [e,n, e + e_step,n + n_step]
        com_str = ('gdal_merge.py -of GTIFF ' 
                    + '-o ' + out_name  # output name
                    # map coordinate of the output raster
                    + ' -ul_lr ' + str(e) +' ' + str(n) +' '+ str(e+e_step) +' '+ str(n + n_step)
                    + ' -v' # no data value 0 are ignored
                    + ' -n 0 '
                    + tif_list # list of useful raster
                    )
        #os.system(com_str)
        print( '\n',out_name,'completed')
    print('\n',count,'th row complete','\n')

del path,irgb_select,irgb_list, N,k, count , row, row_id, all_row_id
del n, n_max, n_min, n_step  e,e_max,e_min,e_step


# 2. From DEM raster to 2x5km tiles
#------------------------------------
# Read all the dem raster files stored in the data folder
path = '/home/valerie/data/'
os.chdir(path + 'swissALTI3D')
dem_list = glob.glob( '*.tif' )        # list with all the IRGB .tif 
N  = len (dem_list)                    # 585 tif 

# Select only the tiles within area of interest ( North range 1117000:1134000, all East range)
# We will also extract row id for the selected tiles
dem_select =[]
row_id =[]
for k in range(N):
    if  int(dem_list[k][-6:-4])>16:
        dem_select += [dem_list[k]]
        row_id += [dem_list[k][-6:-4]]

N = len(dem_select)            # we have left 140 tif with 25 cm resolution

# Attribute to each row the corresponding dem raster
def select_dem(lst):
    out =''
    for row in lst:
        for k in range(N):
            print('row:',row,'row_id:',row_id[k])
            if row_id[k] == row:
                out += (path + 'swissALTI3D/' + dem_select[k]+' ')
    return out  # contain the paths to several tiff 

# we give to each row the path to the tif files covering this area
row=[]
row += [select_dem(['31','32','33','34'])] # for first line 0
row += [select_dem(['29','30','31'])]     # 1
row += [select_dem(['27','28','29'])]     # 2
row += [select_dem(['25','26','27'])]     # 3
row += [select_dem(['23','24','25'])]     # 4
row += [select_dem(['21','22','23'])]     # 5
row += [select_dem(['19','20','21'])]     # 6 
row += [select_dem(['17','18','19'])]     # 7

#Define the area of interest and the tiles size
n_min  = 1134050
n_max  = n_min -16000 -100 
n_step = -2100
e_min  = 2584950
e_max  = e_min + 35000 +100
e_step = 5100

# We can start to produce tif with gdal 
# Loop over e,n coordinate to extract the 2x5km tiles
count = 0
for n in range(n_min,n_max,n_step):
    tif_list =row[count]
    count += 1
    print('N:',n)
    for e in range (e_min,e_max,e_step):
        out_name = str('/home/valerie/data/tile/dem_'
                    + str( int( ( e+50 )/100 ) )  +'_'
                    + str( int( ( n-50 )/100 ) ) +'.tif'
                    )
        ul_lr = [e,n, e + e_step,n + n_step]
        com_str = ('gdal_merge.py -of GTIFF ' 
                    + '-o ' + out_name  # output name
                    # map coordinate of the output raster
                    + ' -ul_lr ' + str(e) +' ' + str(n) +' '+ str(e+e_step) +' '+ str(n + n_step)
                    + ' -v' 
                    + ' -n 0 '# no data value 0 are ignored
                    + tif_list # list of useful raster
                    )
        os.system(com_str)
        print( '\n',out_name,'completed')
    print('\n',count,'th row complete','\n')

# 3. Stack dem and irgb into a single raster
#-------------------------------------------
from osgeo import gdal
import os, glob
src  = '/home/valerie/data/tile/'
dest = '/home/valerie/data/tile5D/'

# Read the dem and irgb raster
os.chdir(src)
dem = glob.glob( 'dem**.tif' )
dem.sort()        # list with all the IRGB .tif 
N  = len (dem) 
irgb =glob.glob('irgb**.tif')
irgb.sort()

# Verify that each irgb has a matching dem with the same index 
for k in range(N):
    if irgb[k][-15:-4] != dem[k][-15:-4]:
        print('Houston we have a problem !', irgb[k],dem[k])

# We can start producing the stacked raster !
for k in range(N):
    tif_list = str(  src+irgb[k]+' '+src+ dem[k])
    com_string = ( 'gdal_merge.py -o '+ dest + dem[k][-15:-4]+'.tif' # outname
                    + ' -separate '  # dem into a new band
                    + tif_list       # files path
                )
    os.system(com_string)
    print(k, 'file was created !')   
