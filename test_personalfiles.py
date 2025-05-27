import orthority as oty
import os, sys
import numpy as np
from imageio import imwrite
from rasterio import open as r_open
import rasterio as rio
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def _get_point(data):
    point = -1
    row = 0
    while point < 0 and row < data.shape[0]:
        try:
            # point = np.where(np.isfinite(data[row, :]))[0][0]
            point = np.where(data[row, :] > 0)[0][0]
        except:
            row += 1

        # if row == 200:
        #     plt.imshow(data)
        #     plt.show()
        #     break
    # print(point, row)

    # print(data)
    return point, row

# def guess_flight_angle(data):
#     c1 = _get_point(data.T)
#     c2 = _get_point(data.T[::-1, :])
#     d1 = _get_point(data)
#     d2 = _get_point(data[::-1, :])
#     # print(c1, d1, c2, d2)
#     along_flight_angle1 = np.arctan2(c1, d1)
#     along_flight_angle2 = np.arctan2(c2, d2)
#     # print(np.rad2deg(along_flight_angle1), np.rad2deg(along_flight_angle2))
#     along_flight_angle = (along_flight_angle1 + along_flight_angle2) / 2
#     return along_flight_angle

def guess_flight_angle(data):
    d, c = np.shape(data)
    d, c = d - 1, c - 1
    # print("shape: ", d, c)
    c1, c1_row = _get_point(data.T)
    c2, c2_row = _get_point(data.T[::-1, :])
    d1, d1_row = _get_point(data)
    d2, d2_row = _get_point(data[::-1, :])
    # print(c1, d1, c2, d2)
    along_flight_angle = np.arctan2(c1 - d1_row, d1 - c1_row)
    # np.arctan2((d-d2)-c2_row, (c-c2)-d2_row)
    along_swath_angle = np.pi - np.arctan2((c - c1) - d2_row, d2 - c2_row)
    right_angle = along_flight_angle + np.pi / 2
    along_swath_diff = along_swath_angle - right_angle

    side1_length = np.sqrt(
        (c1_row - d1) ** 2 + (c1 - d1_row) ** 2
    )
    print("side1_length: ", side1_length)
    side2_length = np.sqrt(
        (c1_row - d2) ** 2 + (c1-(d - d2_row)) ** 2
    )
    print("side2_length: ", side2_length)
    if side1_length < side2_length:
        along_flight_angle = along_swath_angle
        along_swath_angle = along_flight_angle

    print(
        "flight angles: ", np.rad2deg(along_flight_angle), np.rad2deg(along_swath_angle)
    )
    print("trapezoid angle: ", np.rad2deg(along_swath_diff))
    return along_flight_angle
    #     along_swath_angle,
    #     ((c1_row, c1), (d1, d1_row), (c - c2_row, c2), (d2, d - d2_row)),
    # )

def make_mask(data, along_flight_angle, origin = None, stripwidth=24):
    X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    if origin is None:
        origin = np.array([data.shape[1]//2, data.shape[0]//2])
    distance = (Y-origin[1])*np.cos(-along_flight_angle) - (X-origin[0])*np.sin(-along_flight_angle)
    mask = np.abs(distance) < stripwidth/2
    return mask

def make_fake_data():
    root_dir = "C:/Users/thijs/OneDrive/Documents/PostDoc_UPC/Data_local/Mountain_Fire_processing/day_09_04_2022/"
    root = root_dir+'Metashape/Workswell/Metashape_data_references/Qinertia/' #Qinertia/ Interpolated/
    src_file_source = root_dir+'Workswell_images/fake2.jpg'
    # im = np.tile(np.arange(0, 256, dtype="uint8"), (640, 2))
    # im[::2, :] = np.fliplr(im[::2, :])
    # im = im.T
    im = np.tile(np.arange(0, 256, dtype="uint8"), (640, 2)).T
    print(im.shape)
    imwrite(src_file_source, im)
    return src_file_source

def match_folder(folders, ref_files, delim="-"):
    folder_times = np.array(
        [datetime.strptime(f, f"%H{delim}%M{delim}%S") for f in folders]
    )
    ref_times = np.array(
        [datetime.strptime(f.removesuffix(".txt"), "%H-%M-%S") for f in ref_files]
    )
    matches = []
    for ii, t in enumerate(folder_times):
        match_time = np.where(np.abs(ref_times - t) < timedelta(seconds=15))
        if match_time[0].size > 0:
            matches.append((folders[ii], ref_files[match_time[0][0]]))
    return matches

def _orthorectify(src_file, dem_file, int_param_file, ext_param_file, io_kwargs=dict(crs='EPSG:3857')): #crs='EPSG:4087'
    print(src_file, ext_param_file)
    cameras = oty.FrameCameras(int_param_file, ext_param_file, io_kwargs=io_kwargs)
    camera = cameras.get(src_file)
    print(dir(camera))
    print(camera.pos)
# create Ortho object and orthorectify
    ortho = oty.Ortho(src_file, dem_file, camera=camera, crs=cameras.crs)
    return ortho

def orthorectify(root_dir, time="20-33-30", instrument="Telops"):
    suffix = ".tiff"
    
    # root = root_dir + "Metashape/Workswell/Metashape_data_references/Qinertia/"
    root = root_dir + f"Orthority/{instrument}/"
    # src_file_source = root_dir+'Workswell_images/22-45-29/'  # aerial image
    # time = "20-33-30"
    src_file_source = match_folder(os.listdir(root_dir+f'{instrument}_images/'), [f"{time}.txt"])[0][0]
    src_file_source = root_dir+f'{instrument}_images/' + src_file_source + "/" # aerial image
    # ext_dir = root + "single_entries210339pitch6.7/"
    ext_dir = root + f"single_entriesOffset_{time}/"
    # src_files = [src_file_source + f for f in os.listdir(src_file_source) if f.endswith('.jpg')]
    # dem_file = root_dir+'USGS_mountain_fire_dem.tif'  # DEM covering imaged area
    # dem_file = root_dir+'USGS_Kaibab_Fire_dem.tif'  # DEM covering imaged area
    dem_file = root_dir+'USGS_Cedar_Creek_Fire_bigdem.tif'  # DEM covering imaged area
    int_param_file = root_dir + "Orthority/int_camera_calfide.yaml"  # interior parameters
    # ext_param_file = root + '22-58-55_opk2.geojson'  # exterior parameters
    # ext_param_file = root + '22-45-41_opk3.csv'  # exterior parameters
    ext_param_files = [ext_dir + f for f in os.listdir(ext_dir) if f.endswith('.csv')]  # exterior parameters


    # create a camera model for src_file from interior & exterior parameters
    # nums = [f.split('_')[-1].removesuffix(suffix) for f in os.listdir(src_file_source) if f.endswith(suffix)]
    nums = [f.split('_')[-1].removesuffix(suffix) for f in os.listdir(src_file_source) if f.endswith(suffix)]
    # print(src_files, ext_param_files)
    io_kwargs = dict(crs='EPSG:3857') #crs='EPSG:4087'
    estimate_res = (2.87, 2.87)#(1.6, 1.6)

    for ii in nums[:]:
        src_file = src_file_source + f"frame_{ii}"+suffix
        ext_param_file = ext_dir + f"frame_{ii}.csv"
        ortho = _orthorectify(src_file, dem_file, int_param_file, ext_param_file, io_kwargs=io_kwargs)
        os.makedirs(root+f"results_offset_{time}/", exist_ok=True)
        ortho.process(root+f"results_offset_{time}/ortho_{ii}.tif", resolution=estimate_res, interp=oty.enums.Interp.nearest)

    return 0

def try_angles():
    root_dir = "C:/Users/thijs/OneDrive/Documents/PostDoc_UPC/Data_local/Test_Fire_processing/"
    instrument = "Flir_wide"
    root = root_dir # + "Qinertia/"
    time = "18-22-32"
    # src_file_source = root_dir+f'{instrument}_images/20.33.30/'  # aerial image
    src_file = root_dir+f'Flir_images/Flir_wide/18.22.32_rot180/frame_64497.tiff'  # aerial image _rot180
    # ext_dir = root + "single_entries210339pitch6.7/"
    ext_dir =  root + f"Orthority/{instrument}/"#trypitchrollyaw/"
    dem_file = root_dir+'Merged_DSM.tif'  # DEM covering imaged area
    int_param_file = root_dir + "int_camera_firesense.yaml"  # interior parameters
    pitches = [0]
    rolls = [0]
    yaws = [0]
    
    for p in pitches:
        for r in rolls:
            for y in yaws:
                ext_param_file = ext_dir+"frame_solvePnP.csv"#f"p{p}r{r}y{y}.csv" # exterior parameters   
                # print(src_files, ext_param_files)
                io_kwargs = dict(crs='EPSG:3857') #crs='EPSG:4087'
                estimate_res = (2.87, 2.87)#(1.6, 1.6)
                ortho = _orthorectify(src_file, dem_file, int_param_file, ext_param_file, io_kwargs=io_kwargs)
                os.makedirs(root+f"results_tryangles_{time}/", exist_ok=True)
                ortho.process(root+f"results_tryangles_{time}/solvePnP5.tif", resolution=estimate_res, interp=oty.enums.Interp.nearest)

    # interP_ortho_pitch{p}roll{r}yaw{y}.tif

def median_raster(merged_data, new_data, merged_mask, new_mask, **kwargs):
    """Calculates the median of the overlapping pixels in the input arrays"""
    raise TypeError("Niet te doen ")

def combine_orthos(num = "1"):
    """combines the orthorectified images through the median of the overlapping pixels"""
    # root_dir = "C:/Users/thijs/OneDrive/Documents/PostDoc_UPC/Data_local/Mountain_Fire_processing/day_09_04_2022/"
    # root = root_dir+'Metashape/Workswell/Metashape_data_references/Qinertia/' #Qinertia/ Interpolated/
    root_dir = "C:/Users/thijs/OneDrive/Documents/PostDoc_UPC/Data_local/Test_Fire_processing/"
    root = root_dir #+ "Flir_wide/"
    orthos_uint8 = [r_open(root+f"results_182232/{ii}") for ii in os.listdir(root+f"results_182232/") if ii.endswith('.tif')]
    orthos = [WarpedVRT(ortho, dtype="float32") for ortho in orthos_uint8]
    print(type(orthos[0].read(1)[0, 0]))
    sum, outputT = merge(orthos, method="sum", dtype="float32")
    counts, outputT = merge(orthos, method="count")
    max, outputT = merge(orthos, method="max")
    min, outputT = merge(orthos, method="min")
    first, outputT = merge(orthos, method="first")
    last, outputT = merge(orthos, method="last")
    # counts = counts.astype(float)
    # sum = sum.astype(float)
    print(sum.dtype, counts.dtype)
    mean = np.divide(sum, counts, out=np.zeros_like(sum), where=counts!=0)
    output_meta = orthos_uint8[0].meta.copy()
    output_meta.update(
    {   "dtype": "float32",
        "height": mean.shape[1],
        "width": mean.shape[2],
        "transform": outputT,
    }
)
    
    with rio.open(root+f"ortho_combined_mean{num}.tif", 'w', **output_meta) as dst:
        dst.write(mean)
    with rio.open(root+f"ortho_combined_sum{num}.tif", 'w', **output_meta) as dst:
        dst.write(sum)
    with rio.open(root+f"ortho_combined_counts{num}.tif", 'w', **output_meta) as dst:
        dst.write(counts)
    with rio.open(root+f"ortho_combined_max{num}.tif", 'w', **output_meta) as dst:
        dst.write(max)
    with rio.open(root+f"ortho_combined_min{num}.tif", 'w', **output_meta) as dst:
        dst.write(min)
    with rio.open(root+f"ortho_combined_first{num}.tif", 'w', **output_meta) as dst:
        dst.write(first)
    with rio.open(root+f"ortho_combined_last{num}.tif", 'w', **output_meta) as dst:
        dst.write(last)
    return 0

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def single_trim(num = "20", stripwidth = 24, instrument="Telops"):
    """combines the orthorectified images through the the centerlines of the overlapping pixels"""
    # root_dir = "C:/Users/thijs/OneDrive/Documents/PostDoc_UPC/Data_local/Mountain_Fire_processing/day_09_04_2022/"
    # root = root_dir+'Metashape/Workswell/Metashape_data_references/Qinertia/' #Qinertia/ Interpolated/
    root_dir = "/media/Data/2022_calfide/Cedar_Creek_Fire_processing/day_09_10_2022/"
    root = root_dir + f"Orthority/{instrument}/"
    time = num#"20-33-30"
    source = root+f"results_{time}/"
    orthos_uint8 = [ii for ii in os.listdir(source) if ii.endswith('.tif') and "trimmed" not in ii]
    for ortho_name in orthos_uint8[0:1]:
        ortho = r_open(source+ortho_name, "r")
        meta = ortho.meta
        # warp = WarpedVRT(ortho, dtype="float32")
        data = ortho.read(1).copy()
        
        # print("og_data: ", data[517, len(data[0])//2-12:len(data[0])//2+12])
        along_flight_angle = guess_flight_angle(data)
        # print(np.rad2deg(along_flight_angle))
        mask = make_mask(data, along_flight_angle, stripwidth=stripwidth)
        fig, axes = plt.subplots( 1, 3)
        axes[0].imshow(data)
        axes[1].imshow(mask)
        axes[2].imshow(data*mask)
        plt.show()
    return 0 

def combine_orthosv2(root_dir, num = "20", stripwidth = 24, instrument="Telops"):
    """combines the orthorectified images through the the centerlines of the overlapping pixels"""
    # root_dir = "C:/Users/thijs/OneDrive/Documents/PostDoc_UPC/Data_local/Mountain_Fire_processing/day_09_04_2022/"
    # root = root_dir+'Metashape/Workswell/Metashape_data_references/Qinertia/' #Qinertia/ Interpolated/
    
    root = root_dir + f"Orthority/{instrument}/"
    time = num#"20-33-30"
    source = root+f"results_offset_{time}/"
    orthos_uint8 = [ii for ii in os.listdir(source) if ii.endswith('.tif') and "trimmed" not in ii]
    orthos_numbers = [ii.split("_")[1].rstrip(".tif") for ii in orthos_uint8]
    if len(orthos_uint8) > 500:
        list_of_lists = split(orthos_uint8, len(orthos_uint8)//300)
    else: 
        list_of_lists = [orthos_uint8]
    # orthos = [WarpedVRT(ortho, dtype="float32") for ortho in orthos_uint8]
    for number, orthos_uint8 in enumerate(list_of_lists):
        orthos_trimmed = []
        for ortho_name in orthos_uint8:
            ortho = r_open(source+ortho_name, "r")
            meta = ortho.meta
            # warp = WarpedVRT(ortho, dtype="float32")
            data = ortho.read(1).copy()
            # plt.imshow(data)
            # plt.show()
            # print("og_data: ", data[517, len(data[0])//2-12:len(data[0])//2+12])
            along_flight_angle = guess_flight_angle(data)
            # print(np.rad2deg(along_flight_angle))
            mask = make_mask(data, along_flight_angle, stripwidth=stripwidth)
            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # ax[0].imshow(np.abs(distance))

            # ax[1].imshow(data)
            # ax[1].imshow(mask, alpha=0.5)
            # plt.show()
            # bla
            # X_rot = np.clip(X_rot, 0, data.shape[0]-1).astype(int)
            # Y_rot = np.clip(Y_rot, 0, data.shape[1]-1).astype(int)
            # data_rot = np.zeros_like(data)
            # data_rot[X_rot, Y_rot] = data[X, Y]
            # data[:, len(data[0])//2+12:] = 0
            # data[:, :len(data[0])//2-12] = 0
            # print(data.shape, len(data[0])//2+12, len(data[0])//2-12)
            # print(data[517, len(data[0])//2-12:len(data[0])//2+12])
            # ortho.write(data, indexes=1)
            data = np.where(mask, data, np.nan)
            with rio.open(source+f"trimmed_{ortho.name.split('/')[-1]}", 'w', **meta) as dst:
                dst.write(data, indexes=1)
            ortho = rio.open(source+f"trimmed_{ortho.name.split('/')[-1]}", "r")
            ortho_trim_warped = WarpedVRT(ortho, dtype="float32")
            orthos_trimmed.append(ortho_trim_warped)
        sum, outputT = merge(orthos_trimmed, method="sum", dtype="float32")
        counts, outputT = merge(orthos_trimmed, method="count")
        max, outputT = merge(orthos_trimmed, method="max")
        # min, outputT = merge(orthos_trimmed, method="min")
        # first, outputT = merge(orthos_trimmed, method="first")
        # last, outputT = merge(orthos_trimmed, method="last")
        # counts = counts.astype(float)
        # sum = sum.astype(float)
        print(sum.dtype, counts.dtype)
        mean = np.divide(sum, counts, out=np.zeros_like(sum), where=counts!=0)
        output_meta = ortho.meta.copy()
        output_meta.update(
            {   "dtype": "float32",
                "height": mean.shape[1],
                "width": mean.shape[2],
                "transform": outputT,
            }
        )
        with rio.open(root+f"ortho_combined_mean_offset_strip{num}_{number}.tif", 'w', **output_meta) as dst:
            dst.write(mean)
        with rio.open(root+f"ortho_combined_sum_offset_strip{num}_{number}.tif", 'w', **output_meta) as dst:
            dst.write(sum)
        with rio.open(root+f"ortho_combined_counts_offset_strip{num}_{number}.tif", 'w', **output_meta) as dst:
            dst.write(counts)
        with rio.open(root+f"ortho_combined_max_offset_strip{num}_{number}.tif", 'w', **output_meta) as dst:
            dst.write(max)
    return 0
    
def main():
    # fake_path = make_fake_data()
    
#     root_dir = "C:/Users/thijs/OneDrive/Documents/PostDoc_UPC/Data_local/Mountain_Fire_processing/day_09_04_2022/"
#     root = root_dir+'Metashape/Workswell/Metashape_data_references/Qinertia/' #Qinertia/ Interpolated/
#     fake_path = root_dir+'Workswell_images/fake.jpg'
#     int_param_file = root + "int_camera.yaml"  # interior parameters
#     ext_param_file = root + 'fake_ext.csv'  # exterior parameters
#     io_kwargs = dict(crs='EPSG:4087')
#     dem_file = root_dir+'USGS_mountain_fire_dem.tif'
#     cameras = oty.FrameCameras(int_param_file, ext_param_file, io_kwargs=io_kwargs)
#     camera = cameras.get(fake_path)
#     print(dir(camera))
#     print(camera.pos)
# # create Ortho object and orthorectify
#     ortho = oty.Ortho(fake_path, dem_file, camera=camera, crs=cameras.crs)

#     ortho.process(ortho_file = root+f"fake4_rec.tif", resolution=(1.0, 1.0))
#     result = r_open(root+f"fake4_rec.tif")
#     print(result.read(1).shape)
#     print(result.profile)
#     print(np.sum(result.read(1)>0))
#     print(640*514)

    # try_angles()

    root_dir = "C:/Users/thijs/OneDrive/Documents/PostDoc_UPC/Data_local/Test_Fire_processing/"
    # root = root_dir+'Metashape/Workswell/Metashape_data_references/Qinertia/' #Qinertia/ Interpolated/
    # root_dir = "/media/Data/2022_calfide/Cedar_Creek_Fire_processing/day_09_10_2022/"
    num = "18-22-32"
    # orthorectify(root_dir, time = num, instrument="Flir_wide")
    
    # combine_orthos()
    combine_orthosv2(num=num, stripwidth=90, instrument="Workswell")
    # single_trim(num=num, stripwidth=90, instrument="Workswell")
    return 0

if __name__ == "__main__":
    sys.exit(main())