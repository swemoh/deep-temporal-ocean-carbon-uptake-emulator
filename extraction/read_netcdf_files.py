import xarray as xr
import pandas as pd
import numpy as np
from glob import glob

import warnings
warnings.filterwarnings("ignore")

def mask_land(mask_df, data_df):
    data_df = data_df.merge(mask_df, left_index=True, right_on=['y', 'x'])
    return data_df

def assign_basins(nav_lat, nav_lon):
    '''
    Assigns basins to individual latitude and longitude values. 
    !!! Better use it as a lambda function.
    '''
    if nav_lat > 70.0:
        return 'ARCTIC'
    elif -75.0 <= nav_lon <= 0.0 and 10 <= nav_lat <= 70: 
        return 'NORTH_ATLANTIC'
    elif -10.0 <= nav_lat <= 10.0:
        if 105.0 <= nav_lon <= 180.0 or -180.0 <= nav_lon <= -80.0:
            return 'EQ_PACIFIC'
    elif nav_lat <= -45:
        return 'SOUTHERN_OCEAN'
    else:
        return 'OTHER'

def coriolis_parameter(lat_deg):
    """
    lat_deg : latitude in degrees (scalar or array)
    returns f in s^-1
    Note: f > 0 in NH, f < 0 in SH
    """
    Omega = 7.2921159e-5  # Earth's rotation rate [s^-1]
    lat_rad = np.deg2rad(lat_deg)
    f = 2 * Omega * np.sin(lat_rad)

    return f

def get_mean_over_lat_lon(df, col_name):
    '''
    Get the mean for each pair of coordinate
    '''
    df = df.groupby(['nav_lat','nav_lon','time_counter'], as_index=False)[col_name].mean()
    return df

def calculate_curl(df):
    
    # Sort so shifting is correct
    # df = df.sort_values(["y", "x"])
    df = df.sort_values(["nav_lat", "nav_lon"])
    
    ## -- Currents
    df["term_v"] = df["e2u"] * df["currents_Y"]
    df["term_u"] = df["e1v"] * df["currents_X"]

    # Sort so shifting is correct
    df = df.sort_values(["y", "x"])
    
    # d(term_v)/dx
    df["term_v_shift"] = df.groupby("y")["term_v"].shift(-1)
    df["dterm_v"] = df["term_v_shift"] - df["term_v"]
    
    # d(term_u)/dy
    df["term_u_shift"] = df.groupby("x")["term_u"].shift(-1)
    df["dterm_u"] = df["term_u_shift"] - df["term_u"]

    df["relative_vorticity"] = (df["dterm_v"] - df["dterm_u"]) / (df["e1f"] * df["e2f"])

    ## -- Stress

    df["term_v_s"] = df["e2u"] * df["stress_Y"]
    df["term_u_s"] = df["e1v"] * df["stress_X"]

    # Sort so shifting is correct
    # df = df.sort_values(["y", "x"])
    
    # d(term_v)/dx
    df["term_v_shift_s"] = df.groupby("y")["term_v_s"].shift(-1)
    df["dterm_v_s"] = df["term_v_shift_s"] - df["term_v_s"]
    
    # d(term_u)/dy
    df["term_u_shift_s"] = df.groupby("x")["term_u_s"].shift(-1)
    df["dterm_u_s"] = df["term_u_shift_s"] - df["term_u_s"]

    df["windstress_curl"] = (df["dterm_v_s"] - df["dterm_u_s"]) / (df["e1f"] * df["e2f"])

    df = df[df["x"] < df["x"].max()]     # drop last x
    df = df[df["y"] < df["y"].max()]     # drop last y

    # df = df.drop(columns=["term_v","term_u","term_v_shift","term_u_shift","dterm_v","dterm_u"])
    
    return df


output_folder_num = '1'
path = "PATH"
EXP_NAME = 'EXP-NAME'

print(f"\n---> Reading mask file.")

path_mask = glob("../../mask/mesh_mask_orca_025.nc")
print(path_mask) # "../../mask/mesh_mask_orca_025.nc" 

mask = xr.open_dataset(path_mask[0],chunks={"z":1, "y":100, "x":100}).tmask.isel(z=0).squeeze()
e1t = xr.open_dataset(path_mask[0],chunks={"z":1, "y":100, "x":100}).e1t.squeeze()
e2t = xr.open_dataset(path_mask[0],chunks={"z":1, "y":100, "x":100}).e2t.squeeze()
e1u = xr.open_dataset(path_mask[0],chunks={"z":1, "y":100, "x":100}).e1u.squeeze()
# e2u = xr.open_dataset(path_mask[0],chunks={"z":1, "y":100, "x":100}).e2u.squeeze()
# e1v = xr.open_dataset(path_mask[0],chunks={"z":1, "y":100, "x":100}).e1v.squeeze()
e2v = xr.open_dataset(path_mask[0],chunks={"z":1, "y":100, "x":100}).e2v.squeeze()
# e1f = xr.open_dataset(path_mask[0],chunks={"z":1, "y":100, "x":100}).e1f.squeeze()
# e2f = xr.open_dataset(path_mask[0],chunks={"z":1, "y":100, "x":100}).e2f.squeeze()
mask_df = mask.to_dataframe()
mask_df['e1t'] = e1t.to_dataframe()['e1t']
mask_df['e2t'] = e2t.to_dataframe()['e2t']
mask_df['e1u'] = e1u.to_dataframe()['e1u']
# mask_df['e2u'] = e2u.to_dataframe()['e2u']
# mask_df['e1v'] = e1v.to_dataframe()['e1v']
mask_df['e2v'] = e2v.to_dataframe()['e2v']
# mask_df['e1f'] = e1f.to_dataframe()['e1f']
# mask_df['e2f'] = e2f.to_dataframe()['e2f']

print("\n Mask extracted.")

years = range(1958,2019)
# years = range(2017,2019)

# Loop over 61 years
for yr in years:
    print(f"\n---> Reading {yr}")
    # try:
    print(f"\n---> Processing {yr}")

    print(f"\n Reading SST")
    file_sst = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*sosstsst.nc")
    sst_xr = xr.open_dataset(file_sst[0], chunks={"z":1, "y":100, "x":100})
    sst_df = sst_xr.sosstsst.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading SAL")
    file_sal = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*sosaline.nc")
    sal_xr = xr.open_dataset(file_sal[0], chunks={"z":1, "y":100, "x":100})
    sal_df = sal_xr.sosaline.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading MLD")
    file_mld = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*somxl010.nc")
    mld_xr = xr.open_dataset(file_mld[0], chunks={"z":1, "y":100, "x":100})
    mld_df = mld_xr.somxl010.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading Heat Flux")
    file_heat_flux = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*sohefldo.nc")
    heat_flux_xr = xr.open_dataset(file_heat_flux[0], chunks={"z":1, "y":100, "x":100})
    heat_flux_down_df = heat_flux_xr.sohefldo.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading Water Flux")
    file_water_flux = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*sowaflup.nc")
    water_flux_xr = xr.open_dataset(file_water_flux[0], chunks={"z":1, "y":100, "x":100})
    water_flux_up_df = water_flux_xr.sowaflup.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading ICEMOD")
    file_ice = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*ileadfra.nc")
    ice_xr = xr.open_dataset(file_ice[0], chunks={"z":1, "y":100, "x":100})
    icemod_df = ice_xr.ileadfra.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading Stress_x")
    file_stress_x = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*sozotaux.nc")
    stress_x_xr = xr.open_dataset(file_stress_x[0], chunks={"z":1, "y":100, "x":100})
    stress_x_df = stress_x_xr.sozotaux.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading Stress_y")
    file_stress_y = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*sometauy.nc")
    stress_y_xr = xr.open_dataset(file_stress_y[0], chunks={"z":1, "y":100, "x":100})
    stress_y_df = stress_y_xr.sometauy.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading Current_x")
    file_current_x = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*vozocrtx_k10.nc")
    current_x_xr = xr.open_dataset(file_current_x[0], chunks={"z":1, "y":100, "x":100})
    current_x_df = current_x_xr.vozocrtx.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading Current_y")
    file_current_y = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*vomecrty_k10.nc")
    current_y_xr = xr.open_dataset(file_current_y[0], chunks={"z":1, "y":100, "x":100})
    current_y_df = current_y_xr.vomecrty.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading fco2_pre")
    file_fco2_pre = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*fco2_pre.nc")
    fco2_pre_xr = xr.open_dataset(file_fco2_pre[0], chunks={"z":1, "y":100, "x":100})
    fco2_pre_df = fco2_pre_xr.fco2_pre.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading fco2")
    file_fco2 = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*fco2.nc")
    fco2_xr = xr.open_dataset(file_fco2[0], chunks={"z":1, "y":100, "x":100})
    fco2_df = fco2_xr.fco2.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading co2_flux_pre")
    file_co2_flux_pre = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*co2flux_pre.nc")
    co2_flux_pre_xr = xr.open_dataset(file_co2_flux_pre[0], chunks={"z":1, "y":100, "x":100})
    co2_flux_pre_df = co2_flux_pre_xr.co2flux_pre.squeeze().to_dataframe().reset_index(level=['time_counter'])

    print(f"\n Reading co2_flux")
    file_co2_flux = glob(f"{path}/{EXP_NAME}/ORCA025*1m_{yr}*co2flux.nc")
    co2_flux_xr = xr.open_dataset(file_co2_flux[0], chunks={"z":1, "y":100, "x":100})
    co2_flux_df = co2_flux_xr.co2flux.squeeze().to_dataframe().reset_index(level=['time_counter'])


    print(f"\n Applying mask.")
    sst_df = mask_land(mask_df, sst_df)
    sal_df = mask_land(mask_df, sal_df)
    mld_df = mask_land(mask_df, mld_df)
    heat_flux_down_df = mask_land(mask_df, heat_flux_down_df)
    water_flux_up_df = mask_land(mask_df, water_flux_up_df)
    # height_df = mask_land(mask_df, height_df)

    icemod_df = mask_land(mask_df, icemod_df)

    stress_x_df = mask_land(mask_df, stress_x_df)
    current_x_df = mask_land(mask_df, current_x_df)
    stress_y_df = mask_land(mask_df, stress_y_df)
    current_y_df = mask_land(mask_df, current_y_df)

    fco2_pre_df = mask_land(mask_df, fco2_pre_df)
    fco2_df = mask_land(mask_df, fco2_df)

    co2_flux_pre_df = mask_land(mask_df, co2_flux_pre_df)
    co2_flux_df = mask_land(mask_df, co2_flux_df)

    print(f"\n Building dataframe.")
    # get all columns from SST df
    data_df = sst_df.reset_index().rename(columns={"sosstsst": "SST",})
    # get rest of the features
    data_df['SAL'] = sal_df.reset_index()['sosaline']
    data_df['ice_frac'] = icemod_df.reset_index()['ileadfra']
    data_df['mixed_layer_depth'] = mld_df.reset_index()['somxl010']
    data_df['heat_flux_down'] = heat_flux_down_df.reset_index()['sohefldo']
    data_df['water_flux_up'] = water_flux_up_df.reset_index()['sowaflup']
    # data_df['surface_height'] = height_df.reset_index()['sossheig']
    data_df['stress_X'] = stress_x_df.reset_index()['sozotaux']
    data_df['stress_Y'] = stress_y_df.reset_index()['sometauy']
    data_df['currents_X'] = current_x_df.reset_index()['vozocrtx']
    data_df['currents_Y'] = current_y_df.reset_index()['vomecrty']
    data_df['fco2_pre'] = fco2_pre_df.reset_index()['fco2_pre']
    data_df['fco2'] = fco2_df.reset_index()['fco2']
    data_df['co2flux_pre'] = co2_flux_pre_df.reset_index()['co2flux_pre']
    data_df['co2flux'] = co2_flux_df.reset_index()['co2flux']

    print(f"\n Calculating wind stress curls.")
    # data_df=calculate_curl(df=data_df)

    mask = xr.open_dataset(path_mask[0],chunks={"z":1, "y":100, "x":100})
    
    ds_x = stress_x_xr
    ds_y = stress_y_xr

    # Merge needed variables into one dataset
    merged = xr.merge([
        ds_x[["sozotaux"]],
        ds_y[["sometauy"]],
        mask[["e1u", "e2u", "e1v", "e2v", "e1f", "e2f"]],], compat='override')

    u_U = merged["sozotaux"]
    v_V = merged["sometauy"]
    e2v = merged["e2v"]
    e1u = merged["e1u"]
    
    dudy_F = (u_U.shift(y=-1)-u_U)/e2v
    dvdy_F = (v_V.shift(x=-1)-v_V)/e1u
    curl_F = dvdy_F - dudy_F
    # interpolation onto the T-grid by averageing from the four F-points 
    curl_T = (curl_F.shift(x=1,y=1) + curl_F.shift(x=1) + curl_F.shift(y=1) + curl_F)/4 
    
    curl_T.name = "wind_stress_curl"
    curl_T.attrs["long_name"] = "wind stress curl"
    curl_T.attrs["units"] = "N m-3"
    curl_T_df = curl_T.to_dataframe().reset_index(level=['time_counter', 't'])
    curl_T_df = mask_land(mask_df, curl_T_df)
    data_df['wind_stress_curl'] = curl_T_df.reset_index()['wind_stress_curl']

    print(f"\n Calculating relative vorticity or currents curls.")

    ds_x = current_x_xr
    ds_y = current_y_xr

    # Merge needed variables into one dataset
    merged = xr.merge([
        ds_x[["vozocrtx"]],
        ds_y[["vomecrty"]],
        mask[["e1u", "e2u", "e1v", "e2v", "e1f", "e2f"]],], compat='override')

    u_U = merged["vozocrtx"]
    v_V = merged["vomecrty"]
    e2v = merged["e2v"]
    e1u = merged["e1u"]
    
    dudy_F = (u_U.shift(y=-1)-u_U)/e2v
    dvdy_F = (v_V.shift(x=-1)-v_V)/e1u
    curl_F = dvdy_F - dudy_F
    # interpolation onto the T-grid by averageing from the four F-points 
    curl_T = (curl_F.shift(x=1,y=1) + curl_F.shift(x=1) + curl_F.shift(y=1) + curl_F)/4 
    
    curl_T.name = "rel_vorticity"
    curl_T.attrs["long_name"] = "Realtive vorticity - currents curl"
    curl_T.attrs["units"] = "s-1"
    curl_T_df = curl_T.to_dataframe().reset_index(level=['time_counter', 't', 'depthu', 'depthv'])
    curl_T_df = mask_land(mask_df, curl_T_df)
    data_df['rel_vorticity'] = curl_T_df.reset_index()['rel_vorticity']

    print(f"\n Applying coriolis parameter.")

    data_df["f"] = coriolis_parameter(data_df["nav_lat"].values) # Compute f for all rows at once
    data_df["wind_stress_curl_f"] = data_df["wind_stress_curl"] / data_df["f"]
    data_df["rel_vorticity_f"] = data_df["rel_vorticity"] / data_df["f"]
    
    
    data_df.to_pickle(f"../../OUTPUT/{output_folder_num}/{EXP_NAME}_{yr}_df.pkl")

    print(f"\n---> {yr} processed.\n")