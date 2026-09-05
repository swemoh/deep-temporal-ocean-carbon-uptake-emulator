import numpy as np
import pandas as pd

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

output_folder_num = 1
sim_name = "ADD-NAME"
list_of_files = glob.glob(f"ADD-PATH")

for yr in range(1958, 2019):
    print(yr)
    fn = [file_name for file_name in list_of_files if f"_{str(yr)}_" in file_name]
    df = pd.read_pickle(fn[0])
    df["f"] = coriolis_parameter(df["nav_lat"].values) # Compute f for all rows at once
    df["wind_stress_curl_f"] = df["wind_stress_curl"] / df["f"]
    df["rel_vorticity_f"] = df["rel_vorticity"] / df["f"]