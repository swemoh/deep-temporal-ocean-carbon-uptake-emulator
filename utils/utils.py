import pandas as pd
import numpy as np

def get_year_month(df, yrmonth):
    df['time_counter'] = df['time_counter'].astype("string")
    df = df.loc[df['time_counter'].str.contains(yrmonth, case=False)]
    return df.reset_index()

def round_nav_lat(df):
    '''
    Round up the coordinates to 2 decimal places
    '''
    df['nav_lat'] = df['nav_lat'].apply(lambda x:round(x,2))
    df['nav_lon'] = df['nav_lon'].apply(lambda x:round(x,2))
    return df

def select_nth_coordinates(df, select_n=10):

    # unique lat-lon pairs in order of appearance
    unique_pairs = df[['nav_lat', 'nav_lon']].drop_duplicates().reset_index(drop=True)

    # every nth pair selection
    sampled_pairs = unique_pairs.iloc[::select_n]

    # Filter the DataFrame to keep only those pairs
    # df_sampled = df.merge(sampled_pairs, on=['nav_lat', 'nav_lon'], how='inner')
    return sampled_pairs

def sample_random(df, frac=0.5, random_state = 42):
    # unique lat-lon pairs in order of appearance
    unique_pairs = df[['nav_lat', 'nav_lon']].drop_duplicates().reset_index(drop=True)
    sampled_pairs = unique_pairs.sample(frac=frac, random_state=random_state)
    return sampled_pairs

def get_zoned_df(appended_data):
    '''
    returns multiple dataframes corresponding to 4 different basins
    '''
    
    zone_ARCTIC = appended_data.loc[appended_data['nav_lat'] >= 70.0]
    zone_ARCTIC['zone'] = 'ARCTIC'
        
    zone_NORTH_ATLANTIC_lon= appended_data.loc[(appended_data['nav_lon'] >= -75.0) & (appended_data['nav_lon'] <= 0.0)] ## Subpolar NATL 75 to 45N, Subtropical NATL 45 to 20N
    zone_NORTH_ATLANTIC_SP = zone_NORTH_ATLANTIC_lon.loc[(zone_NORTH_ATLANTIC_lon['nav_lat'] >= 40) & (zone_NORTH_ATLANTIC_lon['nav_lat'] < 70)]
    zone_NORTH_ATLANTIC_SP['zone'] = 'NORTH_ATLANTIC_SP'
    zone_NORTH_ATLANTIC_ST = zone_NORTH_ATLANTIC_lon.loc[(zone_NORTH_ATLANTIC_lon['nav_lat'] >= 10) & (zone_NORTH_ATLANTIC_lon['nav_lat'] < 40)]
    zone_NORTH_ATLANTIC_ST['zone'] = 'NORTH_ATLANTIC_ST'
    
    zone_EQ= appended_data.loc[(appended_data['nav_lat'] >= -10.0) & (appended_data['nav_lat'] <= 10.0)]
    zone_EQ_PACIFIC_1 = zone_EQ.loc[(zone_EQ['nav_lon'] >= 105.0) & (zone_EQ['nav_lon'] <= 180.0)]
    zone_EQ_PACIFIC_2 = zone_EQ.loc[(zone_EQ['nav_lon'] >= -180.0) & (zone_EQ['nav_lon'] <= -80.0)]
    zone_EQ_PACIFIC = pd.concat([zone_EQ_PACIFIC_1, zone_EQ_PACIFIC_2])
    zone_EQ_PACIFIC['zone'] = 'EQ_PACIFIC'
    
    zone_SOUTHERN_OCEAN_ST = appended_data.loc[appended_data['nav_lat'] <= -30 & (appended_data['nav_lat'] > -45)]
    zone_SOUTHERN_OCEAN_SP = appended_data.loc[appended_data['nav_lat'] <= -45]
    zone_SOUTHERN_OCEAN_SP['zone'] = 'SOUTHERN_OCEAN_SP'
    zone_SOUTHERN_OCEAN_ST['zone'] = 'SOUTHERN_OCEAN_ST'
    
    return zone_ARCTIC, zone_NORTH_ATLANTIC_SP, zone_NORTH_ATLANTIC_ST, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN_SP, zone_SOUTHERN_OCEAN_ST

def assign_basins(nav_lat, nav_lon):
    '''
    Assigns basins to individual latitude and longitude values. 
    !!! Better use it as a lambda function.
    
    df['basin'] = df.apply(lambda row: assign_basins(row['nav_lat'], row['nav_lon']), axis=1)
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