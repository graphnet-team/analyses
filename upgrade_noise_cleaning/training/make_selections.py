import sqlite3
import pandas as pd
import os

from graphnet.data.sqlite.sqlite_selection import get_even_track_cascade_indicies

def make_pulse_cleaning_selection(database):
    with sqlite3.connect(database) as con:
        query = 'select event_no from truth where abs(pid) != 1'
        return pd.read_sql(query,con)

def make_track_cascade_classification(database):
    with sqlite3.connect(database) as con:
        query = "select event_no from truth where abs(pid) = 12"
        nu_e = pd.read_sql(query, con)

    with sqlite3.connect(database) as con:
        query = "select event_no from truth where abs(pid) = 14 and interaction_type = 1"
        tracks = pd.read_sql(query, con)

    with sqlite3.connect(database) as con:
        query = "select event_no from truth where abs(pid) = 14 and interaction_type = 2"
        nu_u_nc = pd.read_sql(query, con)

    with sqlite3.connect(database) as con:
        query = "select event_no from truth where abs(pid) = 16 and interaction_type = 2"
        nu_tau = pd.read_sql(query, con)

    cascades = nu_e.append(nu_u_nc, ignore_index = True).append(nu_tau, ignore_index = True).reset_index(drop = True)
    
    if len(tracks)> len(cascades):
        train_events = cascades.append(tracks.sample(len(cascades)), ignore_index = True).sample(frac = 1).reset_index(drop = True)
    else:
        train_events = tracks.append(cascades.sample(len(cascades)), ignore_index = True).sample(frac = 1).reset_index(drop = True)

    return train_events

def make_nu_mu_classification_selection(database):
    with sqlite3.connect(database) as con:
        query = 'select event_no from truth where abs(pid) != 13 and abs(pid) != 1' 
        neutrinos = pd.read_sql(query,con)
        query = 'select event_no from truth where abs(pid) = 13 and abs(pid) != 1'
        muons = pd.read_sql(query,con)

        if len(muons) > len(neutrinos):
            return neutrinos.append(muons.sample(len(neutrinos)), ignore_index = True).sample(frac = 1).reset_index(drop = True)
        else:
            return muons.append(neutrinos.sample(len(muons)), ignore_index = True).sample(frac = 1).reset_index(drop = True)

def make_regression_selection(database):
    with sqlite3.connect(database) as con:
        query = 'select event_no from truth where abs(pid) != 13 and abs(pid) != 1' 
        neutrinos = pd.read_sql(query,con)
    return neutrinos
    
os.makedirs('selections', exist_ok = True)

database = '/mnt/scratch/rasmus_orsoe/databases/upgrade/dev_step4_upgrade_028_with_noise_dynedge_pulsemap_v3_merger_aftercrash/data/dev_step4_upgrade_028_with_noise_dynedge_pulsemap_v3_merger_aftercrash.db'


df = make_track_cascade_classification(database)
df.to_csv('/home/iwsatlas1/oersoe/phd/upgrade_event_selection/training/selections/track_classification.csv')
#df = make_nu_mu_classification_selection(database = database)
#df.to_csv('selections/neutrino_classification.csv')

#df = make_pulse_cleaning_selection(database = database)
#df.to_csv('selections/pulse_cleaning.csv')

#df = make_regression_selection(database)
#df.to_csv('selections/regression_selection.csv')