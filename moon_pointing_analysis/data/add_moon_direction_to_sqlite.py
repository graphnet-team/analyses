from cmath import nan
import sqlite3 as sql
import numpy as np
from pandas import read_sql
from icecube import astro
from icecube.dataclasses import I3Time

# data pathing
indir = "/data/user/mholm/data/moonL4_segspline_exp13_01_redo_00001.db"
outdir = "/data/user/mholm/data/MoonL4_with_time/"

def moonDirection(mjd):
    time = I3Time()
    time.set_mod_julian_time_double(mjd)
    moon = astro.I3GetMoonDirection(time)
    return moon.azimuth, moon.zenith

tablename = "MoonDirection"

with sql.connect(indir) as con:
    # add new columns to database
    # queries
    create_table = """
        CREATE TABLE IF NOT EXISTS '%s' (
            azimuth FLOAT, 
            zenith FLOAT
            );
        """ % tablename
    
    try:
        con.execute(create_table)

    except:
        print("failed to create table.")

    # load data
    query = """SELECT event_time FROM InIceDSTPulses;"""
    sql_data = read_sql(query,con)
    print('event_times has been read in from database')
    
    # calculate moon directions based on the timestamp
    moon_azimuth = []
    moon_zenith = []
    for t in sql_data["event_time"]:
        moon = moonDirection(t)
        moon_azimuth.append(moon[0])
        moon_zenith.append(moon[1])

    for x in range(len(moon_azimuth)):
        addValues = """INSERT INTO '%s' (azimuth, zenith) VALUES ('%s', '%s');""" % (tablename, moon_azimuth[x], moon_zenith[x])
        con.execute(addValues)
      