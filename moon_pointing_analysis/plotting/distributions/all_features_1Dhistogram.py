""" This script is meant purely as an investigation tool of features in a dataset. """

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import argparse
import numpy as np
import sqlite3 as sql
import pandas as pd
import io
from pandas import read_sql
from helper_functions.plot_params import * 

def create_save_directory(args, table):
    folder_name=args.path_to_db[:-3].split('/')[-1]
    save_dir = args.output + folder_name + "/" + table + "/"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def tune_bins(data,counts,bins,args):
    """attempt to tune bins to find a bin with an acceptable range of zeros"""
    # track previous bins
    list_bin_attempt = [len(counts)]
    list_counts = [counts]
    list_bins = [bins]

    # too many zeros bins; auto was not able to find anything meaningful.
    if zero_ratio(list_counts[-1]) < args.cutoff:
        # try to find a bin that is acceptable.
        while zero_ratio(list_counts[-1]) < 0.8 or list_counts[-1].size > 30:
            list_bin_attempt.append(int(len(bins)/2))
            counts, bins = np.histogram(data, bins=list_bin_attempt[-1])
            list_counts.append(counts)
            list_bins.append(bins)
        return list_counts[-2], list_bins[-2], list_bin_attempt[-2]
    else:
        return list_counts[-1], list_bins[-1], list_bin_attempt[-1]

def outlier(df, feature, sigma):
    # calculate summary statistics
    data_mean, data_std = df[feature].mean(), df[feature].std()
    # identify outliers
    cut_off = data_std * sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    # identify outliers
    temp1 = pd.DataFrame({feature+"_no_outliers" : []})
    temp1 = df[feature][df[feature] < lower]
    temp2 = pd.DataFrame({feature+"_no_outliers" : []})
    temp2 = df[feature][df[feature] > upper]
    df_outliers = pd.concat([temp1, temp2])
    print('Identified outliers: %d' % len(df_outliers))

    # remove outliers
    temp1 = pd.DataFrame({feature+"_no_outliers" : []})
    temp1 = df[feature][df[feature] >= lower]
    temp2 = pd.DataFrame({feature+"_no_outliers" : []})
    temp2 = df[feature][df[feature] <= upper]
    df_no_outliers = pd.concat([temp1, temp2])
    print('Non-outlier observations: %d' % len(df_no_outliers))

    return df_outliers, df_no_outliers

def custom_bin_range(df):
    try:
        bin_range = int(df.max() - df.min())+1
        if bin_range > 100:
            bin_range = 50
    except:
        bin_range = 50
    return bin_range

def zero_ratio(counts):
    try:
        x = counts[counts != 0].size / counts[counts == 0].size
    except ZeroDivisionError: # no zeros
        x = 1
    return x

parser = argparse.ArgumentParser(
    description="plotting all the features as 1-d histograms in a given dataset"
)
parser.add_argument(
    "-db",
    "--database",
    dest="path_to_db",
    type=str,
    help="path to database [str]",
    default="/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/monte_carlo/11069/Level2_nugen_numu_IC86.2012.011069_InIcePulses_InIceDSTPulses_SplitInIcePulses_SplitInIceDSTPulses_SRTInIcePulses.db",
)
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    help="the output path [str]",
    default="/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/plots/distributions/",
)
parser.add_argument(
    "-s",
    "--sigma",
    dest="sigma",
    type=int,
    help="the sigma outlier for cutting data; happen if too many bins in dataset. [int]",
    default=2,
)
parser.add_argument(
    "-c",
    "--cutoff",
    dest="cutoff",
    type=float,
    help="the zero ratio cutoff used in bin tuning; how many zeros instances compared to non zero instances. [int]",
    default=0.8,
)
parser.add_argument(
    "-l",
    "--log",
    dest="log",
    type=int,
    help="if a number exceeds the specified, will scale the graph as log. [int]",
    default=5000,
)
args = parser.parse_args()

ext = (args.path_to_db.split('.')[-1])
if ext == "db":
    # find the name of all tables
    with sql.connect(args.path_to_db) as con:
        table_query = """SELECT name FROM sqlite_master WHERE type='table';"""
        tables = read_sql(table_query, con)

    for table in tables.name:
        with sql.connect(args.path_to_db) as con:
            feature_query = """SELECT * FROM %s; """ % (table)
            feature_data = read_sql(feature_query, con)
        
        # create a folder to save results into
        save_dir = create_save_directory(args, table)

        for feature in feature_data:
            print(f"plotting {feature} in {table}.")
            try:
                # preliminary histogram counts and binning
                counts, bins = np.histogram(feature_data[feature], bins='auto')
                # tuning attempt to reduce excess zeros
                counts, bins, _ = tune_bins(data=feature_data[feature], counts=counts, bins=bins, args=args)

                df_outliers, df_no_outliers = outlier(feature_data, feature, sigma=args.sigma)
                
                fig, ax = plt.subplots(3,1,figsize=(25, 15))

                ax[0].set_title(f"full data")
                ax[0].hist(feature_data[feature], bins=bins)
                ax[0].set_xlabel(feature)
                ax[0].set_ylabel("count")
                if counts.max() > args.log:
                    ax[0].set_yscale("log")

                bin_range = custom_bin_range(df_no_outliers)
                counts1, bins1 = np.histogram(df_no_outliers, bins=bin_range)
                ax[1].set_title(f"outliers under {args.sigma} sigmas")
                ax[1].hist(df_no_outliers, bins=bins1)
                ax[1].set_xlabel(feature)
                ax[1].set_ylabel("count")
                if counts.max() > args.log:
                    ax[1].set_yscale("log")

                bin_range = custom_bin_range(df_outliers)
                counts2, bins2 = np.histogram(df_outliers, bins=bin_range)
                ax[2].set_title(f"outliers over {args.sigma} sigmas")
                ax[2].hist(df_outliers, bins=bins2)
                ax[2].set_xlabel(feature)
                ax[2].set_ylabel("count")
                if counts.max() > args.log:
                    ax[2].set_yscale("log")

                plt.savefig(save_dir+ feature + ".png")

            except Exception as e:
                print(f"plotting {feature} in {table} failed")
                print(e)
                

elif ext == "csv":
    feature_data = pd.read_csv(args.path_to_db)
    extra_index = "Unnamed: 0"
    # if an extra index is imported, drop it.
    if extra_index in feature_data.keys():
        feature_data = feature_data.drop(extra_index, axis=1)

    # create a folder to save results into
    folder_name=args.path_to_db[:-4].split('/')[-1]
    save_dir = args.output + folder_name + "/"
    os.makedirs(save_dir, exist_ok=True)

    for feature in feature_data:
        print(f"plotting {feature}.")
        try:
            x = feature_data[feature]
            bin_range = int((max(x)) - min(x))+1
            if bin_range > 50000 and feature_data[feature].all() > 0:
                bin_range = 25
            counts, bins = np.histogram(x, bins=bin_range)
            
            plt.figure()
            plt.hist(counts, bins=bins)
            plt.title(f"{feature}")
            if counts.mean() > 50000:
                plt.yscale("log")
            plt.savefig(save_dir+ feature + ".png")
            plt.close() # plots are saved in memory, closing to conserve
        except:
            print(f"plotting {feature} failed, data type was of type: {type(feature_data[feature][0])}")

print("Done plotting.")