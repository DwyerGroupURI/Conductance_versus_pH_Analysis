# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:06:42 2021

@author: JRD_U
"""
#     G_vs_pH Data Analysis in Function Format
#
# edit dates:
#%%
# BS 11-23-2021
# BS 11-24-2021
# BS 11-29-2021
# BS 11-30-2021
# BS 12-01-2021
# BS 12-02-2021
# BS 12-03-2021
# BS 12-08-2021
# BS 12-09-2021
# BS 01-10-2022
# BS 01-11-2022
# BS 01-14-2022
# BS 01-19-2022
# BS 01-20-2022
# BS 02-07-2022
# BS 02-09-2022
# BS 02-21-2022

#%% FUNCTIONS
#.....................MODUALS.......................

import numpy as np
import matplotlib.pyplot as plt
import os
# import time
import scipy
# from scipy import optimize
from scipy.stats import linregress
import statistics as st
import glob
import ntpath
# from numpy import diff
import re
import logging
# import sys
from lmfit import Parameters, minimize, fit_report
from matplotlib.offsetbox import AnchoredText

#...................FUNCTIONS........................
"""1. Collect all file names that you want to analyze in a list"""
def list_of_files(path_to_file, common_name, save_file_tag, pH_in_label):
    """
        This function goes to the path (path_to_file) and puts all the file names that contain the (common_title) into a list. Each file name is examined.
        Each name has the pH extracted from the string and the file tag removed from the end. A list of pHs and names w/out file tags is made.
        Finally, a last list is made of the file names with the save_file_tag string added to the end of each name.
        
        FOR pH exctraction to work: the first digit of the pH must be the second group of numbers in the title, with the next two digits of the pH value being the next gouping of numbers.
        (example: BS_p096_GvpH2-81_run1_0808 => pH: 2.81)
        
        1. file_names, list of all the file names that share the (common_title) with file tags removed
        2. save_file_names, list of all the file names with (save_file_tag) added to the end of each
        3. pHs, list of the pHs extracted from the file names
        
        Updated: BS - 01/11/2022
    """
    file_names = []
    save_file_names = []
    pHs = []
    abc_temp = glob.glob( f"{path_to_file}\\*{common_name}*.bin") # grabs all file nomes with the common title
    abc = sorted(abc_temp)
    
    for h in range(len(abc)):
        file_name = ntpath.basename(f"{abc[h]}")
        title = file_name[:-4]
        regex = re.compile(r'\d+')
        numbers = [int(s) for s in regex.findall(title)]
        int_pH = pH_in_label[0]
        dec_pH = pH_in_label[1]
        pH_label1 = numbers[int_pH] # these need to change with the naming scheme - number corisponds to the grouping of numbers
        pH_label2 = numbers[dec_pH]
        pH_label2 = pH_label2/100
        pH = pH_label1 + pH_label2
        pHs.append(pH)
        file_names.append(title)
        save_file_name = [f"{title}_{save_file_tag}"]
        save_file_names.append(save_file_name)
    
    return(file_names, save_file_names, pHs)


"""2. Open data and separate Raw data into raw voltage and raw current"""
def open_bin_data(path_to_file, file_name):
    """
        This function opens a file with the current file name in the path, extracts and separates the (raw_current ) and the (raw_voltage).
        A master index is also made from the total length of the raw data
        1. raw_current, list of the raw current values.
        2. raw_voltage, list of the raw voltage values
        3. x_data_index_master, indexing list for current and voltage (list from 0 to len(raw_current))
        
        Updated: BS - 01/11/2022
    """
    with open(os.path.join(path_to_file, file_name + ".bin"),'r') as current_file:
        data_type = np.dtype('>d') # assign data type: big-endian ordered 64 bit long data format 
        raw_data = np.fromfile(current_file, dtype = data_type) # use numpy module to assign current trace data to [raw_data]
        raw_current = raw_data[0::2] # array of full baseline (current) data with first second and trailing values removed
        raw_voltage = raw_data[1::2]
        x_data_index_master = np.linspace(0,len(raw_current), len(raw_current), endpoint = True, dtype = 'int')
        current_file.close() # closes large data file
        del current_file
    
    return(raw_current, raw_voltage, x_data_index_master)
        

"""3. Read_text_file opens metadata file and reads acquisition rate """
def read_text_file(path_to_file, file_name):
    """
        This function reads in the matadata file that accompanies each file, the LabView program should automatically produce this file.
        The metadata file is needed to input the:
        1. acquisition_rate, single variable
        2. axopatch_gain, single variable
        3. bessel_filter, single variable
        
        Updated: BS - 01/11/2022
    """
    # Opens file and collects acquisition rate
    with open(os.path.join(path_to_file, file_name + ".txt"), 'r') as metadata_file:
        # Read all lines in text file
        MDF = metadata_file.readlines() 
        
        # Acquisition rate
        acquisition_rate = int(MDF[1].split(":")[1].strip()) # .split removes label and isolates number        
        # Gain for axopatch
        axopatch_gain = (MDF[3].split(":")[1].strip())#MDF[3].strip()
        # Bessel filter setting - this is input by the user, so might not be accurate. 
        bessel_filter = (MDF[10].strip())
    metadata_file.close()
    
    return(acquisition_rate, axopatch_gain, bessel_filter)


"""4. Create folders to save the data and plots you make"""
def make_save_folders(path_to_save, analysis_folder_name, plots_folder_name, raw_data_plots_folder_name, noise_plot_folder_name, double_fit_plots_folder_name, fit_vals_plots_folder_name, npy_file_folder_name, lmfit_double_fit_plots_folder_name, lmfit_double_fit_vals_plots_folder_name, lmfit_single_fit_plots_folder_name, lmfit_single_fit_vals_plots_folder_name):
    """
        This function makes folders in (path_to_save) to save the analysis results into. Folder titles are defined with user input strings
        
        Updated: BS - 01/11/2022
    """
    try:
        os.mkdir(os.path.join(path_to_save, analysis_folder_name[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)
    try:
        os.mkdir(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)    
    try:
        os.mkdir(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)
    try:
        os.mkdir(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], noise_plot_folder_name[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)
    try:
        os.mkdir(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], double_fit_plots_folder_name[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)
    try:
        os.mkdir(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_double_fit_plots_folder_name[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)
    try:
        os.mkdir(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_single_fit_plots_folder_name[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)
    try:
        os.mkdir(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], fit_vals_plots_folder_name[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)     
    try:
        os.mkdir(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_single_fit_vals_plots_folder_name[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)    
    try:
        os.mkdir(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_double_fit_vals_plots_folder_name[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)
    try:
        os.mkdir(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)

    return(print("save locations created"))


"""5. Logging Error Messages"""
def create_error_log_file(analysis_title, pHs, save_path, save_file_folder_name, logger_name):
    """
        This function is an attempt to log the errors that occur in the program. There should be no errors but if there are, this function collects them
        and saves them to a file.
        
        Updated: BS - 01/11/2022
    """
    # logger_name = analysis_title
    # logger_name += f"{analysis_title}"
    
    level = logging.DEBUG
    log_format= ('%(asctime)s %(levelname)s - %(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    logging.basicConfig(filename=f'{save_path}\\{save_file_folder_name[0]}\\{logger_name}.log', format= log_format, datefmt='%H:%M:%S')
    
    logging.captureWarnings(True)
    
    # return logger
    logger = logging.getLogger()
    return (print("we are logging errors"), logger)


"""5.5 Plotting raw data"""
def plotting_raw_data(raw_current, raw_voltage, x_data_index_master, acquisition_rate, analysis_title, path_to_save, analysis_folder_name, plots_folder_name, raw_data_plots_folder_name):
    
    # raw data from the start in times 30 seconds to 160 seconds
    
    # start overview 
    
    start_1 = 300000
    stop_1 = 1600000
    
    plot_1_ydata = raw_current[start_1:stop_1]
    xdata_1 = x_data_index_master[start_1:stop_1]
    plot_1_xdata = [element / acquisition_rate for element in xdata_1]
    plot_1_V_ydata = raw_voltage[start_1:stop_1]
    
    avg_cur_1 = round(sum((raw_current[start_1:stop_1])/(stop_1-start_1)),2)
    stdev_1 = round(st.stdev((raw_current[start_1:stop_1])),2)
    avg_V_1 = round(sum((raw_voltage[start_1:stop_1])/(stop_1-start_1)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_1_xdata, plot_1_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_1+500
    min_y_val = avg_cur_1-500
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    # anchored_text = AnchoredText(f" Average Current = {avg_cur_1} pA\n Current STDEV = {stdev_1} pA\n Applied Voltage = {avg_V_1} mV", loc=2)
    # ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_1_xdata, plot_1_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")

    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_start_overview"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # start zero
    
    start_2 = 340000
    stop_2 = 350000
    
    plot_2_ydata = raw_current[start_2:stop_2]
    xdata_2 = x_data_index_master[start_2:stop_2]
    plot_2_xdata = [element / acquisition_rate for element in xdata_2]
    plot_2_V_ydata = raw_voltage[start_2:stop_2]
    
    avg_cur_2 = round(sum((raw_current[start_2:stop_2])/(stop_2-start_2)),2)
    stdev_2 = round(st.stdev((raw_current[start_2:stop_2])),2)
    avg_V_2 = round(sum((raw_voltage[start_2:stop_2])/(stop_2-start_2)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_2_xdata, plot_2_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_2+200
    min_y_val = avg_cur_2-200
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    anchored_text = AnchoredText(f" Average Current = {avg_cur_2} pA\n Current STDEV = {stdev_2} pA\n Applied Voltage = {avg_V_2} mV", loc=2)
    ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_2_xdata, plot_2_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_start_zero"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # start pos
    
    start_21 = 370000
    stop_21 = 380000

    plot_21_ydata = raw_current[start_21:stop_21]
    xdata_21 = x_data_index_master[start_21:stop_21]
    plot_21_xdata = [element / acquisition_rate for element in xdata_21]
    plot_21_V_ydata = raw_voltage[start_21:stop_21]
    
    avg_cur_21 = round(sum((raw_current[start_21:stop_21])/(stop_21-start_21)),2)
    stdev_21 = round(st.stdev((raw_current[start_21:stop_21])),2)
    avg_V_21 = round(sum((raw_voltage[start_21:stop_21])/(stop_21-start_21)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_21_xdata, plot_21_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_21+100
    min_y_val = avg_cur_21-100
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    anchored_text = AnchoredText(f" Average Current = {avg_cur_21} pA\n Current STDEV = {stdev_21} pA\n Applied Voltage = {avg_V_21} mV", loc=2)
    ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_21_xdata, plot_21_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_start_pos"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # start neg
    
    start_3 = 510000
    stop_3 = 520000
    
    plot_3_ydata = raw_current[start_3:stop_3]
    xdata_3 = x_data_index_master[start_3:stop_3]
    plot_3_xdata = [element / acquisition_rate for element in xdata_3]
    plot_3_V_ydata = raw_voltage[start_3:stop_3]
    
    avg_cur_3 = round(sum((raw_current[start_3:stop_3])/(stop_3-start_3)),2)
    stdev_3 = round(st.stdev((raw_current[start_3:stop_3])),2)
    avg_V_3 = round(sum((raw_voltage[start_3:stop_3])/(stop_3-start_3)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_3_xdata, plot_3_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_3+100
    min_y_val = avg_cur_3-100
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    anchored_text = AnchoredText(f" Average Current = {avg_cur_3} pA\n Current STDEV = {stdev_3} pA\n Applied Voltage = {avg_V_3} mV", loc=2)
    ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_3_xdata, plot_3_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_start_neg"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # middle overview
    
    start_1 = 8000000
    stop_1 = 9600000
    
    plot_1_ydata = raw_current[start_1:stop_1]
    xdata_1 = x_data_index_master[start_1:stop_1]
    plot_1_xdata = [element / acquisition_rate for element in xdata_1]
    plot_1_V_ydata = raw_voltage[start_1:stop_1]
    
    avg_cur_1 = round(sum((raw_current[start_1:stop_1])/(stop_1-start_1)),2)
    stdev_1 = round(st.stdev((raw_current[start_1:stop_1])),2)
    avg_V_1 = round(sum((raw_voltage[start_1:stop_1])/(stop_1-start_1)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_1_xdata, plot_1_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_1+500
    min_y_val = avg_cur_1-500
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    # anchored_text = AnchoredText(f" Average Current = {avg_cur_1} pA\n Current STDEV = {stdev_1} pA\n Applied Voltage = {avg_V_1} mV", loc=2)
    # ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_1_xdata, plot_1_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_middle_overview"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # middle zero
    
    start_2 = 8220000
    stop_2 = 8230000
    
    plot_2_ydata = raw_current[start_2:stop_2]
    xdata_2 = x_data_index_master[start_2:stop_2]
    plot_2_xdata = [element / acquisition_rate for element in xdata_2]
    plot_2_V_ydata = raw_voltage[start_2:stop_2]
    
    avg_cur_2 = round(sum((raw_current[start_2:stop_2])/(stop_2-start_2)),2)
    stdev_2 = round(st.stdev((raw_current[start_2:stop_2])),2)
    avg_V_2 = round(sum((raw_voltage[start_2:stop_2])/(stop_2-start_2)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_2_xdata, plot_2_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_2+100
    min_y_val = avg_cur_2-100
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    anchored_text = AnchoredText(f" Average Current = {avg_cur_2} pA\n Current STDEV = {stdev_2} pA\n Applied Voltage = {avg_V_2} mV", loc=2)
    ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_2_xdata, plot_2_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_middle_zero"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # middle pos
    
    start_21 = 8170000
    stop_21 = 8180000

    plot_21_ydata = raw_current[start_21:stop_21]
    xdata_21 = x_data_index_master[start_21:stop_21]
    plot_21_xdata = [element / acquisition_rate for element in xdata_21]
    plot_21_V_ydata = raw_voltage[start_21:stop_21]
    
    avg_cur_21 = round(sum((raw_current[start_21:stop_21])/(stop_21-start_21)),2)
    stdev_21 = round(st.stdev((raw_current[start_21:stop_21])),2)
    avg_V_21 = round(sum((raw_voltage[start_21:stop_21])/(stop_21-start_21)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_21_xdata, plot_21_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_21+100
    min_y_val = avg_cur_21-100
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    anchored_text = AnchoredText(f" Average Current = {avg_cur_21} pA\n Current STDEV = {stdev_21} pA\n Applied Voltage = {avg_V_21} mV", loc=2)
    ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_21_xdata, plot_21_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_middle_pos"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # middle neg
    
    start_3 = 8300000
    stop_3 = 8310000
    
    plot_3_ydata = raw_current[start_3:stop_3]
    xdata_3 = x_data_index_master[start_3:stop_3]
    plot_3_xdata = [element / acquisition_rate for element in xdata_3]
    plot_3_V_ydata = raw_voltage[start_3:stop_3]
    
    avg_cur_3 = round(sum((raw_current[start_3:stop_3])/(stop_3-start_3)),2)
    stdev_3 = round(st.stdev((raw_current[start_3:stop_3])),2)
    avg_V_3 = round(sum((raw_voltage[start_3:stop_3])/(stop_3-start_3)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_3_xdata, plot_3_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_3+100
    min_y_val = avg_cur_3-100
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    anchored_text = AnchoredText(f" Average Current = {avg_cur_3} pA\n Current STDEV = {stdev_3} pA\n Applied Voltage = {avg_V_3} mV", loc=2)
    ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_3_xdata, plot_3_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_middle_neg"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # end overview
    
    start_1 = 16000000
    stop_1 = 17600000
    
    plot_1_ydata = raw_current[start_1:stop_1]
    xdata_1 = x_data_index_master[start_1:stop_1]
    plot_1_xdata = [element / acquisition_rate for element in xdata_1]
    plot_1_V_ydata = raw_voltage[start_1:stop_1]
    
    avg_cur_1 = round(sum((raw_current[start_1:stop_1])/(stop_1-start_1)),2)
    stdev_1 = round(st.stdev((raw_current[start_1:stop_1])),2)
    avg_V_1 = round(sum((raw_voltage[start_1:stop_1])/(stop_1-start_1)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_1_xdata, plot_1_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_1+500
    min_y_val = avg_cur_1-500
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    # anchored_text = AnchoredText(f" Average Current = {avg_cur_1} pA\n Current STDEV = {stdev_1} pA\n Applied Voltage = {avg_V_1} mV", loc=2)
    # ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_1_xdata, plot_1_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_end_overview"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # end zero
    
    start_2 = 16040000
    stop_2 = 16050000
    
    plot_2_ydata = raw_current[start_2:stop_2]
    xdata_2 = x_data_index_master[start_2:stop_2]
    plot_2_xdata = [element / acquisition_rate for element in xdata_2]
    plot_2_V_ydata = raw_voltage[start_2:stop_2]
    
    avg_cur_2 = round(sum((raw_current[start_2:stop_2])/(stop_2-start_2)),2)
    stdev_2 = round(st.stdev((raw_current[start_2:stop_2])),2)
    avg_V_2 = round(sum((raw_voltage[start_2:stop_2])/(stop_2-start_2)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_2_xdata, plot_2_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_2+100
    min_y_val = avg_cur_2-100
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    anchored_text = AnchoredText(f" Average Current = {avg_cur_2} pA\n Current STDEV = {stdev_2} pA\n Applied Voltage = {avg_V_2} mV", loc=2)
    ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_2_xdata, plot_2_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_end_zero"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # end pos
    
    start_21 = 16240000
    stop_21 = 16250000

    plot_21_ydata = raw_current[start_21:stop_21]
    xdata_21 = x_data_index_master[start_21:stop_21]
    plot_21_xdata = [element / acquisition_rate for element in xdata_21]
    plot_21_V_ydata = raw_voltage[start_21:stop_21]
    
    avg_cur_21 = round(sum((raw_current[start_21:stop_21])/(stop_21-start_21)),2)
    stdev_21 = round(st.stdev((raw_current[start_21:stop_21])),2)
    avg_V_21 = round(sum((raw_voltage[start_21:stop_21])/(stop_21-start_21)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_21_xdata, plot_21_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_21+100
    min_y_val = avg_cur_21-100
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    anchored_text = AnchoredText(f" Average Current = {avg_cur_21} pA\n Current STDEV = {stdev_21} pA\n Applied Voltage = {avg_V_21} mV", loc=2)
    ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_21_xdata, plot_21_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_end_pos"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # end neg
    
    start_3 = 16110000
    stop_3 = 16120000
    
    plot_3_ydata = raw_current[start_3:stop_3]
    xdata_3 = x_data_index_master[start_3:stop_3]
    plot_3_xdata = [element / acquisition_rate for element in xdata_3]
    plot_3_V_ydata = raw_voltage[start_3:stop_3]
    
    avg_cur_3 = round(sum((raw_current[start_3:stop_3])/(stop_3-start_3)),2)
    stdev_3 = round(st.stdev((raw_current[start_3:stop_3])),2)
    avg_V_3 = round(sum((raw_voltage[start_3:stop_3])/(stop_3-start_3)),2)
    
    fig,ax = plt.subplots(figsize=(8,4))
    
    ax.scatter(plot_3_xdata, plot_3_ydata, s=5, alpha=0.75, color = "b")
    
    max_y_val = avg_cur_3+100
    min_y_val = avg_cur_3-100
    
    ax.set_ylim(min_y_val,max_y_val)
    
    ax.set_title(f" {analysis_title} ", size=12, weight='bold') #Title
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Current (pA)')
    
    anchored_text = AnchoredText(f" Average Current = {avg_cur_3} pA\n Current STDEV = {stdev_3} pA\n Applied Voltage = {avg_V_3} mV", loc=2)
    ax.add_artist(anchored_text)
    
    ax2=ax.twinx()
    
    ax2.scatter(plot_3_xdata, plot_3_V_ydata, s=2, alpha=0.75, color = "r")
    
    ax2.set_ylim(-100,100)
    
    ax2.set_ylabel('Voltage (mV)', color = "r")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "raw_data_end_neg"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    return(print(f"done plotting raw data for {analysis_title}"))


"""5.75 Plotting All Important Raw Data on a Single Subplot"""
def plot_all_raw_data_on_one_subplot(raw_current, raw_voltage, x_data_index_master, acquisition_rate, analysis_title, path_to_save, analysis_folder_name, plots_folder_name, raw_data_plots_folder_name):
    
    # raw data from the start in times 30 seconds to 160 seconds
    
    # start overview 
    try:
        start_1 = 300000
        stop_1 = 1600000
        
        plot_1_ydata = raw_current[start_1:stop_1]
        xdata_1 = x_data_index_master[start_1:stop_1]
        plot_1_xdata = [element / acquisition_rate for element in xdata_1]
        plot_1_V_ydata = raw_voltage[start_1:stop_1]
        
        avg_cur_1 = round(sum((raw_current[start_1:stop_1])/(stop_1-start_1)),2)
        
        fig, ax = plt.subplots(4, 3, figsize = (22,12))
        
        ax[0,0].scatter(plot_1_xdata, plot_1_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_1+500
        min_y_val = avg_cur_1-500
        
        ax[0,0].set_ylim(min_y_val,max_y_val)
        
        ax[0,0].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[0,0].set_xlabel('Seconds')
        ax[0,0].set_ylabel('Current (pA)')
        
        ax2=ax[0,0].twinx()
        
        ax2.scatter(plot_1_xdata, plot_1_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax2.set_ylim(-100,100)
        
        ax2.set_ylabel('Voltage (mV)', color = "r")
        
        # start zero
        
        start_2 = 340000
        stop_2 = 350000
        
        plot_2_ydata = raw_current[start_2:stop_2]
        xdata_2 = x_data_index_master[start_2:stop_2]
        plot_2_xdata = [element / acquisition_rate for element in xdata_2]
        plot_2_V_ydata = raw_voltage[start_2:stop_2]
        
        avg_cur_2 = round(sum((raw_current[start_2:stop_2])/(stop_2-start_2)),2)
        stdev_2 = round(st.stdev((raw_current[start_2:stop_2])),2)
        avg_V_2 = round(sum((raw_voltage[start_2:stop_2])/(stop_2-start_2)),2)
        
        ax[1,0].scatter(plot_2_xdata, plot_2_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_2+200
        min_y_val = avg_cur_2-200
        
        ax[1,0].set_ylim(min_y_val,max_y_val)
        
        ax[1,0].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[1,0].set_xlabel('Seconds')
        ax[1,0].set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_2} pA\n Current STDEV = {stdev_2} pA\n Applied Voltage = {avg_V_2} mV", loc=2)
        ax[1,0].add_artist(anchored_text)
        
        ax3=ax[1,0].twinx()
        
        ax3.scatter(plot_2_xdata, plot_2_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax3.set_ylim(-100,100)
        
        ax3.set_ylabel('Voltage (mV)', color = "r")
        
        # start pos
        
        start_21 = 370000
        stop_21 = 380000
    
        plot_21_ydata = raw_current[start_21:stop_21]
        xdata_21 = x_data_index_master[start_21:stop_21]
        plot_21_xdata = [element / acquisition_rate for element in xdata_21]
        plot_21_V_ydata = raw_voltage[start_21:stop_21]
        
        avg_cur_21 = round(sum((raw_current[start_21:stop_21])/(stop_21-start_21)),2)
        stdev_21 = round(st.stdev((raw_current[start_21:stop_21])),2)
        avg_V_21 = round(sum((raw_voltage[start_21:stop_21])/(stop_21-start_21)),2)
            
        ax[2,0].scatter(plot_21_xdata, plot_21_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_21+100
        min_y_val = avg_cur_21-100
        
        ax[2,0].set_ylim(min_y_val,max_y_val)
        
        ax[2,0].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[2,0].set_xlabel('Seconds')
        ax[2,0].set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_21} pA\n Current STDEV = {stdev_21} pA\n Applied Voltage = {avg_V_21} mV", loc=2)
        ax[2,0].add_artist(anchored_text)
        
        ax4=ax[2,0].twinx()
        
        ax4.scatter(plot_21_xdata, plot_21_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax4.set_ylim(-100,100)
        
        ax4.set_ylabel('Voltage (mV)', color = "r")
        
        # start neg
        
        start_3 = 510000
        stop_3 = 520000
        
        plot_3_ydata = raw_current[start_3:stop_3]
        xdata_3 = x_data_index_master[start_3:stop_3]
        plot_3_xdata = [element / acquisition_rate for element in xdata_3]
        plot_3_V_ydata = raw_voltage[start_3:stop_3]
        
        avg_cur_3 = round(sum((raw_current[start_3:stop_3])/(stop_3-start_3)),2)
        stdev_3 = round(st.stdev((raw_current[start_3:stop_3])),2)
        avg_V_3 = round(sum((raw_voltage[start_3:stop_3])/(stop_3-start_3)),2)
        
        ax[3,0].scatter(plot_3_xdata, plot_3_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_3+100
        min_y_val = avg_cur_3-100
        
        ax[3,0].set_ylim(min_y_val,max_y_val)
        
        ax[3,0].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[3,0].set_xlabel('Seconds')
        ax[3,0].set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_3} pA\n Current STDEV = {stdev_3} pA\n Applied Voltage = {avg_V_3} mV", loc=2)
        ax[3,0].add_artist(anchored_text)
        
        ax5=ax[3,0].twinx()
        
        ax5.scatter(plot_3_xdata, plot_3_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax5.set_ylim(-100,100)
        
        ax5.set_ylabel('Voltage (mV)', color = "r")
        
        # middle overview
        
        start_1 = 8000000
        stop_1 = 9600000
        
        plot_1_ydata = raw_current[start_1:stop_1]
        xdata_1 = x_data_index_master[start_1:stop_1]
        plot_1_xdata = [element / acquisition_rate for element in xdata_1]
        plot_1_V_ydata = raw_voltage[start_1:stop_1]
        
        avg_cur_1 = round(sum((raw_current[start_1:stop_1])/(stop_1-start_1)),2)
        
        ax[0,1].scatter(plot_1_xdata, plot_1_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_1+500
        min_y_val = avg_cur_1-500
        
        ax[0,1].set_ylim(min_y_val,max_y_val)
        
        ax[0,1].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[0,1].set_xlabel('Seconds')
        ax[0,1].set_ylabel('Current (pA)')
        
        # anchored_text = AnchoredText(f" Average Current = {avg_cur_1} pA\n Current STDEV = {stdev_1} pA\n Applied Voltage = {avg_V_1} mV", loc=2)
        # ax.add_artist(anchored_text)
        
        ax6=ax[0,1].twinx()
        
        ax6.scatter(plot_1_xdata, plot_1_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax6.set_ylim(-100,100)
        
        ax6.set_ylabel('Voltage (mV)', color = "r")
        
        # middle zero
        
        start_2 = 8220000
        stop_2 = 8230000
        
        plot_2_ydata = raw_current[start_2:stop_2]
        xdata_2 = x_data_index_master[start_2:stop_2]
        plot_2_xdata = [element / acquisition_rate for element in xdata_2]
        plot_2_V_ydata = raw_voltage[start_2:stop_2]
        
        avg_cur_2 = round(sum((raw_current[start_2:stop_2])/(stop_2-start_2)),2)
        stdev_2 = round(st.stdev((raw_current[start_2:stop_2])),2)
        avg_V_2 = round(sum((raw_voltage[start_2:stop_2])/(stop_2-start_2)),2)
            
        ax[1,1].scatter(plot_2_xdata, plot_2_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_2+100
        min_y_val = avg_cur_2-100
        
        ax[1,1].set_ylim(min_y_val,max_y_val)
        
        ax[1,1].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[1,1].set_xlabel('Seconds')
        ax[1,1].set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_2} pA\n Current STDEV = {stdev_2} pA\n Applied Voltage = {avg_V_2} mV", loc=2)
        ax[1,1].add_artist(anchored_text)
        
        ax7=ax[1,1].twinx()
        
        ax7.scatter(plot_2_xdata, plot_2_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax7.set_ylim(-100,100)
        
        ax7.set_ylabel('Voltage (mV)', color = "r")
        
        # middle pos
        
        start_21 = 8170000
        stop_21 = 8180000
    
        plot_21_ydata = raw_current[start_21:stop_21]
        xdata_21 = x_data_index_master[start_21:stop_21]
        plot_21_xdata = [element / acquisition_rate for element in xdata_21]
        plot_21_V_ydata = raw_voltage[start_21:stop_21]
        
        avg_cur_21 = round(sum((raw_current[start_21:stop_21])/(stop_21-start_21)),2)
        stdev_21 = round(st.stdev((raw_current[start_21:stop_21])),2)
        avg_V_21 = round(sum((raw_voltage[start_21:stop_21])/(stop_21-start_21)),2)
            
        ax[2,1].scatter(plot_21_xdata, plot_21_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_21+100
        min_y_val = avg_cur_21-100
        
        ax[2,1].set_ylim(min_y_val,max_y_val)
        
        ax[2,1].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[2,1].set_xlabel('Seconds')
        ax[2,1].set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_21} pA\n Current STDEV = {stdev_21} pA\n Applied Voltage = {avg_V_21} mV", loc=2)
        ax[2,1].add_artist(anchored_text)
        
        ax8=ax[2,1].twinx()
        
        ax8.scatter(plot_21_xdata, plot_21_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax8.set_ylim(-100,100)
        
        ax8.set_ylabel('Voltage (mV)', color = "r")
        
        # middle neg
        
        start_3 = 8300000
        stop_3 = 8310000
        
        plot_3_ydata = raw_current[start_3:stop_3]
        xdata_3 = x_data_index_master[start_3:stop_3]
        plot_3_xdata = [element / acquisition_rate for element in xdata_3]
        plot_3_V_ydata = raw_voltage[start_3:stop_3]
        
        avg_cur_3 = round(sum((raw_current[start_3:stop_3])/(stop_3-start_3)),2)
        stdev_3 = round(st.stdev((raw_current[start_3:stop_3])),2)
        avg_V_3 = round(sum((raw_voltage[start_3:stop_3])/(stop_3-start_3)),2)
            
        ax[3,1].scatter(plot_3_xdata, plot_3_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_3+100
        min_y_val = avg_cur_3-100
        
        ax[3,1].set_ylim(min_y_val,max_y_val)
        
        ax[3,1].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[3,1].set_xlabel('Seconds')
        ax[3,1].set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_3} pA\n Current STDEV = {stdev_3} pA\n Applied Voltage = {avg_V_3} mV", loc=2)
        ax[3,1].add_artist(anchored_text)
        
        ax9=ax[3,1].twinx()
        
        ax9.scatter(plot_3_xdata, plot_3_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax9.set_ylim(-100,100)
        
        ax9.set_ylabel('Voltage (mV)', color = "r")
        
        # end overview
        
        start_1 = 16000000
        stop_1 = 17600000
        
        plot_1_ydata = raw_current[start_1:stop_1]
        xdata_1 = x_data_index_master[start_1:stop_1]
        plot_1_xdata = [element / acquisition_rate for element in xdata_1]
        plot_1_V_ydata = raw_voltage[start_1:stop_1]
        
        avg_cur_1 = round(sum((raw_current[start_1:stop_1])/(stop_1-start_1)),2)
        
        ax[0,2].scatter(plot_1_xdata, plot_1_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_1+500
        min_y_val = avg_cur_1-500
        
        ax[0,2].set_ylim(min_y_val,max_y_val)
        
        ax[0,2].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[0,2].set_xlabel('Seconds')
        ax[0,2].set_ylabel('Current (pA)')
        
        ax10=ax[0,2].twinx()
        
        ax10.scatter(plot_1_xdata, plot_1_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax10.set_ylim(-100,100)
        
        ax10.set_ylabel('Voltage (mV)', color = "r")
        
        # end zero
        
        start_2 = 16040000
        stop_2 = 16050000
        
        plot_2_ydata = raw_current[start_2:stop_2]
        xdata_2 = x_data_index_master[start_2:stop_2]
        plot_2_xdata = [element / acquisition_rate for element in xdata_2]
        plot_2_V_ydata = raw_voltage[start_2:stop_2]
        
        avg_cur_2 = round(sum((raw_current[start_2:stop_2])/(stop_2-start_2)),2)
        stdev_2 = round(st.stdev((raw_current[start_2:stop_2])),2)
        avg_V_2 = round(sum((raw_voltage[start_2:stop_2])/(stop_2-start_2)),2)
            
        ax[1,2].scatter(plot_2_xdata, plot_2_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_2+100
        min_y_val = avg_cur_2-100
        
        ax[1,2].set_ylim(min_y_val,max_y_val)
        
        ax[1,2].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[1,2].set_xlabel('Seconds')
        ax[1,2].set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_2} pA\n Current STDEV = {stdev_2} pA\n Applied Voltage = {avg_V_2} mV", loc=2)
        ax[1,2].add_artist(anchored_text)
        
        ax11=ax[1,2].twinx()
        
        ax11.scatter(plot_2_xdata, plot_2_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax11.set_ylim(-100,100)
        
        ax11.set_ylabel('Voltage (mV)', color = "r")
        
        # end pos
        
        start_21 = 16240000
        stop_21 = 16250000
    
        plot_21_ydata = raw_current[start_21:stop_21]
        xdata_21 = x_data_index_master[start_21:stop_21]
        plot_21_xdata = [element / acquisition_rate for element in xdata_21]
        plot_21_V_ydata = raw_voltage[start_21:stop_21]
        
        avg_cur_21 = round(sum((raw_current[start_21:stop_21])/(stop_21-start_21)),2)
        stdev_21 = round(st.stdev((raw_current[start_21:stop_21])),2)
        avg_V_21 = round(sum((raw_voltage[start_21:stop_21])/(stop_21-start_21)),2)
            
        ax[2,2].scatter(plot_21_xdata, plot_21_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_21+100
        min_y_val = avg_cur_21-100
        
        ax[2,2].set_ylim(min_y_val,max_y_val)
        
        ax[2,2].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[2,2].set_xlabel('Seconds')
        ax[2,2].set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_21} pA\n Current STDEV = {stdev_21} pA\n Applied Voltage = {avg_V_21} mV", loc=2)
        ax[2,2].add_artist(anchored_text)
        
        ax12=ax[2,2].twinx()
        
        ax12.scatter(plot_21_xdata, plot_21_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax12.set_ylim(-100,100)
        
        ax12.set_ylabel('Voltage (mV)', color = "r")
        
        # end neg
        
        start_3 = 16110000
        stop_3 = 16120000
        
        plot_3_ydata = raw_current[start_3:stop_3]
        xdata_3 = x_data_index_master[start_3:stop_3]
        plot_3_xdata = [element / acquisition_rate for element in xdata_3]
        plot_3_V_ydata = raw_voltage[start_3:stop_3]
        
        avg_cur_3 = round(sum((raw_current[start_3:stop_3])/(stop_3-start_3)),2)
        stdev_3 = round(st.stdev((raw_current[start_3:stop_3])),2)
        avg_V_3 = round(sum((raw_voltage[start_3:stop_3])/(stop_3-start_3)),2)
            
        ax[3,2].scatter(plot_3_xdata, plot_3_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_3+100
        min_y_val = avg_cur_3-100
        
        ax[3,2].set_ylim(min_y_val,max_y_val)
        
        ax[3,2].set_title(f" {analysis_title} ", size=12, weight='bold') #Title
        ax[3,2].set_xlabel('Seconds')
        ax[3,2].set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_3} pA\n Current STDEV = {stdev_3} pA\n Applied Voltage = {avg_V_3} mV", loc=2)
        ax[3,2].add_artist(anchored_text)
        
        ax13=ax[3,2].twinx()
        
        ax13.scatter(plot_3_xdata, plot_3_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax13.set_ylim(-100,100)
        
        ax13.set_ylabel('Voltage (mV)', color = "r")
    
    except:
        print("error")
        
    fig.tight_layout()
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], "ALL_raw_data"), dpi = PLOT_DPI)#, bbox_inches = 'tight')
    plt.close()
    
    return()


"""6. Index when voltages switch"""
def voltage_switch_index(raw_voltage, voltage_switch_threshold, acquisition_rate, dp_after_spike):
    """
        This function scans through the voltage watching for a change which would indicate a capacitance spike. When there is a change the function 
        saves the index to (current_switch_index) and then jumps a user defined amount of data (dp_after_spike) and comtimues scanning for voltage 
        changes/cap spike locations.
        
        The amount of voltage change is a user defined value (voltage_switch_threshold).
        
        The location for the initial voltage value is set here (initial_v)
        (initial_v) is then updated after every switch detection, the new voltage value is taken at the (dp_after_spike) location.
        
        1. current_switch_index, list of the capacitance spike index's
    
        Updated: BS - 01/11/2022
    """
    initial_v = int(np.mean(raw_voltage[500:1000]))
    
    current_switch_index = []
    loop = len(raw_voltage)
    i = 0
    while i < loop:
        if raw_voltage[i] > initial_v + voltage_switch_threshold or raw_voltage[i] < initial_v - voltage_switch_threshold:
            current_switch_index.append(i)
            initial_v = np.mean(raw_voltage[i+(dp_after_spike):i+(dp_after_spike)+500]) #sets the initial to two seconds into the current 
            i = i + (dp_after_spike) # skips ahead time_after_spike seconds
        else: i += 1
    
    return(current_switch_index)


""""7. Parse current data into its oscilating states with the current switch index"""
def parse_current_from_v_switchs(current_switch_index, raw_current, raw_voltage, acquisition_rate, cap_data_backstep, data_per_cap_spike, voltage_switch_threshold, cond_datapoints):
    """
        This function takes the (current_switch_index) and the (raw_current) list and parses up the (raw_current) data for each capacitance spike from the (current_switch_index)
        The function organizes the positive, negative, and zero applied potential capacitance spikes into their respective tuples, along with index tuples
        
        (cap_data_backstep) is a user defined value that sets the number of datapoints to backstep from the voltage switch so that the start of the capacitance spikes are fully collected.
        (data_per_cap_spike) is a user defined value that sets the amount of data to hold for each capacitance spike, aka the amount of data that will be fit later on
        (voltage_switch_threshold) is the same user defined value from funtion #6, and provides the threshold value for sorting
        
        * might want to make (voltage_sorting) location a user defined value *
        
        1. cap_spikes, list of all capacitance spike current values
        2. pos_caps, list of all capacitance spike current values from positive applied voltages
        3. pos_caps_index, list of all capacitance spike index values from positive applied voltages
        4. neg_caps, list of all capacitance spike current values from negative applied voltages
        5. neg_caps_index, list of all capacitance spike index values from negative applied voltages
        6. zero_caps, list of all capacitance spike discharge current values from zero applied voltages
        7. zero_caps_index, list of all capacitance spike discharge index values from zero applied voltages
        
        Updated: BS - 01/11/2022
        Updated: BS - 02/07/2022 -- added an if statement to break the loop if the (last_index) goes past the length of the (raw_data)
    """
    cap_spikes = []
    pos_caps = []
    pos_caps_index = []
    zero_caps = []
    zero_caps_index = []
    neg_caps = []
    neg_caps_index = []
    pos_cond_current = []
    pos_cond_current_index = []
    neg_cond_current = []
    neg_cond_current_index =  []
    zero_cond_current = []
    zero_cond_current_index = []
    pos_cond_master = []
    neg_cond_master = []
    zero_cond_master = []
    all_cond_index = []
    
    # global first_index
    # global last_index
    
    for l in range(len(current_switch_index)): # loop through the switch index to separate oscilating current
        first_int_index = current_switch_index[l] # 
        first_index = int(first_int_index - cap_data_backstep) # index includes 100 datapints bepore voltage indexed current spike
        last_index = int(first_int_index + data_per_cap_spike) # last index is one second after voltage switch
        if last_index > len(raw_voltage): break
        current_values = raw_current[first_index:last_index]
        current_values = np.abs(current_values) # this might not work
        cond_current_values = raw_current[last_index:last_index + cond_datapoints]
        cond_mean = sum(cond_current_values)/len(cond_current_values)
        cap_spikes.append(current_values)
        voltage_sorting = first_int_index + 10        # grabbing a stable voltage (THIS MAY NEED TO BE A USER DEFINED VARIEBLE!?!?!)
        voltage_temp = raw_voltage[voltage_sorting:voltage_sorting + 1000]
        voltage = sum(voltage_temp)/len(voltage_temp)
        if raw_voltage[voltage_sorting] > voltage_switch_threshold:
            current_values = np.abs(current_values)
            pos_caps.append(current_values)
            pos_caps_index.append(first_int_index)
            pos_cond_current.append(cond_current_values)
            pos_cond_current_index.append(last_index)
            pos_cond_master.append((cond_mean, cond_current_values, last_index, voltage))
        if raw_voltage[voltage_sorting] < -voltage_switch_threshold:
            current_values = np.abs(current_values)
            neg_caps.append(current_values)
            neg_caps_index.append(first_int_index)
            neg_cond_current.append(cond_current_values)
            neg_cond_current_index.append(last_index)
            neg_cond_master.append((cond_mean, cond_current_values, last_index, voltage))
        if raw_voltage[voltage_sorting] > -voltage_switch_threshold and raw_voltage[voltage_sorting]< voltage_switch_threshold:
            current_values = np.abs(current_values)
            zero_caps.append(current_values)
            zero_caps_index.append(first_int_index)
            zero_cond_current.append(cond_current_values)
            zero_cond_current_index.append(last_index)
            zero_cond_master.append((cond_mean, cond_current_values, last_index, voltage))
        all_cond_index.append(last_index)
    
    return(cap_spikes, pos_caps, pos_caps_index, neg_caps, neg_caps_index, zero_caps, zero_caps_index, all_cond_index, pos_cond_master, neg_cond_master, zero_cond_master)


"""7.25 Plotting All Applied Voltage and Current"""
def plotting_all_applied_voltage_and_current(pos_caps_index, neg_caps_index, zero_caps_index, raw_current, raw_voltage, acquisition_rate, pos_time, neg_time, zero_time, analysis_title, path_to_save, analysis_folder_name, plots_folder_name, raw_data_plots_folder_name):
    
    
    for i in range(len(zero_caps_index)):
        
        start_1 = zero_caps_index[i]
        stop_1 = start_1 + (zero_time*acquisition_rate)
        
        plot_1_ydata = raw_current[start_1:stop_1]
        xdata_1 = x_data_index_master[start_1:stop_1]
        plot_1_xdata = [element / acquisition_rate for element in xdata_1]
        plot_1_V_ydata = raw_voltage[start_1:stop_1]
        time = int(plot_1_xdata[0])
        
        avg_cur_1 = round(sum((raw_current[(start_1+acquisition_rate):stop_1])/(stop_1-start_1)),2)
        stdev_1 = round(st.stdev((raw_current[(start_1+acquisition_rate):stop_1])),2)
        avg_V_1 = round(sum((raw_voltage[(start_1+acquisition_rate):stop_1])/(stop_1-(start_1+acquisition_rate))),2)
    
        fig,ax = plt.subplots(figsize=(8,4))
        
        ax.scatter(plot_1_xdata, plot_1_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_1+500
        min_y_val = avg_cur_1-500
        
        ax.set_ylim(min_y_val,max_y_val)
        
        ax.set_title(f" Zero Voltage at {time}sec\n {analysis_title} ", size=12, weight='bold') #Title
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_1} pA\n Current STDEV = {stdev_1} pA\n Applied Voltage = {avg_V_1} mV", loc=9)
        ax.add_artist(anchored_text)
        
        ax2=ax.twinx()
        
        ax2.scatter(plot_1_xdata, plot_1_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax2.set_ylim(-100,100)
        
        ax2.set_ylabel('Voltage (mV)', color = "r")
    
        
        plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], f"zero_raw_data_at_{time}_seconds"), dpi = PLOT_DPI, bbox_inches = 'tight')
        plt.close()
    
    for i in range(len(pos_caps_index)):
        
        start_1 = pos_caps_index[i]
        stop_1 = start_1 + (pos_time*acquisition_rate)
        
        plot_1_ydata = raw_current[start_1:stop_1]
        xdata_1 = x_data_index_master[start_1:stop_1]
        plot_1_xdata = [element / acquisition_rate for element in xdata_1]
        plot_1_V_ydata = raw_voltage[start_1:stop_1]
        time = int(plot_1_xdata[0])
        
        avg_cur_1 = round(sum((raw_current[(start_1+acquisition_rate):stop_1])/(stop_1-start_1)),2)
        stdev_1 = round(st.stdev((raw_current[(start_1+acquisition_rate):stop_1])),2)
        avg_V_1 = round(sum((raw_voltage[(start_1+acquisition_rate):stop_1])/(stop_1-(start_1+acquisition_rate))),2)
    
        fig,ax = plt.subplots(figsize=(8,4))
        
        ax.scatter(plot_1_xdata, plot_1_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_1+500
        min_y_val = avg_cur_1-500
        
        ax.set_ylim(min_y_val,max_y_val)
        
        ax.set_title(f" Pos Voltage at {time}sec\n {analysis_title} ", size=12, weight='bold') #Title
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_1} pA\n Current STDEV = {stdev_1} pA\n Applied Voltage = {avg_V_1} mV", loc=9)
        ax.add_artist(anchored_text)
        
        ax2=ax.twinx()
        
        ax2.scatter(plot_1_xdata, plot_1_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax2.set_ylim(-100,100)
        
        ax2.set_ylabel('Voltage (mV)', color = "r")
    
        
        plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], f"pos_raw_data_at_{time}_seconds"), dpi = PLOT_DPI, bbox_inches = 'tight')
        plt.close()
    
    for i in range(len(neg_caps_index)):
        
        start_1 = neg_caps_index[i]
        stop_1 = start_1 + (neg_time*acquisition_rate)
        
        plot_1_ydata = raw_current[start_1:stop_1]
        xdata_1 = x_data_index_master[start_1:stop_1]
        plot_1_xdata = [element / acquisition_rate for element in xdata_1]
        plot_1_V_ydata = raw_voltage[start_1:stop_1]
        time = int(plot_1_xdata[0])
        
        avg_cur_1 = round(sum((raw_current[(start_1+acquisition_rate):stop_1])/(stop_1-start_1)),2)
        stdev_1 = round(st.stdev((raw_current[(start_1+acquisition_rate):stop_1])),2)
        avg_V_1 = round(sum((raw_voltage[(start_1+acquisition_rate):stop_1])/(stop_1-(start_1+acquisition_rate))),2)
    
        fig,ax = plt.subplots(figsize=(8,4))
        
        ax.scatter(plot_1_xdata, plot_1_ydata, s=5, alpha=0.75, color = "b")
        
        max_y_val = avg_cur_1+500
        min_y_val = avg_cur_1-500
        
        ax.set_ylim(min_y_val,max_y_val)
        
        ax.set_title(f" Neg Voltage at {time}sec\n {analysis_title} ", size=12, weight='bold') #Title
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Current (pA)')
        
        anchored_text = AnchoredText(f" Average Current = {avg_cur_1} pA\n Current STDEV = {stdev_1} pA\n Applied Voltage = {avg_V_1} mV", loc=9)
        ax.add_artist(anchored_text)
        
        ax2=ax.twinx()
        
        ax2.scatter(plot_1_xdata, plot_1_V_ydata, s=2, alpha=0.75, color = "r")
        
        ax2.set_ylim(-100,100)
        
        ax2.set_ylabel('Voltage (mV)', color = "r")
    
        
        plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], raw_data_plots_folder_name[0], f"neg_raw_data_at_{time}_seconds"), dpi = PLOT_DPI, bbox_inches = 'tight')
        plt.close()
    
    return()


"""7.5 Conductance calculations"""
def conductance_calculation(all_cond_index, pos_cond_master, neg_cond_master, zero_cond_master, raw_current, raw_voltage, cond_datapoints, time_steps, conductance_plot_data):
    
    global data_for_cond_calc_master
    data_for_cond_calc_master = []
    data_for_cond_calc = []
    
    for i in range(len(all_cond_index)):
        first_index = all_cond_index[i]
        last_index = first_index + cond_datapoints
        current_values = raw_current[first_index:last_index]
        current_mean = sum(current_values)/len(current_values)
        voltage_values = raw_voltage[first_index:last_index]
        voltage_mean = sum(voltage_values)/len(voltage_values)
        data_for_cond_calc.append((last_index, current_mean, voltage_mean))
    
    data_for_cond_calc_master.append(data_for_cond_calc)
    
    slope = []
    k = 0
    
    while (k+4) < len(data_for_cond_calc):
        data_master_temp_1 = data_for_cond_calc[k]
        data_master_temp_2 = data_for_cond_calc[k+1]
        data_master_temp_3 = data_for_cond_calc[k+2]
        data_master_temp_4 = data_for_cond_calc[k+3]
        k = k + 4
        #global x_fit_data
        x_fit_data = [data_master_temp_1[2], data_master_temp_2[2], data_master_temp_3[2], data_master_temp_4[2]]
        #global y_fit_data
        y_fit_data = [data_master_temp_1[1], data_master_temp_2[1], data_master_temp_3[1], data_master_temp_4[1]]
        
        #global fit_data
        fit_data = scipy.stats.linregress(x_fit_data, y_fit_data, alternative='two-sided')
        # scipy.stats.linregress information = https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
        
        slope.append(fit_data[0])
    
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    global chunk_size        
    chunk_size = (int(len(slope)/time_steps))
    if chunk_size == 1: 
        chunk_size = 2
    time_chunks = list(chunks(slope, chunk_size))
    cond_time_chunks_master.append(time_chunks)
    end_temp = len(slope) - conductance_plot_data
    end_cond = list(slope[end_temp:-1])
    end_cond_master.append(end_cond)
    
    return(cond_time_chunks_master, end_cond_master)


"""7.75 Plotting global conductance vs time vs pH trends"""
def plotting_global_conductance_trends(cond_time_chunks_master, pHs, time_steps_in_minutes_for_legends, total_voltage_cycle_time, longest_run, path_to_save, analysis_folder_name, plots_folder_name, global_conductance_trends_folder, PLOT_DPI):
    """
            
        
        Updated: BS 02/21/2022 - reformatted to use loops so that the plotting can handle any number of pHs and any time parsing
    """
    # global data_master
    data_master = []
    
    for i in range(len(cond_time_chunks_master)):       # loop through the pHs with (i)
        data_temp = cond_time_chunks_master[i]
        data_packet_temp = []
        for n in range(len(data_temp)):         # loop through the time_chunks with n, time_chunks is a user defined amount
            if len(data_temp[n]) == 1: 
                continue
            elif len(data_temp[n]) < 1:
                data_packet_temp.append([0,0]) 
                continue
            data_chunk_mean_temp = (sum(data_temp[n])/len(data_temp[n]))
            data_chunk_stdev_temp = st.stdev(data_temp[n])
            data_packet_temp.append([data_chunk_mean_temp, data_chunk_stdev_temp])
        data_master.append(data_packet_temp)
    
    # global plot_data_master
    plot_data_master = []
    temp_data = data_master[0]
    i = 0
    n = 0
    while n in range(len(temp_data)):       # cycle through the time 
        i = 0
        mean_temp_list = []
        stdev_temp_list = []
        while i in range(len(data_master)):#        cycle through the pHs
            temp_data = data_master[i]
            data_packet_temp = temp_data[n]
            mean_temp = data_packet_temp[0]
            mean_temp_list.append(mean_temp)
            stdev_temp = data_packet_temp[1]
            stdev_temp_list.append(stdev_temp)
            i = i + 1
        plot_data_master.append([mean_temp_list, stdev_temp_list])
        n = n + 1
    
    # steps = 0
    # legend_steps = []
    # for i in range(len(plot_data_master)):
    #     steps = round((steps + time_steps_in_minutes_for_legends),1)
    #     legend_steps.append(steps)
    
    # global legend_steps_z
    
    legend_steps_z = []
    
    temp_data_z = cond_time_chunks_master[0]
    steps_end_z = 0
    for i in range(len(temp_data_z)):
        steps_start_z = steps_end_z
        steps_end_z = round(steps_end_z+((len(temp_data_z[i])*total_voltage_cycle_time)/60), 1)
        steps_z =  ([steps_start_z, steps_end_z])
        legend_steps_z.append(steps_z)
    
    plot_colors = ["k", "dimgray", "lightgray", "rosybrown", "indianred", "brown", "maroon", "red", "tomato", "coral", "orange", "sienna", "chocolate", "peru", "darkorange", "tan", "darkgoldenrod", "gold", "khaki", "darkkhaki", "olive", "yellow", "yellowgreen", "darkolivegreen", "chartreuse", "darkseagreen", "palegreen", "limegreen", "green", "lime", "springgreen", "aquamarine", "turquoise", "lightseagreen", "darkslategray", "darkcyan", "cyan", "deepskyblue", "lightskyblue", "dodgerblue", "cornflowerblue", "midnightblue", "blue", "slateblue", "darkslateblue", "rebeccapurple", "indigo", "darkorchid", "mediumorchid", "thistle", "plum", "violet", "purple", "fuchsia", "orchid", "mediumvioletred", "deeppink", "hotpink", "palevioletred", "crimson", "lightcoral", "indianred", "firebrick", "darkred", "tomato", "coral", "sienna", "bisque", "tan", "orange", "darkgoldenrod", "gold", "darkkhaki", "olive", "olivedrab", "darkolivegreen", "lawngreen", "forestgreen", "lime", "springgreen", "mediumspringgreen", "aquamarine" ]                      
    
    pH_labels = pHs
    
    # plot_label = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    
    fig = plt.figure(figsize=(12,8))
    ax = plt.gca()
    
    # for i in range(len(pH)):
    for i in range(len(plot_data_master)):      # cycle through the times, this will be detuminerd by the used defined (time_chunks)
        temp_data = plot_data_master[i]
        mean_list = temp_data[0]
        stdev_list = temp_data[1]
        plt.scatter(pH_labels, mean_list, s=5, alpha=0.75, color = plot_colors[i], label = legend_steps_z[i])
        plt.errorbar(pH_labels, mean_list, yerr = stdev_list, ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        pH_labels = [element + 0.01 for element in pH_labels]
        # plt.plot(pHs, mean_list, color = plot_colors[i], alpha=0.75, linewidth = 1)
        
    ax.set_title(" Conductance Over Time ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Conductance (nS)')
    ax.legend(bbox_to_anchor=(1.2,1), loc="upper right")
    
    
    # ax.set_yticks(np.arange(16, 26, 2))
    # ax.set_xticks(np.arange(3, 9, 1))
    # ax.tick_params(axis = 'x', which = 'both', direction = 'inout', length = 7, width = 1.5, labelsize = 11)
    # ax.tick_params(axis = 'y', which = 'both', direction = 'inout', length = 7, width = 1.5, labelsize = 11)
    # for axis in ['left', 'bottom', 'top', 'right']:
    #     ax.spines[axis].set_linewidth(1)
    # ax.legend(bbox_to_anchor = (1, 0.8), loc = 0)
    # ax.set_title(f" {analysis_title} Pos Fast Tau ", size=12, weight='bold') #Title
    # ax.set_xlabel('pH', size = 12)
    # ax.set_ylabel('Tau (s)',size = 12)
    # ax.set_ylim([0, 800])
    # ax.set_xlim([2.25, 8.75])
    # ax.legend(loc = 0, prop = {'size':9})
    
    
    plt.savefig(os.path.join(path_to_save, global_conductance_trends_folder[0], "conductance_vs_pH_vs_time_large"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    pH_labels = pHs
    
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    
    # for i in range(len(pH)):
    for i in range(len(plot_data_master)):      # cycle through the times, this will be detuminerd by the used defined (time_chunks)
        temp_data = plot_data_master[i]
        mean_list = temp_data[0]
        stdev_list = temp_data[1]
        plt.scatter(pH_labels, mean_list, s=5, alpha=0.75, color = plot_colors[i], label = legend_steps_z[i])
        plt.errorbar(pH_labels, mean_list, yerr = stdev_list, ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        pH_labels = [element + 0.01 for element in pH_labels]
        # plt.plot(pHs, mean_list, color = plot_colors[i], alpha=0.75, linewidth = 1)
        
    ax.set_title(" Conductance Over Time ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Conductance (nS)')
    ax.legend(bbox_to_anchor=(1.3,1), loc="upper right")
    
    
    # ax.set_yticks(np.arange(16, 26, 2))
    # ax.set_xticks(np.arange(3, 9, 1))
    # ax.tick_params(axis = 'x', which = 'both', direction = 'inout', length = 7, width = 1.5, labelsize = 11)
    # ax.tick_params(axis = 'y', which = 'both', direction = 'inout', length = 7, width = 1.5, labelsize = 11)
    # for axis in ['left', 'bottom', 'top', 'right']:
    #     ax.spines[axis].set_linewidth(1)
    # ax.legend(bbox_to_anchor = (1, 0.8), loc = 0)
    # ax.set_title(f" {analysis_title} Pos Fast Tau ", size=12, weight='bold') #Title
    # ax.set_xlabel('pH', size = 12)
    # ax.set_ylabel('Tau (s)',size = 12)
    # ax.set_ylim([0, 800])
    # ax.set_xlim([2.25, 8.75])
    # ax.legend(loc = 0, prop = {'size':9})
    
    
    plt.savefig(os.path.join(path_to_save, global_conductance_trends_folder[0], "conductance_vs_pH_vs_time"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    pH_labels = pHs
    
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    
    # for i in range(len(pH)):
    for i in range(len(plot_data_master)):      # cycle through the times, this will be detuminerd by the used defined (time_chunks)
        if i == 0: continue    
        temp_data = plot_data_master[i]
        mean_list = temp_data[0]
        stdev_list = temp_data[1]
        plt.scatter(pH_labels, mean_list, s=5, alpha=0.75, color = plot_colors[i], label = legend_steps_z[i])
        plt.errorbar(pH_labels, mean_list, yerr = stdev_list, ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        pH_labels = [element + 0.01 for element in pH_labels]
        # plt.plot(pHs, mean_list, color = plot_colors[i], alpha=0.75, linewidth = 1)
        
    ax.set_title(" Conductance Over Time ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Conductance (nS)')
    # ax.legend()
    
    
    # ax.set_yticks(np.arange(16, 26, 2))
    # ax.set_xticks(np.arange(3, 9, 1))
    # ax.tick_params(axis = 'x', which = 'both', direction = 'inout', length = 7, width = 1.5, labelsize = 11)
    # ax.tick_params(axis = 'y', which = 'both', direction = 'inout', length = 7, width = 1.5, labelsize = 11)
    # for axis in ['left', 'bottom', 'top', 'right']:
    #     ax.spines[axis].set_linewidth(1)
    # ax.legend(bbox_to_anchor = (1, 0.8), loc = 0)
    # ax.set_title(f" {analysis_title} Pos Fast Tau ", size=12, weight='bold') #Title
    # ax.set_xlabel('pH', size = 12)
    # ax.set_ylabel('Tau (s)',size = 12)
    # ax.set_ylim([0, 800])
    # ax.set_xlim([2.25, 8.75])
    # ax.legend(loc = 0, prop = {'size':9})
    
    
    plt.savefig(os.path.join(path_to_save, global_conductance_trends_folder[0], "conductance_vs_pH_vs_time_first_5_removed"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    pH_labels = pHs
    
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    
    # for i in range(len(pH)):
    for i in range(len(plot_data_master)):      # cycle through the times, this will be detuminerd by the used defined (time_chunks)
        temp_data = plot_data_master[i]
        mean_list = temp_data[0]
        stdev_list = temp_data[1]
        plt.scatter(pH_labels, mean_list, s=5, alpha=0.75, color = plot_colors[i], label = legend_steps_z[i])
        plt.errorbar(pH_labels, mean_list, yerr = stdev_list, ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        pH_labels = [element + 0.01 for element in pH_labels]
        # plt.plot(pHs, mean_list, color = plot_colors[i], alpha=0.75, linewidth = 1)
        
    ax.set_title(" Conductance Over Time ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Conductance (nS)')
    ax.legend(bbox_to_anchor=(1.3,1), loc="upper right")
    
    
    # ax.set_yticks(np.arange(16, 26, 2))
    # ax.set_xticks(np.arange(3, 9, 1))
    # ax.tick_params(axis = 'x', which = 'both', direction = 'inout', length = 7, width = 1.5, labelsize = 11)
    # ax.tick_params(axis = 'y', which = 'both', direction = 'inout', length = 7, width = 1.5, labelsize = 11)
    # for axis in ['left', 'bottom', 'top', 'right']:
    #     ax.spines[axis].set_linewidth(1)
    # ax.legend(bbox_to_anchor = (1, 0.8), loc = 0)
    # ax.set_title(f" {analysis_title} Pos Fast Tau ", size=12, weight='bold') #Title
    # ax.set_xlabel('pH', size = 12)
    # ax.set_ylabel('Tau (s)',size = 12)
    ax.set_ylim([0, 3])
    # ax.set_xlim([2.25, 8.75])
    # ax.legend(loc = 0, prop = {'size':9})
    
    
    plt.savefig(os.path.join(path_to_save, global_conductance_trends_folder[0], "conductance_vs_pH_vs_time_scale"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    pH_labels = pHs
    
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    
    # for i in range(len(pH)):
    for i in range(len(plot_data_master)):      # cycle through the times, this will be detuminerd by the used defined (time_chunks)
        temp_data = plot_data_master[i]
        mean_list = temp_data[0]
        stdev_list = temp_data[1]
        plt.scatter(pH_labels, mean_list, s=5, alpha=0.75, color = plot_colors[i], label = legend_steps_z[i])
        plt.errorbar(pH_labels, mean_list, yerr = stdev_list, ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        pH_labels = [element + 0.01 for element in pH_labels]
        # plt.plot(pHs, mean_list, color = plot_colors[i], alpha=0.75, linewidth = 1)
        
    ax.set_title(" Conductance Over Time ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Conductance (nS)')
    # ax.legend(bbox_to_anchor=(1.3,1), loc="upper right")
    
    
    # ax.set_yticks(np.arange(16, 26, 2))
    # ax.set_xticks(np.arange(3, 9, 1))
    # ax.tick_params(axis = 'x', which = 'both', direction = 'inout', length = 7, width = 1.5, labelsize = 11)
    # ax.tick_params(axis = 'y', which = 'both', direction = 'inout', length = 7, width = 1.5, labelsize = 11)
    # for axis in ['left', 'bottom', 'top', 'right']:
    #     ax.spines[axis].set_linewidth(1)
    # ax.legend(bbox_to_anchor = (1, 0.8), loc = 0)
    # ax.set_title(f" {analysis_title} Pos Fast Tau ", size=12, weight='bold') #Title
    # ax.set_xlabel('pH', size = 12)
    # ax.set_ylabel('Tau (s)',size = 12)
    ax.set_ylim([0, 3])
    # ax.set_xlim([2.25, 8.75])
    # ax.legend(loc = 0, prop = {'size':9})
    
    
    plt.savefig(os.path.join(path_to_save, global_conductance_trends_folder[0], "conductance_vs_pH_vs_time_scale_no_leg"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    
    return(print("done"))


"""7.76 Plotting the final conductance vs pH trend"""
def plotting_the_final_G_v_pH(pHs, end_cond_master, path_to_save, analysis_folder_name, plots_folder_name, global_conductance_trends_folder, PLOT_DPI):
    
    data_mean = []
    data_stdev = []
    
    for i in range(len(end_cond_master)):
        data_temp = end_cond_master[i]
        data_mean_temp = (sum(data_temp)/len(data_temp))
        data_stdev_temp = st.stdev(data_temp)
        data_mean.append(data_mean_temp)
        data_stdev.append(data_stdev_temp)
        
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    
    plt.scatter(pHs, data_mean, s=5, alpha=0.75, color = "b") #, label = "15 min")
    plt.errorbar(pHs, data_mean, yerr = data_stdev, ls = "None", ecolor = "b", elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
    plt.plot(pHs, data_mean, color = "b", alpha=0.75, linewidth = 1)
    
    ax.set_title(" Conductance vs. pH", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Conductance (nS)')
    # ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_conductance_trends_folder[0], "conductance_vs_pH"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    return(print("done"))


"""8. Fitting Capacitance Spikes with lmfit double exponential"""
def fitting_cap_spikes_w_lmfit_double_exp(pos_caps_index, neg_caps_index, raw_current, acquisition_rate, fit_offset, time_steps, data_per_cap_spike):
    """
        This function takes the capacitance spike index's (pos_cap_index, neg_caps_index) and fits the eponential like decay with a DOUBLE exponential equation using the open source lmfit funtion.
        (lmfit DOI: 10.5281/zenodo.5570790)
        
        The fit parameters, variables, and a log of each fit performace are saved. Several other parameters and variables are also saved. 
        
        1. lmfit_parameters, cpntains a tuple of lists that contain all the fit determined variables for each fit (pos_fast_tau, pos_fast_ab, pos_slow_tau, pos_slow_ab, neg_fast_tau, neg_fast_ab, neg_slow_tau, neg_slow_ab, pos_intercept, neg_intercept)
        2. lmfit_cap_varieables, contains a tuple that stores the fit determined variables along with an index and other values that anayze the fit performance (a, k1, b, k2, c, lmfit_y_data, fitting_x_data, cap_index, cap_data, fit_one, fit_two, timing_index)
        3. double_lmfit_log_master, contains a list of the fit performance log
        4. lmfit_cap_varieables_master, contains #3 but for all files (master for ALL double lmfit fits) 
        5. lmfit_double_exp_10min_windows_master, chops the experiment time into 6 sections, then finds the average of each parameter in each section so that the analysis can be broken up and displayed in 10 minute chuncks
            (pos_double_fit_fast_tau_chunk_master, pos_double_fit_slow_tau_chunk_master, neg_double_fit_fast_tau_chunk_master, neg_double_fit_slow_tau_chunk_master)    
            * might want to make the number of section that the data is broken up into a used defined value *
        6. ratios, list of calculated ratios between fit parameter values (pos_tau_ratio, pos_ab_ratio, neg_tau_ratio, neg_ab_ratio)
        
        Updated: BS - 01/11/2022
        Updated: BS - 02/09/2022 - added the user defined value for the number of time chuncks the fitting variables are parsed into (time_steps)
        Updated: BS - 02/21/2022 - reduced the lines of code when parsing for the last X minute plotting with a loop to make the variable. Same variable, new creation method
    """
    def monoExp(x, m, k, h):
        return m * np.exp(-(x * (1/k))) + h
    
    def _2exponential(x, a, k1, b, k2, c):
        return a * np.exp(-(x * (1/k1))) + b * np.exp(-(x * (1/k2))) + c
    
    def power_fitting_lmfit(params,x,y):
        a = params['a']
        k1 = params['k1']
        b = params['b']
        k2 = params['k2']
        c = params['c']
        y_fit = a * np.exp(-(x * (1/k1))) + b * np.exp(-(x * (1/k2))) + c
        return y_fit-y
    
    params = Parameters()
    
    params.add('a', value = 14000, vary = True, min= 1.0, max= 40000)
    params.add('k1', value = 20, vary = True, min= 1.0, max= 1000)
    params.add('b', value = 10000, vary = True, min= 1.0, max= 40000)
    params.add('k2', value = 40, vary = True, min= 1.0, max= 1000)
    params.add('c', value = 400, vary = True, min= 1.0, max= 5000)
    
    pos_cap_index_neg_cap_index = []
    lmfit_cap_varieables = []
    double_lmfit_log_master = []
    fast_tau = []
    slow_tau = []
    fast_ab = []
    slow_ab = []
    intercept = []
    
    pos_cap_index_neg_cap_index.extend(pos_caps_index)
    pos_cap_index_neg_cap_index.extend(neg_caps_index)
    
    for o in range(len(pos_cap_index_neg_cap_index)):
        cap_index = pos_cap_index_neg_cap_index[o]
        cap_data_first_index = cap_index + fit_offset
        cap_data_last_index = cap_data_first_index + data_per_cap_spike       # should this be aquisition_rate or the user defined value of how much to fit?
        cap_data = abs(raw_current[cap_data_first_index:cap_data_last_index])
        timing_index = x_data_index_master[cap_data_first_index:cap_data_last_index]
        fitting_x_data = np.linspace(0, len(cap_data), num = len(cap_data), endpoint = True)
    
        fitted_params = minimize(power_fitting_lmfit, params, args=(fitting_x_data,cap_data), method='least_squares')
        # logging.warning(f'error in lmfit double exp fit # {o}, run #{count}')
        
        a = fitted_params.params['a'].value
        k1 = fitted_params.params['k1'].value
        b = fitted_params.params['b'].value
        k2 = fitted_params.params['k2'].value
        c = fitted_params.params['c'].value
        
        fit_log = fit_report(fitted_params)
        double_lmfit_log_master.append(f"double lmfit number {o}")
        double_lmfit_log_master.append(fit_log)
        
        lmfit_y_data = _2exponential(fitting_x_data, a, k1, b, k2, c)
        fit_one = monoExp(fitting_x_data, a, k1, c)
        fit_two = monoExp(fitting_x_data, b, k2, c)
        
        if k1 > k2:
            fast_tau.append(k1)
            fast_ab.append(a)
            slow_ab.append(b)
            slow_tau.append(k2)
        if k2 > k1:
            fast_tau.append(k2)
            fast_ab.append(b)
            slow_ab.append(a)
            slow_tau.append(k1)
        
        intercept.append(c)
        
        lmfit_cap_var_temp = [a, k1, b, k2, c, lmfit_y_data, fitting_x_data, cap_index, cap_data, fit_one, fit_two, timing_index]
        lmfit_cap_varieables.append(lmfit_cap_var_temp)
    
    # double fit data manipulation
    len_of_pos_caps = len(pos_caps_index)
    
    pos_fast_tau = fast_tau[0:len_of_pos_caps]   
    pos_fast_ab = fast_ab[0:len_of_pos_caps]   
    pos_slow_tau = slow_tau[0:len_of_pos_caps]   
    pos_slow_ab = slow_ab[0:len_of_pos_caps]   
    pos_intercept = intercept[0:len_of_pos_caps]
    # pos_data_packet = [pos_fast_tau, pos_slow_tau]
    
    pos_tau_ratio = []
    pos_ab_ratio = []
    pos_tau_ratio = [i / j for i, j in zip(pos_fast_tau, pos_slow_tau)]
    pos_ab_ratio = [i / j for i, j in zip(pos_fast_ab, pos_slow_ab)]
    
    neg_fast_tau = fast_tau[len_of_pos_caps:]   
    neg_fast_ab = fast_ab[len_of_pos_caps:]   
    neg_slow_tau = slow_tau[len_of_pos_caps:]   
    neg_slow_ab = slow_ab[len_of_pos_caps:] 
    neg_intercept = intercept[len_of_pos_caps:] 
    global tau_data_packet
    tau_data_packet = [pos_fast_tau, pos_slow_tau, neg_fast_tau, neg_slow_tau]
    
    neg_tau_ratio = []
    neg_ab_ratio = []
    neg_tau_ratio = [i / j for i, j in zip(neg_fast_tau, neg_slow_tau)]
    neg_ab_ratio = [i / j for i, j in zip(neg_fast_ab, neg_slow_ab)]
    
    ratios = []
    ratios = [pos_tau_ratio, pos_ab_ratio, neg_tau_ratio, neg_ab_ratio]
    
    lmfit_double_exp_10min_windows_temp = []
    i = 0
    n = time_steps
    for i in range(len(tau_data_packet)):
        temp_data = tau_data_packet[i]
        double_fit_temp = []
        double_fit_chunks = np.array_split(temp_data, n)
        for t in range(len(double_fit_chunks)):
            current_date = double_fit_chunks[t]
            if len(current_date) == 1: 
                mean = current_date[0]
                stdev = 0
                double_fit_temp.append([current_date, mean, stdev])
                continue
            mean = sum(current_date)/len(current_date)
            stdev = st.stdev(current_date)
            double_fit_temp.append([current_date, mean, stdev])
        lmfit_double_exp_10min_windows_temp.append(double_fit_temp)
        
    lmfit_double_exp_10min_windows_master.append(lmfit_double_exp_10min_windows_temp)
    
    lmfit_params_temp = [pos_fast_tau, pos_fast_ab, pos_slow_tau, pos_slow_ab, neg_fast_tau, neg_fast_ab, neg_slow_tau, neg_slow_ab, pos_intercept, neg_intercept]
    lmfit_parameters.append(lmfit_params_temp)
    
    lmfit_cap_varieables_master.append(lmfit_cap_varieables)
        
    return(lmfit_parameters, lmfit_cap_varieables, double_lmfit_log_master, lmfit_cap_varieables_master, lmfit_double_exp_10min_windows_master, ratios)


"""9. Fitting Capacitance Spikes with lmfit single exponential"""
def fitting_cap_spikes_w_lmfit_single_exp(pos_caps_index, neg_caps_index, raw_current, acquisition_rate, fit_offset, time_steps):
    """
        This function takes the capacitance spike index's (pos_cap_index, neg_caps_index) and fits the eponential like decay with a SINGLE exponential equation using the open source lmfit funtion.
        (lmfit DOI: 10.5281/zenodo.5570790)
        
        The fit parameters, variables, and a log of each fit performace are saved. Several other parameters and variables are also saved. 
        
        1. lmfit_single_exp_fit_parameters, contains a tuple of lists that contain all the fit determined variables for each fit (pos_m, pos_k, pos_h, neg_m, neg_k, neg_h)
        2. lmfit_single_exp_fit_cap_varieables, contains a tuple that stores the fit determined variables along with an index and other values that anayze the fit performance (m, k, h, lmfit_y_data, fitting_x_data, cap_index, cap_data, timing_index)
        3. lmfit_single_exp_fit_log_master, contains a list of the fit performance log
        4. lmfit_single_exp_fit_cap_varieables_master, contains #3 but for all files (master for ALL double lmfit fits) 
        5. lmfit_single_exp_10min_windows_master, chops the experiment time into 6 sections, then finds the average of each parameter in each section so that the analysis can be broken up and displayed in 10 minute chuncks
            (pos_single_fit_tau_chunk_master, neg_single_fit_tau_chunk_master)    
            * might want to make the number of section that the data is broken up into a used defined value *
        
        Updated: BS - 01/11/2022
        Updated: BS - 02/09/2022 - added the user defined value for the number of time chuncks the fitting variables are parsed into (time_steps)
    """
    def monoExp(x, m, k, h):
        return m * np.exp(-(x * (1/k))) + h
    
    def power_fitting_lmfit(params,x,y):
        m = params['m']
        k = params['k']
        h = params['h']
        y_fit = m * np.exp(-(x * (1/k))) + h
        return y_fit-y
    
    params = Parameters()
    
    params.add('m', value = 20000, vary = True, min= 1.0, max= 40000)
    params.add('k', value = 40, vary = True, min= 1.0, max= 5000)
    params.add('h', value = 400, vary = True, min= 1.0, max= 5000)
    pos_cap_index_neg_cap_index = []
    lmfit_single_exp_fit_cap_varieables = []
    lmfit_single_exp_fit_log_master = []
    # global m_data
    m_data = []
    
    k_data = []
    intercept = []
    
    pos_cap_index_neg_cap_index.extend(pos_caps_index)
    pos_cap_index_neg_cap_index.extend(neg_caps_index)
    
    for o in range(len(pos_cap_index_neg_cap_index)):
        cap_index = pos_cap_index_neg_cap_index[o]
        cap_data_first_index = cap_index + fit_offset
        cap_data_last_index = cap_data_first_index + acquisition_rate
        cap_data = abs(raw_current[cap_data_first_index:cap_data_last_index])
        timing_index = x_data_index_master[cap_data_first_index:cap_data_last_index]
        fitting_x_data = np.linspace(0, len(cap_data), num = len(cap_data), endpoint = True)
        
        fitted_params = minimize(power_fitting_lmfit, params, args=(fitting_x_data,cap_data,), method='least_squares')
        # logging.warning(f'error in lmfit single exp fit # {o}, run #{i}')
        
        m = fitted_params.params['m'].value
        k = fitted_params.params['k'].value
        h = fitted_params.params['h'].value
        
        fit_log = fit_report(fitted_params)
        lmfit_single_exp_fit_log_master.append(f"single mlfit number {o}")
        lmfit_single_exp_fit_log_master.append(fit_log)
        
        lmfit_y_data = monoExp(fitting_x_data, m, k, h)
        
        m_data.append(m)
        k_data.append(k)
        intercept.append(h)
        
        lmfit_cap_var_temp = [m, k, h, lmfit_y_data, fitting_x_data, cap_index, cap_data, timing_index]
        lmfit_single_exp_fit_cap_varieables.append(lmfit_cap_var_temp)
    
    
    len_of_pos_caps = len(pos_caps_index)
    # len_of_neg_caps = len(neg_caps_index)
    
    pos_m = m_data[0:len_of_pos_caps]   
    pos_k = k_data[0:len_of_pos_caps]   
    pos_h = intercept[0:len_of_pos_caps]   
    
    neg_m = m_data[len_of_pos_caps:]   
    neg_k = k_data[len_of_pos_caps:]   
    neg_h = intercept[len_of_pos_caps:]   
    
    lmfit_params_temp = [pos_m, pos_k, pos_h, neg_m, neg_k, neg_h]
    lmfit_single_exp_fit_parameters.append(lmfit_params_temp)
    
    n = time_steps       # might want to make this a user defined variable
    pos_single_fit_tau_chunk_master = []
    single_fit_tau_temp_temp = []
    pos_single_fit_tau_chuncks = np.array_split(pos_k, n)
    for t in range(len(pos_single_fit_tau_chuncks)):
        current_date = pos_single_fit_tau_chuncks[t]
        if len(current_date) == 1: 
            mean = current_date[0]
            stdev = 0
            single_fit_tau_temp = [current_date, mean, stdev]
            single_fit_tau_temp_temp.append(single_fit_tau_temp)
            continue
        mean = sum(current_date)/len(current_date)
        stdev = st.stdev(current_date)
        single_fit_tau_temp = [current_date, mean, stdev]
        single_fit_tau_temp_temp.append(single_fit_tau_temp)
    pos_single_fit_tau_chunk_master.append(single_fit_tau_temp_temp)
    
    neg_single_fit_tau_chunk_master = []
    single_fit_tau_temp_temp = []
    neg_single_fit_tau_chuncks = np.array_split(neg_k, n)
    for t in range(len(neg_single_fit_tau_chuncks)):
        current_date = neg_single_fit_tau_chuncks[t]
        if len(current_date) == 1: 
            mean = current_date[0]
            stdev = 0
            single_fit_tau_temp = [current_date, mean, stdev]
            single_fit_tau_temp_temp.append(single_fit_tau_temp)
            continue
        mean = sum(current_date)/len(current_date)
        stdev = st.stdev(current_date)
        single_fit_tau_temp = [current_date, mean, stdev]
        single_fit_tau_temp_temp.append(single_fit_tau_temp)
    neg_single_fit_tau_chunk_master.append(single_fit_tau_temp_temp)
    
    lmfit_single_exp_10min_windows_temp = [pos_single_fit_tau_chunk_master, neg_single_fit_tau_chunk_master]
    lmfit_single_exp_10min_windows_master.append(lmfit_single_exp_10min_windows_temp)
    
    lmfit_single_exp_fit_cap_varieables_master.append(lmfit_single_exp_fit_cap_varieables)
        
    return(lmfit_single_exp_fit_parameters, lmfit_single_exp_fit_cap_varieables, lmfit_single_exp_fit_log_master, lmfit_single_exp_fit_cap_varieables_master, lmfit_single_exp_10min_windows_master)


"""10. Ploting lmfit Single Exp fit Cap Spikes and lmfits Fits"""
def plotting_lmfit_single_exp_caps_and_fits(lmfit_single_exp_fit_cap_varieables, raw_current, acquisition_rate, raw_current_data_seen, path_to_save, analysis_folder_name, plots_folder_name, lmfit_single_fit_plots_folder_name, PLOT_DPI):
    """
        This function plots and saves the single exponential lmfit fit and the raw_current data to that the fit can be visually evaluated
        Saves the plots to the (lmfit_single_fit_plots_folder_name) folder that was made earlier
        
        * for help with writing program : lmfit_cap_var_temp = [m, k, h, lmfit_y_data, fitting_x_data, cap_index, cap_data, timing_index] *
        
        Updated: BS - 01/11/2022
        Updated: BS - 02/21/2022 - added the fit data to the legend
    """
    
    for i in range(len(lmfit_single_exp_fit_cap_varieables)):
        index_data = lmfit_single_exp_fit_cap_varieables[i]
        raw_current_y_index = index_data[5]
        raw_current_y_start = raw_current_y_index - raw_current_data_seen
        raw_current_y_end = raw_current_y_start + acquisition_rate
        raw_current_y_data = abs(raw_current[raw_current_y_start:raw_current_y_end])
        raw_current_x_data = x_data_index_master[raw_current_y_start:raw_current_y_end]
        Scalar = int(index_data[0])
        Tau = int(index_data[1])
        Intercept = int(index_data[2])
        all_exp_fit_x_data = index_data[7]
        
        single_exp_fit_y_data = index_data[3]
        intercept = index_data[2]
        
        plot_x_lower_bound = index_data[5] - raw_current_data_seen
        plot_x_upper_bound = (index_data[5] + (acquisition_rate*0.05))
        
        fig = plt.figure(figsize=(4,3))
        ax = plt.gca()
        ax.scatter(raw_current_x_data, raw_current_y_data, color = 'k', s = 2, label="data", zorder = 1)
        ax.plot(all_exp_fit_x_data, single_exp_fit_y_data, '--', color = 'r', label="fit1", linewidth = 1, zorder = 2)
        ax.hlines(y=intercept, xmin = plot_x_lower_bound, xmax = plot_x_upper_bound, linewidth=1, color='m', linestyles='--', label = "intercept")
        ax.set_title(f" single fit #{i} ", size=12, weight='bold') #Title
        ax.set_xlabel('Datapoints',  fontsize=12)
        ax.set_ylabel('Current (pA)',  fontsize=12)
        ax.legend()
        # anchored_text = AnchoredText(f" scalar = {Scalar}\n tau = {Tau}\n intercept = {Intercept}", loc=2)
        # ax.add_artist(anchored_text)
        # ax.text(0.5, 0.5, f" scalar = {Scalar}\n tau = {Tau}\n intercept = {Intercept}",  fontsize=12)
        ax.set_xlim(plot_x_lower_bound,plot_x_upper_bound)
        
        plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_single_fit_plots_folder_name[0], f"lmfit_single_exp_fit_#{i}"), dpi = PLOT_DPI, bbox_inches = 'tight')
        plt.close()
        
    return(print('done plotting single exp lmfit fits'))


"""11. plotting lmfit Single Exp fit Cap Spike Variables"""
def plotting_lmfit_single_fit_parameters(lmfit_single_exp_fit_parameters, path_to_save, analysis_folder_name, plots_folder_name, lmfit_single_fit_vals_plots_folder_name, total_voltage_cycle_time, PLOT_DPI):
    """
        This function plots all the fit variables from the single exponential lmfit analysis and saves them in the (lmfit_single_fit_vals_plots_folder_name) flder that was made earlier
        
        Updated: BS - 01/11/2022
        Updated: BS - 02/09/2022 - making the x-axid units into seconds for all variable plots
    """
    
    master = lmfit_single_exp_fit_parameters[0]
    # global pos_tau
    pos_tau = master[1]
    
    neg_tau = master[4]
    
    pos_ab = master[0]
    
    neg_ab = master[3]
    pos_intercept_y_data = master[2]
    neg_intercept_y_data = master[5]
    
    one_cycle_seconds = total_voltage_cycle_time
    
    pos_tau_x_data_temp = (len(pos_tau)*one_cycle_seconds)
    pos_tau_x_data = np.linspace(0, pos_tau_x_data_temp, num = len(pos_tau), endpoint = True)
    pos_ab_x_data_temp = (len(pos_ab)*one_cycle_seconds)
    pos_ab_x_data = np.linspace(0, pos_ab_x_data_temp, num = len(pos_ab), endpoint = True)
    neg_tau_x_data_temp = (len(neg_tau)*one_cycle_seconds)
    neg_tau_x_data = np.linspace(0, neg_tau_x_data_temp, num = len(neg_tau), endpoint = True)
    neg_ab_x_data_temp = (len(neg_ab)*one_cycle_seconds)
    neg_ab_x_data = np.linspace(0, neg_ab_x_data_temp, num = len(neg_ab), endpoint = True)
    pos_intercept_x_data_temp = (len(pos_intercept_y_data)*one_cycle_seconds)
    pos_intercept_x_data = np.linspace(0, pos_intercept_x_data_temp, num = len(pos_intercept_y_data), endpoint = True)
    neg_intercept_x_data_temp = (len(neg_intercept_y_data)*one_cycle_seconds)
    neg_intercept_x_data = np.linspace(0, neg_intercept_x_data_temp, num = len(neg_intercept_y_data), endpoint = True)
    
    # pos fast Tau
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(pos_tau_x_data, pos_tau, color = 'b', s = 1.5)
    ax.set_title(" pos single lmfit Tau ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tau (s)')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_single_fit_vals_plots_folder_name[0], "pos_single_lmfit_tau"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # pos fast ab
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(pos_ab_x_data, pos_ab, color = 'b', s = 1.5)
    ax.set_title(" pos single lmfit a ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ab')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_single_fit_vals_plots_folder_name[0], "pos_single_lmfit_a"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all pos variable
    fig, ax1 = plt.subplots(1, figsize=(8,4))
    ax2 = ax1.twinx()
    ax1.scatter(pos_tau_x_data, pos_tau, color = 'k', s = 1.5, alpha = 0.75, label = "tau")
    ax2.scatter(pos_ab_x_data, pos_ab, color = 'b', s = 1.5, alpha = 0.75,label = "a")
    ax1.scatter(pos_intercept_x_data, pos_intercept_y_data, color = 'r', s = 1, alpha = 0.75,label = "intercept")
    ax1.set_title(" all pos single lmfit variables ", size=12, weight='bold') #Title
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Time (s)')
    ax2.set_ylabel('a')
    fig.legend(loc="upper right")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_single_fit_vals_plots_folder_name[0], "all_pos_lmfit_single_fit_vars"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # neg fast Tau
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(neg_tau_x_data, neg_tau, color = 'b', s = 1.5)
    ax.set_title(" neg single lmfit Tau ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tau (s)')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_single_fit_vals_plots_folder_name[0], "neg_single_lmfit_tau"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # neg fast ab
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(neg_ab_x_data, neg_ab, color = 'b', s = 1.5)
    ax.set_title(" neg single lmfit a ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ab')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_single_fit_vals_plots_folder_name[0], "neg_single_lmfit_a"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all neg variable
    fig, ax1 = plt.subplots(1, figsize=(8,4))
    ax2 = ax1.twinx()
    ax1.scatter(neg_tau_x_data, neg_tau, color = 'k', s = 1.5, alpha = 0.75, label = "tau")
    ax2.scatter(neg_ab_x_data, neg_ab, color = 'b', s = 1.5, alpha = 0.75, label = "a")
    ax1.scatter(neg_intercept_x_data, neg_intercept_y_data, color = 'r', s = 1, alpha = 0.75, label = "intercept")
    ax1.set_title(" all neg single lmfit variables ", size=12, weight='bold') #Title
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Time (s)')
    ax2.set_ylabel('a')
    fig.legend(loc="upper right")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_single_fit_vals_plots_folder_name[0], "all_neg_lmfit_single_fit_vars"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    return(print('done plotting single exp lmfit fit variables'))


"""12. Potting total lmfit Single Exp fit Tau trends"""
def plotting_total_single_lmfit_tau_trends(lmfit_single_exp_10min_windows_master, pHs, path_to_save, analysis_folder_name, plots_folder_name, fit_vals_plots_folder_name, file_tag, time_steps, PLOT_DPI):
    """
        This function plots the trends from the single exponential lmfit fitting time constant (tau) across all of the files that were analyzed and saves them in (fit_vals_plots_folder_name)
        
        Updated: BS - 01/11/2022
        Updated: BS - 02/09/2022 - changed the number of itterations is the second loop in each plotting section to be determined by the (time_steps) not the (pHs)
    """
    global_single_lmfit_trends_folder = [f"global_single_lmfit_trends{file_tag}"]
    
    try:
        os.mkdir(os.path.join(path_to_save, global_single_lmfit_trends_folder[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)
    
    plot_colors = ["k", "b", "g", "r", "m", "y"]
    plot_colors = ["k", "dimgray", "lightgray", "rosybrown", "indianred", "brown", "maroon", "red", "tomato", "coral", "orange", "sienna", "chocolate", "peru", "darkorange", "tan", "darkgoldenrod", "gold", "khaki", "darkkhaki", "olive", "yellow", "yellowgreen", "darkolivegreen", "chartreuse", "darkseagreen", "palegreen", "limegreen", "green", "lime", "springgreen", "aquamarine", "turquoise", "lightseagreen", "darkslategray", "darkcyan", "cyan", "deepskyblue", "lightskyblue", "dodgerblue", "cornflowerblue", "midnightblue", "blue", "slateblue", "darkslateblue", "rebeccapurple", "indigo", "darkorchid", "mediumorchid", "thistle", "plum", "violet", "purple", "fuchsia", "orchid", "mediumvioletred", "deeppink", "hotpink", "palevioletred", "crimson", "lightcoral", "indianred", "firebrick", "darkred", "tomato", "coral", "sienna", "bisque", "tan", "orange", "darkgoldenrod", "gold", "darkkhaki", "olive", "olivedrab", "darkolivegreen", "lawngreen", "forestgreen", "lime", "springgreen", "mediumspringgreen", "aquamarine"]
    plot_label = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65"]
    
    
    # all chuncks single pos tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    r = 0
    for r in range(time_steps):
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)):
            temp_y_data_list = lmfit_single_exp_10min_windows_master[j]
            temp_y_data = temp_y_data_list[0] # this chooses pos, neg  ---- single fit..... 
            temp_y_data_2 = temp_y_data[0]
            if len(temp_y_data_2) <= r: continue # trying to add a catch for "short data" - some data file/experiments dont go as long as others
            temp2_y_data = temp_y_data_2[r] 
            temp_mean.append(temp2_y_data[1])
            temp_stdev.append(temp2_y_data[2])
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):    
        plt.scatter(pHs, y_data_mean[i], s=5, alpha=0.75, color = plot_colors[i], label = plot_label[i])
        plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        # plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" pos single lmfit tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_single_lmfit_trends_folder[0], "pos_single_lmfit_tau_vs_pH_vs_time"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all chuncks single pos tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    r = 0
    for r in range(time_steps):
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)):
            temp_y_data_list = lmfit_single_exp_10min_windows_master[j] # selects the pH
            temp_y_data = temp_y_data_list[0] # this chooses pos, neg  ---- single fit..... 
            temp_y_data_2 = temp_y_data[0] # one level deeper .... weird
            if len(temp_y_data_2) <= r: continue # trying to add a catch for "short data" - some data file/experiments dont go as long as others
            temp2_y_data = temp_y_data_2[r] 
            temp_mean.append(temp2_y_data[1])
            temp_stdev.append(temp2_y_data[2])
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):    
        plt.scatter(pHs, y_data_mean[i], s=5, alpha=0.75, color = plot_colors[i], label = plot_label[i])
        # plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        # plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" pos single lmfit tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_single_lmfit_trends_folder[0], "pos_single_lmfit_tau_vs_pH_vs_time-no-error"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all chuncks single neg tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    r = 0
    for r in range(time_steps):
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)):
            temp_y_data_list = lmfit_single_exp_10min_windows_master[j]
            temp_y_data = temp_y_data_list[1] # this chooses pos fast, pos slow, neg fast, neg slow
            temp_y_data_2 = temp_y_data[0]
            if len(temp_y_data_2) <= r: continue # trying to add a catch for "short data" - some data file/experiments dont go as long as others
            temp2_y_data = temp_y_data_2[r] 
            temp_mean.append(temp2_y_data[1])
            temp_stdev.append(temp2_y_data[2])
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):    
        plt.scatter(pHs, y_data_mean[i], color = plot_colors[i], s=5, alpha=0.75, label = plot_label[i])
        plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        # plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" neg single lmfit tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_single_lmfit_trends_folder[0], "neg_single_fit_tau_vs_pH_vs_time"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all chuncks single neg tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    r = 0
    for r in range(time_steps):
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)):
            temp_y_data_list = lmfit_single_exp_10min_windows_master[j]
            temp_y_data = temp_y_data_list[1] # this chooses pos fast, pos slow, neg fast, neg slow
            temp_y_data_2 = temp_y_data[0]
            if len(temp_y_data_2) <= r: continue # trying to add a catch for "short data" - some data file/experiments dont go as long as others
            temp2_y_data = temp_y_data_2[r] 
            temp_mean.append(temp2_y_data[1])
            temp_stdev.append(temp2_y_data[2])
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):    
        plt.scatter(pHs, y_data_mean[i], color = plot_colors[i], s=5, alpha=0.75, label = plot_label[i])
        # plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        # plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" neg single lmfit tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_single_lmfit_trends_folder[0], "neg_single_fit_tau_vs_pH_vs_time-no-error"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    return(print('done plotting single lmfit global trends'))


"""13. Ploting lmfit Double Exp fit Cap Spikes and lmfits Fits"""
def plotting_lmfit_double_exp_caps_and_fits(lmfit_parameters, lmfit_cap_varieables, fit_log_master, raw_current, acquisition_rate, raw_current_data_seen, path_to_save, analysis_folder_name, plots_folder_name, lmfit_double_fit_plots_folder_name, PLOT_DPI):
    """
        This function plots the double exponential lmfit fit and the raw_current data to that the fit can be visually evaluated
        Saves the plots to the (lmfit_double_fit_plots_folder_name) folder that was made earlier
        
        * for help with writing program : lmfit_cap_var_temp = [m, k, h, lmfit_y_data, fitting_x_data, cap_index, cap_data, timing_index] *
        
        Updated: BS - 01/11/2022
        Updated: BS - 02/21/2022 - added the fit data to the legend of the plots 
    """
    for i in range(len(lmfit_cap_varieables)):
        index_data = lmfit_cap_varieables[i]
        # double_fit_data = index_data[5]
        raw_current_y_index = index_data[7]
        raw_current_y_start = raw_current_y_index - raw_current_data_seen
        raw_current_y_end = raw_current_y_start + acquisition_rate
        raw_current_y_data = abs(raw_current[raw_current_y_start:raw_current_y_end])
        raw_current_x_data = x_data_index_master[raw_current_y_start:raw_current_y_end]
        
        Scalar_a = int(index_data[0])
        Tau_k1 = int(index_data[1])
        Scalar_b = int(index_data[2])
        Tau_k2 = int(index_data[3])
        Intercept = int(index_data[4])
        
        all_exp_fit_x_data = index_data[11]
        
        double_exp_fit_y_data = index_data[5]
        first_exp_fit_y_data = index_data[9]
        second_exp_fit_y_data = index_data[10]
        intercept = index_data[4]
        
        plot_x_lower_bound = index_data[7] - raw_current_data_seen
        plot_x_upper_bound = (index_data[7] + (acquisition_rate*0.05))
        
        fig = plt.figure(figsize=(8,4))
        ax = plt.gca()
        ax.scatter(raw_current_x_data, raw_current_y_data, color = 'k', s = 2, label="data", zorder = 1)
        ax.plot(all_exp_fit_x_data, first_exp_fit_y_data, '--', color = 'b', label="fit1", linewidth = 1)
        ax.plot(all_exp_fit_x_data, second_exp_fit_y_data, '--', color = 'g', label="fit2", linewidth = 1)
        ax.plot(all_exp_fit_x_data, double_exp_fit_y_data, '--', color = 'r', label="double fit", linewidth = 1)
        ax.hlines(y=intercept, xmin = plot_x_lower_bound, xmax = plot_x_upper_bound, linewidth=1, color='m', linestyles='--', label = "intercept")
        ax.set_title(f" double fit #{i} ", size=12, weight='bold') #Title
        ax.set_xlabel('Datapoints')
        ax.set_ylabel('Current (pA)')
        ax.legend()
        anchored_text = AnchoredText(f"scalar a = {Scalar_a}\n tau 1 = {Tau_k1}\n scalar b = {Scalar_b}\n tau 2 = {Tau_k2}\n intercept = {Intercept}", loc=2)
        ax.add_artist(anchored_text)
        # ax.text(0.5, 0.5, f"scalar a = {Scalar_a}\n tau 1 = {Tau_k1}\n scalar b = {Scalar_b}\n tau 2 = {Tau_k2}\n intercept = {Intercept}", fontsize=12)
        ax.set_xlim(plot_x_lower_bound,plot_x_upper_bound)
        
        plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_double_fit_plots_folder_name[0], f"lmfit_double_exp_fit_#{i}"), dpi = PLOT_DPI, bbox_inches = 'tight')
        plt.close()
        
    return(print('done plotting double exp lmfit fits'))


"""14. plotting lmfit Double Exp fit Cap Spike Variables"""
def plotting_lmfit_double_fit_parameters(lmfit_parameters, ratios, path_to_save, analysis_folder_name, plots_folder_name, lmfit_fit_vals_plots_folder_name, PLOT_DPI):
    """
        This function plots all the fit variables from the double exponential lmfit analysis and saves them in the (lmfit_double_fit_vals_plots_folder_name) flder that was made earlier
        
        Updated: BS - 01/11/2022
    """
    
    master = lmfit_parameters[0]
    pos_fast_tau = master[0]
    neg_fast_tau = master[4]
    pos_fast_ab = master[1]
    neg_fast_ab = master[5]
    pos_slow_tau = master[2]
    neg_slow_tau = master[6]
    pos_slow_ab = master[3]
    neg_slow_ab = master[7]
    pos_intercept_y_data = master[8]
    neg_intercept_y_data = master[9]
    
    pos_fast_tau_x_data = np.linspace(0, len(pos_fast_tau), num = len(pos_fast_tau), endpoint = True)
    pos_fast_ab_x_data = np.linspace(0, len(pos_fast_ab), num = len(pos_fast_ab), endpoint = True)
    neg_fast_tau_x_data = np.linspace(0, len(neg_fast_tau), num = len(neg_fast_tau), endpoint = True)
    neg_fast_ab_x_data = np.linspace(0, len(neg_fast_ab), num = len(neg_fast_ab), endpoint = True)
    pos_slow_tau_x_data = np.linspace(0, len(pos_slow_tau), num = len(pos_slow_tau), endpoint = True)
    pos_slow_ab_x_data = np.linspace(0, len(pos_slow_ab), num = len(pos_slow_ab), endpoint = True)
    neg_slow_tau_x_data = np.linspace(0, len(neg_slow_tau), num = len(neg_slow_tau), endpoint = True)
    neg_slow_ab_x_data = np.linspace(0, len(neg_slow_ab), num = len(neg_slow_ab), endpoint = True)
    pos_intercept_x_data = np.linspace(0, len(pos_intercept_y_data), num = len(pos_intercept_y_data), endpoint = True)
    neg_intercept_x_data = np.linspace(0, len(neg_intercept_y_data), num = len(neg_intercept_y_data), endpoint = True)
    
    # pos fast Tau
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(pos_fast_tau_x_data, pos_fast_tau, color = 'b', s = 1.5)
    ax.set_title(" pos double lmfit fast Tau ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tau (s)')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_fit_vals_plots_folder_name[0], "pos_double_lmfit_fast_tau"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # pos fast ab
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(pos_fast_ab_x_data, pos_fast_ab, color = 'b', s = 1.5)
    ax.set_title(" pos double lmfit fast ab ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ab')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_fit_vals_plots_folder_name[0], "pos_double_lmfit_fast_ab"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # pos slow Tau
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(pos_slow_tau_x_data, pos_slow_tau, color = 'b', s = 1.5)
    ax.set_title(" pos double lmfit slow Tau ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tau (s)')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_fit_vals_plots_folder_name[0], "pos_double_lmfit_slow_tau"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # pos slow ab
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(pos_slow_ab_x_data, pos_slow_ab, color = 'b', s = 1.5)
    ax.set_title(" pos double lmfit slow ab ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ab')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_fit_vals_plots_folder_name[0], "pos_double_lmfit_slow_ab"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all pos variable
    fig, ax1 = plt.subplots(1, figsize=(8,4))
    ax2 = ax1.twinx()
    ax1.scatter(pos_fast_tau_x_data, pos_fast_tau, color = 'green', s = 1.5, alpha = 0.75, label = "fast tau")
    ax2.scatter(pos_fast_ab_x_data, pos_fast_ab, color = 'lime', s = 1.5, alpha = 0.75,label = "fast ab")
    ax1.scatter(pos_slow_tau_x_data, pos_slow_tau, color = 'b', s = 1.5, alpha = 0.75, label = "slow tau")
    ax2.scatter(pos_slow_ab_x_data, pos_slow_ab, color = 'cornflowerblue', s = 1.5, alpha = 0.75,label = "slow ab")
    ax1.scatter(pos_intercept_x_data, pos_intercept_y_data, color = 'r', s = 1, alpha = 0.75,label = "intercept")
    ax1.set_title(" all pos double lmfit variables ", size=12, weight='bold') #Title
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Time (s)')
    ax2.set_ylabel('ab')
    fig.legend(loc="upper right")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_fit_vals_plots_folder_name[0], "all_pos_lmfit_double_fit_vars"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # neg fast Tau
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(neg_fast_tau_x_data, neg_fast_tau, color = 'b', s = 1.5)
    ax.set_title(" neg double lmfit fast Tau ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tau (s)')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_fit_vals_plots_folder_name[0], "neg_double_lmfit_fast_tau"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # neg fast ab
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(neg_fast_ab_x_data, neg_fast_ab, color = 'b', s = 1.5)
    ax.set_title(" neg double lmfit fast ab ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ab')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_fit_vals_plots_folder_name[0], "neg_double_lmfit_fast_ab"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # neg slow Tau
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(neg_slow_tau_x_data, neg_slow_tau, color = 'b', s = 1.5)
    ax.set_title(" neg double lmfit slow Tau ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tau (s)')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_fit_vals_plots_folder_name[0], "neg_double_lmfit_slow_tau"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # neg slow ab
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.scatter(neg_slow_ab_x_data, neg_slow_ab, color = 'b', s = 1.5)
    ax.set_title(" neg double lmfit slow ab ", size=12, weight='bold') #Title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ab')
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_fit_vals_plots_folder_name[0], "neg_double_lmfit_slow_ab"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all neg variable
    fig, ax1 = plt.subplots(1, figsize=(8,4))
    ax2 = ax1.twinx()
    ax1.scatter(neg_fast_tau_x_data, neg_fast_tau, color = 'green', s = 1.5, alpha = 0.75, label = "fast tau")
    ax2.scatter(neg_fast_ab_x_data, neg_fast_ab, color = 'lime', s = 1.5, alpha = 0.75, label = "fast ab")
    ax1.scatter(neg_slow_tau_x_data, neg_slow_tau, color = 'b', s = 1.5, alpha = 0.75, label = "slow tau")
    ax2.scatter(neg_slow_ab_x_data, neg_slow_ab, color = 'cornflowerblue', s = 1.5, alpha = 0.75, label = "slow ab")
    ax1.scatter(neg_intercept_x_data, neg_intercept_y_data, color = 'r', s = 1, alpha = 0.75, label = "intercept")
    ax1.set_title(" all neg double lmfit variables ", size=12, weight='bold') #Title
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Time (s)')
    ax2.set_ylabel('ab')
    fig.legend(loc="upper right")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_fit_vals_plots_folder_name[0], "all_neg_lmfit_double_fit_vars"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    pos_tau_ratios = ratios[0]
    pos_ab_ratios = ratios[1]
    neg_tau_ratios = ratios[2]
    neg_ab_ratios = ratios[3]
    
    pos_tau_ratio_x_data = np.linspace(0, len(pos_tau_ratios), num = len(pos_tau_ratios), endpoint = True)
    pos_ab_ratio_x_data = np.linspace(0, len(pos_ab_ratios), num = len(pos_ab_ratios), endpoint = True)
    neg_tau_ratio_x_data = np.linspace(0, len(neg_tau_ratios), num = len(neg_tau_ratios), endpoint = True)
    neg_ab_rayio_x_data = np.linspace(0, len(neg_ab_ratios), num = len(neg_ab_ratios), endpoint = True)
    
    # all ratios on one plot
    fig, ax1 = plt.subplots(1, figsize=(8,4))
    ax2 = ax1.twinx()
    ax1.scatter(pos_tau_ratio_x_data, pos_tau_ratios, color = 'green', s = 1.5, alpha = 0.75, label = "pos tau")
    ax2.scatter(pos_ab_ratio_x_data, pos_ab_ratios, color = 'y', s = 1.5, alpha = 0.75, label = "pos ab")
    ax1.scatter(neg_tau_ratio_x_data, neg_tau_ratios, color = 'b', s = 1.5, alpha = 0.75, label = "neg tau")
    ax2.scatter(neg_ab_rayio_x_data, neg_ab_ratios, color = 'cornflowerblue', s = 1.5, alpha = 0.75, label = "neg ab")
    ax1.scatter(pos_intercept_x_data, pos_intercept_y_data, color = 'r', s = 1, alpha = 0.75, label = "pos intercept")
    ax1.scatter(neg_intercept_x_data, neg_intercept_y_data, color = 'k', s = 1, alpha = 0.75, label = "neg intercept")
    ax1.set_title(" all lmfit variables ratios ", size=12, weight='bold') #Title
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('fast_tau/slow_tau')
    ax2.set_ylabel('fast_ab/slow_ab')
    fig.legend(loc="upper right")
    
    plt.savefig(os.path.join(path_to_save, analysis_folder_name[0], plots_folder_name[0], lmfit_fit_vals_plots_folder_name[0], "all_lmfit_double_fit_var_ratios"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    return(print('done plotting double exp lmfit fit variables'))


"""15. Potting total lmfit Double Exp fit Tau trends"""
def plotting_total_double_lmfit_tau_trends(lmfit_double_exp_10min_windows_master, pHs, path_to_save, analysis_folder_name, plots_folder_name, fit_vals_plots_folder_name, file_tag, time_steps, PLOT_DPI):
    """
        This function plots the trends from the double exponential lmfit fitting time constants (fast tau, and slow tau) across all of the files that were analyzed and saves them in (fit_vals_plots_folder_name)
        
        Updated: BS - 01/11/2022
        Updated: BS - 02/09/2022 - changed the number of itterations is the second loop in each plotting section to be determined by the (time_steps) not the (pHs)
    """
    global_double_lmfit_trends_folder = [f"global_double_lmfit_trends_{file_tag}"]
    
    try:
        os.mkdir(os.path.join(path_to_save, global_double_lmfit_trends_folder[0]))    # makes a folder to save everything in (master folder)
    except OSError as error:
        print(error)
    
    plot_colors = ["k", "b", "g", "r", "m", "y"]
    plot_colors = ["k", "dimgray", "lightgray", "rosybrown", "indianred", "brown", "maroon", "red", "tomato", "coral", "orange", "sienna", "chocolate", "peru", "darkorange", "tan", "darkgoldenrod", "gold", "khaki", "darkkhaki", "olive", "yellow", "yellowgreen", "darkolivegreen", "chartreuse", "darkseagreen", "palegreen", "limegreen", "green", "lime", "springgreen", "aquamarine", "turquoise", "lightseagreen", "darkslategray", "darkcyan", "cyan", "deepskyblue", "lightskyblue", "dodgerblue", "cornflowerblue", "midnightblue", "blue", "slateblue", "darkslateblue", "rebeccapurple", "indigo", "darkorchid", "mediumorchid", "thistle", "plum", "violet", "purple", "fuchsia", "orchid", "mediumvioletred", "deeppink", "hotpink", "palevioletred", "crimson", "lightcoral", "indianred", "firebrick", "darkred", "tomato", "coral", "sienna", "bisque", "tan", "orange", "darkgoldenrod", "gold", "darkkhaki", "olive", "olivedrab", "darkolivegreen", "lawngreen", "forestgreen", "lime", "springgreen", "mediumspringgreen", "aquamarine"]
    plot_label = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65"]
    
    global y_data_mean
    global y_data_stdev
    global i
    
    # all chuncks double lmfit pos fast tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    for r in range(time_steps): # cycles through time
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)): # cycles pH
            temp_y_data_pH = lmfit_double_exp_10min_windows_master[j]# this chooses pH
            temp_y_data_tau = temp_y_data_pH[0] # this chooses pos fast, pos slow, neg fast, neg slow
            temp_y_data_time = temp_y_data_tau[r] # this chooses the time step
            temp_y_data_mean = temp_y_data_time[1]
            temp_y_data_stdev = temp_y_data_time[2]
            temp_mean.append(temp_y_data_mean)
            temp_stdev.append(temp_y_data_stdev)
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):    
        plt.scatter(pHs, y_data_mean[i], s=5, alpha=0.75, color = plot_colors[i], label = plot_label[i])
        plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" pos double lmfit fast tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_double_lmfit_trends_folder[0], "pos_double_lmfit_fast_tau_vs_pH_vs_time"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all chuncks double lmfit pos fast tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    for r in range(time_steps):
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)):
            temp_y_data_pH = lmfit_double_exp_10min_windows_master[j]# this chooses pH
            temp_y_data_tau = temp_y_data_pH[0] # this chooses pos fast, pos slow, neg fast, neg slow
            temp_y_data_time = temp_y_data_tau[r] # this chooses the time step
            temp_y_data_mean = temp_y_data_time[1]
            temp_y_data_stdev = temp_y_data_time[2]
            temp_mean.append(temp_y_data_mean)
            temp_stdev.append(temp_y_data_stdev)
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):    
        plt.scatter(pHs, y_data_mean[i], s=5, alpha=0.75, color = plot_colors[i], label = plot_label[i])
        # plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" pos double lmfit fast tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_double_lmfit_trends_folder[0], "pos_double_lmfit_fast_tau_vs_pH_vs_time-no-error"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    
    # all chuncks double pos slow tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    for r in range(time_steps):
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)):
            temp_y_data_pH = lmfit_double_exp_10min_windows_master[j]# this chooses pH
            temp_y_data_tau = temp_y_data_pH[1] # this chooses pos fast, pos slow, neg fast, neg slow
            temp_y_data_time = temp_y_data_tau[r] # this chooses the time step
            temp_y_data_mean = temp_y_data_time[1]
            temp_y_data_stdev = temp_y_data_time[2]
            temp_mean.append(temp_y_data_mean)
            temp_stdev.append(temp_y_data_stdev)
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):    
        plt.scatter(pHs, y_data_mean[i], color = plot_colors[i], s=5, alpha=0.75, label = plot_label[i])
        plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" pos double fit slow tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_double_lmfit_trends_folder[0], "pos_double_fit_slow_tau_vs_pH_vs_time"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all chuncks double pos slow tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    for r in range(time_steps):
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)):
            temp_y_data_pH = lmfit_double_exp_10min_windows_master[j]# this chooses pH
            temp_y_data_tau = temp_y_data_pH[1] # this chooses pos fast, pos slow, neg fast, neg slow
            temp_y_data_time = temp_y_data_tau[r] # this chooses the time step
            temp_y_data_mean = temp_y_data_time[1]
            temp_y_data_stdev = temp_y_data_time[2]
            temp_mean.append(temp_y_data_mean)
            temp_stdev.append(temp_y_data_stdev)
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):    
        plt.scatter(pHs, y_data_mean[i], color = plot_colors[i], s=5, alpha=0.75, label = plot_label[i])
        # plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ls = "None", ecolor = plot_colors[i], elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" pos double fit slow tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_double_lmfit_trends_folder[0], "pos_double_fit_slow_tau_vs_pH_vs_time-no-error"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all chuncks double neg fast tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    for r in range(time_steps):
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)):
            temp_y_data_pH = lmfit_double_exp_10min_windows_master[j]# this chooses pH
            temp_y_data_tau = temp_y_data_pH[2] # this chooses pos fast, pos slow, neg fast, neg slow
            temp_y_data_time = temp_y_data_tau[r] # this chooses the time step
            temp_y_data_mean = temp_y_data_time[1]
            temp_y_data_stdev = temp_y_data_time[2]
            temp_mean.append(temp_y_data_mean)
            temp_stdev.append(temp_y_data_stdev)
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):    
        plt.scatter(pHs, y_data_mean[i], color = plot_colors[i], s=5, alpha=0.75, label = plot_label[i])
        plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ecolor = plot_colors[i], ls = "None", elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" neg double fit fast tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_double_lmfit_trends_folder[0], "neg_double_fit_fast_tau_vs_pH_vs_time"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all chuncks double neg fast tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    for r in range(time_steps):
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)):
            temp_y_data_pH = lmfit_double_exp_10min_windows_master[j]# this chooses pH
            temp_y_data_tau = temp_y_data_pH[2] # this chooses pos fast, pos slow, neg fast, neg slow
            temp_y_data_time = temp_y_data_tau[r] # this chooses the time step
            temp_y_data_mean = temp_y_data_time[1]
            temp_y_data_stdev = temp_y_data_time[2]
            temp_mean.append(temp_y_data_mean)
            temp_stdev.append(temp_y_data_stdev)
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):    
        plt.scatter(pHs, y_data_mean[i], color = plot_colors[i], s=5, alpha=0.75, label = plot_label[i])
        # plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ecolor = plot_colors[i], ls = "None", elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" neg double fit fast tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_double_lmfit_trends_folder[0], "neg_double_fit_fast_tau_vs_pH_vs_time-no-error"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all chuncks double neg slow tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    for r in range(time_steps):
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)):
            temp_y_data_pH = lmfit_double_exp_10min_windows_master[j]# this chooses pH
            temp_y_data_tau = temp_y_data_pH[3] # this chooses pos fast, pos slow, neg fast, neg slow
            temp_y_data_time = temp_y_data_tau[r] # this chooses the time step
            temp_y_data_mean = temp_y_data_time[1]
            temp_y_data_stdev = temp_y_data_time[2]
            temp_mean.append(temp_y_data_mean)
            temp_stdev.append(temp_y_data_stdev)
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):
        plt.scatter(pHs, y_data_mean[i], color = plot_colors[i], s=5, alpha=0.75, label = plot_label[i])
        plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ecolor = plot_colors[i], ls = "None", elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" neg double fit slow tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_double_lmfit_trends_folder[0], "neg_double_fit_slow_tau_vs_pH_vs_time"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    # all chuncks double neg slow tau trends
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    y_data_mean = []
    y_data_stdev = []
    for r in range(time_steps):
        temp_mean = []
        temp_stdev = []
        for j in range(len(pHs)):
            temp_y_data_pH = lmfit_double_exp_10min_windows_master[j]# this chooses pH
            temp_y_data_tau = temp_y_data_pH[3] # this chooses pos fast, pos slow, neg fast, neg slow
            temp_y_data_time = temp_y_data_tau[r] # this chooses the time step
            temp_y_data_mean = temp_y_data_time[1]
            temp_y_data_stdev = temp_y_data_time[2]
            temp_mean.append(temp_y_data_mean)
            temp_stdev.append(temp_y_data_stdev)
        y_data_mean.append(temp_mean)
        y_data_stdev.append(temp_stdev)  
    for i in range(time_steps):    
        plt.scatter(pHs, y_data_mean[i], color = plot_colors[i], s=5, alpha=0.75, label = plot_label[i])
        # plt.errorbar(pHs, y_data_mean[i], yerr = y_data_stdev[i], ecolor = plot_colors[i], ls = "None", elinewidth = 0.75, capsize = 3, capthick = 0.75 , zorder = 0)
        plt.plot(pHs, y_data_mean[i], color = plot_colors[i], alpha=0.75, linewidth = 1)
    ax.set_title(" neg double fit slow tau vs time vs pH ", size=12, weight='bold') #Title
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (s)')
    ax.legend()
    
    plt.savefig(os.path.join(path_to_save, global_double_lmfit_trends_folder[0], "neg_double_fit_slow_tau_vs_pH_vs_time-no-error"), dpi = PLOT_DPI, bbox_inches = 'tight')
    plt.close()
    
    return(print('done plotting lmfit global trends'))


"""21. Saving double lmfit fit info"""
def saving_double_lmfit_fitting_data(path_to_save, analysis_folder_name, npy_file_folder_name, lmfit_parameters, lmfit_cap_varieables, double_lmfit_log_master, lmfit_cap_varieables_master):
    
    analysis_title = files_to_analyze[i]
    # np.savetxt(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"lmfit_parameters_{analysis_title}.txt"), lmfit_parameters, delimiter = ', ', fmt='%s')
    # np.savetxt(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"lmfit_cap_varieables_{analysis_title}.txt"), lmfit_cap_varieables, delimiter = ', ', fmt='%s')
    np.savetxt(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"double_lmfit_fit_log_master_{analysis_title}.txt"), double_lmfit_log_master, delimiter = ', ', fmt='%s')
    # np.savetxt(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"lmfit_cap_varieables_master_{analysis_title}.txt"), lmfit_cap_varieables_master, delimiter = ', ', fmt='%s')
    
    np.save(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"lmfit_parameters_{analysis_title}.npy"), lmfit_parameters)
    np.save(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"lmfit_cap_varieables_{analysis_title}.npy"), lmfit_cap_varieables)
    np.save(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"double_lmfit_fit_log_master_{analysis_title}.npy"), double_lmfit_log_master)
    np.save(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"lmfit_cap_varieables_master_{analysis_title}.npy"), lmfit_cap_varieables_master)
    
    return(print('double lmfit info saved'))


"""22. Saving single lmfit fit info"""
def saving_single_lmfit_fitting_data(path_to_save, analysis_folder_name, npy_file_folder_name, lmfit_single_exp_fit_parameters, lmfit_single_exp_fit_cap_varieables, lmfit_single_exp_fit_log_master, lmfit_single_exp_fit_cap_varieables_master):
    
    
    analysis_title = files_to_analyze[i]
    # np.savetxt(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"lmfit_parameters_{analysis_title}.txt"), lmfit_parameters, delimiter = ', ', fmt='%s')
    # np.savetxt(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"lmfit_cap_varieables_{analysis_title}.txt"), lmfit_cap_varieables, delimiter = ', ', fmt='%s')
    np.savetxt(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"single_lmfit_fit_log_master_{analysis_title}.txt"), lmfit_single_exp_fit_log_master, delimiter = ', ', fmt='%s')
    # np.savetxt(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"lmfit_cap_varieables_master_{analysis_title}.txt"), lmfit_cap_varieables_master, delimiter = ', ', fmt='%s')
    
    np.save(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"single_lmfit_parameters_{analysis_title}.npy"), lmfit_single_exp_fit_parameters)
    np.save(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"single_lmfit_cap_varieables_{analysis_title}.npy"), lmfit_single_exp_fit_cap_varieables)
    np.save(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"single_lmfit_fit_log_master_{analysis_title}.npy"), lmfit_single_exp_fit_log_master)
    np.save(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"single_lmfit_cap_varieables_master_{analysis_title}.npy"), lmfit_single_exp_fit_cap_varieables_master)
    np.save(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"single_lmfit_10min_window_master_{analysis_title}.npy"), lmfit_single_exp_10min_windows_master)
    
    return(print('single lmfit info saved'))


"""23. Saving Conductance Data"""
def saving_conductance_calulations(cond_time_chunks_master, path_to_save, analysis_folder_name, npy_file_folder_name):
    # np.savetxt(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], f"all_conducntance_caluculations.txt"), lmfit_single_exp_fit_log_master, delimiter = ', ', fmt='%s')
    
    np.save(os.path.join(path_to_save, analysis_folder_name[0], npy_file_folder_name[0], "all_conducntance_caluculations.npy"), cond_time_chunks_master)
    
    return('done saving conductance stuff')
    
#%%

#....................DATA INPUT..................
"""USER INPUT REQUIRED""" # assign the path/location of data to be analyzed, the common tag of the data, and the tag at the end of the saved data folder

path = "ENTER FILE PATH"

save_path = "ENTER SAVE PATH"     # this is the path for the master folder

common_name = ("ENTER COMMON TITLE")
file_tag = ("ENTER FILE TAG TO ADD TO SAVE FOLDERS")
pH_in_label = (1, 2)         # numbers corrispond to the groups of numbers ex. BS_p130_GvpH1_pH3-51_3_1302 === (2,3) === pH 3.51
logger_name = common_name
logger_name += file_tag
PLOT_DPI = 180      

parameter_master = []
lmfit_parameters = []
lmfit_cap_varieables_master = []
lmfit_single_exp_fit_cap_varieables_master = []
lmfit_single_exp_fit_parameters = []
lmfit_double_exp_10min_windows_master = []
lmfit_single_exp_10min_windows_master = []
cond_time_chunks_master = []
end_cond_master = []
pH_time_steps = []

global_conductance_trends_folder = [f"global_conductance_trends_{file_tag}"]

try:
    os.mkdir(os.path.join(save_path, global_conductance_trends_folder[0]))    # makes a folder to save everything in (master folder)
except OSError as error:
    print(error)

plots_folder_name = ["plot_files"]
raw_data_plots_folder_name = ["raw_data_plots"]
noise_plot_folder_name = ["noise_plots"]
double_fit_plots_folder_name = ["double_fit_plots"]
fit_vals_plots_folder_name = ["fit_vals_plots"]
npy_file_folder_name = ["data_sets"]
lmfit_double_fit_plots_folder_name = ["lmfit_double_exp caps_and_fits"]
lmfit_double_fit_vals_plots_folder_name = ["lmfit_double_exp_fit_variables"]
lmfit_single_fit_plots_folder_name = ["lmfit_single_exp_caps_and_fits"]
lmfit_single_fit_vals_plots_folder_name = ["lmfit_single_exp_fit_variables"]

files_to_analyze, save_file_names, pHs = list_of_files(path, common_name, file_tag, pH_in_label)         # Function #1

# test = [1]
for i in range(len(files_to_analyze)):
# for i in range(len(test)):
    # global count
    count = i
    analysis_title = files_to_analyze[i]
    # logger_name = analysis_title
    raw_current, raw_voltage, x_data_index_master = open_bin_data(path, analysis_title)         # Function #2
    acquisition_rate, gain, bessel_filter = read_text_file(path, analysis_title)        # Function #3

    """USER INPUT REQUIRED""" #assign the path/location for the analysis to be saved
    save_file_folder_name = save_file_names[i]
    
    make_save_folders(save_path, save_file_folder_name, plots_folder_name, raw_data_plots_folder_name, noise_plot_folder_name, double_fit_plots_folder_name, fit_vals_plots_folder_name, npy_file_folder_name, lmfit_double_fit_plots_folder_name, lmfit_double_fit_vals_plots_folder_name, lmfit_single_fit_plots_folder_name, lmfit_single_fit_vals_plots_folder_name)        # Function #4
    
    # plotting_raw_data(raw_current, raw_voltage, x_data_index_master, acquisition_rate, analysis_title, save_path, save_file_folder_name, plots_folder_name, raw_data_plots_folder_name)
    
    # plot_all_raw_data_on_one_subplot(raw_current, raw_voltage, x_data_index_master, acquisition_rate, analysis_title, save_path, save_file_folder_name, plots_folder_name, raw_data_plots_folder_name)
    
    # create_error_log_file(analysis_title, pHs, save_path, save_file_folder_name, logger_name)       # Function #5
    
    pos_time = 3        # enter amount of time spent applying positive voltege
    neg_time = 3        # enter amount of time spent applying negative voltege
    zero_time = 10      # enter amount of time spent applying zero voltege
    total_voltage_cycle_time = pos_time + neg_time + (zero_time*2)       # input the total time it takes to complete one voltage cycle: 3 + 10 + 3 + 10 = 26 seconds
    longest_run = 0       # enter which run was your longest to help with labeling, starts at 0.
    
    time_steps_seconds = 52           #used for global trend plotting, must be a multiple of 26 seconds (156 = 2.6 minutes, 312 = slightly above 5 minutes (5.2 min), 624 = slightly above ten minutes)
    
    time_steps = int(round(((len(raw_voltage)/acquisition_rate)/time_steps_seconds),0))         # used for global trend plotting
    
    pH_time_steps.append(time_steps)
    
    time_steps_in_minutes_for_legends = time_steps_seconds/60
    
    conductance_final_plot_data_seconds = 312         # input the number of seconds of data to use from the end of each pH run for the "final" conductance plot, must be a multiple of 26 seconds (312 = slightly above 5 minutes, 624 = slightly above ten minutes)
    conductance_plot_data = int(conductance_final_plot_data_seconds/total_voltage_cycle_time)
    
    voltage_switch_threshold = 5        # input value theat will be used to signal a voltage switch (example: applied_voltage +/- voltage_switch_threshold = 100 +/- 5 = (105 or 95), (5 or-5), (-95 or -105))
    
    seconds_after_spike = 2         # input the amount of time for the index to jump forward after detecting a voltage change (cap. spike) AND the location for the voltage values to use for the detection of the next switch
    dp_after_spike = (seconds_after_spike * acquisition_rate)
    
    cap_data_backstep_seconds = 0.01        # input value for the number of seconds to backstep when collecting capacitance spike current_values (for function #7)
    cap_data_backstep = (cap_data_backstep_seconds * acquisition_rate)
    
    data_per_cap_spike_seconds = 1          # input the mount of data to include fit with the exponential decay
    data_per_cap_spike = (data_per_cap_spike_seconds * acquisition_rate)
    
    cond_data_location_seconds = 1        # amount of data to use for the time chunked conductance calculations. starts where cap spike data ends and goes for this user defined duration
    cond_datapoints = (cond_data_location_seconds * acquisition_rate)
    
    current_switch_index = voltage_switch_index(raw_voltage, voltage_switch_threshold, acquisition_rate, dp_after_spike)        # Function #6
    
    cap_spikes, pos_caps, pos_caps_index, neg_caps, neg_caps_index, zero_caps, zero_caps_index, all_cond_index, pos_cond_master, neg_cond_master, zero_cond_master = parse_current_from_v_switchs(current_switch_index, raw_current, raw_voltage, acquisition_rate, cap_data_backstep, data_per_cap_spike, voltage_switch_threshold, cond_datapoints)        # Function #7

    cond_time_chunks_master, end_cond_master = conductance_calculation(all_cond_index, pos_cond_master, neg_cond_master, zero_cond_master, raw_current, raw_voltage, cond_datapoints, time_steps, conductance_plot_data)
    
    # plotting_all_applied_voltage_and_current(pos_caps_index, neg_caps_index, zero_caps_index, raw_current, raw_voltage, acquisition_rate, pos_time, neg_time, zero_time, analysis_title, save_path, save_file_folder_name, plots_folder_name, raw_data_plots_folder_name)
    
    """USER INPUT REQUIRED""" # assign the number of points to remove in the current data starting from the voltage switch index, do this to handle repeating values when there is an overload during the cap. spike
    fit_method = ('lm') # lm = Levenberg-Marquardt algorithm through scipy.optimize.leastsq
    fit_offset = 50        # number of datapoints to move forward after voltage switch signal before fitting. (when cap spikes overload this is needed so that the oveload is not included in the fit)
    number_of_fit_iteration = 1600
    
    # lmfit_parameters, lmfit_cap_varieables, double_lmfit_log_master, lmfit_cap_varieables_master, lmfit_double_exp_10min_windows_master, ratios = fitting_cap_spikes_w_lmfit_double_exp(pos_caps_index, neg_caps_index, raw_current, acquisition_rate, fit_offset, time_steps, data_per_cap_spike)          # Function #8
    
    # lmfit_single_exp_fit_parameters, lmfit_single_exp_fit_cap_varieables, lmfit_single_exp_fit_log_master, lmfit_single_exp_fit_cap_varieables_master, lmfit_single_exp_10min_windows_master = fitting_cap_spikes_w_lmfit_single_exp(pos_caps_index, neg_caps_index, raw_current, acquisition_rate, fit_offset, time_steps)         # Function #9
    
    """USER INPUT REQUIRED""" 
    raw_current_data_seen = 100         # assign the number of data points you want to see befor the capacitance spike (for plotting only)
    
    # plotting_lmfit_double_exp_caps_and_fits(lmfit_parameters, lmfit_cap_varieables, double_lmfit_log_master, raw_current, acquisition_rate, raw_current_data_seen, save_path, save_file_folder_name, plots_folder_name, lmfit_double_fit_plots_folder_name, PLOT_DPI)         # Function #13

    # plotting_lmfit_double_fit_parameters(lmfit_parameters, ratios, save_path, save_file_folder_name, plots_folder_name, lmfit_double_fit_vals_plots_folder_name, PLOT_DPI)        # Function #14

    # plotting_lmfit_single_exp_caps_and_fits(lmfit_single_exp_fit_cap_varieables, raw_current, acquisition_rate, raw_current_data_seen, save_path, save_file_folder_name, plots_folder_name, lmfit_single_fit_plots_folder_name, PLOT_DPI)         # Function #10

    # plotting_lmfit_single_fit_parameters(lmfit_single_exp_fit_parameters, save_path, save_file_folder_name, plots_folder_name, lmfit_single_fit_vals_plots_folder_name, total_voltage_cycle_time, PLOT_DPI)         # Function #11
    
    # saving_double_lmfit_fitting_data(save_path, save_file_folder_name, npy_file_folder_name, lmfit_parameters, lmfit_cap_varieables, double_lmfit_log_master, lmfit_cap_varieables_master)      # Function #21
                                    
    # saving_single_lmfit_fitting_data(save_path, save_file_folder_name, npy_file_folder_name, lmfit_single_exp_fit_parameters, lmfit_single_exp_fit_cap_varieables, lmfit_single_exp_fit_log_master, lmfit_single_exp_fit_cap_varieables_master)      # Function #22
    
# plotting_total_double_lmfit_tau_trends(lmfit_double_exp_10min_windows_master, pHs, save_path, save_file_folder_name, plots_folder_name, fit_vals_plots_folder_name, file_tag, time_steps, PLOT_DPI)       # Function #15

# plotting_total_single_lmfit_tau_trends(lmfit_single_exp_10min_windows_master, pHs, save_path, save_file_folder_name, plots_folder_name, fit_vals_plots_folder_name, file_tag, time_steps, PLOT_DPI)       # Function #12

plotting_global_conductance_trends(cond_time_chunks_master, pHs, time_steps_in_minutes_for_legends, total_voltage_cycle_time, longest_run, save_path, save_file_folder_name, plots_folder_name, global_conductance_trends_folder, PLOT_DPI)

plotting_the_final_G_v_pH(pHs, end_cond_master, save_path, save_file_folder_name, plots_folder_name, global_conductance_trends_folder, PLOT_DPI)

saving_conductance_calulations(cond_time_chunks_master, save_path, save_file_folder_name, npy_file_folder_name)

#%%






















check_y = (raw_current[0:-1])
check_x_temp = np.linspace(1, len(check_y), num = len(check_y), endpoint = True)
check_x = [x / acquisition_rate for x in check_x_temp]

# mean_y = window_mean_master[0:-1]
# mean_x = window_mean_index_master[0:-1]

# small_mean_y = small_window_mean_master[0:-1]
# small_mean_x = small_window_index_master[0:-1]

# state_0_events_y_data = state_0_event_current_data[0:-1]
# state_0_events_x_data = state_0_event_index_data[0:-1]

# state_1_events_y_data = state_1_event_current_data[0:-1]
# state_1_events_x_data = state_1_event_index_data[0:-1]


fig = plt.figure(figsize=(15,10))
plt.scatter(check_x, check_y, s = 2, zorder = 0, alpha = 0.75)
# plt.scatter(mean_x, mean_y, c = 'k', s=0.75, zorder = 2)
# plt.scatter(small_mean_x, small_mean_y, c = 'y', s=0.5, zorder = 1)
# plt.scatter(state_0_events_x_data, state_0_events_y_data, c = 'r', s=1, zorder = 1)
# plt.scatter(state_1_events_x_data, state_1_events_y_data, c = 'r', s=1, zorder = 1)
ax = plt.gca()
ax.set_xlabel('Seconds')
ax.set_ylabel('Current (pA)')
# ax.set_title(f'initial_window_size = {initial_window_size}\nwindow_size_limit = {window_size_limit}\nwindow_stdev_limit = {window_stdev_limit}\nbaseline_focusing_val = {baseline_focusing_val}')
plt.ylim(-200,22500)
plt.xlim(9.98,10.1)
# plt.savefig(os.path.join(path_to_save, final_analysis_folder_name[0], plots_file_folder[0], "1_min"), dpi = PLOT_DPI, bbox_inches = 'tight')
# plt.close()






















