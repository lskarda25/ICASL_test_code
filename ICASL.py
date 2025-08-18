# Collection of functions for plotting, testing, and working with multidimensional data. 
# Use the outline in the bottom left (VScode) to navigate functions. Press "Collapse all" in top right of that menu.

######### Importing #########
# Copy/paste the functions you want, or import the file into your code and write ICASL.example_function() to call them.
# Two ways to import:
# 1) Move into the same directory as your code and write "import ICASL" at top
# 2) Move into any higher directory than your code and copy/paste the commented out section of code below. I'll likely
#    place this file somewhere high up in the Sharepoint, or properly publish as a python package

# Copy, paste, and uncomment (ctrl+/) the following into a python/jupyter file to load these functions from any higher 
# directory that contains ICASL.py
###
# Loads ICASL.py from a higher directory
# import os
# import importlib.util
# quiet=True
# current_dir = "."
# target_file = "ICASL.py"
# target_name = "ICASL"
# while True:
#     if not quiet: print(current_dir)
#     # Check if ICASL is in the current directory
#     ICASL_path = os.path.join(current_dir, target_file)
#     if not quiet: print(ICASL_path)
#     if os.path.isfile(ICASL_path):
#         # Loads ICASL.py
#         spec = importlib.util.spec_from_file_location(target_name, ICASL_path)
#         ICASL = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(ICASL)
#         break
#     else:
#         # Move one level up
#         parent_dir = os.path.join(current_dir, "..")
#         if not quiet: print(parent_dir)
#         if current_dir == parent_dir:  # Reached the root directory
#             raise ValueError("ICASL.py not found")
#         else:
#             current_dir = parent_dir
# ICASL.test()
###

######### Summary of Functions #########
# arange() - performs numpy.arange(), but fixes many of the bugs and stupid behavior in it.
# read_csv() - performs pandas.read_csv(), but adjusts for utf-16 encoding when needed.

### Plotting ###
# start_plot() - Applies default plotting settings. Styles built-in to choose from, can add more.
# add_legend_text() - Adds an annotation to the bottom of the legend. Clean/organized way to add multiple of them.
# finish_plot() - Saves, shows, closes, colors lines, applies legend styling, adds annotations.

### Calcs ###
# limit_data() - Removes dataframe data outside of a high and/or low range for a particular column
# tran_peak(), tran_settle(), tran_start() - find useful points in transient response (can be used for settling time and delta_V)
# find_closest() - find closest value in a list
# convert_from_decibels()
# calc_gain_margin()
# calc_phase_margin()
# calc_gain_bandwidth()
# calc_crossover_freq()

### Presentation ###
# scale() - Scales a list of nums such that the highest is between 1-999 (useful for Engineering Notation)
# scale_by() - performs scale(), but uses a different list as a reference for determining precision. Consistency across multiple lines.
# prefix() - Returns the corresponding Engineering Notation prefix, when provided an unscaled list
# set_precision() - Pass an element, and a list it belongs to. Returns element with proper trailing zeros. Makes file names sort better.

### Testing ###
# execute_cells_by_tag() - Allows for running cells in groups. (Can't call, must copy/paste)
# generate_experiment_name() - Makes folder name for experiment. Attribute-based. Adds [No_Plots] attribute by default.
# rename_dir_once_plotted() - Removes attribute (such as [No_Plots])

### Reading in data ###
# There are two types of csv input formats we often need to plot. I refer to them in this code as:
# 1) Tree (Tests). Branching directories, each named for a constant value of a sweep variable (temp=100), which end in 
# multiple csv files, also named for a constant value (VREF=.8), which contain 2-column data within.
# 2) Cadence (Sims). Virtuoso's default csv format is to stack 2 column data horizontally into one big file, embedding the 
#    constant values for each simulation within the metadata of the header cells. Harder to write code for.
# The rest of these functions are designed to provide either conversion or dual functionality between these two formats.
# Whether using them is worth the learning curve is debatable. AI is already good enough to do most of our plotting for us.
# They are:
# reformat() - Converts a cadence csv into a tree structure of 2-column csv files
# read_cadence_csv() - (USEFUL!!!) Reads a cadence-style csv into a tree of dictionaries, an easy data structure for plotting
# read_tree_csv() - Reads a tree of directories into a tree of dictionaries
# read_tree_names() - Reads names of constants that were swept (temp, VREF) from a tree of directories. Makes assumptions.
# read_cadence_metadata_names() - Reads names of constants that were swept (temp, VREF) from a cadence csv's metadata
# read_tree_values() - Reads swept values of constants from directory tree
# read_cadence_values() - Reads swept values of constants from cadence csv file

# Most other functions are simply meant to be used internally by other functions. Don't bother looking at all of these.

import matplotlib.pyplot as plt   # Plotting graphs and visualizing data
import matplotlib.offsetbox as ob # An internal Matplotlib data structure. Used for cleaner annotations.
import numpy as np                # Numerical operations, particularly with arrays
import os                         # Interact with the operating system, such as handling file paths
import pandas as pd               # Intuitive labeled 2D data
import os                         # Interface with Operating System
import re                         # Regular expressions for text searching
import math                       # Does math
import collections                # I use this to check whether input to a function is a collection (list,array,etc.)
import matplotlib.legend as lg    # I use this to check whether input to a function is a matplotlib legend
from datetime import date         # Read current date

def test():
    print("ICASL is loaded!")

# Performs numpy's arange() function (which generates a list of numbers from a start, stop, and step)
# The returned list will include the 'stop' value, which numpy usually excludes
def arange(start, stop, step, adjustment=1e-12):
    # Adding one more step size makes the stop value inclusive. 
    # Adding 1e-12 accounts for exclusion due to numpy's imprecision. (See next comments)
    listA = np.arange(start, stop+step+adjustment, step)
    # Rounds to a far out decimal place. Because of how floats are stored, np.arange could return bizarre values otherwise. 
    # i.e. 2 -> 1.99999999999999995
    listA = [float(round(element, 12)) for element in listA]
    return listA

# Adjusts for utf-16 encoding when needed
def read_csv(path):
    try:
        df = pd.read_csv(path, encoding='utf-8')
        return df
    except UnicodeDecodeError:
        pass
    try:
        df = pd.read_csv(path, encoding='utf-16')
        return df
    except UnicodeDecodeError:
        print("ICASL.read_csv() may not yet account for this encoding scheme.")
        raise UnicodeDecodeError()

# Apply default parameters that are shared by all plots. Any property can be changed afterwards if needed.
def start_plot(title, xlabel, ylabel, style="a,c", cm_num=13):
    fig, ax = plt.subplots(layout='constrained')
    
    # Text and tick settings
    if ("b" in style): # Bigger & bolder text
        ax.set_title(title, fontdict={'fontsize': 16, 'fontweight': 'bold'}, y = 1.03)
        ax.set_xlabel(xlabel, fontdict={'fontsize': 12})
        ax.set_ylabel(ylabel, fontdict={'fontsize': 12})
        ax.tick_params(axis='both', which='major', labelsize=10)
    else:
        # Matplotlib defaults
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Color settings
    # Finish_plot() does this as well, with requiring a cm_num. But that function's not as intuitive to use overall.
    if ("c" in style):
        cm=plt.get_cmap('gist_rainbow')
        ax.set_prop_cycle('color', cm(np.linspace(0, 1, cm_num)))

    # Removes margins on left/right side - lines no longer stop abruptly before edge of plot. Not always desired.
    if ("m" in style):
        pass
    else:
        ax.margins(x=0)

    if ("ng" in style):
        pass
    else:
        ax.grid()
    
    return fig, ax

# Adds an annotation into the bottom of the legend. Centered text, legend box will expand if text is large.
def add_legend_text(legend, text):
    if (not isinstance(text, str)):
        print("Error: Legend annotation was passed non-string value: {text}. Not adding to plot.")
        return legend
    txt = ob.TextArea(text)
    box = legend._legend_box
    box.get_children().append(txt)
    box.set_figure(legend.get_figure(root=False))
    return legend

# In finish_plot(): fig and ax are POSITIONAL arguments - when calling a function, you must insert those first and 
# in the same order. All arguments that are defined with an "=" are KEYWORD arguments - when calling the function, 
# you write out their name, an '=', then the value for each.
# Keyword arguments are optional. If not provided, the default values in the definition below will be assumed.

# This is all the code that can only run after a plot is filled. 
# Honestly, this one's a bit messy (despite best effort). Might not be worth learning/using?
# Adding colors afterwards is nice - don't need to know the number beforehand. Completely abstracted away.
# Legend_style is searched for 1-2 character sequences to determine how to construct the legend. Need to add delimiter eventually.
def finish_plot(fig, ax, save_dir="none", save_file="none", dpi=100, cm=plt.get_cmap('gist_rainbow'), 
                legend=None, legend_style="#", annotations=None, close=True, show=False):
    
    # Check if there's multiple axes (left and right side)
    multi_ax = False
    if isinstance(ax, collections.abc.Iterable):
        multi_ax = True
        axes = ax # Rename to be more sensical

    # Sets line color. Reads number of lines on its own
    if cm != None:
        if multi_ax == False:
            lines = ax.lines
            colors = cm(np.linspace(0, 1, len(lines)))
            for line, color in zip(lines, colors):
                line.set_color(color)
        else:
            all_lines = []
            for a in axes:
                for line in a.lines:
                    all_lines.append(line)
            colors = cm(np.linspace(0, 1, len(all_lines)))
            for line, color in zip(all_lines, colors):
                line.set_color(color)

    if legend_style == None:
        raise ValueError("legend_style cannot be NoneType. Use 'n' to avoid all legend behavior.")
    
    # Legend styling. Adds a legend if you don't pass one to this function.
    if ((legend != None or legend_style != "#" or len(lines) > 1) and "n" not in legend_style):
        if (legend == None):
            # Determine location
            if "b2" in legend_style: bbox=(1.15, 1) # box on the right side
            elif "b" in legend_style: bbox=(1.02, 1) # box on the right side, further right (for second axis)
            else: bbox = None

            if "ur" in legend_style: loc = "upper right"
            elif "ul" in legend_style: loc = "upper left"
            elif "lr" in legend_style: loc = "lower right"
            elif "ll" in legend_style: loc = "lower left"

            # Set legend
            if multi_ax == False:
                if bbox == None and loc == None: legend = ax.legend()
                elif bbox == None and loc != None: legend = ax.legend(loc=loc)
                elif bbox != None and loc == None: legend = ax.legend(bbox_to_anchor=bbox)
            else:
                all_lines = []
                all_labels = []
                for a in axes:
                    lines, labels = a.get_legend_handles_labels()
                    for line in lines:
                        all_lines.append(line)
                    for label in labels:
                        all_labels.append(label)
                if bbox == None: legend = axes[0].legend(all_lines, all_labels)
                else: legend = axes[0].legend(all_lines, all_labels, bbox_to_anchor=bbox)
                if bbox == None and loc == None: legend = axes[0].legend(all_lines, all_labels)
                elif bbox == None and loc != None: legend = axes[0].legend(all_lines, all_labels, loc=loc)
                elif bbox != None and loc == None: legend = axes[0].legend(all_lines, all_labels, bbox_to_anchor=bbox)
                
        elif (not isinstance(legend, lg.Legend)):
            raise ValueError(f"legend is of invalid type {type(legend)}. It should be the object that ax.legend() returns.")
        # Style legend
        if (legend_style == "#"): # If you pass your legend but not a style, I default to this one
            legend_style = "kc"
        if ("k" in legend_style): # Thickens widths of the legend's example lines, NOT the actual lines on the plot
            for legend_line in legend.get_lines(): 
                legend_line.set_linewidth(2.5)
        if ("c" in legend_style): # Colors legend lines the same as actual lines
            if cm != None:
                for legend_line, color in zip(legend.get_lines(), colors):
                    legend_line.set_color(color)
    
    # Adds annotations to legend (an organized way to place them, little funky though.)
    if (annotations != None and legend != None):
        if isinstance(annotations, str):
            legend = add_legend_text(legend, annotations)
        elif isinstance(annotations, collections.abc.Iterable):
            for annotation in annotations:
                legend=add_legend_text(legend, annotation)
        else:
            print("Error: Improper type for 'annotations'. Should be either a string or a collection of strings.")
    
    # Saves and/or shows
    if (save_dir == "None" and save_file != "None"):
        print("Warning: You specified a save_file but not a save_dir. Figure not saved.")
        show = True
    if (save_dir != "None" and save_file == "None"):
        print("Warning: You specified a save_dir but not a save_file. Figure not saved.")
        show = True
    elif (save_dir != "None" and save_file != "None"):
        os.makedirs(save_dir, exist_ok=True)
        if os.path.isdir(save_dir):
            fig.savefig(os.path.join(save_dir, save_file), dpi=dpi)
        else:
            raise ValueError("Directory provided ({save_dir}) does not exist. Figure not saved.")
    else:
        show = True # If you don't save the figure, shows plot regardless of if you ask it to
    if (show):
        plt.show()
    
    # Closes
    if (close):
        plt.close(fig) # Saves memory, gets rid of warnings.

##########################################################################################################################
##################################################### Plotting Calcs #####################################################
##########################################################################################################################

# Limits the data of a dataframe. Good to use before calcs. Also, when plotting after data that's been passed through this, 
# matplotlib will adjust its y to fit your new data, which it won't always do using the intended xlimit function.
def limit_data(df, column_name, start=None, end=None):
    # This honestly doesn't need to be a function, barely saves any lines of code. I just don't like the pandas syntax.
    if start == None and end == None:
        print("Warning: No reason to use limit data with no start or end limit")
    if start != None:
        df = df[df[column_name] > start]
    if end != None:
        df = df[df[column_name] < end]
    return df

# Determines coordinates of peak
# Limit data to only show one transient step response before calling
def tran_peak(times, values, settling_direction, log=False):
    if (settling_direction.lower() in ["up", "rising"]): peak_value = min(values)
    elif (settling_direction.lower() in ["down", "falling"]): peak_value = max(values)
    peak_time = times[np.where(values == peak_value)[0]][0]
    if log:
        print(f"Peak Time: {peak_time}")
        print(f"Peak Value: {peak_value}")
    return peak_time, peak_value

# Determines coordinates where transient is officially settled
# Limit data to only show one transient step response before calling
def tran_settle(times, values, peak_value, percent_settling_threshold, log=False):
    settled_value = np.mean(values[-3:-1])
    threshold = abs(peak_value*percent_settling_threshold/100)
    # These are the indices of every piece of data whose y value is not within the settled threshold
    unsettled_indices = np.where(np.logical_or(values > settled_value+threshold, values < settled_value-threshold))[0]
    if len(unsettled_indices) == 0: 
        print("Error: Signal is always settled.")
        settled_time = times[0]
    else:
        last_unsettled_time = times[unsettled_indices[-1]]
        if log: print(f"Last Unsettled Time: {last_unsettled_time}")
        if times[-1] != last_unsettled_time:
            settled_time = times[unsettled_indices[-1]+1] # If your data range is too narrow and signal is oscillating, this may run while unsettled.
        else:
            print("Warning: Not settled by final time")
            settled_time = times[-1]
    if log:
        print(f"Settled Value: {settled_value}")
        print(f"Unsettled Indices: {unsettled_indices}")
        print(f"Unsettled Times: {times[unsettled_indices]}")
        print(f"Unsettled Values: {values[unsettled_indices]}")
        print(f"Within Threshold Times: {np.array([x for x in times if x not in times[unsettled_indices]])}")
        print(f"Within Threshold Values: {np.array([x for x in values if x not in values[unsettled_indices]])}")
        print(f"Settled Time: {settled_time}")
    return settled_time, settled_value

# Determines coordinates where transient first begins
# Limit data to only show one transient step response before calling
# Works best with interpolation. Won't work at all if you have less than [std_width] points in the starting, settled region of your data.
def tran_start(times, values, log=False, std_width=10):
    # I thought it'd be fun to detect the start of transients (although uneccesary, usually already known) -- I trace through each time step, 
    # finding the standard deviation shortly before and after, then take a ratio of those two stds. This ratio is within 20 and .05 everywhere 
    # except around start of the transient, where it spikes to the 1,000's (with interpolation, stays within 5 and .2 and spikes to 1,000,000s)
    # To find the exact point where the transient begins, I find the maximum drop in this ratio from one time step to the next. Works so far.
    std_ratios = []
    drops = []
    if (len(values) < std_width*2+1):
        print(f"Error: only {len(values)} data points. Cannot compute tran_start calculation.")
        starting_time = 0
        starting_value = 0
    else:
        for i, value, time in zip(range(len(values)), values, times):
            if (i < std_width or i+std_width >= len(values)): std_ratios.append(1) ; drops.append(1) ; continue
            if (values[i-std_width] == values[i-std_width+1]): std_ratios.append(1) ; drops.append(1) ; continue
            if (values[i+std_width] == values[i+std_width-1]): std_ratios.append(1) ; drops.append(1) ; continue
            past_std = np.std(values[i-std_width:i])
            future_std = np.std(values[i:i+std_width])
            std_ratio = future_std/past_std
            std_ratios.append(std_ratio)
            drops.append(std_ratios[i-1]/std_ratios[i])
            if log: print(f"Index {i} : Time {time}: Value {value} : Ratio {future_std/past_std}")
        starting_time = times[drops.index(max(drops))-2]
        starting_value = values[np.where(times == starting_time)[0]][0]
    if log:
        print(f"Starting Value: {starting_value}")
        print(f"Starting Time: {starting_time}")
    return starting_time, starting_value

def find_closest(listA, target):
    return min(listA, key=lambda x: abs(x - target)) # Minimum of difference between a value and every value in a list -- finds closest value

def convert_from_decibels(dB):
    return 10**(dB/20)

# Assumes gains and phases are perfectly aligned -- would need to use frequency as a middleman otherwise
def calc_gain_margin(gains, phases, log=False):
    inverted_phase = find_closest(phases, phases[0]-180)
    gain_reference = gains[phases == inverted_phase]
    gain_margin = 0-gain_reference
    if log: print(f"Gain Margin: {gain_margin}")
    return gain_margin

# Assumes gains and phases are perfectly aligned
def calc_phase_margin(gains, phases, log=False):
    zero_gain = find_closest(gains, 0)
    phase_reference = phases[gains == zero_gain]-phases[0]
    phase_margin = (phase_reference+180)
    if log: print(f"Phase Margin: {phase_margin}")
    return phase_margin

# Provide a frequency solidly on the one pole rolloff
def calc_gain_bandwidth(freqs, gains, test_frequency, log=False):
    freq = find_closest(freqs, test_frequency)
    GB = convert_from_decibels(gains[freqs == freq]) * freq
    if log: print(f"Gain Bandwidth: {GB} (Hz)")
    return GB

# This might always be the preferred method of finding GB - not sure.
def calc_crossover_freq(freqs, gains):
    zero_gain = find_closest(gains, 0)
    crossover_freq = freqs[gains == zero_gain]
    return crossover_freq

###########################################################################################################################
################################################## Plotting Presentation ##################################################
###########################################################################################################################

# Determine the appropriate scaling for generated plots in Engineering Notation
def scale(list):
    array = np.asarray(list, dtype='float') # Python's lists don't support multiplication by a float. Numpy's arrays do.
    max = np.max(np.abs(array))
    k = 10**(int(math.floor(math.log10(abs(max)) / 3) * 3))
    return array/k # Return the scaled array

# Scales a list, but uses a larger list as a reference for how much to scale by. Useful for multiple scaled lines on one plot.
def scale_by(list, reference):
    array = np.asarray(reference, dtype='float') # Python's lists don't support multiplication by a float. Numpy's arrays do.
    max = np.max(np.abs(array))
    k = 10**(int(math.floor(math.log10(abs(max)) / 3) * 3))
    array2 = np.asarray(list, dtype='float')
    return array2/k # Return the scaled array

# Determine the appropriate prefix for arrays that use scale(). 
# PASS THE ORIGINAL LIST, NOT THE SCALED
def prefix(list):
    array = np.asarray(list, dtype='float')
    max = np.max(np.abs(array))
    if max < 1e-18:
        print("Prefix not implemented")
        prefix = "?"
    elif max < 1e-15: prefix = 'a'
    elif max < 1e-12: prefix = 'f'
    elif max < 1e-9:  prefix = 'p'
    elif max < 1e-6:  prefix = 'n'
    elif max < 1e-3:  prefix = 'µ'
    elif max < 1:     prefix = 'm'
    elif max < 1e3:   prefix = ''
    elif max < 1e6:   prefix = 'k'
    elif max < 1e9:   prefix = 'M'
    elif max < 1e12:  prefix = 'G'
    elif max < 1e15:  prefix = "T"
    else:
        print("Prefix not implemented")
        prefix = "?"
    return prefix

# Determines and returns the highest precision of all elements
# Accepts ints, floats, or string equivalents, either as single elements or within a collection
def determine_precision(input):
    if isinstance(input, collections.abc.Iterable) and not isinstance(input, str):
        # Round to remove float imprecision (occurs around 16th digit for doubles)
        listA = [round(float(element), 13) for element in input]
        precisions = []
        for element in listA:
            element_str = str(element).rstrip('0') # Removes trailing zeros
            if '.' in element_str:
                digits_after_decimal = len(element_str.split('.')[1])
                precisions.append(digits_after_decimal)
            else:
                precisions.append(0)
        return max(precisions)
    # Also works for a single element
    elif isinstance(input, int) or isinstance(input, float) or isinstance(input, str):
        element = float(input)
        element_str = str(element).rstrip('0') # Removes trailing zeros
        if '.' in element_str:
            return len(element_str.split('.')[1]) # Number of digits after '.'
        else:
            return 0
    else:
        raise ValueError(f"Unsupported Type: {type(input)}")

# Determines the highest precision of all elements in a list, returns specified element as a string with that precision. 
# Adds trailing zeros. This avoids poorly sorted and inconsistent file names. 
def set_precision(imprecise_element, list):
    imprecise_element = float(imprecise_element)
    return f"{imprecise_element:.{determine_precision(list)}f}"

##########################################################################################################################
###################################################### Experimental ######################################################
##########################################################################################################################

def execute_cells_by_tag(group_tag):
    # Placeholder for outline. Actual code is below. 
    pass

# This runs every cell in its notebook that has the specified tag. Allows for groupings.
# You'll need to copy and paste this in, can't call it from another file.
# Turn on File->'Auto Save' if you want to use this. Reads cell data from disk, so otherwise there could be misalignments.
#
# import nbformat # For manipulating notebooks
# def execute_cells_by_tag(group_tag):
#     filepath = globals()['__vsc_ipynb_file__'] # Returns path to this file
#     print(filepath)
#     nb = nbformat.read(open(filepath, 'r', encoding='utf-8'), as_version=4) # Saves this notebook as JSON
#     ip = get_ipython()                                                      # Gets the "global InteractiveShell instance"
#     print(ip)
#     for cell_number in range(len(nb.cells)):
#         cell=nb.cells[cell_number]
#         #print(cell)
#         if 'tags' in cell.metadata:
#             if group_tag in cell.metadata.tags:
#                 ip.run_cell(cell.source)
#                 print(f"Ran Cell {cell_number}")
#
# All outputs will be printed here
# This code is janky and might depreciate fast
# Only other way to do this would be through javascript - I had kernal issues. I don't think VSCode supports switching between javascript and python in the middle of running code
# Or an extension, but the ones that do this are more obscure (and seem poorly written) and I don't want to download malicious code to UT's network

def generate_experiment_name(run_type, local_temps, device_name, info = ""):
    DATE =  str(date.today())
    DATE_DIR = os.path.join(f"{device_name}_Results", f"{DATE}")
    i = 1
    # Starts directory name for run (inside DEVICE_NAME/DATE/) with run_type and a number, which is generated based off existing runs.
    while (True) : 
        run_exists = False
        if os.path.isdir(DATE_DIR) == False:
            run_dir = os.path.join(DATE_DIR, f"{run_type}_{i}_")
            break
        dirs = os.listdir(DATE_DIR)
        for dir in dirs:
            if f"{run_type}_{i}" in dir:
                run_exists = True
                break
        if run_exists == False:
            run_dir = os.path.join(DATE_DIR, f"{run_type}_{i}_")
            break
        else:
            i = i+1
            continue
    descriptors = []
    # To-do: Revise this so it doesn't have to be a string
    if ((not isinstance(local_temps, collections.abc.Iterable)) or isinstance(local_temps, str)):
        descriptors.append(f"{local_temps}°C")    # Adds temperature if there's only one (such as a practice run)
    elif len(local_temps) == 1:
        descriptors.append(f"{local_temps[0]}°C")
    descriptors.append("No_Plots")
    if isinstance(info, str) and info != "":      # Adds extra info if desired
        descriptors.append(f"{info}")
    for descriptor in descriptors:
        run_dir = run_dir + f"[{descriptor}]"
    os.makedirs(run_dir, exist_ok=True)           # Makes directory
    return run_dir
    # If the directory already exists, it won't raise an error due to exist_ok=True

def rename_dir_once_plotted(run_dir, remove_attribute):
    # Changes name of run directory once some plots are in it.
    try:
        if not isinstance(remove_attribute, str): remove_attribute = str(remove_attribute)
        if remove_attribute[0] == "[": remove_attribute = remove_attribute[1:]
        if remove_attribute[-1] == "]": remove_attribute = remove_attribute[0:-1]
        if f"[{remove_attribute}]" in run_dir:
            new_run_dir = run_dir.replace(f"[{remove_attribute}]", "")
            if new_run_dir[-1] == "_":
                new_run_dir = new_run_dir[0:-1]
            os.rename(run_dir, new_run_dir)
            run_dir = new_run_dir
            return run_dir
    except PermissionError:
        # Operating System might get mad if you have the directory open and try to rename. This simply moves on if it does.
        pass

##########################################################################################################################
############### The next 15-ish functions are not that useful to know/use. Mostly behind the scenes stuff. ###############
##########################################################################################################################

# Converts ints, floats, and strings - either alone, in a collection, or in a dataframe - into either ints or floats
# depending on their precision. (all collections will convert to lists)
def convert_to_num(input):
    if isinstance(input, pd.core.frame.DataFrame):
        df = input
        for column in df.columns:
            column_array = df[column].values
            column_precision = determine_precision(column_array)
            if column_precision == 0:
                df[column] = df[column].astype(int)
            elif column_precision > 0:
                df[column] = df[column].astype(float)
        return df
    elif isinstance(input, collections.abc.Iterable) and not isinstance(input, str):
        if (determine_precision(input) == 0):
            return [int(element) for element in input]
        elif (determine_precision(input) > 0):
            return [float(element) for element in input]
    elif isinstance(input, int) or isinstance(input, float) or isinstance(input, str):
        if (determine_precision(input) == 0):
            return int(input)
        elif (determine_precision(input) > 0):
            return float(input)
    else:
        raise ValueError(f"Unsupported Type: {type(input)}")

def fix_dataframe(df):
    df = df.dropna(how='all')
    # df = df.astype(str) # Converts all to strings. Next line will throw an error if they're not. 
    # df = df[df[df.columns[0]].str.strip().astype(bool)] # Clears empty rows and whitespace rows
    # df = convert_to_num(df) # Convert back to either ints or floats - decides which.
    return df

# Sorts, removes duplicates, converts to either floats or ints depending on precision, rounds off imprecision.
def clean_list(listA, quiet=True):
    if not quiet: print(f"Cleaning list of nums: {listA}")
    listA = [element for element in listA if element is not None] # Remove None Type
    listA = convert_to_num(listA) # Convert to int/float
    listA = list(set(listA)) # Removes duplicates
    listA = sorted(listA) # Sorts
    listA = [round(element, 12) for element in listA]
    return listA

# Similar to clean_list(), but designed for string inputs and string outputs.
def clean_list_strings(listA, quiet=True):
    if not quiet: print(f"Cleaning list of strings: {listA}")
    listA = [element for element in listA if element is not None] # Remove None Type
    listA = list(set(listA)) # Removes duplicates
    listA = sorted(listA) # Sorts
    return listA

# Input: header cell of a cadence-style csv (Ex: df.columns[0])
# Output: Custom data structure of names and values
def read_cell_metadata(header_cell, quiet=True):
    metadata = re.findall(r'\((.*?)\)', header_cell) # Finds all within text in parentheses, saves into list
    if metadata == []:
        if not quiet: print("Warning: Input cell is not a .csv and does not contain parenthesis. Returning empty list.")
        return []
    if len(metadata) > 1:
        if not quiet: print("Warning: Multiple sets of parentheses found within metadata. Used the first.")
    constants = re.split(r',\s*', metadata[0]) # Splits first text in parenthesis using comma as delimiter
    split_constants = []
    for constant in constants:
        constant_dict = {}
        constant_dict["name"] = re.split(r'=\s*', constant)[0]
        constant_dict["value"] = convert_to_num(re.split(r'=\s*', constant)[1])
        split_constants.append(constant_dict)
    if not quiet: print(f"Read constants from a header cell:{split_constants}")
    return split_constants # Returns ints or floats, not strings

# Input: Cadence-style csv file path
# Output: Custom data structure of names and values (sourced from the first column's metadata)
def read_file_metadata(read_file, quiet=True):
    if (read_file[-4:] != ".csv"):
        if not quiet: print("Warning: File path does not point to a .csv. Returning empty list.")
        return []
    df = pd.read_csv(read_file)
    header_cell = df.columns[0]
    return read_cell_metadata(header_cell, quiet)

# Determines which of the above two functions to call by examining input. Returns same structure.
# Input: EITHER a header cell from a cadence-style csv OR the csv file path
# Output: Custom data structure of names and values
def read_cadence_metadata(header_cell_or_file, quiet=True):
    hcof = str(header_cell_or_file)
    if len(hcof) > 4 and hcof[-4:] == ".csv":
        # Input is csv file
        if not quiet: print(f"Determined input '{hcof}' is a csv file.")
        return read_file_metadata(hcof, quiet)
    else:
        if "(" not in hcof or ")" not in hcof:
            # Input is neither csv file or cell with parenthesis to extract metadata from
            if not quiet: print(f"Warning: Input '{hcof}' is not a .csv and does not contain parenthesis. " + 
                                f"Returning empty list.")
            return []
        else:
            if not quiet: print(f"Assuming input: '{hcof}' is a header cell.")
            return read_cell_metadata(hcof, quiet)

# Input: EITHER a header cell OR the csv file path
# Output: List of names (no values)
def read_cadence_metadata_names(header_cell, quiet=True):
    split_constants = read_cadence_metadata(header_cell, quiet)
    names = []
    for constant in split_constants:
        names.append(constant["name"])
    if not quiet: print(f"Extracted names: {names} from metadata: {split_constants}")
    return names
    
# Input: EITHER a header cell OR the csv file path
# Output: Prints out names of variables in metadata out, with each on a new line. Returns nothing.
def print_cadence_metadata_names(header_cell_or_file, quiet=True):
    names = read_cadence_metadata_names(header_cell_or_file, quiet)
    for name in names:
        print(name)

# Input: string in a form similar to "temp=100". Needs identification_string (i.e. "temp")
# Output: The following number. (i.e. 100)
def parse_num_from_string(full_string, identification_string, quiet=True):
    if f"{identification_string}" in full_string:
        # Removes everything before (and through) the equals sign
        str = full_string[full_string.find(f"{identification_string}")+len(identification_string)+1:] 
        # Pulls out int or float from start of the remaining string.
        num = re.match(r'[+-]?(([0-9]+\.?[0-9]*)|(\.[0-9]+))([eE][+-]?\d+)?', str) 
        if num == None:
            if not quiet: print(f"Parsed {full_string} with ID {identification_string}. Found no number after.")
            return(None)
        else:
            if not quiet: print(f"Parsed {full_string} with ID {identification_string}. Found {num.group()}.")
            return convert_to_num(num.group())
    else:
         if not quiet: print(f"Parsed {full_string}. It does not contain ID {identification_string}.")

# Input: string in a form similar to "temp=100". 
# Output: All text that comes before the last number. (i.e. "temp")
def parse_constant_name_from_string(full_string):
    num = re.search(r'[+-]?(([0-9]+\.?[0-9]*)|(\.[0-9]+))([eE][+-]?\d+)?', full_string) # Pulls out int or float
    if num == None:
        return(None)
    else:
        # last_num = num.group(len(num.group())-1)
        # index = full_string.rfind(last_num)
        # trimmed_string = full_string[:index] # Remove last number
        trimmed_string = full_string
        index = -1
        num_parts = ['0','1','2','3','4','5','6','7','8','9','.','-','+','e', 'E']
        # Moves index backwards from end until it sees part of a number
        while (True):
            if trimmed_string[index] not in num_parts:
                index = index-1
            else:
                break
        # Moves index backwards through number until it sees a character that's not numerical
        while (True):
            if trimmed_string[index] in num_parts:
                index = index-1
            else:
                break
        # Removes last string composed of chars from num_parts, and everything after it.
        trimmed_string = trimmed_string[:index+1]
        trimmed_string = trimmed_string.replace("=", "") # Remove ALL equals signs
        if (trimmed_string[-1] == '_'): # Remove last character, if it is an underline
            trimmed_string = trimmed_string[:-1]
        return trimmed_string

# Reads a tree directory structure, returns the names of the constants varied within it (as well as the csv file names
# with the constant removed)
# There's no guaranteed way to know this (without perfect, standardized directory creation). I simply chose the most
# likely options, loosely following (not exclusively) the naming scheme we've been using thus far.
# If this doesn't succeed, you may need different or cleaner directory structures or names
# If we start using double underscores in file names between test names and the last swept variable, split_strategy 'c'
# should always succeed.
# It follows the last branch it reads each time it delves deeper.
def read_tree_names(read_dir_path, split_strategy="default", quiet=True):
    current_dir = read_dir_path
    names = []
    file_identifiers = []
    while(True):
        name_counts = {}
        # Find names that precede numbers in file names throughout directory. Count quantity of each
        relevant_names = [file_name for file_name in os.listdir(current_dir) 
                          if (os.path.isdir(os.path.join(current_dir, file_name)) or file_name[-4:] == ".csv")]
        if (quiet == False): print(f"Relevant names: {relevant_names}")
        if relevant_names == []:
            raise ValueError(f"{current_dir} was read as a branch in a tree of data, but did not " +
                             f"contain any directories or csv's within.")
        for file_name in relevant_names:
            name = parse_constant_name_from_string(file_name)
            if name == None:
                continue
            else:
                if name not in name_counts:
                    name_counts[name] = 1
                else:
                    name_counts[name] = name_counts[name]+1
        if (quiet == False): print(f"Name counts: {name_counts}")
        # Find the name that appears most
        blacklist = []
        while(True):
            first = True
            repeat = False
            for name in name_counts.keys():
                if name in blacklist:
                    continue
                if first:
                    max_val = name_counts[name]
                    max_name = name
                elif name_counts[name] > max_val:
                    max_val = name_counts[name]
                    max_name = name
            for i in range(len(max_name)):
                if max_name[i:] in names:
                    repeat = True
            if repeat == True:
                print(f"Warning: {max_name} was detected in two separate depths of tree. Using next most likely " +
                      f"constant name at lower depth.")
                blacklist.append(max_name)
                continue
            elif repeat == False:
                break
        if (quiet == False): print(f"Highest count name: {max_name}. Its count: {max_val}")
        # Find if there's more csvs or more directories that have that name
        csv_amount = 0
        dir_amount = 0
        for file_name in relevant_names:
            if max_name in file_name:
                if file_name[-4:] == ".csv":
                    csv_amount = csv_amount + 1
                elif os.path.isdir(os.path.join(current_dir, file_name)):
                    dir_amount = dir_amount + 1
                    save_dir = os.path.join(current_dir, file_name)
        if (quiet == False): print(f"CSVs:{csv_amount}. Dirs: {dir_amount}")
        if csv_amount >= dir_amount:
            # Assume we're on the last layer
            # Need to seperate file names from constant name
            index = 0
            while (True):
                # Compare last characters of names. Move index backwards, looking at larger substrings at end of each.
                index = index-1
                trimmed_occurances = 0
                for name in name_counts.keys():
                    if len(max_name) < index*-1:
                        # Ran out of name length. No notable drop in similarity. Assume no file names, just the constant.
                        constant_name = max_name
                        names.append(constant_name)
                        if (quiet == False): print(f"Names: {names}")
                        if (quiet == False): print(f"File_identifiers: {file_identifiers}")
                        return names, file_identifiers
                    if len(name) < index*-1:
                        continue
                    if name[index:] == max_name[index:]:
                        trimmed_occurances = trimmed_occurances + name_counts[name]
                if (quiet == False): print(f"Matches: {trimmed_occurances}")
                if (index == -1):
                    highest_trimmed_occurances = trimmed_occurances
                if (highest_trimmed_occurances-trimmed_occurances >= max_val): 
                    # Drop in similarity as great as the most commonly seen file name.
                    # Assume this is the boundary between constant name and test/sim-specific name
                    constant_name = max_name[index+1:]
                    if (split_strategy == "default"):
                        if (max_name.find("__") == -1):
                            split_strategy = "b" # Default if there's no double underscore
                        else:
                            split_strategy = "c" # Default if there is a double underscore
                    if not quiet: print(f"Applying split strategy {split_strategy}")
                    if (split_strategy == "a"):
                        # Assume all identical text is part of the variable name
                        pass
                    elif (split_strategy == "b"):
                        # Assume the variable name is separated by an underscore, and has no underscores in the middle.
                        if not quiet: print(f"Found _ at index {constant_name.rfind("_")}")
                        new_index = (len(constant_name)-constant_name.rfind("_"))*-1
                        if new_index > index:
                            constant_name = constant_name[new_index:]
                    elif (split_strategy == "c"):
                        # Assume the variable name is separated by a double underscore, 
                        # and has no double underscores in the middle.
                        new_index = (len(constant_name)-constant_name.rfind("__"))*-1
                        if new_index > index:
                            constant_name = constant_name[new_index:]
                    if constant_name[0] == "_":
                        if not quiet: print("Removing underscore at start")
                        constant_name = constant_name[1:]
                    names.append(constant_name)
                    for name in relevant_names:
                        if constant_name in name and name[-4:] == ".csv":
                            parsed = parse_constant_name_from_string(name)
                            if parsed != None:
                                file_identifiers.append(parsed[:index+1])
                    file_identifiers = clean_list_strings(file_identifiers, quiet=quiet)
                    for i,file in enumerate(file_identifiers):
                        if file[-1] == "_":
                            file_identifiers[i] = file[:-1]
                    if (quiet == False): print(f"Sufficient matches lost at index: {index}. Returning:")
                    if (quiet == False): print(f"Names: {names}")
                    if (quiet == False): print(f"File_identifiers: {file_identifiers}")
                    return names, file_identifiers
        else:
            names.append(max_name)
            if (quiet == False): print(f"Names so far: {names}")
            current_dir = save_dir # Pick random directory (last one, actually) to enter
            continue
# Man that one was worse than I thought - definitely not worth the time. Oh well.

# Reads a directory's contents and generates a list of numbers (swept values) from the names of its files/folders. 
# Must have a "identification_string=" before each value to match with.
def read_dir_values(read_dir_path, identification_string):
    listA = []
    for filename in os.listdir(read_dir_path):
        #print(filename)
        num = parse_num_from_string(filename, identification_string)
        if num is not None:
            listA.append(num)
    if (len(listA) == 0):
        print(f"Found no files/directories in {read_dir_path} containing {identification_string} " +
              f"followed by an int or float")
        return []
    return clean_list(listA) # converts to ints/float, sorts, filters

# Given list of constants (in proper order), navigates a tree of directories and returns the swept values of those 
# constants (as a dict). Follows the last branch read from each directory,
def read_tree_values(read_dir_path, constant_names=None, quiet=True, split_strategy="default"):
    current_dir = read_dir_path
    values = {}
    if (constant_names==None):
        constant_names, _ = read_tree_names(read_dir_path, quiet=quiet, split_strategy=split_strategy)
    if (isinstance(constant_names, str)):
        constant_names = [constant_names]
    for constant_name in constant_names:
        listA = []
        for file_name in os.listdir(current_dir):
            num = parse_num_from_string(file_name, constant_name)
            if not quiet: print(num)
            if num is not None:
                listA.append(num)
                save_path = os.path.join(current_dir, file_name)
        if (len(listA) == 0):
            return None
        values[constant_name] = clean_list(listA, quiet=quiet) # converts to ints/float, sorts, filters
        current_dir = save_path
    return values
    
# Reads a cadence generated csv and generates a list of swept values from the metadata of each column
# Must have an "identification_string=" before each value to match with.
def read_cadence_values(read_csv_path, identification_string):
    if identification_string[-1] == '=':
        identification_string[:-1] # Remove '='
    df = pd.read_csv(read_csv_path)
    listA = []
    for i in range(0, len(df.columns)-1, 2):
        md = read_cell_metadata(df.columns[i])
        if identification_string not in md:
            print("Identification_string not found in this cell")
        else:
            listA.append(md[identification_string])
    return clean_list(listA) # converts to ints/float, sorts, filters

# Reads cadence metadata, compares to a pre-defined list of constants. Sorts/filters to be equivalent to those defined 
# constants and produces a helpful blurb of text if something doesn't match.
# Input: cadence-style csv's header cell, a list of constants for comparison, and details for outputting helpful error
def match_metadata_to_defined_constants(header_cell, defined_constants, error_read_file_path, error_column_num, quiet=True):
    constants = read_cadence_metadata(header_cell, quiet)
    #print(f"Constants: {constants}")
    constant_names = read_cadence_metadata_names(header_cell, quiet)
    for constant in defined_constants:
        if constant not in constant_names:
            print(f"\nERROR: The string \"{constant}\" from constant_names_in_order was not found in the " + 
                  f"header cell of column {error_column_num} for the following file: \n{error_read_file_path}")
            print(f"The constant names in that cell are:")
            print_cadence_metadata_names(error_read_file_path, quiet)
            raise ValueError("See above")
    filtered_constants = [constant for constant in constants if constant["name"] in defined_constants]
    sorted_constants = sorted(filtered_constants, key=lambda constant: defined_constants.index(constant["name"]))
    #print(f"Sorted/filtered: {filtered_constants}")
    return sorted_constants

# Strips metadata out of Cadence column name
def remove_text_in_parentheses(string):
    trimmed_string = re.sub(r'\(.*?\)', '', string)
    trimmed_string = re.sub(r'  ', '_', trimmed_string)
    return trimmed_string

#############################################################################
###################### These next three can be useful #######################
#############################################################################

# Converts a cadence-style csv into a tree of directories terminated with 2 column csvs.
def reformat(write_dir_name, read_file_path, write_file_name, constant_names_in_order, 
             new_constant_names_in_order="default", new_x_label="default", new_y_label="default", quiet=True):
    df = pd.read_csv(read_file_path)
    working_dir = os.path.dirname(read_file_path)
    write_dir = os.path.join(working_dir, write_dir_name)
    os.makedirs(write_dir, exist_ok=True)

    # One name makes more sense in this code, one makes more sense when setting inputs
    defined_constants = constant_names_in_order

    # Convert new names from list to dictionary
    new_constant_names = {}
    for i in range(len(new_constant_names_in_order)):
        new_constant_names[defined_constants[i]] = new_constant_names_in_order[i]

    # If no new names are specified, use defaults
    if new_x_label == "default":
        new_x_label = df.columns[0]
    if new_y_label == "default":
        new_y_label = df.columns[1]
    if new_constant_names_in_order == "default":
        new_constant_names_in_order = defined_constants

    # This is so I can set the precision later. Some error-checking.
    all_vals = {}
    for constant in defined_constants:
        all_vals[constant] = []
    for i in range(0, len(df.columns)-1, 2):
        col1, col2 = df.columns[i], df.columns[i + 1]
        #print(f"\nProcessing columns: {col1}, {col2}")

        if (read_cell_metadata(col1) != read_cell_metadata(col2)):
            print("ERROR: X and Y metadata do not match")
        if (df[col1].size != df[col2].size):
            print("ERROR: X and Y columns differ in length")

        filtered_constants = match_metadata_to_defined_constants(col1, defined_constants, read_file_path, i, quiet)
        for constant in filtered_constants:
            all_vals[constant["name"]].append(constant["value"])  

    # Produces csv file hierarchy 
    for i in range(0, len(df.columns)-1, 2): # Iterate through columns in steps of 2
        filtered_constants = match_metadata_to_defined_constants(df.columns[i], defined_constants, read_file_path, i, quiet)
        write_current = write_dir
        for constant in filtered_constants:
            value = set_precision(float(constant['value']), all_vals[constant['name']])
            name = new_constant_names[constant['name']]
            if constant["name"] != filtered_constants[-1]["name"]: # if not last
                # Build directory
                write_current = os.path.join(write_current, f"{name}={value}")
            else: 
                # Last in hierarchy, so we'll make the actual file
                os.makedirs(write_current, exist_ok=True)
                filename = f"{write_file_name}_{name}={value}.csv"
                write_current = os.path.join(write_current, filename)
                new_df = df.iloc[:, [i, i+1]]
                new_df.columns = [new_x_label, new_y_label]
                new_df = new_df.astype(str) # Converts all to strings. Next line will throw an error if they're not. 
                new_df = new_df[new_df[new_df.columns[0]].str.strip().astype(bool)] # Clears empty rows and whitespace rows
                new_df.to_csv(write_current, index=False)
                if quiet == False:
                    print(f"Writing to {write_current}")
                break

#############################################################################
########################## Especially this one!!! ###########################
#############################################################################

# Reads a cadence-style csv (sims) into a tree of dictionaries. Allows for shared sim/test plotting code.
# To-Do: Functionality for multiple csv files at final branch?
def read_cadence_csv(read_file_path, constant_names_in_order="default", new_x_label="default", 
                     new_y_label="default", quiet=True):
    df = pd.read_csv(read_file_path)
    if constant_names_in_order == "default":
        constant_names_in_order = read_cadence_metadata_names(read_file_path, quiet)
    defined_constants = constant_names_in_order
    data = {}
    all_values = {}

    if new_x_label == "default":
        new_x_label = remove_text_in_parentheses(df.columns[0])
        print(f"x-axis data label set to: '{new_x_label}'. Add argument 'new_x_label=' to choose your own name")
    if new_y_label == "default":
        new_y_label = remove_text_in_parentheses(df.columns[1])
        print(f"y-axis data label set to: '{new_y_label}'. Add argument 'new_y_label=' to choose your own name")
    for i in range(0, len(df.columns) - 1, 2): # Iterate through columns in steps of 2
        filtered_constants = match_metadata_to_defined_constants(df.columns[i], defined_constants, read_file_path, i, quiet)
        current_dict = data
        for constant in filtered_constants:
            name = constant["name"]
            value = constant["value"]
            # Update all_values
            if name not in all_values.keys():
                all_values[name] = []
            if value not in all_values[name]:
                all_values[name].append(value)
            # Build dictionary tree
            if name != filtered_constants[-1]["name"]: # if not last
                if value not in current_dict:
                    current_dict[value] = {}
                current_dict = current_dict[value]
            else: 
                # Last, so we'll store the actual 2-column dataframe
                new_df = df.iloc[:, [i, i+1]]
                if new_x_label == "default":
                    x_label = df.columns[i]
                else:
                    x_label = new_x_label
                if new_y_label == "default":
                    y_label = df.columns[i+1]
                else:
                    y_label = new_y_label
                new_df.columns = [x_label, y_label]
                new_df = new_df.astype(str) # Converts all to strings. Next line will throw an error if they're not. 
                new_df = new_df[new_df[new_df.columns[0]].str.strip().astype(bool)] # Clears empty rows and whitespace rows
                new_df = convert_to_num(new_df) # Convert back to either ints or floats - decides which.
                current_dict[constant["value"]] = new_df
                if quiet == False:
                    print(f"Storing dataframe for {current_dict}")
                    print(new_df)
                break
        all_values[name] = clean_list(all_values[name], quiet=quiet)
    i = 1
    values_list = []
    for constant_name in constant_names_in_order:
        values_list.append(all_values[constant_name])
        print(f"Layer {i} of keys for {constant_name}: {all_values[constant_name]}")
        i = i+1
    print(f"Final dataframe columns: ['{new_x_label}', '{new_y_label}']\n")
    return data, values_list

# This is considered bad practice. I don't care too much. Definitely ugly though.
suggested_names = False
suggested_file_IDs = False

# Reads a tree of directories (test data) into a tree of dictionaries. Allows for shared sim/test plotting code.
# To-Do: Functionality for multiple cadence csv files in a directory
def read_tree_csv(read_dir_path, constant_names_in_order, file_identifier="default", quiet=True):
    if file_identifier == "default":
        print("Warning: Unless there's only one .csv to choose, you need a tree_file_identifier=")
    read_tree_names(read_dir_path)

    current_dir = read_dir_path
    global suggested_names
    suggested_names = False
    global suggested_file_IDs
    suggested_file_IDs = False
    data, all_values, final_column_values = read_tree_csv_recursion(current_dir, constant_names_in_order, 
                                                                    file_identifier, quiet)
    i = 1
    values_list = []
    for constant_name in constant_names_in_order:
        values_list.append(all_values[constant_name])
        print(f"Layer of keys {i} for {constant_name}: {all_values[constant_name]}")
        i = i+1
    print(f"Final dataframe columns: {final_column_values}\n")
    return data, values_list

# Data and all_values are both built in reverse
def read_tree_csv_recursion(current_dir, remaining_names_in_order, file_identifier, quiet=True):
    final_column_values = []
    if not quiet: print(f"Remaining names in order: {remaining_names_in_order}. Type: {type(remaining_names_in_order)}")
    if isinstance(remaining_names_in_order, str):
        name = remaining_names_in_order
    elif isinstance(remaining_names_in_order, collections.abc.Iterable):
        if not quiet: print(f"Remaining names in order[0]: {remaining_names_in_order[0]}. " +
                            f"Type: {type(remaining_names_in_order[0])}")
        name = remaining_names_in_order[0]
        if not isinstance(name, str):
            raise ValueError("Collection of constant names contains a non-string value")
    else:
        raise ValueError("Constant_names format unrecognized. Must be a string or collection of strings")
    all_names_values = read_tree_values(current_dir, name)
    if (all_names_values == None):
        global suggested_names
        print(f"Warning: Found no files/directories in {current_dir} containing {name} followed by an int or float")
        if (suggested_names == False):
            print("Perhaps you meant to specify the following constant names:")
            names, tmp = read_tree_names(current_dir)
            for name in names:
                print(name)
            print("")
            suggested_names = True
        return None, None
    values = all_names_values[name]
    if not quiet: print(f"Read Values: {values}")
    current_values = {}
    current_values[name] = []
    if quiet == False: print(f"Processing variable name '{name}' in {current_dir}")
    for value in values:
        if value not in current_values[name]:
            current_values[name].append(value)
    if (len(values) == 0):
        print(f"Warning: Could not find values for {name} inside {current_dir}")
    
    # Traverse directory
    if name != remaining_names_in_order[-1]:  # if not last
        current_data = {}
        # Delve deeper
        dirs = [dir for dir in os.listdir(current_dir) if name in dir and ".csv" not in dir]
        if not quiet: print(f"All sub-dirs with {name} and .csv: {dirs}")
        if len(dirs) == 0:
            print(f"Warning: Found no child directories with {name} in the directory {current_dir}")
        error_checking_values = []
        for dir in dirs:
            value = parse_num_from_string(dir, name, quiet=quiet)
            if value is None:
                continue
            if value in error_checking_values:
                print(f"Warning: the value {value} appears alongside the constant {name} more than once in {current_dir}. "
                      + f"Output will be unpredictable")
            error_checking_values.append(value)
            new_dir = os.path.join(current_dir, dir)

            if not os.path.isdir(new_dir):
                print(f"Warning: Found something with \"{name}\" and \"{value}\" in {current_dir} that should be " +
                      f"a directory, but is not. Skipping.")
                continue
            if quiet == False: print(f"Opening {new_dir}")
            deeper_data, deeper_values, deeper_final_column_values = \
            read_tree_csv_recursion(new_dir, remaining_names_in_order[1:], file_identifier, quiet)
            if deeper_data == None and deeper_values == None:
                continue
            # Process data and all_values on way out
            current_data[value] = deeper_data 
            for key in deeper_values.keys():
                if key in current_values.keys():
                    for val in deeper_values[key]:
                        if val not in current_values[key]: 
                            current_values[key].append(val)
                else:
                    current_values[key] = deeper_values[key]
            for column in deeper_final_column_values:
                if column not in final_column_values:
                    final_column_values.append(column)
    else:
        #Read in dataframes
        current_data = {}
        relevant_files = [file for file in os.listdir(current_dir) if name in file and ".csv" in file]
        if len(relevant_files) > 1 and file_identifier == "default":
            if not quiet: print("No file identifier provided, and more than one available csv. " +
                                "Returning without reading data.")
            return None, None, None
        files = [file for file in os.listdir(current_dir) if file_identifier in file and name in file and ".csv" in file]
        if len(files) == 0:
            global suggested_file_IDs
            print(f"Warning: Found no files with {file_identifier}, {name}, and .csv in the last directory, {current_dir}")
            if (suggested_names == False):
                print("Perhaps you meant to specify the following constant names:")
                names, tmp = read_tree_names(current_dir)
                for name in names:
                    print(name)
                print("")
                suggested_names = True
            if (suggested_file_IDs == False):
                print("Perhaps you meant to specify the following file identifiers:")
                tmp, file_IDs = read_tree_names(current_dir)
                for ID in file_IDs:
                    print(ID)
                print("")
                suggested_file_IDs = True
            return None, None, None
        error_checking_values = []
        for file_name in files:
            value = parse_num_from_string(file_name, name)
            #par
            if value is None:
                continue
            if value in error_checking_values:
                print(f"Warning: the value {value} appears alongside the constant {name} and file identifier " +
                      f"{file_identifier} more than once in {current_dir}. Output will be unpredictable")
            error_checking_values.append(value)
            file_path = os.path.join(current_dir, file_name)
            if (quiet == False): print(f"Reading file: {file_path}")
            current_data[value] = pd.read_csv(file_path) 
            for column in current_data[value].columns:
                if column not in final_column_values:
                    final_column_values.append(column)
    current_values[name] = clean_list(current_values[name], quiet)
    return current_data, current_values, final_column_values