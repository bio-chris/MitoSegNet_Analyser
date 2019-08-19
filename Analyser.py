import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats.mstats import normaltest, mannwhitneyu
from Plot_Significance import significance_bar

import warnings
warnings.simplefilter("ignore", UserWarning)

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.width", None)


# Select parameters
####################

path = "/home/fischer/Desktop/finetune_testfolder/data"

"""
descriptors = ["Area", "Minor Axis Length", "Major Axis Length", "Eccentricity", "Perimeter", "Solidity", 
               "Mean Intensity", "Max Intensity", "Min Intensity"]

stat_values = ["Average", "Median", "Standard Deviation", "Standard Error", "Minimum", "Maximum", "N"]
"""

descriptor = "Area"
stat_value = "Average"

ylab = stat_value.lower() + " " + descriptor.lower()
ylab_size = 16

xlab = ["catp-6", "n2"]

tick_size = 12

####################


file_list = os.listdir(path)
dataframe = pd.DataFrame(columns=file_list)

print(file_list)
print("\n")

max_vals = []
normtest_list = []
for tables in file_list:

    table = pd.read_csv(path + os.sep + tables)

    # how to acess measurements
    meas_table = table[table["Measurement"] == descriptor]

    # how to acess statistical values
    values_list = meas_table[stat_value].tolist()

    # adding values of interest to table for visualization

    dataframe[tables] = values_list

    max_vals.append(np.max(values_list))


    if normaltest(values_list)[1] > 0.05:
        normtest = "| Parametric distribution"
        normtest_list.append(True)
    else:
        normtest = "| Non-parametric distribution"
        normtest_list.append(False)

    print(tables, normtest)

print("\n")

stat_frame = pd.DataFrame(columns=["Data 1", "Data 2", "Hypothesis test p-value", "Effect size"])

def compare_samples(stat_func):

    data1_l = []
    data2_l = []
    pval_l = []
    eff_siz_l = []

    for a, b in itertools.combinations(file_list, 2):


        pval = stat_func(dataframe[a],dataframe[b])
        eff_siz = cohens_d(dataframe[a],dataframe[b])

        data1_l.append(a)
        data2_l.append(b)
        pval_l.append(pval[1])
        eff_siz_l.append(eff_siz)

    return data1_l, data2_l, pval_l, eff_siz_l

# pooled standard deviation for calculation of effect size (cohen's d)
def cohens_d(data1, data2):

    p_std = np.sqrt(((len(data1)-1)*np.var(data1)+(len(data2)-1)*np.var(data2))/(len(data1)+len(data2)-2))

    cohens_d = np.abs(np.average(data1) - np.average(data2)) / p_std

    return cohens_d


if False in normtest_list:

    data1_l, data2_l, pval_l, eff_siz_l = compare_samples(mannwhitneyu)

else:
    data1_l, data2_l, pval_l, eff_siz_l = compare_samples(mannwhitneyu)



# table with p-values and effect sizes
########
stat_frame["Data 1"] = data1_l
stat_frame["Data 2"] = data2_l
stat_frame["Hypothesis test p-value"] = pval_l
stat_frame["Effect size"] = eff_siz_l
########

print(stat_frame)

increase = 0
for index, row in stat_frame.iterrows():


    if row["Hypothesis test p-value"] > 0.05:
        p = 0


    elif 0.01 < row["Hypothesis test p-value"] < 0.05:
        p = 1


    elif 0.001 < row["Hypothesis test p-value"] < 0.01:
        p = 2

    else:
        p = 3

    max_bar = np.max(max_vals)

    x1= file_list.index(row["Data 1"])
    x2 = file_list.index(row["Data 2"])


    significance_bar(pos_y=max_bar+0.1*max_bar+increase, pos_x=[x1, x2], bar_y=max_bar*0.05, p=p, y_dist=max_bar*0.02,
                     distance=0.05)

    increase+=max_bar*0.1


# select plot
plot = sb.boxplot(data=dataframe, color="white", fliersize=0)
sb.swarmplot(data=dataframe, color="black")


# label the y axis
plt.ylabel(ylab, fontsize=ylab_size)

# label the x axis
plt.xticks(list(range(len(xlab))), xlab)

# determine fontsize of x and y ticks
plot.tick_params(labelsize=tick_size)

plt.show()


