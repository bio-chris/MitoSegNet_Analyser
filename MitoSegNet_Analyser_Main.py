"""

Class Control

Contains all functions that are necessary for the entire program




"""

from tkinter import *
import tkinter.font
import tkinter.messagebox
import tkinter.filedialog
import os
import webbrowser
import collections
import copy
import itertools

import numpy as np
import cv2
import pandas as pd
from skimage import img_as_bool, io, color
from skimage.morphology import skeletonize
from skan import summarise
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats.mstats import normaltest, mannwhitneyu, ttest_ind
from Plot_Significance import significance_bar

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


# GUI
####################

class Control:
    """
    Adds functions that can be accessed from all windows in the program
    """

    def __init__(self):
        pass

    # close currently open window
    def close_window(self, window):

        window.destroy()

    # opens link to documentation of how to use the program
    def help(self, window):

        webbrowser.open_new("https://github.com/bio-chris/MitoSegNet")

    # open new window with specified width and height
    def new_window(self, window, title, width, height):

        window.title(title)

        window.minsize(width=int(width / 2), height=int(height / 2))
        window.geometry(str(width) + "x" + str(height) + "+0+0")

    # adds menu to every window, which contains the above functions close_window, help and go_back
    def small_menu(self, window):

        menu = Menu(window)
        window.config(menu=menu)

        submenu = Menu(menu)
        menu.add_cascade(label="Menu", menu=submenu)

        submenu.add_command(label="Help", command=lambda: self.help(window))

        # creates line to separate group items
        submenu.add_separator()

        submenu.add_command(label="Go Back", command=lambda: self.go_back(window, root))
        submenu.add_command(label="Exit", command=lambda: self.close_window(window))

    def place_text(self, window, text, x, y, height, width):

        if height is None or width is None:
            Label(window, text=text, bd=1).place(bordermode=OUTSIDE, x=x, y=y)
        else:
            Label(window, text=text, bd=1).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)

    def place_button(self, window, text, func, x, y, height, width):

        Button(window, text=text, command=func).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)

    def place_entry(self, window, text, x, y, height, width):

        Entry(window, textvariable=text).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)


class Get_Measurements(Control):

    def __init__(self):
        Control.__init__(self)

    def get_measurements_window(self):

        gm_root = Tk()

        self.new_window(gm_root, "MitoSegNet Analyser - Get measurements", 500, 380)
        self.small_menu(gm_root)

        table_name = StringVar(gm_root)
        imgpath = StringVar(gm_root)
        labpath = StringVar(gm_root)
        dirpath = StringVar(gm_root)
        #meas_var = StringVar(gm_root)

        def askopenimgs():
            set_imgpath = tkinter.filedialog.askdirectory(parent=gm_root, title='Choose a directory')
            imgpath.set(set_imgpath)

        def askopenlabels():
            set_labpath = tkinter.filedialog.askdirectory(parent=gm_root, title='Choose a directory')
            labpath.set(set_labpath)

        def askopendir():
            set_dirpath = tkinter.filedialog.askdirectory(parent=gm_root, title='Choose a directory')
            dirpath.set(set_dirpath)

        #### enter table name

        self.place_text(gm_root, "Enter name of measurements table", 15, 20, None, None)
        self.place_entry(gm_root, table_name, 25, 50, 30, 400)

        #### browse for table saving location

        self.place_text(gm_root, "Select directory in which table should be saved", 15, 90, None, None)
        self.place_button(gm_root, "Browse", askopendir, 435, 120, 30, 50)
        self.place_entry(gm_root, dirpath, 25, 120, 30, 400)

        #### browse for raw image data

        self.place_text(gm_root, "Select directory in which 8-bit raw images are stored", 15, 160, None, None)
        self.place_button(gm_root, "Browse", askopenimgs, 435, 190, 30, 50)
        self.place_entry(gm_root, imgpath, 25, 190, 30, 400)

        #### browse for labels

        self.place_text(gm_root, "Select directory in which segmented images are stored", 15, 230, None, None)
        self.place_button(gm_root, "Browse", askopenlabels, 435, 260, 30, 50)
        self.place_entry(gm_root, labpath, 25, 260, 30, 400)

        """
        meas_var.set("CSV Table")
        Label(gm_root, text="Select table datatype", bd=1).place(bordermode=OUTSIDE, x=15, y=300)
        popupMenu_train = OptionMenu(gm_root, meas_var, *set(["CSV Table", "Excel Table"]))
        popupMenu_train.place(bordermode=OUTSIDE, x=25, y=320, height=30, width=130)
        """

        def measure():

            img_list = os.listdir(imgpath.get())

            dataframe = pd.DataFrame(columns=["Image", "Measurement", "Average", "Median", "Standard Deviation",
                                              "Standard Error", "Minimum", "Maximum", "N"])

            dataframe_branch = copy.copy(dataframe)

            n = 0
            n2 = 0

            for i, img in enumerate(img_list):

                print(i, img)

                read_img = cv2.imread(imgpath.get() + os.sep + img, -1)
                read_lab = cv2.imread(labpath.get() + os.sep + img, cv2.IMREAD_GRAYSCALE)

                # skeletonize
                ##########################

                """    
                05-07-19

                for some reason the sumarise function of skan prints out a different number of objects, which is why
                i currently cannot include the branch data in the same table as the morph parameters    
                """

                read_lab_skel = img_as_bool(color.rgb2gray(io.imread(labpath.get() + os.sep + img)))
                lab_skel = skeletonize(read_lab_skel).astype("uint8")

                branch_data = summarise(lab_skel)

                # curv_ind = (branch_data["branch-distance"] - branch_data["euclidean-distance"]) / branch_data["euclidean-distance"]

                curve_ind = []
                for bd, ed in zip(branch_data["branch-distance"], branch_data["euclidean-distance"]):

                    if ed != 0.0:
                        curve_ind.append((bd - ed) / ed)
                    else:
                        curve_ind.append(bd - ed)

                branch_data["curvature-index"] = curve_ind

                grouped_branch_data_mean = branch_data.groupby(["skeleton-id"], as_index=False).mean()

                grouped_branch_data_sum = branch_data.groupby(["skeleton-id"], as_index=False).sum()

                counter = collections.Counter(branch_data["skeleton-id"])

                n_branches = []
                for i in grouped_branch_data_mean["skeleton-id"]:
                    n_branches.append(counter[i])

                branch_len = grouped_branch_data_mean["branch-distance"].tolist()
                tot_branch_len = grouped_branch_data_sum["branch-distance"].tolist()

                curv_ind = grouped_branch_data_mean["curvature-index"].tolist()

                ##########################

                labelled_img = label(read_lab)

                labelled_img_props = regionprops(label_image=labelled_img, intensity_image=read_img, coordinates='xy')

                area = [obj.area for obj in labelled_img_props]
                minor_axis_length = [obj.minor_axis_length for obj in labelled_img_props]
                major_axis_length = [obj.major_axis_length for obj in labelled_img_props]
                eccentricity = [obj.eccentricity for obj in labelled_img_props]
                perimeter = [obj.perimeter for obj in labelled_img_props]
                solidity = [obj.solidity for obj in labelled_img_props]
                mean_int = [obj.mean_intensity for obj in labelled_img_props]
                max_int = [obj.max_intensity for obj in labelled_img_props]
                min_int = [obj.min_intensity for obj in labelled_img_props]

                def add_to_dataframe(df, measure_str, measure, n):

                    df.loc[n] = [img] + [measure_str, np.average(measure), np.median(measure), np.std(measure),
                                         np.std(measure) / np.sqrt(len(measure)), np.min(measure), np.max(measure),
                                         len(measure)]

                meas_str_l = ["Area", "Minor Axis Length", "Major Axis Length", "Eccentricity", "Perimeter", "Solidity",
                              "Mean Intensity", "Max Intensity", "Min Intensity"]
                meas_l = [area, minor_axis_length, major_axis_length, eccentricity, perimeter, solidity, mean_int,
                          max_int,
                          min_int]

                #########

                meas_str_l_branch = ["Number of branches", "Branch length", "Total branch length", "Curvature index"]
                meas_l_branch = [n_branches, branch_len, tot_branch_len, curv_ind]

                #########

                for m_str, m in zip(meas_str_l, meas_l):
                    add_to_dataframe(dataframe, m_str, m, n)
                    n += 1

                for m_str_b, mb in zip(meas_str_l_branch, meas_l_branch):
                    add_to_dataframe(dataframe_branch, m_str_b, mb, n2)
                    n2 += 1


            writer = pd.ExcelWriter(dirpath.get() + os.sep + table_name.get() + "_MorphMeasurements_Table.xlsx",
                                    engine='xlsxwriter')

            dataframe.to_excel(writer, sheet_name="ShapeDescriptors")
            dataframe_branch.to_excel(writer, sheet_name="BranchAnalysis")

            writer.save()

            tkinter.messagebox.showinfo("Done", "Table generated", parent=gm_root)

        self.place_button(gm_root, "Get Measurements", measure, 200, 330, 30, 110)
        gm_root.mainloop()


class MorphComparison(Control):

    def __init__(self):
        Control.__init__(self)

    def morph_analysis(self):

        ma_root = Tk()

        self.new_window(ma_root, "MitoSegNet Analyser - Morphological comparison", 500, 450)
        self.small_menu(ma_root)


        table_path1 = StringVar(ma_root)
        table1_name = StringVar(ma_root)
        table_path2 = StringVar(ma_root)
        table2_name = StringVar(ma_root)
        descriptor = StringVar(ma_root)
        stat_value = StringVar(ma_root)


        def askopentable1():
            set_tablepath1 = tkinter.filedialog.askopenfilename(parent=ma_root, title='Select file')
            table_path1.set(set_tablepath1)

        def askopentable2():
            set_tablepath2 = tkinter.filedialog.askopenfilename(parent=ma_root, title='Select file')
            table_path2.set(set_tablepath2)

        #### browse for first table location

        self.place_text(ma_root, "Select measurements table file 1", 15, 20, None, None)
        self.place_button(ma_root, "Browse", askopentable1, 435, 50, 30, 50)
        self.place_entry(ma_root, table_path1, 25, 50, 30, 400)

        #### enter table 1 name

        self.place_text(ma_root, "Enter name of table 1 (e. g. wild type)", 15, 90, None, None)
        self.place_entry(ma_root, table1_name, 25, 110, 30, 400)

        #### browse for second table location (to compare against table 1)

        self.place_text(ma_root, "Select measurements table file 2 (to compare against table 1)", 15, 160, None, None)
        self.place_button(ma_root, "Browse", askopentable2, 435, 190, 30, 50)
        self.place_entry(ma_root, table_path2, 25, 190, 30, 400)

        #### enter table 2 name

        self.place_text(ma_root, "Enter name of table 2 (e. g. mutant)", 15, 230, None, None)
        self.place_entry(ma_root, table2_name, 25, 250, 30, 400)


        descriptor_list = ["Area", "Minor Axis Length", "Major Axis Length", "Eccentricity", "Perimeter", "Solidity",
                           "Mean Intensity", "Max Intensity", "Min Intensity", "Number of branches", "Branch length",
                           "Total branch length", "Curvature index"]

        descriptor.set("Area")
        Label(ma_root, text="Select shape descriptor to analyze", bd=1).place(bordermode=OUTSIDE, x=15, y=300)
        popupMenu_desc = OptionMenu(ma_root, descriptor, *set(descriptor_list))
        popupMenu_desc.place(bordermode=OUTSIDE, x=25, y=320, height=30, width=160)


        stat_list = ["Average", "Median", "Standard Deviation", "Standard Error", "Minimum", "Maximum", "N"]

        stat_value.set("Average")
        Label(ma_root, text="Select statistical value to analyze", bd=1).place(bordermode=OUTSIDE, x=255, y=300)
        popupMenu_stat = OptionMenu(ma_root, stat_value, *set(stat_list))
        popupMenu_stat.place(bordermode=OUTSIDE, x=265, y=320, height=30, width=160)


        def start_analysis():

            desc = descriptor.get()
            stat_val = stat_value.get()

            tab1_name = table1_name.get()
            tab2_name = table2_name.get()

            ylab = stat_val + " " + desc.lower()
            ylab_size = 32

            xlab = [table1_name.get(), table2_name.get()]

            ####################

            if desc == "Number of branches" or desc == "Branch length" or desc == "Total branch length" or desc == "Curvature index":

                table1 = pd.read_excel(table_path1.get(), sheet_name="BranchAnalysis")
                table2 = pd.read_excel(table_path2.get(), sheet_name="BranchAnalysis")

            else:

                table1 = pd.read_excel(table_path1.get(), sheet_name="ShapeDescriptors")
                table2 = pd.read_excel(table_path2.get(), sheet_name="ShapeDescriptors")

            data_d = {}
            max_vals = []
            normtest_list = []

            for table, table_name in zip([table1, table2], [tab1_name, tab2_name]):

                # how to acess measurements
                meas_table = table[table["Measurement"] == desc]

                # how to acess statistical values
                values_list = meas_table[stat_val].tolist()

                #print(values_list)
                ###
                data_d.update({table_name: values_list})
                ###

                max_vals.append(np.max(values_list))

                if normaltest(values_list)[1] > 0.05:
                    normtest = "| Parametric distribution"
                    normtest_list.append(True)
                else:
                    normtest = "| Non-parametric distribution"
                    normtest_list.append(False)

                print(table_name, normtest, normaltest(values_list)[1])

            print("\n")

            # converting dictionary with different list lengths into a pandas dataframe
            dataframe = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_d.items()]))

            stat_frame = pd.DataFrame(columns=[tab1_name, tab2_name, "Hypothesis test p-value", "Effect size"])

            def compare_samples(stat_func):

                data1_l = []
                data2_l = []
                pval_l = []
                eff_siz_l = []

                for a, b in itertools.combinations([tab1_name, tab2_name], 2):
                    pval = stat_func(dataframe[a].dropna(), dataframe[b].dropna())
                    eff_siz = cohens_d(dataframe[a].dropna(), dataframe[b].dropna())

                    print(np.average(dataframe[a].dropna()))
                    print(np.average(dataframe[b].dropna()))

                    data1_l.append(a)
                    data2_l.append(b)
                    pval_l.append(pval[1])
                    eff_siz_l.append(eff_siz)

                return data1_l, data2_l, pval_l, eff_siz_l

            # pooled standard deviation for calculation of effect size (cohen's d)
            def cohens_d(data1, data2):

                p_std = np.sqrt(((len(data1) - 1) * np.var(data1) + (len(data2) - 1) * np.var(data2)) / (
                            len(data1) + len(data2) - 2))

                cohens_d = np.abs(np.average(data1) - np.average(data2)) / p_std

                return cohens_d

            if False in normtest_list:

                data1_l, data2_l, pval_l, eff_siz_l = compare_samples(mannwhitneyu)

            else:
                data1_l, data2_l, pval_l, eff_siz_l = compare_samples(ttest_ind)

            # table with p-values and effect sizes
            ########
            stat_frame[tab1_name] = data1_l
            stat_frame[tab2_name] = data2_l
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

                x1 = [tab1_name, tab2_name].index(row[tab1_name])
                x2 = [tab1_name, tab2_name].index(row[tab2_name])

                significance_bar(pos_y=max_bar + 0.1 * max_bar + increase, pos_x=[x1, x2], bar_y=max_bar * 0.05, p=p,
                                 y_dist=max_bar * 0.02,
                                 distance=0.05)

                increase += max_bar * 0.1

            print(dataframe)

            # select plot
            plot = sb.boxplot(data=dataframe, color="white", fliersize=0)
            sb.swarmplot(data=dataframe, color="black", size=8)

            # label the y axis
            plt.ylabel(ylab, fontsize=ylab_size)

            # label the x axis
            plt.xticks(list(range(len(xlab))), xlab)

            # determine fontsize of x and y ticks
            plot.tick_params(axis="x", labelsize=34)
            plot.tick_params(axis="y", labelsize=28)

            plt.show()

        self.place_button(ma_root, "Start analysis", start_analysis, 195, 380, 30, 110)
        ma_root.mainloop()

class CorrelationAnalysis(Control):

    def __init__(self):
        Control.__init__(self)

    def corr_analysis(self):

        ca_root = Tk()

        self.new_window(ca_root, "MitoSegNet Analyser - Morphological comparison", 500, 650)
        self.small_menu(ca_root)

        table_path1 = StringVar(ca_root)
        table1_name = StringVar(ca_root)
        table_path2 = StringVar(ca_root)
        table2_name = StringVar(ca_root)

        stat_value = StringVar(ca_root)

        nb = StringVar(ca_root)
        bl = StringVar(ca_root)
        tbl = StringVar(ca_root)
        ci = StringVar(ca_root)
        ar = StringVar(ca_root)
        min_al = StringVar(ca_root)
        maj_al = StringVar(ca_root)
        ecc = StringVar(ca_root)
        per = StringVar(ca_root)
        sol = StringVar(ca_root)
        mean_int = StringVar(ca_root)
        max_int = StringVar(ca_root)
        min_int = StringVar(ca_root)


        def askopentable1():
            set_tablepath1 = tkinter.filedialog.askopenfilename(parent=ca_root, title='Select file')
            table_path1.set(set_tablepath1)

        def askopentable2():
            set_tablepath2 = tkinter.filedialog.askopenfilename(parent=ca_root, title='Select file')
            table_path2.set(set_tablepath2)

        #### browse for first table location

        self.place_text(ca_root, "Select measurements table file 1", 15, 20, None, None)
        self.place_button(ca_root, "Browse", askopentable1, 435, 50, 30, 50)
        self.place_entry(ca_root, table_path1, 25, 50, 30, 400)

        #### enter table 1 name

        self.place_text(ca_root, "Enter name of table 1 (e. g. wild type)", 15, 90, None, None)
        self.place_entry(ca_root, table1_name, 25, 110, 30, 400)

        #### browse for second table location (to compare against table 1)

        self.place_text(ca_root, "Select measurements table file 2 (to compare against table 1)", 15, 160, None, None)
        self.place_button(ca_root, "Browse", askopentable2, 435, 190, 30, 50)
        self.place_entry(ca_root, table_path2, 25, 190, 30, 400)

        #### enter table 2 name

        self.place_text(ca_root, "Enter name of table 2 (e. g. mutant)", 15, 230, None, None)
        self.place_entry(ca_root, table2_name, 25, 250, 30, 400)

        stat_list = ["Average", "Median", "Standard Deviation", "Standard Error", "Minimum", "Maximum", "N"]

        stat_value.set("Average")
        Label(ca_root, text="Select statistical value to analyze", bd=1).place(bordermode=OUTSIDE, x=15, y=300)
        popupMenu_stat = OptionMenu(ca_root, stat_value, *set(stat_list))
        popupMenu_stat.place(bordermode=OUTSIDE, x=25, y=320, height=30, width=160)

        def place_checkbutton(window, text, variable, x, y, width):
            variable.set(False)
            hf_button = Checkbutton(window, text=text, variable=variable, onvalue=True, offvalue=False)
            hf_button.place(bordermode=OUTSIDE, x=x, y=y, height=30, width=width)

        self.place_text(ca_root, "Select up to 4 variables to", 250, 300, None, None)
        self.place_text(ca_root, "compare against each other ", 250, 320, None, None)

        # left side
        width = 200
        place_checkbutton(ca_root, "Number of branches", nb, 180, 350, width)
        place_checkbutton(ca_root, "Branch length", bl, 180, 380, width)
        place_checkbutton(ca_root, "Total branch length", tbl, 180, 410, width)
        place_checkbutton(ca_root, "Curvature index", ci, 180, 440, width)
        place_checkbutton(ca_root, "Solidity", sol, 180, 470, width)
        place_checkbutton(ca_root, "Mean Intensity", mean_int, 180, 500, width)
        place_checkbutton(ca_root, "Min Intensity", min_int, 180, 530, width)

        # right side
        place_checkbutton(ca_root, "Area", ar, 360, 350, 100)
        place_checkbutton(ca_root, "Minor Axis Length", min_al, 360, 380, 120)
        place_checkbutton(ca_root, "Major Axis Length", maj_al, 360, 410, 120)
        place_checkbutton(ca_root, "Eccentricity", ecc, 360, 440, 100)
        place_checkbutton(ca_root, "Perimeter", per, 360, 470, 100)
        place_checkbutton(ca_root, "Max Intensity", max_int, 360, 500, 100)

        def pair_analysis():

            descriptor_list = ["Number of branches", "Branch length", "Total branch length", "Curvature index",
                               "Area", "Minor Axis Length", "Major Axis Length", "Eccentricity", "Perimeter",
                               "Solidity", "Mean Intensity", "Max Intensity", "Min Intensity", "Image", "Data"]

            variable_list = [nb, bl, tbl, ci, ar, min_al, maj_al, ecc, per, sol, mean_int, max_int, min_int]

            def read_reshape_table(table_path, table_name):

                fin_table = pd.DataFrame(columns=descriptor_list)

                table1_ba = pd.read_excel(table_path, sheet_name="BranchAnalysis")
                table1_sd = pd.read_excel(table_path, sheet_name="ShapeDescriptors")

                table_ba = table1_ba[["Image", "Measurement", stat_value.get()]]
                table_sd = table1_sd[["Image", "Measurement", stat_value.get()]]

                img_list = list(set(table_ba["Image"]))

                for count, img in enumerate(img_list):

                    new_table_ba = table_ba[table1_ba["Image"] == img]
                    new_table_sd = table_sd[table1_sd["Image"] == img]

                    new_table_ba = new_table_ba.transpose()
                    new_table_sd = new_table_sd.transpose()

                    new_table = pd.concat([new_table_ba, new_table_sd], axis=1, sort=False)

                    temp_list = new_table.iloc[2].tolist()
                    temp_list.append(img)
                    temp_list.append(table_name)

                    fin_table.loc[count] = temp_list

                return fin_table

            table = read_reshape_table(table_path1.get(), table1_name.get())
            table2 = read_reshape_table(table_path2.get(), table2_name.get())

            final_table = table2.append(table)

            print(final_table)

            selection_list = ["Data"]
            for name, var in zip(descriptor_list, variable_list):

                if int(var.get()) == 1:

                    selection_list.append(name)

                if len(selection_list) == 5:
                    break

            print(selection_list)

            final_table = final_table[selection_list]
            sb.pairplot(final_table, hue="Data")

            #sb.lmplot(data=final_table, x="Number of branches", y="Branch length", hue="Data")

            plt.title(stat_value.get())
            plt.show()


        self.place_button(ca_root, "Start analysis", pair_analysis, 195, 580, 30, 110)
        ca_root.mainloop()


if __name__ == '__main__':
    """
    Main (starting) window
    """

    control_class = Control()
    get_measurements = Get_Measurements()
    morph_comparison = MorphComparison()
    correlation = CorrelationAnalysis()

    root = Tk()

    control_class.new_window(root, "MitoSegNet Analyser", 300, 300)
    control_class.small_menu(root)

    control_class.place_button(root, "Get measurements", get_measurements.get_measurements_window, 85, 20, 60, 150)
    control_class.place_button(root, "Morphological comparison", morph_comparison.morph_analysis, 85, 100, 60, 150)
    control_class.place_button(root, "Correlation analysis", correlation.corr_analysis, 85, 180, 60, 150)

    root.mainloop()






