import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
import argparse
from pathlib import Path
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.preprocessing import StandardScaler, LabelEncoder #, RobustScaler


#main_plotをクラスタリングするかどうかをのちに追加

#-------------------------------------------------------------------
#--------------------------Sample Usage--------------------------
#-------------------------------------------------------------------
"""
ex1 ) python super_heatmap.py {input_file} {main_col_str} {other_options**}

ex2 ) python super_heatmap.py {input_file} {main_col_str} -adf {add_file} -ctgc {column1} {column2} {column3} -fgs {float_w} {float_h} {other_options**}
"""
#-------------------------------------------------------------------
#-----------------------------Functions-----------------------------
#-------------------------------------------------------------------

def parseArgs():
    
    parser = argparse.ArgumentParser()
    
    #Required arguments
    parser.add_argument("input_file", type=str, help="Input file path")
    parser.add_argument("main_col_str", type=str, help="String common to all main_plot columns")

    #--------------------Data processing, input/output configuration--------------------
    
    #If entered, merged into main_data
    parser.add_argument("-adf", "--add_file", type=str, default=None,
                        help="If a file path is entered, it will be merged to the right of the main data.")
    
    #Arguments to identify columns
    parser.add_argument("-sid", "--sample_id", type=str, default=None,
                        help="The column include sample_id. if None, the first column of dataset will be used.")
    parser.add_argument("-clsc", "--cluster_col", type=str, default=None,
                        help="The column include clusters. if None, the first column of sub_plot will be used.")
    
    #Arguments related to preprocessing of data frames
    parser.add_argument("-thc", "--thd_col", type=str, nargs="*", default=None,
                        help="the columns list you want to threshold.")
    parser.add_argument("-th", "--threshold", type=int, nargs="*", default=None,
                        help="the threshold list you want to use with thd_col.")
    parser.add_argument("-flc", "--fillna_col", type=str, nargs="*", default=None,
                        help="the columns list you want to convert to missing values.")
    parser.add_argument("-msv", "--missing_value", type=str, nargs="*", default=None,
                        help="the value or string list you want to use with fillna_col.")
    parser.add_argument("-ucnc", "--unic_norm_col", type=str, nargs="*", default=None,
                        help="the columns including double-byte characters.")
    parser.add_argument("-ctgc", "--ctg_col", type=str, nargs="*", default=None,
                        help="the columns that need to be label encoded. if None, all categorical columns in sub_plot will be used.")
    parser.add_argument("-mscl", "--main_scale", action='store_true', default=False,
                        help="if true, main_plot columns will be standardized.")
    
    #Arguments related to the appearance of the plotting
    parser.add_argument("-spo", "--sub_plot_order", type=str, nargs="*", default=None,
                        help="the columns name and order for sub_plot. if None, the others than sample_id and main_plot columns will be used.")
    parser.add_argument("-srtc", "--sort_col", type=str, nargs="*", default=None,
                        help="the columns you want to sort by. if None, sorting will be performed only on the cluster_col.")
    parser.add_argument("-cml", "--cmap_list", type=str, nargs="*", default=None,
                        help="the colors list you want to use with plotting.")
    parser.add_argument("-rsc", "--right_side_col", type=str, nargs="*", default=None,
                        help="the columns list you want to plot on right side in add_plot")
    
    #Output settings
    parser.add_argument("-oc", "--output_csv", action='store_true', default=False) #if True, the csv file is also output.
    parser.add_argument("-od", "--output_dir", type=str, default=None) #if None, same as input_dir
    parser.add_argument("-on", "--output_name", type=str, default="heatmap") #if None, same as input_dir
    parser.add_argument("-of", "--output_format", type=str, default="png", choices=['png', 'jpg', 'pdf'])
    parser.add_argument("-dpi", "--dpi", type=float, default=100)
    
    
    #--------------------Setting of plotting area--------------------
    
    #Setting of figure area
    parser.add_argument("-fgs", "--figsize", type=float, nargs="*", default=(8, 8))
    
    #Setting of master_grid
    parser.add_argument("-smhr", "--sub_main_hratio", type=float, nargs="*", default=(3,7))
    parser.add_argument("-mwr", "--master_w_ratio", type=float, nargs="*", default=(7,3))
    parser.add_argument("-mhs", "--master_hspace", type=float, default=0.05)
    parser.add_argument("-mws", "--master_wspace", type=float, default=0.01)
    
    #Setting of the text next to sub_plot
    parser.add_argument("-stxx", "--sub_text_x", type=float, default=0)
    parser.add_argument("-stxy", "--sub_text_y", type=float, default=0.1)
    parser.add_argument("-stxfs", "--sub_text_font_size", type=float, default=12)
    
    #Setting of cbar_grid
    parser.add_argument("-cbhr", "--cbar_h_ratio", type=float, nargs="*", default=(5,5))
    parser.add_argument("-cbwr", "--cbar_w_ratio", type=float, nargs="*", default=(1,9))
    parser.add_argument("-cbws", "--cbar_wspace", type=float, default=0.2) #the margin on the right side of the z-score
    
    #Setting of cbar
    parser.add_argument("-cbtcfs", "--cbar_tick_font_size", type=float, default=8)
    parser.add_argument("-cbt", "--cbar_title", type=str, default="Z-score")
    parser.add_argument("-cbtfs", "--cbar_title_font_size", type=float, default=8)
    parser.add_argument("-ctp", "--cbar_title_pad", type=float, default=8) #the margin above the title of cbar
    parser.add_argument("-cbkws", "--cbar_kws", type=dict, default={"ticks":[-2,-1,0,1,2]})
    parser.add_argument("-cbvmn", "--cbar_vmin", type=float, default=-2)
    parser.add_argument("-cbvmx", "--cbar_vmax", type=float, default=2)
    parser.add_argument("-cbctr", "--cbar_center", type=float, default=0)
    
    #Setting of add_grid
    parser.add_argument("-adhr", "--add_h_ratio", type=float, nargs="*", default=(-0.25, 10))
    parser.add_argument("-adwr", "--add_w_ratio", type=float, nargs="*", default=(1,1))
    parser.add_argument("-adws", "--add_wspace", type=float, default=0.0) #the margin between plot columns
    parser.add_argument("-adhs", "--add_hspace", type=float, default=0.8) #top and bottom margins for each add_plot
    
    #Setting of add_plot
    parser.add_argument("-adtfs", "--add_title_font_size", type=float, default=8)
    parser.add_argument("-adtp", "--add_title_pad", type=float, default=3.5) #the margin above the title of add_plot
    parser.add_argument("-adtr", "--add_tick_rotate", type=float, default=0)
    parser.add_argument("-adtcfs", "--add_tick_font_size", type=float, default=7)
    parser.add_argument("-adtcp", "--add_tick_pad", type=float, default=0) #the margin on the right side of the label
    parser.add_argument("-adlc", "--add_line_color", type=str, default="white") #add_plotの枠線の色
    parser.add_argument("-adlw", "--add_line_width", type=float, default=0) #the border thickness of add_plot
    
    #Setting of vlines
    parser.add_argument("-vlw", "--vlines_width", type=float, default=0.8)
    parser.add_argument("-vlc", "--vlines_color", type=str, default="white")
    
    
    #Printing arguments to the command line
    args = parser.parse_args()

    print("Called with args:")
    print(f"{args}\n")

    return args


#Function to convert full-width notation to half-width notation("ⅡＢ" ---> "IIB")
def unic_norm(df, cols):
    
    for col in cols:
        not_null = df[col][df[col].notnull()]
        normalized = not_null.map(lambda x:unicodedata.normalize('NFKC', x))
        df[col] = pd.Series(normalized, index=not_null.index)
    
    return df


#Function for label encoding
def label_encode(df, cols):
    
    le_dic = {}
    for col in cols:
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
        le_dic[col] = le
    
    return df, le_dic


#Create variables to be used for add_plot
#Adjust columns to have the same number of uniques
def add_plot_adjuster(df, columns, right_side_col):

    #The list containing the number of uniques and column names
    uni_list = [[len(df[c].unique()), c] for c in columns]

    n_left = 0
    n_right = 0
    right = []
    left = []
    #In order of n_uni largest (easier to adjust the total value)
    for li in sorted(uni_list)[::-1]:

        #If right_side_col is specified
        if right_side_col != None:
            if li[1] in right_side_col:
                n_right += li[0]
                right.append([li[0], li[1]])
            elif n_left > n_right:
                n_right += li[0]
                right.append([li[0], li[1]])
            else:
                n_left += li[0]
                left.append([li[0], li[1]])

        #If right_side_col is not specified
        else:
            if n_left > n_right:
                n_right += li[0]
                right.append([li[0], li[1]])
            else:
                n_left += li[0]
                left.append([li[0], li[1]])

    #The ratio of heights used in the grid
    left_ratio = [n[0] for n in sorted(left)]
    right_ratio = [n[0] for n in sorted(right)]

    #If there is a difference, a dummy space is created for the smaller list.
    if n_right > n_left: left_ratio.append(n_right - n_left)
    elif n_right < n_left: right_ratio.append(n_left - n_right)
    
    #The columns of add_plot (in order of plotting)
    add_plot = [li[1] for li in sorted(left) + sorted(right)]
    
    return add_plot, left_ratio, right_ratio


#Function to combine data frames
def check_and_merge(path_main, path_add):
    
    if path_add != None:
        
        if Path(path_main).suffix == ".xlsx": df_main = pd.read_excel(path_main)
        elif Path(path_main).suffix == ".csv": df_main = pd.read_csv(path_main)
        elif Path(path_main).suffix == ".tsv": df_main = pd.read_table(path_main, sep="\t")
        else: 
            print("INPUT_ERROR: the input_file must be .xlsx, .csv, or .tsv")
            exit()
        
        if Path(path_add).suffix == ".xlsx": df_add = pd.read_excel(path_add)
        elif Path(path_add).suffix == ".csv": df_add = pd.read_csv(path_add)
        elif Path(path_add).suffix == ".tsv": df_add = pd.read_table(path_add, sep="\t")
        else: 
            print("INPUT_ERROR: the add_file must be .xlsx, .csv, or .tsv")
            exit()
        
        df = df_main.merge(df_add)
    
    else:
        if Path(path_main).suffix == ".xlsx": df = pd.read_excel(path_main)
        elif Path(path_main).suffix == ".csv": df = pd.read_csv(path_main)
        elif Path(path_main).suffix == ".tsv": df = pd.read_table(path_main, sep="\t")
        else: 
            print("INPUT_ERROR: the input_file must be .xlsx, .csv, or .tsv")
            exit()
        
    return df

    
#-------------------------------------------------------------------
#--------------------------------main-------------------------------
#-------------------------------------------------------------------

def main():
    
    args = parseArgs()  
    
    #Output path (if not specified, it is the same as the input directory)
    if args.output_dir == None: output_dir = Path(args.input_file).absolute().parent
    else: output_dir = Path(args.output_dir).absolute()
    #If the output directory does not exist, create it.
    if output_dir.exists() == False : output_dir.mkdir()
    
    #------------------------------making dataframe、sort------------------------------
    
    #If add_file is specified, data is combined.
    df = check_and_merge(args.input_file, args.add_file)
    
    #If the column name contains " ", an error occurs in specifying the argument
    columns = [re.sub(" ", "_", col) for col in df.columns]
    df.columns = columns
    
    #Columns not to be ploted (if not specified, the first column of df)
    if args.sample_id == None: sample_id = [df.columns[0]]
    else: sample_id = args.sample_id
    
    #Column of main_plot
    main_plot = [col for col in df.columns if args.main_col_str in col]

    #Column of sub_plot
    #If not specified, columns other than ID and main_plot are used in their original order
    if args.sub_plot_order == None: sub_plot = list(df.drop(sample_id + main_plot, axis=1).columns)
    else: sub_plot = args.sub_plot_order
    
    #Column including cluster information
    #If not specified, the first column of sub_plot
    if args.cluster_col == None: cluster_col = sub_plot[0]
    else: cluster_col = args.cluster_col
    
    #Sorting of data frame columns
    df = df[sample_id+sub_plot+main_plot]
    #If not specified, sort by cluster column only
    if args.sort_col == None: df.sort_values(cluster_col, ignore_index=True, inplace=True)
    else: df.sort_values(args.sort_col, ignore_index=True, inplace=True)
    
    
    #------------------------------Data pre-processing------------------------------
    
    #Converting full-width characters to half-width characters(ex:"ⅡＢ" ---> "IIB")
    if args.unic_norm_col != None:
        df = unic_norm(df, args.unic_norm_col)
    
    #Threshold processing
    if args.thd_col != None:
        for c, t in zip(args.thd_col, args.threshold):
            df[c] = df[c].map(lambda x: f">{t}" if int(x) > t else f"<={t}")
            
    #Missing-value processing(ex:"-" ---> None)
    if args.fillna_col != None:
        for c, v in zip(args.fillna_col, args.missing_value):
            df[c] = df[c].map(lambda x: None if x==v else x)
    
    #Columns to label-encode (if not specified, columns of type object in df)
    if args.ctg_col == None: ctg_col = df.columns[df.dtypes == "object"]
    else: ctg_col = args.ctg_col
    #Label encoding (at the same time, encoder dictionary {"column": "encoder"} is also obtained)
    df, le_dic = label_encode(df, ctg_col)
    
    #Create mask data frame to be used later
    sub_mask = df[sub_plot].isnull()
    #Complete missing values with 0
    df.fillna(0, inplace=True)
    #Conversion to int type
    df[sub_plot] = df[sub_plot].astype("int64")
    
    #If True, standardize main_plot column
    if args.main_scale:
        ss = StandardScaler()
        df[main_plot] = ss.fit_transform(df[main_plot])
    
    #If True, output csv file
    if args.output_csv:
        csv_path = output_dir.joinpath("merged_file.csv")
        df.to_csv(csv_path, index=False)
    
    
    #------------------------------Variables used with plotting------------------------------
    
    #List of colors used in plotting
    if args.cmap_list == None:
        cmap_list = ["coolwarm", "Set1", "flag", "Dark2", "Set2", "tab20",
                     "Set3", "Paired", "tab10", "tab20b", "Pastel2", "coolwarm"]
    elif len(args.cmap_list) != len(sub_plot)+1:
        print("ValueError: The length of cmap_list must be the same as the number of columns to be plotted.")
        exit()
    else: cmap_list = args.cmap_list

    #Create a dictionary that maps colors to columns to be used
    cmap_dic = {}
    for column, color in zip(sub_plot+["main"], cmap_list):
        cmap_dic[column] = color
        
    #add_plot: Column order of the corresponding table of sub_plot plotted in the lower right corner
    #left/right_ratio: Ratio of the heights of each column on the left and right
    add_plot, left_ratio, right_ratio = add_plot_adjuster(df, sub_plot, args.right_side_col)
    
    #Ratio of height of drawing area of sub_plot and main_plot
    sub_main_hratio = args.sub_main_hratio
    #Adjust height ratio of master_grid
    master_h_ratio = [sub_main_hratio[0] / len(sub_plot) for _ in range(len(sub_plot))] + [sub_main_hratio[1]]

    #Position of vlines
    line_x = df.value_counts(cluster_col).values[0]
    line_max = df[main_plot].shape[1]

    #Create output path
    output_format = args.output_format
    output_name = args.output_name
    figname = output_dir.joinpath(f"{output_name}.{output_format}")
    
    
    #------------------------------Plotting heatmap------------------------------
    
    figure = plt.figure(figsize=args.figsize)

    #----------gridspec----------
    #master_grid
    gs_master = GridSpec(nrows=len(sub_plot)+1, ncols=2, height_ratios=master_h_ratio,
                         width_ratios=args.master_w_ratio, hspace=args.master_hspace, wspace=args.master_wspace)

    #cbar_grid
    gs_cbar = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_master[-1,1], width_ratios=args.cbar_w_ratio,
                                      height_ratios=args.cbar_h_ratio, wspace=args.cbar_wspace)

    #add_grid
    gs_add = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_cbar[:,1], height_ratios=args.add_h_ratio,
                                     width_ratios=args.add_w_ratio, wspace=args.add_wspace)
    gs_left = GridSpecFromSubplotSpec(len(left_ratio), 1, subplot_spec=gs_add[1,0],
                                      hspace=args.add_hspace, height_ratios=left_ratio)
    gs_right = GridSpecFromSubplotSpec(len(right_ratio), 1, subplot_spec=gs_add[1,1],
                                       hspace=args.add_hspace, height_ratios=right_ratio)

    #----------heatmap in sub_plot----------
    for i, c in enumerate(sub_plot):

        #plotting area of heatmap
        ax_s = figure.add_subplot(gs_master[i,0])
        #vlines
        plt.vlines(x=line_x, ymin=0, ymax=line_max, colors=args.vlines_color, linewidth=args.vlines_width)

        #plotting heatmap
        mask = sub_mask[c] #欠損データフレームのマスク
        data = df[c].values.reshape(-1,1)
        sns.heatmap(data=data.T, ax=ax_s, cbar=False, cmap=cmap_dic[c], mask=mask.T)
        ax_s.axis("off")

        #the text next to sub_plot
        ax_t = figure.add_subplot(gs_master[i,1])
        ax_t.text(x=args.sub_text_x, y=args.sub_text_y, s=c, size=args.sub_text_font_size)
        ax_t.axis("off")


    #----------heatmap in main_plot----------
    #plotting area of heatmap
    ax_m = figure.add_subplot(gs_master[-1,0])
    #vlines
    plt.vlines(x=line_x, ymin=0, ymax=line_max, colors=args.vlines_color, linewidth=args.vlines_width)

    #setting of cbar
    ax_c = figure.add_subplot(gs_cbar[1,0])
    ax_c.tick_params(labelsize=args.cbar_tick_font_size, right=False)
    ax_c.set_title(args.cbar_title, loc="left", size=args.cbar_title_font_size, y=1.0, pad=args.cbar_title_pad)

    #plotting heatmap
    data = df.loc[:, main_plot]
    sns.heatmap(data=data.T, ax=ax_m, cbar_ax=ax_c, cmap=cmap_dic["main"], cbar_kws=args.cbar_kws,
                vmin=args.cbar_vmin, vmax=args.cbar_vmax, center=args.cbar_center)
    ax_m.axis("off")


    #----------heatmap in add_plot----------
    for i, c in enumerate(add_plot):

        #assign to right and left
        if i+1 <= len(left_ratio): ax_a = figure.add_subplot(gs_left[i])
        else: ax_a = figure.add_subplot(gs_right[i-len(left_ratio)])

        #unique data
        uni = df[c].unique()
        #label_encode inverse
        if c in le_dic.keys(): yticks = le_dic[c].inverse_transform(uni)
        else: yticks = uni.tolist()
        #plotting heatmap
        sns.heatmap(uni.reshape(-1,1), cbar=False, ax=ax_a, square=True, cmap=cmap_dic[c], xticklabels=False,
                    yticklabels=yticks, linecolor=args.add_line_color, linewidths=args.add_line_width)
        ax_a.set_title(c, size=args.add_title_font_size, loc='left', y=1.0, pad=args.add_title_pad)
        ax_a.yaxis.tick_right()
        ax_a.tick_params(axis='y', labelrotation=args.add_tick_rotate,
                         labelsize=args.add_tick_font_size, right=False, pad=args.add_tick_pad)


    plt.savefig(figname, pad_inches=0, dpi=args.dpi)


if __name__ == "__main__":
    main()
