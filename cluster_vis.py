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


#----------------------------------------------------------------------
#------------------------------util------------------------------
#----------------------------------------------------------------------

#read_***を自動で判別する(引数の設定考える)
def auto_pdread(path, columns=None, header=None):
    
    if Path(path).suffix == ".xlsx": df = pd.read_excel(path, columns, header)
    elif Path(path).suffix == ".csv": df = pd.read_csv(path, columns, header)
    elif Path(path).suffix == ".tsv": df = pd.read_table(path, sep="\t", columns, header)
    else: 
        print("INPUT_ERROR: the input_file must be .xlsx, .csv, or .tsv")
        df = None
        
    return df



#欠損値があってもエンコード or スケール処理をしてくれる関数("unicode", "label", "zscore"を実装)
#Function to convert full-width notation to half-width notation("ⅡＢ" ---> "IIB")
def conv_with_null(df, cols=None, mode="standard", return_dic=False):
    
    if cols == None: cols = df.columns
    dic = {}

    if mode == "unicode": 
        for col in cols:
            not_null = df[col][df[col].notnull()]
            x = not_null.map(lambda x: unicodedata.normalize("NFKC", x))
            df[col] = pd.Series(x, index=not_null.index)

        return df

    elif any((mode == "label", mode == "zscore")) :
        for col in cols:
            if mode == "label": encoder = LabelEncoder()
            else: encoder = StandardScaler()

            not_null = df[col][df[col].notnull()]
            x = encoder.fit_transform(not_null)
            df[col] = pd.Series(x, index=not_null.index)
            
            dic[col] = encoder

        if return_dic: return df, dic
        else: return df

    else:
        print("ValueError: mode must be 'unicode', 'label' or 'zscore'")
        return None



#for文の中に入れることで進捗を表示することができる
def progress_bar(n_trial, i, ppb=50):
    
    prg = int(i // (n_trial/ppb))

    if i+1 == n_trial:
        bar = "="*(ppb-1) + ">"
        print(f"\r\033[K[{bar}] 100.0% ({i+1}/{n_trial})\n")
    
    elif i % int(round(n_trial/ppb)) == 0:
        
        if prg == 0: bar = " "*ppb        
        else: bar = "="*(prg-1) + ">" + " "*(ppb-prg)
        
        print(f"\r\033[K[{bar}] {i/n_trial*100:.1f}% ({i}/{n_trial})", end="")



#----------------------------------------------------------------------
#------------------------------plotting------------------------------
#----------------------------------------------------------------------

#Create variables to be used for add_plot
#Adjust columns to have the same number of uniques
def add_plot_adjuster(df, n_col=2, right_side_col=None):

    #The list containing the number of uniques and column names
    uni_list = [[len(df[c].unique()), c] for c in df.columns]

    ratio_list = [[] for _ in range(n_col)]  #[ [], [], ...]
    #n_list = [[0] for  _ in range(n_col)]   #[ [0], [0], ...]

    for li in sorted(uni_list)[::-1]:
        if li[1] in right_side_col:
            ratio_list[-1][1].append(li[0], li[1])
            n_list[-1][0] += li[0]














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


    #------------------------------Plotting heatmap------------------------------

def aaaaa(main_df, sub_df, figsize=(1,1), )
    
    figure = plt.figure(figsize=args.figsize)

    main = main_df.columns
    sub = sub_df.columns

    #----------gridspec----------
    #master_grid    --->   dendrogramを描画する
    gs_master = GridSpec(nrows=3, ncols=3, height_ratios=(1, 9), width_ratios=(1, 9), hspace=0, wspace=0)

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











    #----------gridspec----------
    #master_grid
    gs_master = GridSpec(nrows=len(sub)+1, ncols=2, height_ratios=master_h_ratio,
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
