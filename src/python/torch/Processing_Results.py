# This script is used to collect and analyze the results
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 17})
import pandas as pd

if __name__ == '__main__':
    learning_rate = 1e-3
    hidden_size = 1024
    IDs = [30600, 30700, 30800, 30900, 30958, 30959, 54000, 54001, 54002, 54003, 54100, 54200, 54300, 60900, 61000]
    ws_arr = [24]

    case_1_dict = {'ID':[], '2016':[], '2017':[], '2018':[]}
    case_2_dict = {'ID': [], '2016': [], '2017': [], '2018': []}
    for id in IDs:
        for ws in ws_arr:
            root_dir = 'LSTM_{}_IW_{}_OW_{}_LR_{}_ID_{}'.format(hidden_size, ws, ws, learning_rate, id)

            # process case 1
            file_dir = os.path.join(root_dir, 'case1_csv')
            case_1_file_2016 = os.path.join(file_dir, 'without_df_skimmed_xtal_{}_2016_train.csv'.format(id))
            case_1_file_2017 = os.path.join(file_dir, 'without_df_skimmed_xtal_{}_2017_val.csv'.format(id))
            case_1_file_2018 = os.path.join(file_dir, 'without_df_skimmed_xtal_{}_2018_test.csv'.format(id))

            case_1_df_2016 = pd.read_csv(case_1_file_2016)
            case_1_map_2016 = case_1_df_2016['MAP'].values[0]
            case_1_df_2017 = pd.read_csv(case_1_file_2017)
            case_1_map_2017 = case_1_df_2017['MAP'].values[0]
            case_1_df_2018 = pd.read_csv(case_1_file_2018)
            case_1_map_2018 = case_1_df_2018['MAP'].values[0]

            case_1_dict['ID'].append(id)
            case_1_dict['2016'].append(case_1_map_2016)
            case_1_dict['2017'].append(case_1_map_2017)
            case_1_dict['2018'].append(case_1_map_2018)

            # process case 2
            file_dir = os.path.join(root_dir, 'case2_csv')
            case_2_file_2016 = os.path.join(file_dir, 'with_df_skimmed_xtal_{}_2016_train.csv'.format(id))
            case_2_file_2017 = os.path.join(file_dir, 'with_df_skimmed_xtal_{}_2017_val.csv'.format(id))
            case_2_file_2018 = os.path.join(file_dir, 'with_df_skimmed_xtal_{}_2018_test.csv'.format(id))
            case_2_df_2016 = pd.read_csv(case_2_file_2016)
            case_2_map_2016 = case_2_df_2016['MAP'].values[0]
            case_2_df_2017 = pd.read_csv(case_2_file_2017)
            case_2_map_2017 = case_2_df_2017['MAP'].values[0]
            case_2_df_2018 = pd.read_csv(case_2_file_2018)
            case_2_map_2018 = case_2_df_2018['MAP'].values[0]
            case_2_dict['ID'].append(id)
            case_2_dict['2016'].append(case_2_map_2016)
            case_2_dict['2017'].append(case_2_map_2017)
            case_2_dict['2018'].append(case_2_map_2018)

    #### now, let's make a plot
    fig_dir = 'Results'
    for cur_dir in [fig_dir]:
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
    # plot for case 1
    plt.figure(figsize=(16, 9))
    plt.plot(case_1_dict['2016'], label='2016 (case1)', color='r', marker='o', ms=16, lw=3)
    plt.plot(case_1_dict['2017'], label='2017 (case1)', color='g', marker='s', ms=16, lw=3)
    plt.plot(case_1_dict['2018'], label='2018 (case1)', color='b', marker='*', ms=22, lw=3)

    plt.plot(case_2_dict['2016'], label='2016 (case2)', color='r', marker='o', ms=16, lw=3, alpha=0.3, ls='-.')
    plt.plot(case_2_dict['2017'], label='2017 (case2)', color='g', marker='s', ms=16, lw=3, alpha=0.3, ls='-.')
    plt.plot(case_2_dict['2018'], label='2018 (case2)', color='b', marker='*', ms=22, lw=3, alpha=0.3, ls='-.')

    scale_ls = range(len(case_1_dict['ID']))
    index_ls = case_1_dict['ID']
    _ = plt.xticks(scale_ls, index_ls, rotation=30)
    plt.title('Different Crystals (only trained on ID-54000 Y-2016), MAPE')
    plt.grid()
    plt.xlabel('Crystal ID')
    plt.ylabel('MAPE')
    plt.legend()
    fig_name = os.path.join(fig_dir, 'diff_IDs.png')
    plt.savefig(fig_name, dpi=300)
    plt.close()

    # # plot for case 2
    # plt.figure()
    # plt.plot(case_2_dict['2016'], label='2016-Train')
    # plt.plot(case_2_dict['2017'], label='2017-Test')
    # plt.plot(case_2_dict['2018'], label='2018-Test')
    # scale_ls = range(len(case_2_dict['WS']))
    # index_ls = case_2_dict['WS']
    # _ = plt.xticks(scale_ls, index_ls)
    # plt.title('ID:54000, Case 2, MAP')
    # plt.legend()
    # fig_name = os.path.join(fig_dir, 'case2_54000.png')
    # plt.savefig(fig_name, dpi=300)
    # plt.close()










