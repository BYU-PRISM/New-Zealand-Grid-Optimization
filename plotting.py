import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt
import models
import numpy as np
import matplotlib.dates as mdates
import pickle

def choose_ouput_file(pattern: str) -> str:
    files = glob(pattern)
    for i, f in enumerate(files):
        print(f'{i}: {f}')
    index = int(input('Choose the index for the desired file:'))
    return files[index]

def pairplot(data=None, combined=True):
    if data is None:
        f = choose_ouput_file('results/dist*')
        data = pd.read_csv(f)
    if combined:
        data['Renewables'] = data.fv_6 + data.fv_7
        data['wind fraction'] = data['fv_6'] / data['Renewables'] * 100
        data = data[['obj', 'Renewables', 'fv_8', 'fv_9']] # Limit to just the columns we care about
        data.rename({
            'obj': 'LCOE',
            'fv_8': 'Transmission Limit',
            'fv_9': 'Battery Capacity'}, inplace=True, axis=1)
        a: sns.PairGrid = sns.pairplot(data, hue='LCOE')
        a.figure.set_figheight(5)
        a.figure.set_figwidth(8)
        # a.axes[0][0].set_ylabel("Renewables (MW)")
        # a.axes[1][0].set_ylabel('Transmision Limit (MW)')
        # a.axes[2][0].set_ylabel('Battery Capacity (MWh)')

        # a.axes[1][0].set_xlabel("Renewables (MW)")
        # a.axes[1][1].set_xlabel("Transmission Limit (MW)")
        # a.axes[1][2].set_xlabel("Battery Capacity (MWh)")
    else:
        data = data[['obj', 'fv_6', 'fv_7', 'fv_8', 'fv_9']] # Limit to just the columns we care about
        data.rename({
            'obj': 'LCOE',
            'fv_6': 'Wind Capacity',
            'fv_7': 'Solar Capacity',
            'fv_8': 'Transmission Limit',
            'fv_9': 'Battery Capacity'
        }, inplace=True, axis=1)
        a: sns.PairGrid = sns.pairplot(data)
        a.figure.set_figheight(5)
        a.figure.set_figwidth(8)

        # a.axes[0][0].set_ylabel("Wind Capacity (MW)")
        # a.axes[1][0].set_ylabel("Solar Capacity (MW)")
        # a.axes[2][0].set_ylabel('Transmision Limit (MW)')
        # a.axes[3][0].set_ylabel('Battery Capacity (MWh)')

        # a.axes[1][0].set_xlabel("Wind Capacity (MW)")
        # a.axes[1][1].set_xlabel("Solar Capacity (MW)")
        # a.axes[1][2].set_xlabel("Transmission Limit (MW)")
        # a.axes[1][3].set_xlabel("Battery Capacity (MWh)")
    plt.show()

def plot_metrics(data=None):
    import matplotlib.tri as tri
    if data is None:
        f = choose_ouput_file('results/metrics*')
        data = pd.read_csv(f)

    # Shouldn't be needed anymore, but just in case
    # data.rename({
    #     '0': 'Infeasibilities',
    #     '1': 'LCOE',
    #     '2': 'Wind Capacity',
    #     '3': 'Wind Capacity Factor',
    #     '4': 'Solar Capacity',
    #     '5': 'Solar Capacity Factor',
    #     '6': 'Grid Capacity',
    #     '7': 'Battery Capacity',
    #     '8': 'Total Load',
    #     '9': 'Percent Local Generation',
    #     '10': 'Max Transmission Demand',
    # }, inplace=True, axis=1)

    data.drop([
        'Total Load', # This is always the same unless there was an infeasibility
        'Unnamed: 0', # The index that is somehow always added
    ], axis=1, inplace=True)

    x = data['Wind Capacity']
    y = data['Solar Capacity']
    z = data['Percent Local Generation']

    plt.plot(data['Wind Capacity'], data['Solar Capacity'], '.')
    plt.xlabel('Wind Capacity (MW)')
    plt.ylabel('Solar Capacity (MW)')

    triag = tri.Triangulation(x, y)
    # FIXME: Will need to add levels here to help it know where to plot.
    # This is actually not working yet because
    plt.tricontour(triag, z)

    plt.show()

def plot_electrification_dynamics():
    d = pd.read_csv('./data/data.csv', skiprows=2)
    d['date'] = pd.date_range(start='2022-01-01', periods=8760, freq='H')
    myFmt = mdates.DateFormatter('%d %b')
    n_days = 14
    n_hrs = 24*n_days
    start = 2352 # starts at the beginning of a week
    end = start+n_hrs

    x_2wk = d.date[start:end]
    y_2023_2wk = d.industrial_base[start:end]
    y_2050_2wk = d.Industrial_2050[start:end]

    x_day = d.date[start:start+24]
    y_2023_day = d.industrial_base[start:start+24]
    y_2050_day = d.Industrial_2050[start:start+24]

    fig, ax = plt.subplots(2, 1)
    ax[0].xaxis.set_major_formatter(myFmt)
    ax[0].plot(d.date, d.industrial_base, label='2023')
    ax[0].plot(d.date, d.Industrial_2050, label='2050')
    ax[0].set_ylabel('MW')

    ax[1].xaxis.set_major_formatter(myFmt)
    ax[1].plot(x_2wk, y_2023_2wk, label='2023')
    ax[1].plot(x_2wk, y_2050_2wk, label='2050')
    ax[1].set_ylabel('MW')

    # ax[2].xaxis.set_major_formatter(myFmt)
    # ax[2].plot(x_day, y_2023_day, label='2023')
    # ax[2].plot(x_day, y_2050_day, label='2050')
    # ax[2].set_ylabel('MW')

    fig.subplots_adjust(top=0.970,
        bottom=0.075,
        left=0.060,
        right=0.965,
        hspace=0.556,
        wspace=0.2
    )
    fig.set_figheight(4)
    fig.set_figwidth(10)

    plt.legend()
    plt.show()

def plot_transmission_costing():
    import numpy as np
    x = np.linspace(4, 100, 101)

    a = 40e6#20e6# 10e6
    b = -6
    c = 4.27e7+1585948.7118#2.27e7 #11.4e6
    d = 4000000

    y = a*np.arctan(x+b) + c
    y0 = a*np.arctan(x-b) + c
    y1 = a*np.arctan(x+b) + c + d*(x - 4)

    plt.plot(x, y, label='Current')
    plt.plot(x, y0, label='Prev')
    plt.plot(x, y1, label='Added linear')

    plt.legend()
    plt.show()

def plot_transmission_cost():
    p = models.params_2050
    a = p['grid_cap_a']
    b = p['grid_cap_b']
    c = p['grid_cap_c']
    d = p['grid_cap_d']

    lb = 4
    ub = 40

    x = np.linspace(lb, ub, 101)
    y = (a*np.arctan(x+b) + c + d*(x-4))/1e6

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, y)
    ax.set_xlabel('Grid Exit Point Capacity (MW)')
    ax.set_ylabel('Required Investment Cost (NZD)')
    ax.yaxis.set_major_formatter('{x:.2f} M')
    plt.show()

def pairplot_final(case='2023', curtailed=True):
    if case == '2023':
        if curtailed:
            data_full = pd.read_csv('results/dists_2023_curtailed_tau240.csv')
            exact_m = pickle.load(open('./results/exact_2023_curtailed.pkl', 'rb'))
        else:
            data_full = pd.read_csv('results/dists_2023_uncurtailed_tau240.csv')
            exact_m = pickle.load(open('./results/exact_2023_uncurtailed.pkl', 'rb'))
        wind_cutoff = 20
        data = data_full[data_full.fv_6 <= wind_cutoff] # Cut out the outliers with extreme
        print('NRows droped: ', data_full.shape[0] - data.shape[0], 'or pct:', (data_full.shape[0] - data.shape[0])/data_full.shape[0])
        print('Wind cutoff:', wind_cutoff)
    else:
        if curtailed:
            data_full = pd.read_csv('results/dists_2050_curtailed_tau240.csv')
            exact_m = pickle.load(open('./results/exact_2050_curtailed.pkl', 'rb'))
        else:
            data_full = pd.read_csv('results/dists_2050_uncurtailed_tau240.csv')
            exact_m = pickle.load(open('./results/exact_2050_uncurtailed.pkl', 'rb'))
        wind_cutoff = 150
        solar_cutoff = 100
        data = data_full[data_full.fv_6 <= wind_cutoff]
        data = data[data.fv_7 <= solar_cutoff]
        print('NRows droped: ', data_full.shape[0] - data.shape[0], 'or pct:', (data_full.shape[0] - data.shape[0])/data_full.shape[0])
        print('Wind cutoff:', wind_cutoff, 'solar cutoff:', solar_cutoff)

    data = data[['obj', 'fv_6', 'fv_7', 'fv_8', 'fv_9']] # Limit to just the columns we care about
    data.rename({
        'obj': 'LCOE',
        'fv_6': 'Wind Capacity',
        'fv_7': 'Solar Capacity',
        'fv_8': 'Transmission Limit',
        'fv_9': 'Battery Capacity'
    }, inplace=True, axis=1)

    cap_wind = exact_m.model.cap_wind[0]
    cap_solar = exact_m.model.cap_solar[0]
    cap_battery = exact_m.model.cap_battery[0]
    cap_grid = exact_m.model.cap_grid[0]
    LCOE = exact_m.model.LCOE[-1]
    total_local_load = sum(exact_m.model.ts_resid) + sum(exact_m.model.ts_indus)
    total_import = sum(exact_m.model.gimport)
    pct_local_gen = (1 - total_import/total_local_load) * 100

    print(f'''Caps:
        Wind:    {cap_wind:.4f} MW
        Solar:   {cap_solar:.4f} MW
        Battery: {cap_battery:.4f} MWh
        Grid:    {cap_grid:.4f} MW
        LCOE:    {LCOE:.4f} NZD/MWh
        PCT Lcl: {pct_local_gen:.4f} %''')

    a: sns.PairGrid = sns.pairplot(data, corner=True)#, diag_kws={'bins': 'scott'})
    a.figure.set_figheight(6)
    a.figure.set_figwidth(10)

    a.axes[0][0].set_ylabel('LCOE (NZD/MWh)')
    a.axes[1][0].set_ylabel('Wind Capacity\n(MW)')
    a.axes[2][0].set_ylabel('Solar Capacity\n(MW)')
    a.axes[3][0].set_ylabel('Transmission\nLimit (MW)')
    a.axes[4][0].set_ylabel('Battery Capacity\n(MWh)')
    a.axes[4][0].set_xlabel('LCOE (NZD/MWh)')
    a.axes[4][1].set_xlabel('Wind Capacity (MW)')
    a.axes[4][2].set_xlabel('Solar Capacity (MW)')
    a.axes[4][3].set_xlabel('Transmission Limit (MW)')
    a.axes[4][4].set_xlabel('Battery Capacity (MWh)')

    if case == '2023':
        a.axes[3][0].set_ylim(3.75, 4.25)
        a.axes[4][3].set_xlim(3.75, 4.25)

    a.axes[1][0].plot(LCOE, cap_wind, 'C1*')
    a.axes[2][0].plot(LCOE, cap_solar, 'C1*')
    a.axes[3][0].plot(LCOE, cap_grid, 'C1*')
    a.axes[4][0].plot(LCOE, cap_battery, 'C1*')

    a.axes[2][1].plot(cap_wind, cap_solar, 'C1*')
    a.axes[3][1].plot(cap_wind, cap_grid, 'C1*')
    a.axes[4][1].plot(cap_wind, cap_battery, 'C1*')

    a.axes[3][2].plot(cap_solar, cap_grid, 'C1*')
    a.axes[4][2].plot(cap_solar, cap_battery, 'C1*')

    a.axes[4][3].plot(cap_grid, cap_battery, 'C1*')
    a.figure.subplots_adjust(
        top=1.0,
        bottom=0.085,
        left=0.08,
        right=0.984,
        hspace=0.053,
        wspace=0.059
    )

    for i in range(4):
        a.axes[i][i].autoscale()


    # return a
    plt.show()

def plot_metrics_2023_final():
    data = pd.read_csv('./results/metrics_2023_curtailed_overlapping_tau240_2.csv')
    exact_curt = pickle.load(open('./results/exact_2023_curtailed.pkl', 'rb'))
    exact_uncurt = pickle.load(open('./results/exact_2023_uncurtailed.pkl', 'rb'))
    # exact_nocap = pickle.load(open('./results/zero_caps_2023.pkl', 'rb'))

    # data = pd.read_csv('./results/metrics_2050_curtailed_overlapping_tau240.csv')
    # exact_m = pickle.load(open('./results/exact_2050_curtailed.pkl', 'rb'))
    data.drop(['Unnamed: 0',
            '13', # gekko model column
            '3', # wind capacity factor
            '5', # solar capacity factor
            '8', # total load
              ], axis=1, inplace=True)

    data.rename({
        '0': 'Infeasibilities',
        '1': 'LCOE',
        '2': 'Wind Capacity',
        '4': 'Solar Capacity',
        '6': 'Grid Capacity',
        '7': 'Battery Capacity',
        '9': 'Percent Local Generation',
        '10': 'Max Transmission Demand',
        '11': 'Curtailed Hours',
        '12': 'Curtailed Total'
    }, axis=1, inplace=True)

    infeas = data[data['Infeasibilities'] > 0]
    # infeas = infeas[infeas['Wind Capacity'] < 65]
    d = data[data['Infeasibilities'] == 0]
    # d = d[d['Wind Capacity'] < 65]
    # d = d[d['Solar Capacity'] < 150]

    print('NRows droped: ', data.shape[0] - d.shape[0], 'or pct:', (data.shape[0] - d.shape[0])/data.shape[0]*100)

    cap_wind = exact_curt.model.cap_wind[0]
    cap_solar = exact_curt.model.cap_solar[0]
    cap_battery = exact_curt.model.cap_battery[0]
    LCOE = exact_curt.model.LCOE[-1]
    cap_wind_uncurt = exact_uncurt.model.cap_wind[0]
    cap_solar_uncurt = exact_uncurt.model.cap_solar[0]
    cap_battery_uncurt = exact_uncurt.model.cap_battery[0]
    LCOE_uncurt = exact_uncurt.model.LCOE[-1]
    # LCOE_npcap = exact_nocap.LCOE[-1]

    total_local_load = sum(exact_curt.model.ts_resid) + sum(exact_curt.model.ts_indus)
    total_import = sum(exact_curt.model.gimport)
    pct_local_gen = (1 - total_import/total_local_load) * 100
    total_local_load_uncurt = sum(exact_uncurt.model.ts_resid) + sum(exact_uncurt.model.ts_indus)
    total_import_uncurt = sum(exact_uncurt.model.gimport)
    pct_local_gen_uncurt = (1 - total_import_uncurt/total_local_load_uncurt) * 100
    total_curtailment = np.sum(exact_curt.model.curtailment.VALUE)

    print(f'LCOE curtailed: {LCOE:.4f} LCOE uncurtailed: {LCOE_uncurt:.4f}')
    # print(f'LCOE curtailed: {LCOE:.4f} LCOE uncurtailed: {LCOE_uncurt:.4f} LCOE nocap {LCOE_npcap:.4f}')

    # Grid of 2D scatters
    x = d['Wind Capacity']
    y = d['Solar Capacity']
    keys = ['LCOE', 'Percent Local Generation', 'Battery Capacity', 'Curtailed Total']
    labels = [
        'LCOE (NZD/MWh)',
        'Pct. Local Generation (%)',
        'Battery Capacity (MWh)',
        'Total Curtailment (MWh)'
    ]
    exacts = [LCOE, pct_local_gen, cap_battery, total_curtailment]
    exacts_uncurt = [LCOE_uncurt, pct_local_gen_uncurt, cap_battery_uncurt, 0.0]

    fig, axs = plt.subplots(2, 2)
    fig.set_figwidth(10)
    fig.set_figheight(6)
    k = 0
    for i in range(2):
        for j in range(2):
            norm = plt.Normalize(np.min(d[keys[k]]), np.max(d[keys[k]]))
            cb: plt.colorbar = axs[i][j].scatter(x, y, c=d[keys[k]], norm=norm)
            axs[i][j].scatter(cap_wind, cap_solar, c='white', s=50, norm=norm,
                              marker='*')
            # axs[i][j].scatter(cap_wind, cap_solar, c=exacts[k], s=50, norm=norm,
            #                   marker='*', edgecolors='white', linewidths=0.5)
            if k<3:
                # Exclude plotting curtailed details of uncurtailed results
                axs[i][j].scatter(cap_wind_uncurt, cap_solar_uncurt, c='red', s=50,
                                norm=norm, marker='*')
                # axs[i][j].scatter(cap_wind_uncurt, cap_solar_uncurt, c=exacts_uncurt[k], s=50,
                #                 norm=norm, marker='*', edgecolors='red', linewidths=0.5)

            plt.colorbar(cb, label=labels[k])
            axs[i][j].set_xlabel('Wind Capacity (MW)')
            axs[i][j].set_ylabel('Solar Capacity (MW)')
            k += 1

    fig.subplots_adjust(
        top=0.935,
        bottom=0.095,
        left=0.065,
        right=0.97,
        hspace=0.28,
        wspace=0.25
    )
    plt.show()

def plot_2050_metrics_final():
    data = pd.read_csv('./results/metrics_2050_curtailed_overlapping_tau240_3.csv')
    exact_m = pickle.load(open('./results/exact_2050_curtailed.pkl', 'rb'))

    # data = pd.read_csv('./results/metrics_2050_curtailed_overlapping_tau240.csv')
    # exact_m = pickle.load(open('./results/exact_2050_curtailed.pkl', 'rb'))
    data.drop(['Unnamed: 0',
            '13', # gekko model column
            '3', # wind capacity factor
            '5', # solar capacity factor
            '8', # total load
              ], axis=1, inplace=True)

    data.rename({
        '0': 'Infeasibilities',
        '1': 'LCOE',
        '2': 'Wind Capacity',
        '4': 'Solar Capacity',
        '6': 'Grid Capacity',
        '7': 'Battery Capacity',
        '9': 'Percent Local Generation',
        '10': 'Max Transmission Demand',
        '11': 'Curtailed Hours',
        '12': 'Curtailed Total'
    }, axis=1, inplace=True)

    infeas = data[data['Infeasibilities'] > 0]
    # infeas = infeas[infeas['Wind Capacity'] < 65]
    d = data[data['Infeasibilities'] == 0]
    # d = d[d['Wind Capacity'] < 150]
    # d = d[d['Solar Capacity'] < 100]

    print('NRows droped: ', data.shape[0] - d.shape[0], 'or pct:', (data.shape[0] - d.shape[0])/data.shape[0]*100)

    cap_wind = exact_m.model.cap_wind[0]
    cap_solar = exact_m.model.cap_solar[0]
    cap_battery = exact_m.model.cap_battery[0]
    cap_grid = exact_m.model.cap_grid[0]
    LCOE = exact_m.model.LCOE[-1]
    total_local_load = sum(exact_m.model.ts_resid) + sum(exact_m.model.ts_indus)
    total_import = sum(exact_m.model.gimport)

    # x = d['Wind Capacity'] + d['Solar Capacity']
    # # y = d['Solar Capacity']
    # y = d['Grid Capacity']
    # z = d['Battery Capacity']
    # c = d['LCOE']
    x = d['Wind Capacity']
    y = d['Solar Capacity']
    z = d['Grid Capacity']
    c = d['Battery Capacity']
    # c = d['LCOE']
    # 3D scatter
    x_infeas = infeas['Wind Capacity'] + infeas['Solar Capacity']
    # y_infeas = infeas['Solar Capacity']
    y_infeas = infeas['Grid Capacity']
    z_infeas = infeas['Battery Capacity']


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cb = ax.scatter(x, y, z, c=c)
    # ax.scatter(x_infeas, y_infeas, z_infeas, c='r')
    # ax.scatter(cap_wind + cap_solar, cap_grid, cap_battery, c='k')
    # ax.scatter(cap_wind, cap_solar, cap_battery, c='k')
    ax.set_xlabel('Wind Capacity (MW)')
    ax.set_ylabel('Solar Capacity (MW)')
    # ax.set_zlabel('Battery Capacity (MWh)')
    ax.set_zlabel('Grid Capacity (MW)')
    fig.colorbar(cb, label='Battery Capacity (MWh)')

    fig = plt.figure()
    ax = fig.add_subplot()
    cb = ax.scatter(x, y, c=c)
    ax.set_xlabel('Wind Capacity (MW)')
    ax.set_ylabel('Solar Capacity (MW)')
    fig.colorbar(cb, label='Battery Capacity (MWh)')

    d = d[d['Wind Capacity'] + d['Solar Capacity'] <= 150]
    x = d['Wind Capacity'] + d['Solar Capacity']
    y = d['Grid Capacity']
    keys = ['LCOE', 'Percent Local Generation', 'Battery Capacity', 'Curtailed Total']
    labels = [
        'LCOE (NZD/MWh)',
        'Pct. Local Generation (%)',
        'Battery Capacity (MWh)',
        'Total Curtailment (MWh)'
    ]

    total_local_load = sum(exact_m.model.ts_resid) + sum(exact_m.model.ts_indus)
    total_import = sum(exact_m.model.gimport)
    pct_local_gen = (1 - total_import/total_local_load) * 100
    total_curtailment = np.sum(exact_m.model.curtailment.VALUE)
    exacts = [LCOE, pct_local_gen, cap_battery, total_curtailment]

    fig, axs = plt.subplots(2, 2)
    fig.set_figwidth(10)
    fig.set_figheight(6)
    k = 0
    for i in range(2):
        for j in range(2):
            norm = plt.Normalize(np.min(d[keys[k]]), np.max(d[keys[k]]))
            cb: plt.colorbar = axs[i][j].scatter(x, y, c=d[keys[k]], norm=norm)
            axs[i][j].scatter(x_infeas, y_infeas, c='r', s=20)
            axs[i][j].scatter(cap_wind+cap_solar, cap_grid, c='white', s=80, norm=norm,
                              marker='*')

            plt.colorbar(cb, label=labels[k])
            axs[i][j].set_xlabel('Renewables Capacity (MW)')
            axs[i][j].set_ylabel('Grid Capacity (MW)')
            k += 1

    fig.subplots_adjust(
        top=0.935,
        bottom=0.095,
        left=0.065,
        right=0.97,
        hspace=0.28,
        wspace=0.25
    )

    plt.show()

