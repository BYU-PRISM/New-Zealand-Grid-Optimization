import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import optimization as opt
import plotting as plot
import models

def lcoe_calcs():
    # Calculate the LCOE for just meeting the load from the grid
    d = pd.read_csv('./data/data.csv', skiprows=2)

    d['base_total_load'] = d['residential_base'] + d['industrial_base']
    d['netZero_total_load'] = 1.19*d['residential_base'] + d['Industrial_2050'] + d['residential_evs'] + 1.19*d['carpark_evs']

    base_total_mw = d['base_total_load'].sum()
    netZero_total_mw = d['netZero_total_load'].sum()

    base_max_lcoe = (d['base_total_load'] * d['grid_price_base']).sum() / base_total_mw # NZD/MW
    netZero_max_lcoe = (d['netZero_total_load'] * d['grid_price_base']).sum() / netZero_total_mw # NZD/MW

    print(f''' Maximum LCOEs (Meeting using only grid)
    Base: {base_max_lcoe:.4f} NZD/MW
    2050: {netZero_max_lcoe:.4f} NZD/MW
    ''')


if __name__ == '__main__':
    params = models.params_2023
    curtailment = False
    m = models.Model(24*365, params, curtailment)
    tau = 72
    n_samples = 5
    fv_indicies = opt.determine_indicies(m.model, m.get_fvs())
    lower = [90 for f in fv_indicies]
    upper = [100 for f in fv_indicies]

    # dists = pd.read_csv('./results/dists_2023.csv')
    # stuff = opt.measure_technoeconomics(dists.iloc[120], m, tau)
    # stuff = opt.measure_technoeconomics_overlapping(m, tau, dists.iloc[120], int(tau/2))

    distributions = opt.determine_distributions(m.model, tau, fv_indicies, m.model_generator)
    # stuff = opt.determine_feasibility_surface(dists, tau, m, n_samples)

