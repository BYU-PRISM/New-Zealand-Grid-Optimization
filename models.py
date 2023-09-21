from gekko import GEKKO
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from util import get_gk_val_real

NZD_per_USD = 1.65/1 # estimated 5 June 2023
hrs_per_yr = 8760 # number of hours in a year

data_file = pd.read_csv('./data/data.csv', skiprows=2)

params_2023 = {
    # Estimated technology lifetimes
    'solar_lifetime': 20,                   # summary report (Timilsina)
    'wind_lifetime': 20,                    #  (Timilsina)
    'system_lifetime': 30,
    'grid_lifetime': 60,
    'battery_lifetime': 15,                  # NREL Utility-scale battery storage

    # Component Economics
    'wind_cap_cost': 1560*1000*NZD_per_USD, # NZD/MW, World Energy Outlook 2020
    'wind_fix_cost': 0.0,                   # World Energy Outlook 2020
    'wind_var_cost': 10*NZD_per_USD,        # NZD/MWh, World Energy Outlook 2020
    'solar_cap_cost': 1200*1000*NZD_per_USD,# NZD/MW, summary report (Timilsina)
    'solar_fix_cost': 0.0,                  # World Energy Outlook 2020
    'solar_var_cost': 10*NZD_per_USD,       # NZD/MWh, World Energy Outlook 2020
    'battery_cap_cost': 350*1000*NZD_per_USD,# $/MWh, NREL Cost Projections for Utility-Scale Battery Storage: 2021 Update
    'battery_fix_cost': 15*1000*NZD_per_USD,# $/MWh, NREL Cost Projections for Utility-Scale Battery Storage: 2021 Update
    'battery_var_cost': 0,                  # $/MW yr, NREL Cost Projections for Utility-Scale Battery Storage: 2021 Update

    # Grid Economics
    'buyback_price': 0.07*1000,             # NZD/MWh should range between 7-17 Ncents/kWh
    'grid_cap_a': 40e6,#20e6# 10e6
    'grid_cap_b': -6,
    'grid_cap_c': 4.27e7+1585948.7118,#2.27e7 #11.4e6
    'grid_cap_d': 4000000, #2000000, #200000,

    # Time Series for 2050 case
    'ts_wind': data_file['wind'],       # MW
    'ts_solar': data_file['solar'],     # MW
    'ts_resid_solar': 0.0*data_file['solar'],     # MW, no additional rooftop solar
    'ts_industrial': data_file['industrial_base'],  # MW, the current meat factory load
    'ts_residential': data_file['residential_base'], # MW, current for 1000 averaged homes
    'ts_grid_price': data_file['grid_price_base'] + 110, # NZD/MW, Adding the line charge
}

params_2050 = {
    # Estimated technology lifetimes
    'solar_lifetime': 20,                   # summary report (Timilsina)
    'wind_lifetime': 20,                    #  (Timilsina)
    'system_lifetime': 30,
    'grid_lifetime': 60,
    'battery_lifetime': 15,                  # NREL Utility-scale battery storage

    # Component Economics (2050)
    'solar_cap_cost': 700*1000*NZD_per_USD, # NZD/MW, Economics of Utility-Scale Solar in Aotearoa New Zealand
    'solar_fix_cost': 0.0,                  # NZD/MW-yr, World Energy Outlook 2020
    'solar_var_cost': 10*NZD_per_USD,       # NZD/MWh, World Energy Outlook 2020
    'wind_cap_cost': 1440*1000*NZD_per_USD, # NZD/MW, World Energy Outlook 2020
    'wind_fix_cost': 0.0,                   # NZD/MW-yr, World Energy Outlook 2020
    'wind_var_cost': 10*NZD_per_USD,        # USD/MWh, World Energy Outlook 2020
    'battery_cap_cost': 150*1000*NZD_per_USD,# $/MWh, NREL Cost Projections for Utility-Scale Battery Storage: 2021 Update
    'battery_fix_cost': 15*1000*NZD_per_USD,# $/MWh-yr, NREL Cost Projections for Utility-Scale Battery Storage: 2021 Update
    'battery_var_cost': 0,                  # $/MWh, NREL Cost Projections for Utility-Scale Battery Storage: 2021 Update

    # Grid Economics
    'buyback_price': 0.07*1000,             # NZD/MWh should range between 7-17 Ncents/kWh
    'grid_cap_a': 40e6,#20e6# 10e6
    'grid_cap_b': -6,
    'grid_cap_c': 4.27e7+1585948.7118,#2.27e7 #11.4e6
    'grid_cap_d': 4000000, #2000000, #200000,

    # Time Series for 2050 case
    'ts_wind': data_file['wind'],       # MW
    'ts_solar': data_file['solar'],     # MW
    'ts_resid_solar': 1.12*data_file['solar'],     # MW, the amount added from rooftop solar
    'ts_industrial': data_file['Industrial_2050'],  # MW, the fully-electrified meat factory load
    'ts_residential': 1.19*data_file['residential_base']+1.19*data_file['residential_evs'], # MW, for 1190 averaged homes (population increase)
    'ts_grid_price': data_file['grid_price_base'] + 110, # NZD/MW, Adding the line charge

}

solar_ub = 200 #120 #80
wind_ub = 200 #120 #80
grid_ub = 200 #100 #40
battery_ub = 400 #200 #120

class Model(object):
    def __init__(self, hours, params, curtailment) -> None:
        self.params = params
        self.curtailment = curtailment
        self.model = self.model_generator(hours)

    def model_generator(self, nhrs, nshift=0, cap_wind=None, cap_solar=None, cap_battery=None, cap_grid=None):
        p = self.params
        m = GEKKO(remote=False)
        m.time = np.linspace(0, nhrs-1, nhrs) + nshift

        # Load in the time series
        ts_wind = m.Param(p['ts_wind'].values[nshift:nhrs+nshift], name='wind_ts')      # MW
        ts_solar = m.Param(p['ts_solar'].values[nshift:nhrs+nshift], name='solar_ts')   # MW
        ts_industrial = p['ts_industrial'].values[nshift:nhrs+nshift]                   # MW
        ts_residential = p['ts_residential'].values[nshift:nhrs+nshift]
        m.avg_MW = (sum(ts_industrial) + sum(ts_residential))/nhrs
        # The Gekko parameter versions for compatibility in the equations below
        m.ts_indus = m.Param(ts_industrial, name='ts_indus')
        m.ts_resid = m.Param(ts_residential, name='ts_resid')
        m.ts_grid_price = m.Param(value=p['ts_grid_price'].values[nshift:nhrs+nshift], name='grid_pc')    # NZD/MW
        m.ts_resid_solar = m.Param(p['ts_resid_solar'].values[nshift:nhrs+nshift], name='ts_resid_solar') # MW

        # Set up the capacity variables
        if cap_wind is None:
            m.cap_wind = m.FV(value=1, lb=0, ub=wind_ub, fixed_initial=False, name='cap_wind')
            m.cap_wind.STATUS = 1
        else:
            m.cap_wind = m.FV(value=cap_wind, name='cap_wind')
            m.cap_wind.STATUS = 0
        if cap_solar is None:
            m.cap_solar = m.FV(value=1, lb=0, ub=solar_ub, fixed_initial=False, name='cap_solar')
            m.cap_solar.STATUS = 1
        else:
            m.cap_solar = m.FV(value=cap_solar, name='cap_solar')
            m.cap_solar.STATUS = 0
        if cap_grid is None:
            m.cap_grid = m.FV(value=1, lb=4, ub=grid_ub, fixed_initial=False, name='cap_grid')
            m.cap_grid.STATUS = 1
        else:
            m.cap_grid = m.FV(value=cap_grid, name='cap_grid')
            m.cap_grid.STATUS = 0
        if cap_battery is None:
            m.cap_battery = m.FV(value=1, lb=0, ub=battery_ub, fixed_initial=False, name='cap_battery')
            m.cap_battery.STATUS = 1
        else:
            m.cap_battery = m.FV(value=cap_battery, name='cap_battery')
            m.cap_battery.STATUS = 0


        m.wind = m.Var(name='wind')
        m.solar = m.Var(name='solar')
        m.gimport = m.Var(lb=0, ub=grid_ub+10, name='gimport')
        m.gexport = m.Var(lb=0, ub=grid_ub+10, name='gexport')
        m.battery = m.Var(lb=-battery_ub, ub=battery_ub, name='battery')
        m.battery_soc = m.Var(lb=0, ub=battery_ub, name='battery_soc')
        if self.curtailment:
            m.curtailment = m.Var(lb=0, ub=1000, name='curtailment')

        m.Equation(m.wind == m.cap_wind*ts_wind)
        m.Equation(m.solar == m.cap_solar*ts_solar)
        m.Equation(m.gimport <= m.cap_grid)
        m.Equation(m.gexport <= m.cap_grid)
        m.Equation(m.battery_soc <= m.cap_battery)
        m.Equation(m.battery_soc.dt() == m.battery)

        if self.curtailment:
            m.Equation(0 == m.wind+m.solar+m.ts_resid_solar+m.gimport-m.ts_resid-m.ts_indus-m.gexport-m.curtailment + m.battery)
        else:
            m.Equation(0 == m.wind+m.solar+m.ts_resid_solar+m.gimport-m.ts_resid-m.ts_indus-m.gexport + m.battery)
        sum_wind = m.Intermediate(m.integral(m.wind), name='sum_wind')
        sum_solar = m.Intermediate(m.integral(m.solar), name='sum_solar')
        sum_battery = m.Intermediate(m.integral(m.battery), name='sum_battery')
        sum_grid_cost = m.Intermediate(m.integral(m.gimport * m.ts_grid_price - m.gexport * p['buyback_price']), name='sum_grid_cost')

        m.grid_cap_cost = m.Intermediate(p['grid_cap_a']*m.atan(m.cap_grid+p['grid_cap_b'])+p['grid_cap_c'] + p['grid_cap_d']*(m.cap_grid-4))

        m.cap_costs = m.Intermediate(
            p['solar_cap_cost'] * m.cap_solar * p['system_lifetime']/p['solar_lifetime'] +
            p['wind_cap_cost'] * m.cap_wind * p['system_lifetime']/p['wind_lifetime'] +
            m.grid_cap_cost*p['system_lifetime']/p['grid_lifetime'] +
            p['battery_cap_cost'] * m.cap_battery*p['system_lifetime']/p['battery_lifetime'], name='cap_costs')

        m.fixed_costs = m.Intermediate(p['solar_fix_cost']*m.cap_solar +
                                       p['wind_fix_cost']*m.cap_wind +
                                       p['battery_fix_cost']*m.cap_battery, name='fixed_costs')

        m.var_costs = m.Intermediate(
            p['solar_var_cost'] * sum_solar +
            p['wind_var_cost'] * sum_wind +
            sum_grid_cost + p['battery_var_cost']*sum_battery, name='var_costs'
        )

        final = np.zeros(nhrs)
        final[-1] = 1
        f = m.Param(final, name='final')

        m.LCOE = m.Var(name='LCOE')
        m.Equation(m.LCOE == f*(m.cap_costs +
                                m.fixed_costs*p['system_lifetime'] +
                                m.var_costs / nhrs * hrs_per_yr * p['system_lifetime'])
            / (m.avg_MW*hrs_per_yr*p['system_lifetime']))

        m.Obj(m.LCOE)

        m.options.IMODE = 6
        m.options.solver = 3
        m.solver_options = ['tol 1e-2']
        return m

    def plot(self):
        fig = make_subplots(3, 1)

        fig.add_trace(go.Scatter(x=self.model.time, y=self.model.wind.value, name='wind'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.model.time, y=self.model.solar.value, name='solar'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.model.time, y=self.model.gimport.value, name='grid import'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.model.time, y=self.model.ts_resid_solar.value, name='Resid Solar'), row=1, col=1)
        y = np.array(self.model.gimport)+np.array(self.model.solar)+np.array(self.model.wind.value)
        fig.add_trace(go.Scatter(x=self.model.time, y=y, name='Total Generation'), row=1, col=1)

        fig.add_trace(go.Scatter(x=self.model.time, y=self.model.ts_indus.value, name='Industrial'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.model.time, y=self.model.ts_resid.value, name='Residential'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.model.time, y=self.model.gexport.value, name='grid export'), row=2, col=1)
        y = np.array(self.model.ts_resid) + np.array(self.model.ts_indus) + np.array(self.model.gexport)
        fig.add_trace(go.Scatter(x=self.model.time, y=y, name='Total Consumption'), row=2, col=1)

        fig.add_trace(go.Scatter(x=self.model.time, y=self.model.battery_soc.value, name='battery SOC'), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.model.time, y=self.model.battery.value, name='battery'), row=3, col=1)

        fig.update_layout(template='plotly_dark')
        fig.show()

    def get_fvs(self):
        return [self.model.cap_wind, self.model.cap_solar, self.model.cap_grid, self.model.cap_battery]

    @classmethod
    def get_metrics(self, model, params, names_only=False):
        '''Calculate various performance metrics for a dispatched model'''
        if names_only:
            if hasattr(model, 'curtailment'):
                return ['LCOE', 'wind_capacity', 'wind_capacity_factor', 'solar_capacity', 'solar_capacity_factor',
                        'grid_capacity', 'battery_capacity','total_load', 'percent_local_generation', 'max_transmission_demand',
                        'curtailment_hours', 'curtailment_total']
            else:
                return ['LCOE', 'wind_capacity', 'wind_capacity_factor', 'solar_capacity', 'solar_capacity_factor',
                        'grid_capacity', 'battery_capacity','total_load', 'percent_local_generation', 'max_transmission_demand']

        try:
            if model.cap_wind.VALUE[0] > 0.0:
                wind_cap_fac = sum(model.wind.VALUE)/(model.cap_wind.VALUE[0] * len(model.wind.VALUE))
            else:
                wind_cap_fac = 0.0
            if model.cap_solar.VALUE[0] > 0.0:
                solar_cap_fac = sum(model.solar.VALUE)/(model.cap_solar.VALUE[0] * len(model.solar.VALUE))
            else:
                solar_cap_fac = 0.0

            total_local_load = sum(model.ts_resid) + sum(model.ts_indus)
            total_import = sum(model.gimport)
            pct_local_gen = (1 - total_import/total_local_load) * 100
            # if model.cap_wind.VALUE[0] > 0.0 or model.cap_solar.VALUE[0] > 0.0:
            #     total_local_gen = 0.0 # Only the part meeting local load, not oversupply
            #     for t in range(len(model.ts_resid)): # Uses ts_resid instead of m.time as if there are failed windows ts_resid is shorter
            #         load = model.ts_resid[t] + model.ts_indus[t]
            #         gen = model.wind[t] + model.solar[t]
            #         total_local_gen += min(load, gen)
            #     pct_local_gen = total_local_gen/total_local_load
            # else:
            #     pct_local_gen = 0.0

            cap_grid = get_gk_val_real(model.cap_grid)[0]
            grid_cap_cost = params['grid_cap_a']*np.arctan(cap_grid+params['grid_cap_b'])+ \
                            params['grid_cap_c'] + params['grid_cap_d']*(cap_grid-4)
            # try:
            #     # Need to make sure this is a float so it doesn't trigger operator overloading for the symbolic AML
            #     grid_cap_cost = model.grid_cap_cost.VALUE[0]
            # except:
            #     # If the optimization fails at any point then this will remain an int
            #     grid_cap_cost = get_gk_val_real(model.grid_cap_cost)

            # Intermediates are not copied over properly (or at all) so these must be recalculated
            cap_costs = params['solar_cap_cost']*model.cap_solar.VALUE[0]*params['system_lifetime']/params['solar_lifetime'] + \
                        params['wind_cap_cost']*model.cap_wind.VALUE[0]*params['system_lifetime']/params['wind_lifetime'] + \
                        grid_cap_cost*params['system_lifetime']/params['grid_lifetime'] + \
                        params['battery_cap_cost'] * model.cap_battery.VALUE[0]*params['system_lifetime']/params['battery_lifetime']
            fixed_costs = params['solar_fix_cost']*model.cap_solar.VALUE[0] + \
                        params['wind_fix_cost']*model.cap_wind.VALUE[0] + \
                        params['battery_fix_cost']*model.cap_battery.VALUE[0]
            var_costs = params['solar_var_cost']*np.sum(model.solar.VALUE) + \
                        params['wind_var_cost']*np.sum(model.wind.VALUE) + \
                        np.sum(np.array(model.gimport.VALUE)*np.array(model.ts_grid_price.VALUE) - \
                        np.array(model.gexport.VALUE)*params['buyback_price'])
            nhrs = len(model.time)
            lcoe = (cap_costs + fixed_costs*params['system_lifetime'] + var_costs/nhrs*hrs_per_yr*params['system_lifetime'])/ \
                    (model.avg_MW*hrs_per_yr*params['system_lifetime'])

            metrics = {
                'LCOE': lcoe, # Unfortunately we really do need to recalculate this from scratch
                'wind_capacity': model.cap_wind[0],
                'wind_capacity_factor': wind_cap_fac,
                'solar_capacity': model.cap_solar[0],
                'solar_capacity_factor': solar_cap_fac,
                'grid_capacity': model.cap_grid[0],
                'battery_capacity': model.cap_battery[0],
                'total_load': total_local_load,
                'percent_local_generation': pct_local_gen,
                'max_transmission_demand': max(np.max(model.gimport.VALUE[1:-1]), np.max(model.gexport.VALUE[1:-1])),
            }

            if hasattr(model, 'curtailment'):
                metrics['curtailment_hours'] = len([v for v in model.curtailment.VALUE if v > 0])
                metrics['curtailment_total'] = np.sum(model.curtailment.VALUE)

            return metrics
        except:
            print('Failed getting metrics. Model does not appear to have solved correctly.')
            metrics = {
                'LCOE': 0.0, # Unfortunately we really do need to recalculate this from scratch
                'wind_capacity': model.cap_wind.VALUE,
                'wind_capacity_factor': 0,
                'solar_capacity': model.cap_solar.VALUE,
                'solar_capacity_factor': 0,
                'grid_capacity': model.cap_grid.VALUE,
                'battery_capacity': model.cap_battery.VALUE,
                'total_load': 0.0,
                'percent_local_generation': 0.0,
                'max_transmission_demand': 0.0,
            }

            if hasattr(model, 'curtailment'):
                metrics['curtailment_hours'] = 0
                metrics['curtailment_total'] = 0

            return metrics

