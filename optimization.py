import psutil
import math
import tqdm
import copy
import numpy as np
import pandas as pd
from collections.abc import Iterable
from multiprocessing import Pool
from scipy.stats.qmc import LatinHypercube, scale as sample_scaler
from scipy import stats
import multiprocessing as mp
import time
from util import get_gk_val_real


def get_gk_val_len(gk_val):
    # Robustly determine the "length" of the GK_Value
    try:
        val = gk_val.value.value.value
    except:
        try:
            val = gk_val.value.value
        except:
            val = gk_val.value
    l = len(val) if isinstance(val, Iterable) else 1
    return l

def make_windows_overlapping(model, tau, overlap=1):
    '''Break a complete time horizon for a model up into windows based on tau.
    Windows overlaping by the sepcified overlap will be returned.'''
    try:
        model_len = len(model.time)
    except Exception as e:
        print("GEKKO model must be dynamic. m.time must be defined")
        raise(e)

    if overlap >= tau:
        raise Exception('The window overlap must be smaller than tau')
    windows = []
    start = 0
    end = 0
    while end < model_len:
        end = start + tau

        if end > model_len:
            end = model_len
        windows.append({
            'start': start,
            'end': end
        })
        start = end - overlap
    return windows


def make_windows(model, tau, dispatch_only=True):
    '''Break a complete time horizon for a model up into windows based on tau
    If dispatch_only is True windows overlapping by 1 point will be returned,
    otherwise, all possible contiguous windows of length tau will be returned.'''
    try:
        model_len = len(model.time)
    except Exception as e:
        print("GEKKO model must be dynamic. m.time must be defined")
        raise(e)
    windows = []
    start = 0
    end = 0
    while end < model_len:
        if dispatch_only:
            end = end + tau
        else:
            end = start + tau
        if end > model_len:
            end = model_len
        windows.append({
            'start': start,
            'end': end
        })
        if dispatch_only:
            start = end - 1
        else:
            start = start + 1
    return windows

def solve_window(w):
    # Make and solve the model. Done here to minimize RAM usage
    model = adjust_window(w['model_generator'],  w['start'], w['end'])
    try:
        model.solve(disp=False)
        fvs = {f'fv_{fv}': float(model._parameters[fv].VALUE[0]) for fv in w['fv_indicies']}
    except Exception as e:
        print(e)
        fvs = {f'fv_{fv}': None for fv in w['fv_indicies']}
    model.cleanup() # Clean files from disk to reduce disk usage

    return {
        'obj': float(model.options.OBJFCNVAL),
        'time': float(model.options.SOLVETIME),
        'LCOE': float(model.LCOE[-1]),
        **fvs
    }

def adjust_window(model_generator, start, end, prev_win=None, overlap=1, **kwargs):
    # Make a new model over the desired sub-window
    model = model_generator(end-start, nshift=start, **kwargs)
    if prev_win:
        # Set the new window to start where the previous window ended
        for i, v in enumerate(model._variables):
            l = get_gk_val_len(v)
            try:
                v.value[0] = prev_win._variables[i].value[-overlap]
            except:
                v.value = prev_win._variables[i].value[-overlap]
    return model

def determine_indicies(model, fvs):
    '''Determine the indicies for the model FVs to allow direct accessing into the GEKKO model'''
    fv_indicies = []

    # 0. Find the FVs and assemble the FV map (reference: index)
    for fv in fvs:
        for i, p in enumerate(model._parameters):
            if p is fv:
                fv_indicies.append(i)

    if len(fv_indicies) < len(fvs):
        print(fv_indicies)
        raise Exception('Failed to find one or more specified FVs in the model')

    return fv_indicies

def measure_technoeconomics_overlapping(model_class, window_length, r, overlap):
    # FIXME: This has been limited to just the particular model and case I am using right now
    # to speed getting to the results
    start = time.time()
    fv_indicies = []
    obj_vals = []
    infeas = 0

    # 0. Find the FVs and assemble the FV map (reference: index)
    for fv in model_class.get_fvs():
        for i, p in enumerate(model_class.model._parameters):
            if p is fv:
                fv_indicies.append(i)

    if len(fv_indicies) < len(model_class.get_fvs()):
        print(fv_indicies)
        raise Exception('Failed to find one or more specified FVs in the model')

    # Make a new base model with capacities fixed at the chosen sizes
    fixed_model = model_class.model_generator(24*365, cap_wind=r.fv_6, cap_solar=r.fv_7,
                                        cap_grid=r.fv_8, cap_battery=r.fv_9)

    def fixed_model_generator(hrs, nshift=0):
        # FIXME: Generalize this to work with all models
        fixed_model = model_class.model_generator(hrs, nshift=nshift,
                                cap_wind=r.fv_6, cap_solar=r.fv_7, cap_grid=r.fv_8, cap_battery=r.fv_9)
        # for i, fv in enumerate(fv_indicies):
        #     fixed_model._parameters[fv].STATUS = 0
        #     fixed_model._parameters[fv].VALUE = r[f'fv_{fv}']
        return fixed_model

    # 5. Perform rolling-window dispatch with the fixed sizes
    sum_obj_funs = 0
    windows = make_windows_overlapping(fixed_model, window_length, overlap=overlap)
    for i, w in enumerate(windows):
        # Solve the window using the previous window
        prev_win = None
        if i != 0:
            prev_win = windows[i-1]['model']
        win_model = adjust_window(fixed_model_generator, w['start'], w['end'], prev_win=prev_win, overlap=overlap)
        # if not self.silent:
        #     print(f'Solving window {i+1}/{len(windows)}...', end='')

        win_model.solver_options = ['max_iter 10000']
        try:
            win_model.solve(disp=False)
            win_model.cleanup() # helps keep the disk from filling up. :)
            w['model'] = win_model
            if win_model.options.APPSTATUS != 1:
                w['model'] = None
                infeas += 1
            else:
                obj_vals.append(win_model.options.OBJFCNVAL)
        except:
            win_model.cleanup() # helps keep the disk from filling up. :)
            w['model'] = None
            infeas += 1

        sum_obj_funs += win_model.options.OBJFCNVAL
        # if not self.silent:
        #     print('\tSuccess' if win_model.options.APPSTATUS == 1 else f'\tFailed APPSTATUS: {win_model.options.APPSTATUS}')

    # Assemble the full solution
    # Iterate over each of the parameters and variables for each solved window
    # FIXME: the overlapping is not verified!
    i = 0
    model = model_class.model
    for w in windows:
        if w['model'] is not None:
            for j, v in enumerate(w['model']._variables):
                if i == 0:
                    model._variables[j].VALUE = get_gk_val_real(v)[:-overlap]
                # Not sure why appending doesn't work here, but it doesn't
                else:
                    model._variables[j].VALUE = get_gk_val_real(model._variables[j].VALUE) + get_gk_val_real(v)[:-overlap]
            for j, p in enumerate(w['model']._parameters):
                if i == 0:
                    model._parameters[j].VALUE = get_gk_val_real(p)[:-overlap]
                else:
                    model._parameters[j].VALUE = get_gk_val_real(model._parameters[j].VALUE) + get_gk_val_real(p)[:-overlap]
            i += 1
    # Write out the fixed capacities
    for i, fv in enumerate(fv_indicies):
        model._parameters[fv].VALUE = r[f'fv_{fv}'] * np.ones(len(model_class.model.time))

    return infeas, *model_class.get_metrics(model, model_class.params).values(), model


def determine_distributions(model, tau, fv_indicies, model_generator, silent=False):
    '''Run the novel combined design and dispatch optimization algorithm on a model.
    Note that the model passed in must not be solved beforehand to allow for deepcopying internally

    model: the model to be optimized
    tau: the critical time length to be used
    checkpoint: whether to write out a checkpoint file of the optimal subwindows
    '''
    if not silent:
        print(f'Problem Info - hrs: {len(model.time)}, tau: {tau}, fvs: {len(fv_indicies)}')

    # 1. Break the time horizon into time windows
    windows = make_windows(model, tau, dispatch_only=False)

    # 2. Solve each window
    # Note that the previous window normally must be solved before the model for the next window can be created
    # but in this case we are not requiring the windows to line up correctly. The dispatch of each window is
    # independent of the others around it.
    solved_windows = []
    nthreads = psutil.cpu_count() - 2

    # Need to batch the runs into groups of 1000 or less so we don't run out of RAM...
    batch_size = 1000
    nBatches = math.ceil(len(windows)/batch_size)
    for b in range(nBatches):
        window_batch = windows[batch_size*b:(b+1)*batch_size]

        for i, w in enumerate(window_batch):
            w['model_generator'] = model_generator
            w['fv_indicies'] = fv_indicies

        with Pool(processes=nthreads) as pool:
            for result in tqdm.tqdm(pool.imap(solve_window, window_batch), total=len(window_batch), desc=f'Batch {b+1}/{nBatches}'):
                solved_windows.append(result)

    data = pd.DataFrame(solved_windows)
    return data

def generate_representative_samples_copula(dists: pd.DataFrame, fvs, n_samples: int):
    # Fit each of the variables to a random variable
    # Using uniform distributions for right now. Could change this to fit the real distributions,
    # but this could be quite challenging particularly when the bounds are active

    # The battery turns out to always be the same value, so no sampling required
    fvs = fvs[:-1]
    batt_cap = max(dists[f'fv_{fvs[-1]}'])

    rvs = []
    uniform = stats.uniform()
    for fv in fvs:
        # Get the bounds on the right MV
        lower = dists[f'fv_{fv}'].min()
        upper = dists[f'fv_{fv}'].max()
        rvs.append({
            'lower': lower,
            'upper': upper
        })
        print(f'fv: {fv}, lower: {lower}, upper: {upper}')

    # Generate the covariance matrix
    cov = dists[[f'fv_{fv}' for fv in fvs]].corr()

    # Generate the copula for sampling.
    L = np.linalg.cholesky(cov)

    # Sample the required number of samples
    samples = []
    for _ in range(n_samples):
        Z = [uniform.rvs() for fv in range(len(fvs))]
        # Z = [rvs[fv]['lower'] + uniform.rvs()*(rvs[fv]['upper'] - rvs[fv]['lower']) for fv in range(len(fvs))]
        X = L@Z
        u = stats.uniform.cdf(X)
        sample = [ rvs[fv]['lower'] + s*(rvs[fv]['upper'] - rvs[fv]['lower']) for s, fv in zip(stats.uniform().ppf(u), range(len(fvs)))]
        samples.append([*sample, batt_cap])
    return samples

def measure_technoeconomics(r: pd.Series, model_class, tau):
    infeas = 0
    obj_vals = []
    # Run the dispatch problem here, restarting with every infeasibility and incrementing the timer
    fv_indicies = []

    # 0. Find the FVs and assemble the FV map (reference: index)
    for fv in model_class.get_fvs():
        for i, p in enumerate(model_class.model._parameters):
            if p is fv:
                fv_indicies.append(i)

    if len(fv_indicies) < len(model_class.get_fvs()):
        print(fv_indicies)
        raise Exception('Failed to find one or more specified FVs in the model')

    # Make a new base model with capacities fixed at the chosen sizes
    def fixed_model_generator(hrs, nshift=0):
        # FIXME: Generalize this to work with all models
        fixed_model = model_class.model_generator(hrs, nshift=nshift,
                                cap_wind=r.fv_6, cap_solar=r.fv_7, cap_grid=r.fv_8, cap_battery=r.fv_9)
        # for i, fv in enumerate(fv_indicies):
        #     fixed_model._parameters[fv].STATUS = 0
        #     fixed_model._parameters[fv].VALUE = r[f'fv_{fv}']
        return fixed_model

    # 5. Perform rolling-window dispatch with the fixed sizes
    windows = make_windows(model_class.model, tau)

    for i, w in enumerate(windows):
        # Solve the window using the previous window
        prev_win = None
        if i != 0:
            prev_win = windows[i-1]['model']
        win_model = adjust_window(fixed_model_generator, w['start'], w['end'], prev_win=prev_win)
        try:
            win_model.solve(disp=False)
            win_model.cleanup() # helps keep the disk from filling up. :)
            w['model'] = win_model
            if win_model.options.APPSTATUS != 1:
                w['model'] = None
                infeas += 1
            else:
                obj_vals.append(win_model.options.OBJFCNVAL)
        except:
            win_model.cleanup() # helps keep the disk from filling up. :)
            w['model'] = None
            infeas += 1

    # Reassemble a full model
    model = model_class.model
    i = 0
    for w in windows:
        # Note! failed windows are simply left out. This makes the coresponding metrics invalid.
        if w['model'] is not None:
            for j, v in enumerate(w['model']._variables):
                if i == 0:
                    model._variables[j].VALUE = get_gk_val_real(v)
                # Not sure why appending doesn't work here, but it doesn't
                else:
                    model._variables[j].VALUE = get_gk_val_real(model._variables[j].VALUE) + get_gk_val_real(v)[1:]
            for j, p in enumerate(w['model']._parameters):
                if i == 0:
                    model._parameters[j].VALUE = get_gk_val_real(p)
                else:
                    model._parameters[j].VALUE = get_gk_val_real(model._parameters[j].VALUE) + get_gk_val_real(p)[1:]
            i += 1 #Tracking this way handles failed first windows properly.
    # Write out the fixed capacities
    for i, fv in enumerate(fv_indicies):
        model._parameters[fv].VALUE = r[f'fv_{fv}'] * np.ones(len(model.time))

    # return infeas, *model_class.get_metrics(model, model_class.params).values()
    return infeas, *model_class.get_metrics(model, model_class.params).values(), model

def determine_feasibility_surface(dists: pd.DataFrame, tau, model_class,
                                  n_samples):
    '''Evaluates the system at multivariate samples of the distributions between the lower and upper percentage bounds.
    Creates a techno-economic surface describing both the feasibility of the system at given capacities as well as the
    corresponding LCOE'''

    samples = dists.sample(n=n_samples, axis=0)

    # Columns are 0, 1, 2, ...
    d = pd.DataFrame(samples)

    available_metrics = model_class.get_metrics(None, None, names_only=True)

    # Calculate the resulting feasibility and LCOE values
    # d[['dispatch_infeas', *available_metrics]] = d.apply(measure_technoeconomics, axis=1, result_type='expand')
    # with mp.Pool(mp.cpu_count()) as pool:
    #     big_iterable = [(r[1], model_class, tau) for r in d.iterrows()]
    #     stuff = pool.starmap(measure_technoeconomics, big_iterable)
    with mp.Pool(mp.cpu_count()) as pool:
        big_iterable = [(model_class, tau, r[1], int(tau/2)) for r in d.iterrows()]
        stuff = pool.starmap(measure_technoeconomics_overlapping, big_iterable)
    # def measure_technoeconomics_overlapping(model, window_length, r, overlap, params):
    # def measure_technoeconomics(r: pd.Series, model_class, tau):

    data = pd.DataFrame(stuff)
    data.rename({
        '0': 'Infeasibilities',
        '1': 'LCOE',
        '2': 'Wind Capacity',
        '3': 'Wind Capacity Factor',
        '4': 'Solar Capacity',
        '5': 'Solar Capacity Factor',
        '6': 'Grid Capacity',
        '7': 'Battery Capacity',
        '8': 'Total Load',
        '9': 'Percent Local Generation',
        '10': 'Max Transmission Demand',
    }, inplace=True, axis=1)
    return data


def analyze_capacities(model, model_generator, capacities, tau, fv_indicies, silent=False):
    '''Dispatch a model using rolling windows for windows of length tau. Returns a single, full, solved model'''
    # 4. Choose the sizes

    # Make a new base model with capacities fixed at the chosen sizes
    fixed_model = copy.deepcopy(model)
    if aggregator is None:
        aggregator = max
    for i, fv in enumerate(fv_indicies):
        fixed_model._parameters[fv].STATUS = 0
        fixed_model._parameters[fv].VALUE = capacities[i]

    # 5. Perform rolling-window dispatch with the fixed sizes
    windows = make_windows(fixed_model, tau)
    for i, w in enumerate(windows):
        # Solve the window using the previous window
        prev_win = None
        if i != 0:
            prev_win = windows[i-1]['model']
        win_model = adjust_window(model_generator, w['start'], w['end'], prev_win=prev_win)
        if not silent:
            print(f'Solving window {i+1}/{len(windows)}...', end='')
        win_model.solve(disp=False)
        win_model.cleanup() # helps keep the disk from filling up. :)
        if not silent:
            print('\tSuccess' if win_model.options.APPSTATUS == 1 else '\tFailed')
        w['model'] = win_model

    # 6. Return the full solution
    # Iterate over each of the parameters and variables for each solved window
    for i, w in enumerate(windows):
        for j, v in enumerate(w['model']._variables):
            if i == 0:
                model._variables[j].VALUE = get_gk_val_real(v)
            # Not sure why appending doesn't work here, but it doesn't
            else:
                model._variables[j].VALUE = get_gk_val_real(model._variables[j].VALUE) + get_gk_val_real(v)[1:]
        for j, p in enumerate(w['model']._parameters):
            if i == 0:
                model._parameters[j].VALUE = get_gk_val_real(p)
            else:
                model._parameters[j].VALUE = get_gk_val_real(model._parameters[j].VALUE) + get_gk_val_real(p)[1:]

    # Write out the fixed capacities
    for fv in fv_indicies:
        model._parameters[fv].VALUE = capacities[i] * np.ones(len(model.time))

    return model
