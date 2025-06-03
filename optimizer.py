# optimizer.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def _evaluate_combo(args):
    """
    Avalia uma combinação (n_groups, alvo) e retorna um dict com métricas.
    """
    (n_groups, alvo, vectors, weight_loss, weight_cost,
     generate_initial_groups, balancear_grupos,
     define_route, compute_joule_loss, compute_cable_cost) = args

    init_groups = generate_initial_groups(vectors, n_groups)
    balanced   = balancear_grupos(init_groups, vectors, alvo=alvo)
    route      = define_route(balanced)
    loss       = compute_joule_loss(route)
    cost       = compute_cable_cost(route)
    denom      = loss + cost if (loss + cost) > 0 else 1.0
    score      = weight_loss * (loss/denom) + weight_cost * (cost/denom)

    return {
        'n_groups': n_groups,
        'alvo'    : alvo,
        'loss'    : loss,
        'cost'    : cost,
        'score'   : score
    }

def optimize_parameters(vectors,
                                 n_groups_list,
                                 alvo_list,
                                 generate_initial_groups,
                                 balancear_grupos,
                                 define_route,
                                 compute_joule_loss,
                                 compute_cable_cost,
                                 weight_loss=0.5,
                                 weight_cost=0.5):
    """
    Grid search paralelo sobre n_groups e alvo minimizando score ponderado.
    """
    # monte lista de tarefas
    combos = []
    for n in n_groups_list:
        for a in alvo_list:
            combos.append((
                n, a, vectors,
                weight_loss, weight_cost,
                generate_initial_groups,
                balancear_grupos,
                define_route,
                compute_joule_loss,
                compute_cable_cost
            ))

    # paralelize com Pool
    n_procs = min(cpu_count(), len(combos))
    pbar = tqdm(total=len(combos), desc="Grid Search (paralelo)", unit="comb")
    with Pool(n_procs) as pool:
        results = []
        for res in pool.imap_unordered(_evaluate_combo, combos):
            results.append(res)
            pbar.update(1)
    pbar.close()

    df = pd.DataFrame(results)
    best_idx = df['score'].idxmin()
    best     = df.loc[best_idx].to_dict()
    return best, df
