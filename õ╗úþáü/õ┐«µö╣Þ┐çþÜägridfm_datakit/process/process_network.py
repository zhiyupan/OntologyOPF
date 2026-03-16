import os
import sys
import yaml
import time
import shutil
import gc
import copy
import traceback
from datetime import datetime
from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import pandapower as pp
from queue import Queue
from gridfm_datakit.utils.config import PQ, PV, REF
from pandapower.auxiliary import pandapowerNet
from gridfm_datakit.process.solvers import run_opf, run_pf
from gridfm_datakit.utils.stats import Stats, plot_stats, plot_feature_distributions
from gridfm_datakit.utils.param_handler import NestedNamespace, get_load_scenario_generator, initialize_topology_generator, initialize_generation_generator, initialize_admittance_generator
from gridfm_datakit.network import load_net_from_pp, load_net_from_file, load_net_from_pglib
from gridfm_datakit.perturbations.load_perturbation import load_scenarios_to_df, plot_load_scenarios_combined
from gridfm_datakit.utils.utils import Tee
from gridfm_datakit.save import save_edge_params, save_bus_params
from gridfm_datakit.perturbations.topology_perturbation import TopologyGenerator
from gridfm_datakit.perturbations.generator_perturbation import GenerationGenerator
from gridfm_datakit.perturbations.admittance_perturbation import AdmittanceGenerator
from gridfm_datakit.perturbations.topology_perturbation import TopologyGenerator
from pandapower import makeYbus_pypower
from pandapower.pypower.idx_brch import BR_STATUS




def network_preprocessing(net: pandapowerNet) -> None:
    """Adds names to bus dataframe and bus types to load, bus, gen, sgen dataframes.

    This function performs several preprocessing steps:

    1. Assigns names to all network components
    2. Determines bus types (PQ, PV, REF)
    3. Assigns bus types to connected components
    4. Performs validation checks on the network structure

    Args:
        net: The power network to preprocess.

    Raises:
        AssertionError: If network structure violates expected constraints:
            - More than one load per bus
            - REF bus not matching ext_grid connection
            - PQ bus definition mismatch
    """
    # Clean-Up things in Data-Frame // give numbered item names
    for i, row in net.bus.iterrows():
        net.bus.at[i, "name"] = "Bus " + str(i)
    for i, row in net.load.iterrows():
        net.load.at[i, "name"] = "Load " + str(i)
    for i, row in net.sgen.iterrows():
        net.sgen.at[i, "name"] = "Sgen " + str(i)
    for i, row in net.gen.iterrows():
        net.gen.at[i, "name"] = "Gen " + str(i)
    for i, row in net.shunt.iterrows():
        net.shunt.at[i, "name"] = "Shunt " + str(i)
    for i, row in net.ext_grid.iterrows():
        net.ext_grid.at[i, "name"] = "Ext_Grid " + str(i)
    for i, row in net.line.iterrows():
        net.line.at[i, "name"] = "Line " + str(i)
    for i, row in net.trafo.iterrows():
        net.trafo.at[i, "name"] = "Trafo " + str(i)

    num_buses = len(net.bus)
    bus_types = np.zeros(num_buses, dtype=int)

    # assert one slack bus
    assert len(net.ext_grid) == 1
    indices_slack = np.unique(np.array(net.ext_grid["bus"]))

    indices_PV = np.union1d(
        np.unique(np.array(net.sgen["bus"])),
        np.unique(np.array(net.gen["bus"])),
    )
    indices_PV = np.setdiff1d(
        indices_PV,
        indices_slack,
    )  # Exclude slack indices from PV indices

    indices_PQ = np.setdiff1d(
        np.arange(num_buses),
        np.union1d(indices_PV, indices_slack),
    )

    bus_types[indices_PQ] = PQ  # Set PV bus types to 1
    bus_types[indices_PV] = PV  # Set PV bus types to 2
    bus_types[indices_slack] = REF  # Set Slack bus types to 3

    net.bus["type"] = bus_types

    # assign type of the bus connected to each load and generator
    net.load["type"] = net.bus.type[net.load.bus].to_list()
    net.gen["type"] = net.bus.type[net.gen.bus].to_list()
    net.sgen["type"] = net.bus.type[net.sgen.bus].to_list()

    # there is no more than one load per bus:
    assert net.load.bus.unique().shape[0] == net.load.bus.shape[0]

    # REF bus is bus with ext grid:
    assert (
        np.where(net.bus["type"] == REF)[0]  # REF bus indicated by case file
        == net.ext_grid.bus.values
    ).all()  # Buses connected to an ext grid

    # PQ buses are buses with no gen nor ext_grid, only load or nothing connected to them
    assert (
        (net.bus["type"] == PQ)  # PQ buses indicated by case file
        == ~np.isin(
            range(net.bus.shape[0]),
            np.concatenate(
                [net.ext_grid.bus.values, net.gen.bus.values, net.sgen.bus.values],
            ),
        )
    ).all()  # Buses which are NOT connected to a gen nor an ext grid


def pf_preprocessing(net: pandapowerNet) -> pandapowerNet:
    """Sets variables to the results of OPF.

    Updates the following network components with OPF results:

    - sgen.p_mw: active power generation for static generators
    - gen.p_mw, gen.vm_pu: active power and voltage magnitude for generators

    Args:
        net: The power network to preprocess.

    Returns:
        The updated power network with OPF results.
    """
    net.sgen[["p_mw"]] = net.res_sgen[
        ["p_mw"]
    ]  # sgens are not voltage controlled, so we set P only
    net.gen[["p_mw", "vm_pu"]] = net.res_gen[["p_mw", "vm_pu"]]
    return net


def pf_post_processing(net: pandapowerNet, dcpf: bool = False) -> np.ndarray:
    """Post-processes PF data to build the final data representation.

    Creates a matrix of shape (n_buses, 10) or (n_buses, 12) for DC power flow,
    with columns: (bus, Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF) plus (Vm_dc, Va_dc)
    for DC power flow.

    Args:
        net: The power network to process.
        dcpf: Whether to include DC power flow results. Defaults to False.

    Returns:
        numpy.ndarray: Matrix containing the processed power flow data.
    """
    X = np.zeros((net.bus.shape[0], 12 if dcpf else 10))
    all_loads = (
        pd.concat([net.res_load])[["p_mw", "q_mvar", "bus"]].groupby("bus").sum()
    )

    all_gens = (
        pd.concat([net.res_gen, net.res_sgen, net.res_ext_grid])[
            ["p_mw", "q_mvar", "bus"]
        ]
        .groupby("bus")
        .sum()
    )

    assert (net.bus.index.values == list(range(X.shape[0]))).all()

    X[:, 0] = net.bus.index.values

    # Active and reactive power demand
    X[all_loads.index, 1] = all_loads.p_mw  # Pd
    X[all_loads.index, 2] = all_loads.q_mvar  # Qd

    # Active and reactive power generated
    X[net.bus.type == PV, 3] = all_gens.p_mw[
        net.res_bus.type == PV
    ]  # active Power generated
    X[net.bus.type == PV, 4] = all_gens.q_mvar[
        net.res_bus.type == PV
    ]  # reactive Power generated
    X[net.bus.type == REF, 3] = all_gens.p_mw[
        net.res_bus.type == REF
    ]  # active Power generated
    X[net.bus.type == REF, 4] = all_gens.q_mvar[
        net.res_bus.type == REF
    ]  # reactive Power generated

    # Voltage
    X[:, 5] = net.res_bus.vm_pu  # voltage magnitude
    X[:, 6] = net.res_bus.va_degree  # voltage angle
    X[:, 7:10] = pd.get_dummies(net.bus["type"]).values

    if dcpf:
        X[:, 10] = net.bus["Vm_dc"]
        X[:, 11] = net.bus["Va_dc"]
    return X
def opf_post_processing(net: pandapowerNet) -> np.ndarray:
    n = net.bus.shape[0]
    X = np.zeros((n, 10))
    all_loads = pd.concat([net.res_load])[["p_mw", "q_mvar", "bus"]].groupby("bus").sum()
    all_gens = pd.concat([net.res_gen, net.res_sgen, net.res_ext_grid])[["p_mw", "q_mvar", "bus"]].groupby("bus").sum()
    X[:, 0] = net.bus.index.values
    X[all_loads.index, 1] = all_loads.p_mw; X[all_loads.index, 2] = all_loads.q_mvar
    X[all_gens.index, 3] = all_gens.p_mw; X[all_gens.index, 4] = all_gens.q_mvar
    X[:, 5] = net.res_bus.vm_pu; X[:, 6] = net.res_bus.va_degree
    X[:, 7:10] = pd.get_dummies(net.bus["type"]).values
    return X

def opf_edge_post_processing(net: pandapowerNet) -> pd.DataFrame:
    """从 OPF 结果中提取线路和变压器的潮流数据。"""
    line_results = net.res_line[["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar", "loading_percent"]].copy()
    line_results['element_type'] = 'line'
    
    trafo_results = net.res_trafo[["p_hv_mw", "q_hv_mvar", "p_lv_mw", "q_lv_mvar", "loading_percent"]].copy()
    trafo_results.rename(columns={"p_hv_mw": "p_from_mw", "q_hv_mvar": "q_from_mvar", "p_lv_mw": "p_to_mw", "q_lv_mvar": "q_to_mvar"}, inplace=True)
    trafo_results['element_type'] = 'trafo'

    all_edge_results = pd.concat([line_results, trafo_results], ignore_index=True)
    return all_edge_results

def get_adjacency_list(net: pandapowerNet) -> np.ndarray:
    """Gets adjacency list for network.

    Creates an adjacency list representation of the network's bus admittance matrix,
    including real and imaginary components of the admittance.

    Args:
        net: The power network.

    Returns:
        numpy.ndarray: Array containing edge indices and attributes (G, B).
    """
    ppc = net._ppc
    Y_bus, Yf, Yt = makeYbus_pypower(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    i, j = np.nonzero(Y_bus)
    # note that Y_bus[i,j] can be != 0 even if a branch from i to j is not in service because there might be other branches connected to the same buses

    s = Y_bus[i, j]
    G = np.real(s)
    B = np.imag(s)

    edge_index = np.column_stack((i, j))
    edge_attr = np.stack((G, B)).T
    adjacency_lists = np.column_stack((edge_index, edge_attr))
    return adjacency_lists


def get_branch_idx_removed(branch: np.ndarray) -> List[int]:
    """Gets indices of removed branches in the network.

    Args:
        branch: Branch data array from the network.

    Returns:
        List of indices of branches that are out of service (= removed when applying topology perturbations)
    """
    in_service = branch[:, BR_STATUS]
    return np.where(in_service == 0)[0].tolist()


def process_scenario_contingency(
    net: pandapowerNet,
    scenarios: np.ndarray,
    scenario_index: int,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    no_stats: bool,
    local_csv_data: List[np.ndarray],
    local_adjacency_lists: List[np.ndarray],
    local_branch_idx_removed: List[List[int]],
    local_stats: Union[Stats, None],
    error_log_file: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]], Union[Stats, None]]:
    """Processes a load scenario for contingency analysis.

    Args:
        net: The power network.
        scenarios: Array of load scenarios.
        scenario_index: Index of the current scenario.
        topology_generator: Topology perturbation generator.
        generation_generator: Generator cost perturbation generator.
        admittance_generator: Line admittance perturbation generator.
        no_stats: Whether to skip statistics collection.
        local_csv_data: List to store processed CSV data.
        local_adjacency_lists: List to store adjacency lists.
        local_branch_idx_removed: List to store removed branch indices.
        local_stats: Statistics object for collecting network statistics.
        error_log_file: Path to error log file.

    Returns:
        Tuple containing:
            - List of processed CSV data
            - List of adjacency lists
            - List of removed branch indices
            - Statistics object
    """
    net = copy.deepcopy(net)

    # apply the load scenario to the network
    net.load.p_mw = scenarios[:, scenario_index, 0]
    net.load.q_mvar = scenarios[:, scenario_index, 1]

    # first run OPF to get the gen set points
    try:
        run_opf(net)
    except Exception as e:
        with open(error_log_file, "a") as f:
            f.write(
                f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n",
            )
        return (
            local_csv_data,
            local_adjacency_lists,
            local_branch_idx_removed,
            local_stats,
        )

    net_pf = copy.deepcopy(net)
    net_pf = pf_preprocessing(net_pf)

    # Generate perturbed topologies
    perturbations = topology_generator.generate(net_pf)

    # Apply generation perturbations
    perturbations = generation_generator.generate(perturbations)

    # Apply admittance perturbations
    perturbations = admittance_generator.generate(perturbations)

    # to simulate contingency, we apply the topology perturbation after OPF
    for perturbation in perturbations:
        try:
            # run DCPF for benchmarking purposes
            pp.rundcpp(perturbation)
            perturbation.bus["Vm_dc"] = perturbation.res_bus.vm_pu
            perturbation.bus["Va_dc"] = perturbation.res_bus.va_degree
            # run AC-PF to get the new state of the network after contingency (we don't model any remedial actions)
            run_pf(perturbation)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} when solving dcpf or in in run_pf function: {e}\n",
                )

                continue

                # TODO: What to do when the network does not converge for AC-PF? -> we dont have targets for regression!!

        # Append processed power flow data
        local_csv_data.extend(pf_post_processing(perturbation, dcpf=True))
        local_adjacency_lists.append(get_adjacency_list(perturbation))
        local_branch_idx_removed.append(
            get_branch_idx_removed(perturbation._ppc["branch"]),
        )
        if not no_stats:
            local_stats.update(perturbation)

    return local_csv_data, local_adjacency_lists, local_branch_idx_removed, local_stats


def process_scenario_chunk(
    mode, start_idx: int, end_idx: int, scenarios: np.ndarray, net: pandapowerNet,
    progress_queue, topology_generator, generation_generator, admittance_generator,
    no_stats: bool, error_log_path,
) -> Tuple:
    """
    Processes a chunk of scenarios by calling process_scenario for each one.
    This version is upgraded to handle PF and a full set of OPF results (node, edge, cost).
    """

    try:
        local_stats = Stats() if not no_stats else None
        local_pf_node_data, local_pf_edge_data, local_pf_branch_removed = [], [], []
        local_opf_node_data, local_opf_edge_data, local_opf_cost_data, local_opf_gen_data = [], [], [], []
        local_opf_times = []  # ✏️ 时间列表

        for scenario_index in range(start_idx, end_idx):

            if mode == "pf":
                process_scenario(
                    net, scenarios, scenario_index,
                    topology_generator, generation_generator, admittance_generator,
                    no_stats,
                    local_pf_node_data, local_pf_edge_data, local_pf_branch_removed,
                    local_stats, error_log_path,
                    local_opf_node_data, local_opf_edge_data, local_opf_cost_data, local_opf_gen_data,
                    local_opf_times  # ✏️ 传递时间列表
                )
            
            progress_queue.put(1)

        return (
            None, None,
            local_pf_node_data, local_pf_edge_data, local_pf_branch_removed,
            local_stats,
            local_opf_node_data, local_opf_edge_data, local_opf_cost_data, local_opf_gen_data,
            local_opf_times  # ✏️ 返回时间数据
        )
        
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Caught an exception in process_scenario_chunk function: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        return e, traceback.format_exc(), None, None, None, None, None, None, None, None


def process_scenario(
    net: 'pandapowerNet',
    scenarios: 'np.ndarray',
    scenario_index: int,
    topology_generator,
    generation_generator,
    admittance_generator,
    no_stats: bool,
    local_pf_node_data: list,
    local_pf_edge_data: list,
    local_pf_branch_removed: list,
    local_stats: 'Stats | None',
    error_log_file: str,
    local_opf_node_data: list,
    local_opf_edge_data: list,
    local_opf_cost_data: list,
    local_opf_gen_data: list,
    local_opf_times: list  # ✏️ 时间列表参数
) -> None:
    """
    ✏️ MODIFIED: 使用连续编号记录每条成功扰动的OPF时间
    处理单个场景：每个扰动执行 OPF + PF，并记录 OPF 时间。
    """
    import copy

    net.load.p_mw = scenarios[:, scenario_index, 0]
    net.load.q_mvar = scenarios[:, scenario_index, 1]

    perturbations = topology_generator.generate(net)
    perturbations = generation_generator.generate(perturbations)
    perturbations = admittance_generator.generate(perturbations)

    # ✏️ MODIFIED: 使用局部计数器，不再依赖原始scenario_index和perturbation_id
    # 时间记录将在成功时使用len(local_opf_times)作为连续编号
    
    for perturbation_id, perturbation in enumerate(perturbations):
        t0 = time.time()
        opf_success = False
        
        try:
            run_opf(perturbation)
            opf_success = True

            # --- 收集 OPF 数据 ---
            opf_node_matrix = opf_post_processing(perturbation)
            local_opf_node_data.extend(opf_node_matrix)

            opf_edge_df = opf_edge_post_processing(perturbation)
            local_opf_edge_data.append(opf_edge_df)

            local_opf_cost_data.append(perturbation.res_cost)

            opf_gen_df = perturbation.res_gen[["p_mw", "q_mvar", "vm_pu"]].copy()
            opf_gen_df["gen_bus"] = perturbation.gen["bus"].values
            opf_gen_df["scenario_id"] = scenario_index
            local_opf_gen_data.append(opf_gen_df)

        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index}, "
                    f"perturbation {perturbation_id} in run_opf: {e}\n"
                )

        elapsed = time.time() - t0
        
        # ✏️ MODIFIED: 只有成功的扰动才记录时间，并使用连续编号
        if opf_success:
            # 使用当前列表长度作为连续编号
            continuous_scenario_id = len(local_opf_times)
            local_opf_times.append({
                'scenario_id': continuous_scenario_id,  # ✏️ 连续编号
                'elapsed_time': elapsed,
                'success': True
            })

        # === PF 部分逻辑 ===
        if not opf_success:
            continue
            
        net_pf = copy.deepcopy(perturbation)
        net_pf = pf_preprocessing(net_pf)
        try:
            run_pf(net_pf)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index}, "
                    f"perturbation {perturbation_id} in run_pf: {e}\n"
                )
            continue

        local_pf_node_data.extend(pf_post_processing(net_pf))
        local_pf_edge_data.append(get_adjacency_list(net_pf))
        local_pf_branch_removed.append(get_branch_idx_removed(net_pf._ppc["branch"]))
        if not no_stats and local_stats is not None:
            local_stats.update(net_pf)