import pandapower as pp
import numpy as np
import pandas as pd
from pandapower.auxiliary import pandapowerNet
import os
from pandapower.pypower.idx_brch import T_BUS, F_BUS, RATE_A
from pandapower.pypower.makeYbus import branch_vectors
from typing import List


def save_edge_params(net: pandapowerNet, path: str):
    """Saves edge parameters for the network to a CSV file.

    Extracts and saves branch parameters including admittance matrices and rate limits.

    Args:
        net: The power network.
        path: Path where the edge parameters CSV file should be saved.
    """
    pp.rundcpp(net)  # need to run dcpp to create the ppc structure
    ppc = net._ppc
    to_bus = np.real(ppc["branch"][:, T_BUS])
    from_bus = np.real(ppc["branch"][:, F_BUS])
    Ytt, Yff, Yft, Ytf = branch_vectors(ppc["branch"], ppc["branch"].shape[0])
    Ytt_r = np.real(Ytt)
    Ytt_i = np.imag(Ytt)
    Yff_r = np.real(Yff)
    Yff_i = np.imag(Yff)
    Yft_r = np.real(Yft)
    Yft_i = np.imag(Yft)
    Ytf_r = np.real(Ytf)
    Ytf_i = np.imag(Ytf)

    rate_a = np.real(ppc["branch"][:, RATE_A])
    edge_params = pd.DataFrame(
        np.column_stack(
            (
                from_bus,
                to_bus,
                Yff_r,
                Yff_i,
                Yft_r,
                Yft_i,
                Ytf_r,
                Ytf_i,
                Ytt_r,
                Ytt_i,
                rate_a,
            ),
        ),
        columns=[
            "from_bus",
            "to_bus",
            "Yff_r",
            "Yff_i",
            "Yft_r",
            "Yft_i",
            "Ytf_r",
            "Ytf_i",
            "Ytt_r",
            "Ytt_i",
            "rate_a",
        ],
    )
    # comvert everything to float32
    edge_params = edge_params.astype(np.float32)
    edge_params.to_csv(path, index=False)


def save_bus_params(net: pandapowerNet, path: str):
    """Saves bus parameters for the network to a CSV file.

    Extracts and saves bus parameters including voltage limits and base values.

    Args:
        net: The power network.
        path: Path where the bus parameters CSV file should be saved.
    """
    idx = net.bus.index
    base_kv = net.bus.vn_kv
    bus_type = net.bus.type
    vmin = net.bus.min_vm_pu
    vmax = net.bus.max_vm_pu

    bus_params = pd.DataFrame(
        np.column_stack((idx, bus_type, vmin, vmax, base_kv)),
        columns=["bus", "type", "vmin", "vmax", "baseKV"],
    )
    bus_params.to_csv(path, index=False)


def save_branch_idx_removed(branch_idx_removed: List[List[int]], path: str):
    """Saves indices of removed branches for each scenario.

    Appends the removed branch indices to an existing CSV file or creates a new one.

    Args:
        branch_idx_removed: List of removed branch indices for each scenario.
        path: Path where the branch indices CSV file should be saved.
    """
    if os.path.exists(path):
        existing_df = pd.read_csv(path, usecols=["scenario"])
        if not existing_df.empty:
            last_scenario = existing_df["scenario"].iloc[-1]
    else:
        last_scenario = -1

    scenario_idx = np.arange(
        last_scenario + 1,
        last_scenario + 1 + len(branch_idx_removed),
    )
    branch_idx_removed_df = pd.DataFrame(branch_idx_removed)
    branch_idx_removed_df.insert(0, "scenario", scenario_idx)
    branch_idx_removed_df.to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )  # append to existing file or create new one


def save_node_edge_data(net: pandapowerNet, node_path: str, edge_path: str, csv_data: list, adjacency_lists: list, mode: str = "pf"):
    n_buses = net.bus.shape[0]
    last_scenario = -1
    if os.path.exists(node_path):
        existing_df = pd.read_csv(node_path, usecols=["scenario"])
        if not existing_df.empty: last_scenario = existing_df["scenario"].iloc[-1]
    columns = ["bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"]
    if mode == "contingency": columns.extend(["Vm_dc", "Va_dc"])
    df = pd.DataFrame(csv_data, columns=columns)
    df["bus"] = df["bus"].astype("int64")
    scenario_indices = np.repeat(range(last_scenario + 1, last_scenario + 1 + (df.shape[0] // n_buses)), n_buses)
    df.insert(0, "scenario", scenario_indices)
    df.to_csv(node_path, mode="a", header=not os.path.exists(node_path), index=False)
    if adjacency_lists:
        adj_df = pd.DataFrame(np.concatenate(adjacency_lists), columns=["index1", "index2", "G", "B"])
        adj_df[["index1", "index2"]] = adj_df[["index1", "index2"]].astype("int64")
        scenario_indices_edge = np.concatenate([np.full(adjacency_lists[i].shape[0], last_scenario + 1 + i, dtype="int64") for i in range(len(adjacency_lists))])
        adj_df.insert(0, "scenario", scenario_indices_edge)
        adj_df.to_csv(edge_path, mode="a", header=not os.path.exists(edge_path), index=False)


def save_opf_node_data(net: pandapowerNet, path: str, opf_node_data: List[np.ndarray]):
    # 和 PF 一样：读取历史最后场景号 -> 连续编号 -> 追加写
    if not opf_node_data:
        return
    n_buses = net.bus.shape[0]

    last_scenario = -1
    if os.path.exists(path):
        existing_df = pd.read_csv(path, usecols=["scenario"])
        if not existing_df.empty:
            last_scenario = existing_df["scenario"].iloc[-1]

    # 本批场景数
    if len(opf_node_data) % n_buses != 0:
        raise ValueError("OPF node rows is not multiple of n_buses")
    num_scen_this_batch = len(opf_node_data) // n_buses

    columns = ["bus","Pd","Qd","Pg","Qg","Vm","Va","PQ","PV","REF"]
    df = pd.DataFrame(opf_node_data, columns=columns)
    df["bus"] = df["bus"].astype("int64")

    scenario_indices = np.repeat(
        range(last_scenario + 1, last_scenario + 1 + num_scen_this_batch),
        n_buses
    )
    df.insert(0, "scenario", scenario_indices)

    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_opf_edge_data(path: str, opf_edge_data: List[pd.DataFrame]):
    # 和 PF 一样：读取历史最后场景号 -> 连续编号 -> 追加写
    if not opf_edge_data:
        return

    last_scenario = -1
    if os.path.exists(path):
        try:
            existing_df = pd.read_csv(path, usecols=["scenario"])
            if not existing_df.empty:
                last_scenario = existing_df["scenario"].iloc[-1]
        except Exception:
            last_scenario = -1

    dfs = []
    for i, df in enumerate(opf_edge_data):
        d = df.copy()
        # 统一列名为 scenario（去掉旧的 scenario_id）
        if "scenario_id" in d.columns:
            d = d.drop(columns=["scenario_id"])
        d.insert(0, "scenario", last_scenario + 1 + i)
        dfs.append(d)

    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(path, mode="a", header=not os.path.exists(path), index=False)



def save_opf_cost_data(path: str, opf_cost_data: List[float]):
    # 和 PF 一样：读取历史最后场景号 -> 连续编号 -> 追加写
    if not opf_cost_data:
        return

    last_scenario = -1
    if os.path.exists(path):
        try:
            existing_df = pd.read_csv(path, usecols=["scenario"])
            if not existing_df.empty:
                last_scenario = existing_df["scenario"].iloc[-1]
        except Exception:
            last_scenario = -1

    scen = np.arange(last_scenario + 1, last_scenario + 1 + len(opf_cost_data), dtype="int64")
    df = pd.DataFrame({"scenario": scen, "total_cost": opf_cost_data})

    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_opf_gen_data(path: str, opf_gen_data: List[pd.DataFrame]):
    # 和 PF 一样：读取历史最后场景号 -> 连续编号 -> 追加写
    if not opf_gen_data:
        return

    last_scenario = -1
    if os.path.exists(path):
        try:
            existing_df = pd.read_csv(path, usecols=["scenario"])
            if not existing_df.empty:
                last_scenario = existing_df["scenario"].iloc[-1]
        except Exception:
            last_scenario = -1

    dfs = []
    for i, df in enumerate(opf_gen_data):
        d = df.copy()
        if "scenario_id" in d.columns:
            d = d.drop(columns=["scenario_id"])
        d.insert(0, "scenario", last_scenario + 1 + i)
        dfs.append(d)

    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
