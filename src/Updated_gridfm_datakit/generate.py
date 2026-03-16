"""Main data generation module for gridfm_datakit."""

import numpy as np
import os
from gridfm_datakit.save import (
    save_edge_params,
    save_bus_params,
    save_branch_idx_removed,
    save_node_edge_data,
)
from gridfm_datakit.process.process_network import (
    network_preprocessing,
    process_scenario,
    process_scenario_contingency,
    process_scenario_chunk,
)
from gridfm_datakit.utils.stats import (
    plot_stats,
    Stats,
    plot_feature_distributions,
)
from gridfm_datakit.utils.param_handler import (
    NestedNamespace,
    get_load_scenario_generator,
    initialize_topology_generator,
    initialize_generation_generator,
    initialize_admittance_generator,
)
from gridfm_datakit.network import (
    load_net_from_pp,
    load_net_from_file,
    load_net_from_pglib,
)
from gridfm_datakit.perturbations.load_perturbation import (
    load_scenarios_to_df,
    plot_load_scenarios_combined,
)
from pandapower.auxiliary import pandapowerNet
import gc
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, Manager
import shutil
from gridfm_datakit.utils.utils import write_ram_usage_distributed, Tee
import yaml
from typing import List, Tuple, Any, Dict, Optional, Union
import sys
import pandas as pd


def _setup_environment(
    config: Union[str, Dict, NestedNamespace],
) -> Tuple[NestedNamespace, str, Dict[str, str]]:
    """Setup the environment for data generation.

    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)

    Returns:
        Tuple of (args, base_path, file_paths)
    """
    # Load config from file if a path is provided
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    # Convert dict to NestedNamespace if needed
    if isinstance(config, dict):
        args = NestedNamespace(**config)
    else:
        args = config

    # Setup output directory
    base_path = os.path.join(args.settings.data_dir, args.network.name, "raw")
    if os.path.exists(base_path) and args.settings.overwrite:
        shutil.rmtree(base_path)
    os.makedirs(base_path, exist_ok=True)

    # Setup file paths
    file_paths = {
        "tqdm_log": os.path.join(base_path, "tqdm.log"),
        "error_log": os.path.join(base_path, "error.log"),
        "args_log": os.path.join(base_path, "args.log"),
        "node_data": os.path.join(base_path, "pf_node.csv"),
        "edge_data": os.path.join(base_path, "pf_edge.csv"),
        "branch_indices": os.path.join(base_path, "branch_idx_removed.csv"),
        "edge_params": os.path.join(base_path, "edge_params.csv"),
        "bus_params": os.path.join(base_path, "bus_params.csv"),
        "scenarios": os.path.join(base_path, f"scenarios_{args.load.generator}.csv"),
        "scenarios_plot": os.path.join(
            base_path,
            f"scenarios_{args.load.generator}.html",
        ),
        "scenarios_log": os.path.join(
            base_path,
            f"scenarios_{args.load.generator}.log",
        ),
        "feature_plots": os.path.join(base_path, "feature_plots"),
        "opf_node": os.path.join(base_path, "opf_node.csv"),
        "opf_edge": os.path.join(base_path, "opf_edge.csv"),
        "opf_cost": os.path.join(base_path, "opf_cost.csv"),
        "opf_gen": os.path.join(base_path, "opf_gen.csv"),
        "opf_times": os.path.join(base_path, "opf_times.csv"),  # ✏️ 时间文件路径
    }

    # Initialize logs
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for log_file in [
        file_paths["tqdm_log"],
        file_paths["error_log"],
        file_paths["scenarios_log"],
        file_paths["args_log"],
    ]:
        with open(log_file, "a") as f:
            f.write(f"\nNew generation started at {timestamp}\n")
            if log_file == file_paths["args_log"]:
                yaml.dump(config if isinstance(config, dict) else vars(config), f)

    return args, base_path, file_paths


def _prepare_network_and_scenarios(
    args: NestedNamespace,
    file_paths: Dict[str, str],
) -> Tuple[pandapowerNet, Any]:
    """Prepare the network and generate load scenarios.

    Args:
        args: Configuration object
        file_paths: Dictionary of file paths

    Returns:
        Tuple of (network, scenarios)
    """
    # Load network
    if args.network.source == "pandapower":
        net = load_net_from_pp(args.network.name)
    elif args.network.source == "pglib":
        net = load_net_from_pglib(args.network.name)
    elif args.network.source == "file":
        net = load_net_from_file(
            os.path.join(args.network.network_dir, args.network.name) + ".m",
        )
    else:
        raise ValueError("Invalid grid source!")

    network_preprocessing(net)
    assert (net.sgen["scaling"] == 1).all(), "Scaling factor >1 not supported yet!"

    # Generate load scenarios
    load_scenario_generator = get_load_scenario_generator(args.load)
    scenarios = load_scenario_generator(
        net,
        args.load.scenarios,
        file_paths["scenarios_log"],
    )
    scenarios_df = load_scenarios_to_df(scenarios)
    scenarios_df.to_csv(file_paths["scenarios"], index=False)
    plot_load_scenarios_combined(scenarios_df, file_paths["scenarios_plot"])
    save_edge_params(net, file_paths["edge_params"])
    save_bus_params(net, file_paths["bus_params"])

    return net, scenarios


def _save_generated_data(
    net: pandapowerNet,
    csv_data: List,
    adjacency_lists: List,
    branch_idx_removed: List,
    global_stats: Optional[Stats],
    file_paths: Dict[str, str],
    base_path: str,
    args: NestedNamespace,
) -> None:
    """Save the generated data to files.

    Args:
        net: Pandapower network
        csv_data: List of CSV data
        adjacency_lists: List of adjacency lists
        branch_idx_removed: List of removed branch indices
        global_stats: Optional statistics object
        file_paths: Dictionary of file paths
        base_path: Base output directory
        args: Configuration object
    """
    if len(adjacency_lists) > 0:
        save_node_edge_data(
            net,
            file_paths["node_data"],
            file_paths["edge_data"],
            csv_data,
            adjacency_lists,
            mode=args.settings.mode,
        )
        save_branch_idx_removed(branch_idx_removed, file_paths["branch_indices"])
        if not args.settings.no_stats and global_stats:
            global_stats.save(base_path)
            plot_stats(base_path)


def generate_power_flow_data(
    config: Union[str, Dict, NestedNamespace],
) -> Dict[str, str]:
    """Generate power flow data based on the provided configuration using sequential processing.

    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)
            The config must include settings, network, load, and topology_perturbation configurations.

    Returns:
        Dictionary containing paths to generated files.
    """
    # Setup environment
    args, base_path, file_paths = _setup_environment(config)

    # Prepare network and scenarios
    net, scenarios = _prepare_network_and_scenarios(args, file_paths)

    # Initialize generators
    topology_generator = initialize_topology_generator(args.topology_perturbation, net)
    generation_generator = initialize_generation_generator(
        args.generation_perturbation,
        net,
    )
    admittance_generator = initialize_admittance_generator(
        args.admittance_perturbation,
        net,
    )

    csv_data = []
    adjacency_lists = []
    branch_idx_removed = []
    global_stats = Stats() if not args.settings.no_stats else None
    opf_node_data, opf_edge_data, opf_cost_data, opf_gen_data = [],[], [], []
    opf_times = []  # ✏️ 时间列表

    # Process scenarios sequentially
    with open(file_paths["tqdm_log"], "a") as f:
        with tqdm(
            total=args.load.scenarios,
            desc="Processing scenarios",
            file=Tee(sys.stdout, f),
            miniters=5,
        ) as pbar:
            for scenario_index in range(args.load.scenarios):
                local_times = []    

                if args.settings.mode in ["pf", "opf", "datakit_opf"]:
                    process_scenario(
                        net,
                        scenarios,
                        scenario_index,
                        topology_generator,
                        generation_generator,
                        admittance_generator,
                        args.settings.no_stats,
                        csv_data,
                        adjacency_lists,
                        branch_idx_removed,
                        global_stats,
                        file_paths["error_log"],
                        opf_node_data,
                        opf_edge_data,
                        opf_cost_data,
                        opf_gen_data,
                        local_times        # ✏️ 传递时间列表
                    )
                elif args.settings.mode == "contingency":
                    csv_data, adjacency_lists, branch_idx_removed, global_stats = (
                        process_scenario_contingency(
                            net,
                            scenarios,
                            scenario_index,
                            topology_generator,
                            generation_generator,
                            admittance_generator,
                            args.settings.no_stats,
                            csv_data,
                            adjacency_lists,
                            branch_idx_removed,
                            global_stats,
                            file_paths["error_log"],
                        )
                    )
                opf_times.extend(local_times)  
                pbar.update(1)

    # Save final data
    _save_generated_data(
        net,
        csv_data,
        adjacency_lists,
        branch_idx_removed,
        global_stats,
        file_paths,
        base_path,
        args,
    )
    
    from gridfm_datakit.save import (
        save_opf_node_data, save_opf_edge_data, save_opf_cost_data, save_opf_gen_data
    )

    opf_node_path = os.path.join(base_path, "opf_node.csv")
    opf_edge_path = os.path.join(base_path, "opf_edge.csv")
    opf_cost_path = os.path.join(base_path, "opf_cost.csv")
    opf_gen_path = os.path.join(base_path, "opf_gen.csv")

    save_opf_node_data(net, opf_node_path, opf_node_data)
    save_opf_edge_data(opf_edge_path, opf_edge_data)
    save_opf_cost_data(opf_cost_path, opf_cost_data)
    save_opf_gen_data(opf_gen_path, opf_gen_data)
    
    # ✏️ MODIFIED: 保存OPF时间数据（连续编号）
    if opf_times:
        opf_times_df = pd.DataFrame(opf_times)
        opf_times_df.to_csv(file_paths["opf_times"], index=False)
        print(f"Saved {len(opf_times)} OPF time records to {file_paths['opf_times']}")

    # Plot features
    if os.path.exists(file_paths["node_data"]):
        plot_feature_distributions(
            file_paths["node_data"],
            file_paths["feature_plots"],
            net.sn_mva,
        )
    else:
        print("No node data file generated. Skipping feature plotting.")

    return file_paths


def generate_power_flow_data_distributed(
    config: Union[str, Dict, NestedNamespace],
) -> Dict[str, str]:
    """Generate power flow data using distributed multiprocessing."""

    # ========== 1. 基本初始化 ==========
    args, base_path, file_paths = _setup_environment(config)
    net, scenarios = _prepare_network_and_scenarios(args, file_paths)

    topology_generator = initialize_topology_generator(args.topology_perturbation, net)
    generation_generator = initialize_generation_generator(args.generation_perturbation, net)
    admittance_generator = initialize_admittance_generator(args.admittance_perturbation, net)

    # ⭐ 只全局累积时间，其余所有数据仍然 chunk 内保存
    all_final_opf_times: List[Dict[str, Any]] = []

    manager = Manager()
    progress_queue = manager.Queue()

    # 拆 chunk（例如 200 场景拆成 4 块）
    large_chunks = np.array_split(
        range(args.load.scenarios),
        np.ceil(args.load.scenarios / args.settings.large_chunk_size).astype(int),
    )

    # ========== 2. 逐 chunk 处理 ==========
    with open(file_paths["tqdm_log"], "a") as f:
        with tqdm(
            total=args.load.scenarios,
            desc="Processing scenarios",
            file=Tee(sys.stdout, f),
            miniters=5,
        ) as pbar:

            for large_chunk in large_chunks:
                write_ram_usage_distributed(f)

                # 该 chunk 的任务切分给多个 worker
                scenario_chunks = np.array_split(
                    large_chunk,
                    args.settings.num_processes,
                )

                tasks = [
                    (
                        args.settings.mode,
                        chunk[0],
                        chunk[-1] + 1,
                        scenarios,
                        net,
                        progress_queue,
                        topology_generator,
                        generation_generator,
                        admittance_generator,
                        args.settings.no_stats,
                        file_paths["error_log"],
                    )
                    for chunk in scenario_chunks
                ]

                # ===== chunk 内的局部累积器（不会跨 chunk）=====
                csv_data = []
                adjacency_lists = []
                branch_idx_removed = []
                opf_node_data = []
                opf_edge_data = []
                opf_cost_data = []
                opf_gen_data  = []
                stats_chunk = Stats() if not args.settings.no_stats else None
                opf_times_chunk = []

                # ========== 3. 多进程处理 ==========
                with Pool(processes=args.settings.num_processes) as pool:
                    results = [pool.apply_async(process_scenario_chunk, task) for task in tasks]

                    completed = 0
                    while completed < len(large_chunk):
                        progress_queue.get()
                        pbar.update(1)
                        completed += 1

                    for result in results:
                        (
                            e,
                            traceback,
                            local_csv_data,
                            local_adj,
                            local_branch_removed,
                            local_stats,
                            local_opf_node,
                            local_opf_edge,
                            local_opf_cost,
                            local_opf_gen,
                            local_opf_times,
                        ) = result.get()

                        if isinstance(e, Exception):
                            print("Error in worker:", e)
                            print(traceback)
                            sys.exit(1)

                        # chunk 内部累积
                        csv_data.extend(local_csv_data)
                        adjacency_lists.extend(local_adj)
                        branch_idx_removed.extend(local_branch_removed)
                        opf_node_data.extend(local_opf_node)
                        opf_edge_data.extend(local_opf_edge)
                        opf_cost_data.extend(local_opf_cost)
                        opf_gen_data.extend(local_opf_gen)
                        opf_times_chunk.extend(local_opf_times)

                        if not args.settings.no_stats and local_stats:
                            if stats_chunk is None:
                                stats_chunk = local_stats
                            else:
                                stats_chunk.merge(local_stats)

                    pool.close()
                    pool.join()

                # ========== 4. 本 chunk 的时间局部编号，然后加入全局列表 ==========
                for idx, t in enumerate(opf_times_chunk):
                    all_final_opf_times.append(
                        {
                            "scenario_id": idx,           # 局部顺序（最终会重新编号）
                            "elapsed_time": t["elapsed_time"],
                            "success": t["success"],
                        }
                    )

                # ========== 5. 保存本 chunk 的 PF/OPF 数据（不会重复写）==========
                _save_generated_data(
                    net,
                    csv_data,
                    adjacency_lists,
                    branch_idx_removed,
                    stats_chunk,
                    file_paths,
                    base_path,
                    args,
                )

                from gridfm_datakit.save import (
                    save_opf_node_data,
                    save_opf_edge_data,
                    save_opf_cost_data,
                    save_opf_gen_data,
                )

                save_opf_node_data(net, file_paths["opf_node"], opf_node_data)
                save_opf_edge_data(file_paths["opf_edge"], opf_edge_data)
                save_opf_cost_data(file_paths["opf_cost"], opf_cost_data)
                save_opf_gen_data(file_paths["opf_gen"], opf_gen_data)

                gc.collect()

    # ========== 6. 全部 chunk 处理完，统一保存时间 ==========
    if all_final_opf_times:
        df = pd.DataFrame(all_final_opf_times)

        # ⭐ 全局统一编号：从 0 到 N-1，完全连续
        df["scenario_id"] = range(len(df))

        df.to_csv(file_paths["opf_times"], index=False)
        print(f"Saved {len(df)} OPF time records to {file_paths['opf_times']}")

    # ========== 7. 绘图 ==========
    if os.path.exists(file_paths["node_data"]):
        plot_feature_distributions(
            file_paths["node_data"],
            file_paths["feature_plots"],
            net.sn_mva,
        )
    else:
        print("No node data file generated. Skipping feature plotting.")

    return file_paths
