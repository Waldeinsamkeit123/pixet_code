import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_save_root(input_root_file, output_root_file):
    # 读取ROOT树
    with uproot.open(input_root_file) as file:
        tree = file["Tree"]
        
        # 提取所有数据（jagged arrays for clusters/cells, regular for events）
        event_id = tree["event_id"].array()
        event_time = tree["event_time"].array()
        
        # Cluster级
        cluster_index = tree["cluster_index"].array()
        cluster_n_cells = tree["cluster_n_cells"].array()
        cluster_energy = tree["cluster_energy"].array()
        cluster_weighted_x = tree["cluster_weighted_x"].array()
        cluster_weighted_y = tree["cluster_weighted_y"].array()
        cluster_avg_t = tree["cluster_avg_t"].array()
        
        # Cell级
        cell_x = tree["cell_x"].array()
        cell_y = tree["cell_y"].array()
        cell_E = tree["cell_E"].array()
        cell_T = tree["cell_T"].array()
        cell_cluster_id = tree["cell_cluster_id"].array()
        
        # 事件总数
        n_events = len(event_id)
        print(f"总事件数: {n_events}")
        
        discarded_events = 0  # 抛弃事件计数
        merged_clusters = 0   # 合并次数
        new_energies = []     # 收集新的cluster_energy
        
        # 新数据存储：每个event的列表
        new_event_ids = []
        new_event_times = []
        new_cluster_indices = []
        new_cluster_n_cells = []
        new_cluster_energies = []
        new_cluster_weighted_xs = []
        new_cluster_weighted_ys = []
        new_cluster_avg_ts = []
        new_cell_xs = []
        new_cell_ys = []
        new_cell_Es = []
        new_cell_Ts = []
        new_cell_cluster_ids = []
        
        for i in range(n_events):
            n_clusters = len(cluster_energy[i])
            # 修正：event_id 和 event_time 是标量
            curr_event_id = int(event_id[i])
            curr_event_time = float(event_time[i])
            
            # 提取当前event的cell数据（扁平numpy）
            curr_cell_x = ak.to_numpy(cell_x[i])
            curr_cell_y = ak.to_numpy(cell_y[i])
            curr_cell_E = ak.to_numpy(cell_E[i])
            curr_cell_T = ak.to_numpy(cell_T[i])
            curr_cell_cluster_id = ak.to_numpy(cell_cluster_id[i]).copy()  # copy 以允许修改
            
            if n_clusters == 1:
                # 单cluster事件：检查抛弃条件
                if cluster_n_cells[i][0] <= 2 and cluster_energy[i][0] < 30:
                    discarded_events += 1
                    continue
                else:
                    # 保留该event的所有数据
                    new_event_ids.append(curr_event_id)
                    new_event_times.append(curr_event_time)
                    new_cluster_indices.append(ak.to_numpy(cluster_index[i]))
                    new_cluster_n_cells.append(ak.to_numpy(cluster_n_cells[i]))
                    new_cluster_energies.append(ak.to_numpy(cluster_energy[i]))
                    new_cluster_weighted_xs.append(ak.to_numpy(cluster_weighted_x[i]))
                    new_cluster_weighted_ys.append(ak.to_numpy(cluster_weighted_y[i]))
                    new_cluster_avg_ts.append(ak.to_numpy(cluster_avg_t[i]))
                    new_cell_xs.append(curr_cell_x)
                    new_cell_ys.append(curr_cell_y)
                    new_cell_Es.append(curr_cell_E)
                    new_cell_Ts.append(curr_cell_T)
                    new_cell_cluster_ids.append(ak.to_numpy(cell_cluster_id[i]))  # 未修改
                    new_energies.extend(ak.to_numpy(cluster_energy[i]))
            elif n_clusters >= 2:
                # 多cluster事件：提取为NumPy
                energies_np = ak.to_numpy(cluster_energy[i])
                n_cells_np = ak.to_numpy(cluster_n_cells[i])
                index_np = ak.to_numpy(cluster_index[i])
                x_np = ak.to_numpy(cluster_weighted_x[i])
                y_np = ak.to_numpy(cluster_weighted_y[i])
                t_np = ak.to_numpy(cluster_avg_t[i])
                
                n_clusters = len(energies_np)
                
                # 找出所有低能量小cluster：energy <30 且 n_cells <=2
                low_condition = (energies_np < 30) & (n_cells_np <= 2)
                low_idxs = np.where(low_condition)[0]
                
                to_delete = []
                if len(low_idxs) > 0:
                    # 目标cluster：不满足低能量条件的
                    target_mask = ~low_condition
                    target_idxs = np.where(target_mask)[0]
                    if len(target_idxs) > 0:
                        # 对于每个低能量cluster，尝试合并到最近目标
                        merged_count = 0
                        for low_idx in low_idxs:
                            low_energy = energies_np[low_idx]
                            low_n_cells = n_cells_np[low_idx]
                            low_x = x_np[low_idx]
                            low_y = y_np[low_idx]
                            low_t = t_np[low_idx]
                            low_id = index_np[low_idx]
                            
                            # 收集低能量cluster的cells
                            low_cells_mask = (curr_cell_cluster_id == low_id)
                            low_cell_xs = curr_cell_x[low_cells_mask]
                            low_cell_ys = curr_cell_y[low_cells_mask]
                            low_cell_Es = curr_cell_E[low_cells_mask]
                            low_cell_Ts = curr_cell_T[low_cells_mask]
                            
                            # 其他目标cluster的差异
                            other_t = t_np[target_idxs]
                            other_x = x_np[target_idxs]
                            other_y = y_np[target_idxs]
                            other_indices = index_np[target_idxs]
                            
                            dt_diffs = np.abs(other_t - low_t)
                            dx_diffs = np.abs(other_x - low_x)
                            dy_diffs = np.abs(other_y - low_y)
                            
                            # 候选掩码
                            candidate_mask = (dx_diffs <= 10) & (dy_diffs <= 10)
                            candidates_rel_mask = candidate_mask
                            candidates = target_idxs[candidate_mask]
                            
                            if len(candidates) > 0:
                                # 选dt最小的
                                filtered_dt = dt_diffs[candidate_mask]
                                best_rel_idx = np.argmin(filtered_dt)
                                best_idx = candidates[best_rel_idx]
                                best_target_index = other_indices[best_rel_idx]
                                
                                # 找到目标cluster的cells（当前状态，包括之前合并）
                                target_cells_mask = (curr_cell_cluster_id == best_target_index)
                                target_cell_xs = curr_cell_x[target_cells_mask]
                                target_cell_ys = curr_cell_y[target_cells_mask]
                                target_cell_Es = curr_cell_E[target_cells_mask]
                                target_cell_Ts = curr_cell_T[target_cells_mask]
                                
                                # 合并cells
                                merged_cell_xs = np.concatenate([target_cell_xs, low_cell_xs])
                                merged_cell_ys = np.concatenate([target_cell_ys, low_cell_ys])
                                merged_cell_Es = np.concatenate([target_cell_Es, low_cell_Es])
                                merged_cell_Ts = np.concatenate([target_cell_Ts, low_cell_Ts])
                                
                                # 重新计算目标cluster属性
                                total_e = np.sum(merged_cell_Es)
                                if len(merged_cell_Es) > 0:
                                    new_weighted_x = np.sum(merged_cell_xs * merged_cell_Es) / total_e
                                    new_weighted_y = np.sum(merged_cell_ys * merged_cell_Es) / total_e
                                    new_avg_t = np.mean(merged_cell_Ts)
                                    new_n_cells = len(merged_cell_Es)
                                else:
                                    new_weighted_x, new_weighted_y, new_avg_t, new_n_cells = low_x, low_y, low_t, low_n_cells
                                
                                # 更新目标cluster在np中的值
                                energies_np[best_idx] = total_e
                                n_cells_np[best_idx] = new_n_cells
                                x_np[best_idx] = new_weighted_x
                                y_np[best_idx] = new_weighted_y
                                t_np[best_idx] = new_avg_t
                                
                                # 更新cell_cluster_id：低能量cells改为目标index（old id）
                                curr_cell_cluster_id[low_cells_mask] = best_target_index
                                
                                to_delete.append(low_idx)
                                merged_count += 1
                        
                        merged_clusters += merged_count
                
                # 删除已合并的low_idxs（排序逆序删除）
                if len(to_delete) > 0:
                    to_delete = sorted(to_delete, reverse=True)
                    for del_idx in to_delete:
                        energies_np = np.delete(energies_np, del_idx)
                        n_cells_np = np.delete(n_cells_np, del_idx)
                        # 不删除index_np，因为我们需要old values for mapping
                        x_np = np.delete(x_np, del_idx)
                        y_np = np.delete(y_np, del_idx)
                        t_np = np.delete(t_np, del_idx)
                    
                    # Remap: 先收集surviving old ids（基于原始index_np和to_delete）
                    old_index_np = index_np.copy()  # 原始，未删除
                    surviving_old_ids_pre = [old_index_np[j] for j in range(n_clusters) if j not in to_delete]
                    surviving_old_ids = sorted(surviving_old_ids_pre)
                    old_to_new = {old_id: new_id for new_id, old_id in enumerate(surviving_old_ids)}
                    
                    # 新index
                    new_index_np = np.arange(len(energies_np))
                    
                    # Remap cell_cluster_id
                    for old_id, new_id in old_to_new.items():
                        mask = (curr_cell_cluster_id == old_id)
                        if np.any(mask):
                            curr_cell_cluster_id[mask] = new_id
                else:
                    # 无删除
                    new_index_np = index_np.copy()
                
                # 如果剩余clusters >0
                if len(energies_np) > 0:
                    new_event_ids.append(curr_event_id)
                    new_event_times.append(curr_event_time)
                    new_cluster_indices.append(new_index_np)
                    new_cluster_n_cells.append(n_cells_np)
                    new_cluster_energies.append(energies_np)
                    new_cluster_weighted_xs.append(x_np)
                    new_cluster_weighted_ys.append(y_np)
                    new_cluster_avg_ts.append(t_np)
                    new_cell_xs.append(curr_cell_x)
                    new_cell_ys.append(curr_cell_y)
                    new_cell_Es.append(curr_cell_E)
                    new_cell_Ts.append(curr_cell_T)
                    new_cell_cluster_ids.append(curr_cell_cluster_id)
                    new_energies.extend(energies_np)
                else:
                    discarded_events += 1
        
        print(f"抛弃事件数: {discarded_events}")
        print(f"合并cluster次数: {merged_clusters}")
        new_energies = np.array(new_energies)
        
        # 写入新ROOT文件
        with uproot.recreate(output_root_file) as new_file:
            new_file["Tree"] = {
                "event_id": np.array(new_event_ids, dtype=np.int32),
                "event_time": np.array(new_event_times, dtype=np.float64),
                
                # Cluster信息
                "cluster_index": ak.Array(new_cluster_indices),
                "cluster_n_cells": ak.Array(new_cluster_n_cells),
                "cluster_energy": ak.Array(new_cluster_energies),  # 更新后的energy
                "cluster_weighted_x": ak.Array(new_cluster_weighted_xs),
                "cluster_weighted_y": ak.Array(new_cluster_weighted_ys),
                "cluster_avg_t": ak.Array(new_cluster_avg_ts),
                
                # Cell信息（cell_cluster_id已更新）
                "cell_x": ak.Array(new_cell_xs),
                "cell_y": ak.Array(new_cell_ys),
                "cell_E": ak.Array(new_cell_Es),
                "cell_T": ak.Array(new_cell_Ts),
                "cell_cluster_id": ak.Array(new_cell_cluster_ids),
            }
        
        print(f"新ROOT文件已保存: {output_root_file}")
        
        # 绘制分布
        plt.figure(figsize=(8, 6))
        plt.hist(new_energies, bins=100, range=(0, 500), alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('New Cluster Energy (keV)')
        plt.ylabel('Counts')
        plt.title('Distribution of New Cluster Energies After Filtering and Merging')
        plt.grid(True, alpha=0.3)
        plt.savefig('mod_new_cluster_energy_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"分布图已保存为: mod_new_cluster_energy_distribution.png")
        print(f"新cluster总数: {len(new_energies)}, 平均能量: {np.mean(new_energies):.2f} keV")

if __name__ == "__main__":
    input_root = "./TEST-14000-ENERGY.root"  # 输入ROOT文件路径
    output_root = "./TEST-14000-ENERGY_updated.root"  # 输出ROOT文件路径
    analyze_and_save_root(input_root, output_root)