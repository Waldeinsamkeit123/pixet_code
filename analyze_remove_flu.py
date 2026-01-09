import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

def analyze_root(root_file):
    # 读取ROOT树
    with uproot.open(root_file) as file:
        tree = file["Tree"]
        
        # 提取cluster级数据（jagged arrays）
        cluster_energy = tree["cluster_energy"].array()
        cluster_n_cells = tree["cluster_n_cells"].array()
        cluster_weighted_x = tree["cluster_weighted_x"].array()
        cluster_weighted_y = tree["cluster_weighted_y"].array()
        cluster_avg_t = tree["cluster_avg_t"].array()
        
        # 事件总数
        n_events = len(cluster_energy)
        print(f"总事件数: {n_events}")
        
        discarded_events = 0  # 抛弃事件计数
        merged_clusters = 0   # 合并次数
        new_energies = []     # 收集新的cluster_energy
        
        for i in range(n_events):
            n_clusters = len(cluster_energy[i])
            if n_clusters == 1:
                # 单cluster事件：检查抛弃条件
                if cluster_n_cells[i][0] <= 2 and cluster_energy[i][0] < 30:
                    discarded_events += 1
                    continue
                else:
                    # 保留该cluster
                    new_energies.extend(ak.to_numpy(cluster_energy[i]))
            elif n_clusters >= 2:
                # 多cluster事件：提取为NumPy进行修改
                energies_np = ak.to_numpy(cluster_energy[i])
                n_cells_np = ak.to_numpy(cluster_n_cells[i])
                x_np = ak.to_numpy(cluster_weighted_x[i])
                y_np = ak.to_numpy(cluster_weighted_y[i])
                t_np = ak.to_numpy(cluster_avg_t[i])
                
                # 找最小能量cluster
                min_energy_idx = np.argmin(energies_np)
                min_energy = energies_np[min_energy_idx]
                min_n_cells = n_cells_np[min_energy_idx]
                
                if min_energy < 30 and min_n_cells <= 2:
                    # 尝试合并
                    min_x = x_np[min_energy_idx]
                    min_y = y_np[min_energy_idx]
                    min_t = t_np[min_energy_idx]
                    
                    # 其他cluster的索引和差异
                    other_idxs = np.delete(np.arange(n_clusters), min_energy_idx)
                    other_t = t_np[other_idxs]
                    other_x = x_np[other_idxs]
                    other_y = y_np[other_idxs]
                    
                    dt_diffs = np.abs(other_t - min_t)
                    dx_diffs = np.abs(other_x - min_x)
                    dy_diffs = np.abs(other_y - min_y)
                    
                    # 候选掩码：|dx|<=10 且 |dy|<=10
                    candidate_mask = (dx_diffs <= 10) & (dy_diffs <= 10)
                    candidates = other_idxs[candidate_mask]
                    
                    if len(candidates) > 0:
                        # 在候选中选dt最小的
                        filtered_dt = dt_diffs[candidate_mask]
                        best_rel_idx = np.argmin(filtered_dt)
                        best_idx = candidates[best_rel_idx]
                        
                        # 合并能量到best_idx
                        energies_np[best_idx] += min_energy
                        merged_clusters += 1
                        
                        # 删除min_energy_idx（注意：删除后索引会变，但best_idx已在前面）
                        if best_idx > min_energy_idx:
                            best_idx -= 1  # 调整best_idx（因为删除min后，后续索引前移）
                        energies_np = np.delete(energies_np, min_energy_idx)
                        # 无需调整其他np，因为我们只用energies_np追加
                        
                        # 如果剩余为空，跳过
                        if len(energies_np) == 0:
                            discarded_events += 1  # 可选：视为空事件
                            continue
                
                # 追加修改后的energies（或原版如果无合并）
                new_energies.extend(energies_np)
        
        print(f"抛弃事件数: {discarded_events}")
        print(f"合并cluster次数: {merged_clusters}")
        new_energies = np.array(new_energies)
        
        #plot
        plt.figure(figsize=(8, 6))
        plt.hist(new_energies, bins=50, range=(0, 500), alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('New Cluster Energy (keV)')
        plt.ylabel('Counts')
        plt.title('Distribution of New Cluster Energies After Filtering and Merging')
        plt.grid(True, alpha=0.3)
        plt.savefig('new_cluster_energy_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"分布图已保存为: new_cluster_energy_distribution.png")
        print(f"新cluster总数: {len(new_energies)}, 平均能量: {np.mean(new_energies):.2f} keV")

if __name__ == "__main__":
    root_file = "./TEST-14000-ENERGY.root"  # 替换为您的ROOT文件路径
    analyze_root(root_file)