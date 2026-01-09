import uproot
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_save_plots(file_path):
    # 1. 读取数据
    with uproot.open(file_path) as f:
        tree = f["Tree"]
        data = tree.arrays(["cluster_energy", "cell_x", "cell_y", "cell_E", "cluster_n_cells"])

    # --- 数据处理 ---
    flat_cluster_E = ak.to_numpy(ak.flatten(data["cluster_energy"]))
    flat_cluster_n = ak.to_numpy(ak.flatten(data["cluster_n_cells"]))
    flat_cell_x = ak.to_numpy(ak.flatten(data["cell_x"]))
    flat_cell_y = ak.to_numpy(ak.flatten(data["cell_y"]))
    flat_cell_E = ak.to_numpy(ak.flatten(data["cell_E"]))

    # 建立映射
    cluster_map = np.repeat(np.arange(len(flat_cluster_n)), flat_cluster_n)

    # --- 图 1 & 2: Cluster Energy ---
    # (省略部分样式代码以保持简洁，逻辑与之前一致)
    plt.figure(figsize=(8, 6))
    plt.hist(flat_cluster_E, bins=100, range=(0, 370), color='skyblue', edgecolor='black')
    plt.title("Cluster Energy (Full)")
    plt.savefig("1_cluster_energy_full.png", dpi=300)
    plt.close()

    cluster_cut_mask = (flat_cluster_E > 0) & (flat_cluster_E < 40)
    plt.figure(figsize=(8, 6))
    plt.hist(flat_cluster_E[cluster_cut_mask], bins=100, range=(0, 40), color='salmon', edgecolor='black')
    plt.title("Cluster Energy (0-40 keV)")
    plt.savefig("2_cluster_energy_cut.png", dpi=300)
    plt.close()

    # --- 准备 2D Map 数据 ---
    cell_cut_mask = cluster_cut_mask[cluster_map]
    final_x = flat_cell_x[cell_cut_mask]
    final_y = flat_cell_y[cell_cut_mask]
    final_E = flat_cell_E[cell_cut_mask]

    print(f"单次击中 Cell 的最大能量: {np.max(final_E)}")

    # --- 图 3: 2D 总能量分布 (Total Energy Map) ---
    plt.figure(figsize=(9, 7))
    plt.hist2d(final_x, final_y, bins=(256, 256), range=[[0, 256], [0, 256]], 
               cmap='hot', weights=final_E)
    plt.colorbar(label='Total Accumulated Energy (keV)')
    plt.title("2D Total Energy Map (Cluster_E < 40)")
    plt.xlabel("X-pixel")
    plt.ylabel("Y-pixel")
    plt.savefig("3_detector_total_energy_map.png", dpi=300)
    plt.close()

    # --- 图 4: 2D 平均能量分布 (Average Energy Map) ---
    # 使用 numpy 计算矩阵
    # 1. 计算每个像素的能量总和
    sum_energy, x_edges, y_edges = np.histogram2d(
        final_x, final_y, bins=256, range=[[0, 256], [0, 256]], weights=final_E
    )
    # 2. 计算每个像素被击中的次数
    hit_counts, _, _ = np.histogram2d(
        final_x, final_y, bins=256, range=[[0, 256], [0, 256]]
    )

    # 3. 计算平均能量 (处理除以 0 的情况：没有点击的地方设为 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_energy = np.divide(sum_energy, hit_counts)
        avg_energy = np.nan_to_num(avg_energy) # 将 NaN 转为 0

    plt.figure(figsize=(9, 7))
    # 使用 imshow 绘制计算好的矩阵，注意 origin='lower' 保证坐标轴方向正确
    im = plt.imshow(avg_energy.T, origin='lower', extent=[0, 256, 0, 256], 
                    cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Mean Energy per Hit (keV)')
    plt.title("2D Average Energy Map (Cluster_E < 40)")
    plt.xlabel("X-pixel")
    plt.ylabel("Y-pixel")
    plt.savefig("4_detector_avg_energy_map.png", dpi=300)
    plt.close()

    print("所有 4 张图片已保存。")

if __name__ == "__main__":
    analyze_and_save_plots("./TEST-14000-ENERGY.root")