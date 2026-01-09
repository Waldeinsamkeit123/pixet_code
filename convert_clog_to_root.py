import uproot
import numpy as np
import re
import awkward as ak

def convert_clog_to_root(input_file, output_file):
    # Event 级
    event_ids, event_times = [], []
    
    # Cluster 级 (每个 Event 一个列表)
    all_cluster_indices = []
    all_cluster_n_cells = []
    all_cluster_sum_E = []  # 新增：Cluster 总能量
    
    # 新增：Cluster 加权位置和平均T
    all_cluster_weighted_x = []
    all_cluster_weighted_y = []
    all_cluster_avg_t = []
    
    # Cell 级 (Event -> 所有 Cells 的扁平化列表)
    # 这样可以避开 var * var * float 的报错，同时保持数据完整
    all_cell_x, all_cell_y, all_cell_E, all_cell_T = [], [], [], []
    all_cell_cluster = []  # 新增：每个 Cell 所属的 Cluster ID

    frame_re = re.compile(r"Frame\s+(\d+)\s+\(([\d.]+),")
    cell_re = re.compile(r"\[([\d.-]+),\s*([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)\]")

    curr_event_id, curr_event_time = None, None
    
    # 暂存当前 Event 的数据
    c_idx, c_n, c_sum_e = [], [], []
    cw_x, cw_y, avg_t_list = [], [], []  # 新增：暂存当前 Event 的 weighted_x/y 和 avg_t
    cx, cy, ce, ct = [], [], [], []
    cc = []  # 新增：当前 Event 的 cell_cluster 暂存
    
    cluster_counter = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            frame_match = frame_re.match(line)
            if frame_match:
                if curr_event_id is not None:
                    event_ids.append(curr_event_id)
                    event_times.append(curr_event_time)
                    all_cluster_indices.append(c_idx)
                    all_cluster_n_cells.append(c_n)
                    all_cluster_sum_E.append(c_sum_e)
                    all_cluster_weighted_x.append(cw_x)  # 新增：保存当前 Event 的 weighted_x
                    all_cluster_weighted_y.append(cw_y)  # 新增：保存当前 Event 的 weighted_y
                    all_cluster_avg_t.append(avg_t_list)  # 新增：保存当前 Event 的 avg_t
                    all_cell_x.append(cx)
                    all_cell_y.append(cy)
                    all_cell_E.append(ce)
                    all_cell_T.append(ct)
                    all_cell_cluster.append(cc)  # 新增：保存当前 Event 的 cell_cluster

                curr_event_id = int(frame_match.group(1))
                curr_event_time = float(frame_match.group(2))
                c_idx, c_n, c_sum_e = [], [], []
                cw_x, cw_y, avg_t_list = [], [], []  # 新增：重置当前 Event 的 weighted_x/y 和 avg_t
                cx, cy, ce, ct = [], [], [], []
                cc = []  # 新增：重置当前 Event 的 cell_cluster
                cluster_counter = 0
            else:
                cells = cell_re.findall(line)
                if cells:
                    row_energy_sum = 0
                    count = 0
                    weighted_x_num, weighted_y_num = 0.0, 0.0
                    t_sum = 0.0
                    for c in cells:
                        val_x, val_y, val_e, val_t = float(c[0]), float(c[1]), float(c[2]), float(c[3])
                        cx.append(val_x)
                        cy.append(val_y)
                        ce.append(val_e)
                        ct.append(val_t)
                        cc.append(cluster_counter)  # 新增：记录该 cell 所属 cluster ID
                        row_energy_sum += val_e
                        weighted_x_num += val_x * val_e
                        weighted_y_num += val_y * val_e
                        t_sum += val_t
                        count += 1
                    
                    # 新增：计算 weighted_x/y 和 avg_t
                    if count > 0:
                        weighted_x = weighted_x_num / row_energy_sum if row_energy_sum > 0 else 0.0
                        weighted_y = weighted_y_num / row_energy_sum if row_energy_sum > 0 else 0.0
                        avg_t = t_sum / count
                    else:
                        weighted_x, weighted_y, avg_t = 0.0, 0.0, 0.0
                    
                    c_idx.append(cluster_counter)
                    c_n.append(count)
                    c_sum_e.append(row_energy_sum)
                    cw_x.append(weighted_x)  # 新增：暂存 weighted_x
                    cw_y.append(weighted_y)  # 新增：暂存 weighted_y
                    avg_t_list.append(avg_t)  # 新增：暂存 avg_t
                    cluster_counter += 1

        # 保存最后一个
        if curr_event_id is not None:
            event_ids.append(curr_event_id)
            event_times.append(curr_event_time)
            all_cluster_indices.append(c_idx)
            all_cluster_n_cells.append(c_n)
            all_cluster_sum_E.append(c_sum_e)
            all_cluster_weighted_x.append(cw_x)  # 新增：保存最后一个 Event 的 weighted_x
            all_cluster_weighted_y.append(cw_y)  # 新增：保存最后一个 Event 的 weighted_y
            all_cluster_avg_t.append(avg_t_list)  # 新增：保存最后一个 Event 的 avg_t
            all_cell_x.append(cx)
            all_cell_y.append(cy)
            all_cell_E.append(ce)
            all_cell_T.append(ct)
            all_cell_cluster.append(cc)  # 新增：保存最后一个 Event 的 cell_cluster

    # 写入 ROOT
    # 使用 ak.Array 包装，uproot 会将其转为 vector<float> 和 vector<int>
    with uproot.recreate(output_file) as file:
        file["Tree"] = {
            "event_id": np.array(event_ids, dtype=np.int32),
            "event_time": np.array(event_times, dtype=np.float64),
            
            # Cluster 信息 (vector<int> / vector<float>)
            "cluster_index": ak.Array(all_cluster_indices),
            "cluster_n_cells": ak.Array(all_cluster_n_cells),
            "cluster_energy": ak.Array(all_cluster_sum_E),
            "cluster_weighted_x": ak.Array(all_cluster_weighted_x),  # 新增：加权X
            "cluster_weighted_y": ak.Array(all_cluster_weighted_y),  # 新增：加权Y
            "cluster_avg_t": ak.Array(all_cluster_avg_t),  # 新增：平均T
            
            # Cell 信息 (vector<float>)
            # 这里存的是当前 Event 所有的 Cell，通过 cluster_n_cells 和 cell_cluster_id 来区分属于哪个 cluster
            "cell_x": ak.Array(all_cell_x),
            "cell_y": ak.Array(all_cell_y),
            "cell_E": ak.Array(all_cell_E),
            "cell_T": ak.Array(all_cell_T),
            "cell_cluster_id": ak.Array(all_cell_cluster),  # 新增：每个 Cell 所属的 Cluster ID
        }

    print(f"转换成功！输出文件：{output_file}")

if __name__ == "__main__":
    convert_clog_to_root("./TEST-14000-ENERGY.clog", "./TEST-14000-ENERGY.root")