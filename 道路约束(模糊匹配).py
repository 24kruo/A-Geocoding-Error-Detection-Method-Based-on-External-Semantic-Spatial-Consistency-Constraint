import pandas as pd
import tkinter as tk
from tkinter import filedialog
import numpy as np
import math
from difflib import SequenceMatcher
import cpca
from sklearn.cluster import DBSCAN
from collections import Counter
import re

# 距离计算函数
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r

# 投影转换函数
def latlon_to_meters(latlons):
    """将经纬度转为局部米坐标"""
    lat, lon = latlons[:, 1], latlons[:, 0]  # 注意你的coords是[lon, lat]顺序
    R = 6371000  # 地球半径（米）
    lat_mean = lat.mean()
    x = R * np.radians(lon) * np.cos(np.radians(lat_mean))
    y = R * np.radians(lat)
    return np.column_stack([x, y])

# 文本相似度计算函数
def calculate_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

# 创建GUI窗口让用户选择文件
def load_excel_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    return pd.read_excel(file_path)

def save_excel_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    return file_path

def tokenize1(address):
    df = cpca.transform([address])
    city = df['市'].iloc[0] if not pd.isna(df['市'].iloc[0]) else None
    district = df['区'].iloc[0] if not pd.isna(df['区'].iloc[0]) else None
    detail_addr = df['地址'].iloc[0] if not pd.isna(df['地址'].iloc[0]) else None

    street = ""
    number = ""
    if detail_addr:
        s = str(detail_addr)
        # 使用re库：按第一个数字开头的片段分割
        pattern_number = re.compile(r'(\d.+)')
        split_result = pattern_number.split(s, maxsplit=1)

        street = split_result[0].strip() if len(split_result) >= 1 else ""
        number = split_result[1].strip() if len(split_result) >= 2 else ""
    return city, district, street, number

# 计算两地址的文本相似度
def calculate_similarity1(city1, district1, road1, building1, city2, district2, road2, building2):
    weights = {
        'city': 0.4,
        'district': 0.3,
        'road': 0.2,
        'building': 0.1
    }
    similarity = 0
    if city1 == city2:
        similarity += weights['city']
    if district1 == district2:
        similarity += weights['district']
    if road1 == road2:
        similarity += weights['road']
    building_similarity = SequenceMatcher(None, building1, building2).ratio()
    similarity += building_similarity * weights['building']
    return similarity

# 读取Excel表格数据
data = load_excel_file() # 表一
data2 = load_excel_file()  # 表二

address = data["地址2"]
roads = data[["道路1", "道路2", "道路3"]]
start_lon = data["WGS84经度"]
start_lat = data["WGS84纬度"]
distances = data[["距离1", "距离2", "距离3"]]
end_lon = data["高德84经度"]
end_lat = data["高德84纬度"]
data["误差系数"] = 0
data["识别"] = 0
data["点数量"] = 0
data["标签"] = 0
lm = data["路名"]
jl = data["同道路距离"]

address2 = data2["地址名称"]
wgs84_lon2 = data2["WGS84经度"]
wgs84_lat2 = data2["WGS84纬度"]

# 预处理所有地址，提取分词结果
preprocessed_addresses = [tokenize1(addr) for addr in address]
preprocessed_addresses2 = [tokenize1(addr) for addr in address2]  # 表二的分词结果

# 优化点：提前提取表二所有地址的street信息，避免循环中重复计算
# 生成 (street, index) 列表，方便后续匹配
address2_streets = [(preprocessed_addresses2[j][2], j) for j in range(len(address2))]

# 遍历每条数据，计算距离偏差
for i in range(len(data)):
    wgs_lon = start_lon[i]
    wgs_lat = start_lat[i]
    gaode_lon = end_lon[i]
    gaode_lat = end_lat[i]
    deviation = haversine(wgs_lon, wgs_lat, gaode_lon, gaode_lat)

    if deviation >= 0.05 and deviation < 0.15:
        data.at[i, "误差系数"] = 0.1
    elif deviation >= 0.15 and deviation < 0.25:
        data.at[i, "误差系数"] = 0.2
    elif deviation >= 0.25 and deviation < 0.35:
        data.at[i, "误差系数"] = 0.3
    elif deviation >= 0.35 and deviation < 0.45:
        data.at[i, "误差系数"] = 0.4
    elif deviation >= 0.45 and deviation < 0.55:
        data.at[i, "误差系数"] = 0.5
    elif deviation >= 0.55 and deviation < 0.65:
        data.at[i, "误差系数"] = 0.6
    elif deviation >= 0.65 and deviation < 0.75:
        data.at[i, "误差系数"] = 0.7
    elif deviation >= 0.75 and deviation < 0.85:
        data.at[i, "误差系数"] = 0.8
    elif deviation >= 0.85 and deviation < 0.95:
        data.at[i, "误差系数"] = 0.9
    elif deviation >= 0.95:
        data.at[i, "误差系数"] = 1

# 遍历每条数据，进行判断和计算
for i in range(len(data)):
    _, _, road1, building1 = preprocessed_addresses[i]
    # 将roads列转换为列表
    roads_list = roads.iloc[i].dropna().tolist()  # 去除NaN值并转换为列表

    if road1:
        # 存储roads_list中每条道路与road1的相似度（索引, 相似度）
        road_sims = []
        for idx, road in enumerate(roads_list):
            # 跳过空值（避免计算空字符串的相似度）
            if not road:
                continue
            if road in road1:
                sim = 1
            else:
                # 计算相似度（使用你已实现的calculate_similarity函数）
                sim = calculate_similarity(road1, road)
            road_sims.append((idx, road, sim))

        # 筛选出相似度>0.7的道路
        valid_roads = [(idx, road, sim) for idx, road, sim in road_sims if sim > 0.7]

        if valid_roads:  # 存在符合条件的道路
            # 按相似度降序排序，取第一条（最高相似度）
            valid_roads.sort(key=lambda x: x[1], reverse=True)
            best_road_index, best_road, best_sim = valid_roads[0]

            # 后续逻辑使用最佳匹配的道路索引
            road_index = best_road_index  # 0、1、2对应三列

            # 匹配表二的道路时用精确匹配
            matched_indices = [j for (street, j) in address2_streets if street == best_road]

            # 过滤出building分词相互包含或相等的索引
            filtered_indices = []
            for j in matched_indices:
                _, _, _, building2 = preprocessed_addresses2[j]
                similarity = calculate_similarity(building1, building2)
                if building2:
                    if similarity >= 0.8 or (building1 in building2 or building2 in building1 or building1 == building2):
                        filtered_indices.append(j)

            if filtered_indices:
                # 提取所有匹配点的坐标
                coords = np.column_stack([
                    wgs84_lon2[filtered_indices].values,
                    wgs84_lat2[filtered_indices].values
                ])

                # 关键：转换为米坐标后再聚类
                coords_meters = latlon_to_meters(coords)

                # DBSCAN聚类（eps=100 约100米，min_samples=2）
                clustering = DBSCAN(eps=100, min_samples=2).fit(coords_meters)
                labels = clustering.labels_

                # 找到最大聚类
                valid_labels = labels[labels != -1]
                cluster_size = len(filtered_indices)  # 默认值：所有过滤点的数量
                if len(valid_labels) > 0:
                    most_common_label = Counter(valid_labels).most_common(1)[0][0]
                    cluster_mask = labels == most_common_label
                    cluster_coords = coords[cluster_mask]
                    avg_lon = np.mean(cluster_coords[:, 0])
                    avg_lat = np.mean(cluster_coords[:, 1])
                    cluster_size = np.sum(cluster_mask)  # 最大聚类中的点数量
                else:
                    # 无有效聚类时使用全部点
                    avg_lon = np.mean(coords[:, 0])
                    avg_lat = np.mean(coords[:, 1])

                deviation = haversine(end_lon[i], end_lat[i], avg_lon, avg_lat)
                if deviation >= 1:
                    data.at[i, "识别"] = 1
                else:
                    data.at[i, "识别"] = round(deviation, 1)
                data.at[i, "点数量"] = cluster_size  # 使用最大聚类点数
                data.at[i, "标签"] = 1
            else:
                data.at[i, "标签"] = 2
                if distances.iloc[i, road_index] >= 1:
                    data.at[i, "识别"] = 1
                else:
                    # 直接通过索引取distances中对应列的值（0→距离1-1，1→距离1-2，2→距离1-3）
                    data.at[i, "识别"] = distances.iloc[i, road_index]
        else:
            has_parentheses = False
            if building1 is not None:
                building_str = str(building1)
                if '(' in building_str and ')' in building_str:
                    has_parentheses = True

            matched_road_index = None
            if not has_parentheses:  # 只有当不包含()时，才进行道路匹配检查
                for idx, road in enumerate(roads_list):
                    # 道路不为空且building1不为空
                    if road is not None and road in str(building1):
                        matched_road_index = idx
                        break  # 找到第一个匹配的道路即停止

            if matched_road_index is not None:
                distance_val = distances.iloc[i, matched_road_index]
                if distance_val >= 1:
                    data.at[i, "识别"] = 1
                else:
                    data.at[i, "识别"] = distance_val
                data.at[i, "标签"] = 2

            else:
                if pd.notna(lm.iloc[i]) and str(lm.iloc[i]) in road1:
                    data.at[i, "标签"] = 3
                    # 如果相等，将jl的对应值赋给识别列
                    if jl.iloc[i] >= 1:
                        data.at[i, "识别"] = 1
                    else:
                        data.at[i, "识别"] = jl.iloc[i]
                else:
                    data.at[i, "标签"] = 4
                    # 计算三条距离的均值（忽略NaN值）
                    distance_values = distances.iloc[i].dropna()  # 去除NaN值
                    a = round(distance_values.mean(), 1)  # 计算均值
                    if a >= 1:
                        data.at[i, "识别"] = 1
                    else:
                        data.at[i, "识别"] = a
    else:
        data.at[i, "标签"] = 5
        matched_road_index = None
        for idx, road in enumerate(roads_list):
            if str(road) in str(building1):
                matched_road_index = idx
                break  # 找到第一个匹配的道路即停止
        if matched_road_index is not None:
            distance_val = distances.iloc[i, matched_road_index]
            if distance_val >= 1:
                data.at[i, "识别"] = 1
            else:
                data.at[i, "识别"] = distance_val
        else:
            distance_values = distances.iloc[i].dropna()  # 去除NaN值
            a = round(distance_values.mean(), 1)  # 计算均值
            if a >= 1:
                data.at[i, "识别"] = 1
            else:
                data.at[i, "识别"] = a

# 计算精度的评估指标
TP = 0
FP = 0
FN = 0
TN = 0
for i in range(len(address)):
    if data.at[i, "误差系数"] != 0 and 3 > data.at[i, "识别"] > 0:
        TP += 1
    elif data.at[i, "误差系数"] != 0 and data.at[i, "识别"] == 0:
        FN += 1
    elif data.at[i, "误差系数"] == 0 and 3 > data.at[i, "识别"] > 0:
        FP += 1
    elif data.at[i, "误差系数"] == 0 and data.at[i, "识别"] == 0:
        TN += 1

acc = (TP + TN) / (TP + TN + FN + FP) if (TP + TN + FN + FP) > 0 else 0
pre = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
F1score = 2 * pre * recall / (pre + recall) if (pre + recall) > 0 else 0

metrics = {
    "准确率": [acc],
    "精确率": [pre],
    "召回率": [recall],
    "F1score": [F1score]
}
metrics_df = pd.DataFrame(metrics)

save_path = save_excel_file()
if save_path:
    with pd.ExcelWriter(save_path) as writer:
        data.to_excel(writer, sheet_name='数据', index=False)
        metrics_df.to_excel(writer, sheet_name='指标', index=False)
else:
    print("未选择保存路径，操作已取消。")