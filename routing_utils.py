# routing_utils.py
import osmnx as ox
import numpy as np
import pandas as pd
import re
from config import MPH_TO_MPS, ROAD_TYPE_SPEED_MODIFIERS # ROAD_TYPE_SPEED_MODIFIERS vẫn cần thiết

# ... (Hàm parse_maxspeed giữ nguyên) ...
def parse_maxspeed(maxspeed_str, default_speed_mph=None):
    # ... (Nội dung hàm giữ nguyên như lần sửa trước) ...
    if maxspeed_str is None:
        return default_speed_mph
    if isinstance(maxspeed_str, (int, float)): 
        return float(maxspeed_str) 
    if isinstance(maxspeed_str, list): 
        if not maxspeed_str:
            return default_speed_mph
        maxspeed_str = maxspeed_str[0]
        if isinstance(maxspeed_str, (int, float)):
             return float(maxspeed_str)
    try:
        numeric_part = re.findall(r'\d+\.?\d*', str(maxspeed_str))
        if not numeric_part:
            return default_speed_mph
        speed = float(numeric_part[0])
        if "kph" in maxspeed_str.lower() or "km/h" in maxspeed_str.lower():
            return speed / 1.60934 
        elif "mph" in maxspeed_str.lower() or not any(char.isalpha() for char in maxspeed_str.replace(numeric_part[0], "")):
            return speed
        else: 
            return default_speed_mph
    except Exception:
        return default_speed_mph


def get_representative_speed_mps(
    target_hour, 
    target_day_numeric, # 0=Thứ Hai, ..., 6=Chủ Nhật
    median_speed_by_hour, 
    median_speed_by_day_of_week, # Series với index là tên ngày hoặc số 0-6
    overall_median_speed_all_days_mph # Tốc độ trung vị tổng thể của cả tuần
):
    """Lấy tốc độ đại diện (m/s) dựa trên giờ, ngày trong tuần, và tốc độ tổng thể."""
    
    # 1. Lấy tốc độ cơ sở theo giờ
    speed_for_hour_mph = None
    if median_speed_by_hour is not None and not median_speed_by_hour.empty:
        if target_hour in median_speed_by_hour.index:
            speed_for_hour_mph = median_speed_by_hour.loc[target_hour]
        else: 
            speed_for_hour_mph = median_speed_by_hour.median()
            print(f"Cảnh báo: Giờ {target_hour} không có trong median_speed_by_hour, dùng median chung của các giờ.")
    
    if speed_for_hour_mph is None or pd.isna(speed_for_hour_mph):
        print("Cảnh báo: Không thể xác định speed_for_hour_mph. Dùng giá trị mặc định 10mph.")
        speed_for_hour_mph = 10 

    # 2. Tính hệ số điều chỉnh theo ngày trong tuần
    day_of_week_factor = 1.0 # Mặc định không điều chỉnh
    days_map_for_series = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
    
    if median_speed_by_day_of_week is not None and \
       not median_speed_by_day_of_week.empty and \
       overall_median_speed_all_days_mph is not None and \
       overall_median_speed_all_days_mph > 0 and \
       0 <= target_day_numeric < len(days_map_for_series):
        
        target_day_name = days_map_for_series[target_day_numeric] # Chuyển số sang tên ngày (nếu index của series là tên)
        
        # Kiểm tra xem index của median_speed_by_day_of_week là số hay tên
        if isinstance(median_speed_by_day_of_week.index, pd.RangeIndex) or median_speed_by_day_of_week.index.dtype == 'int64':
            # Nếu index là số (0-6)
            if target_day_numeric in median_speed_by_day_of_week.index:
                speed_for_target_day_mph = median_speed_by_day_of_week.loc[target_day_numeric]
                day_of_week_factor = speed_for_target_day_mph / overall_median_speed_all_days_mph
            else:
                print(f"Cảnh báo: Ngày {target_day_numeric} không có trong median_speed_by_day_of_week (index số).")
        elif target_day_name in median_speed_by_day_of_week.index: # Nếu index là tên ngày
            speed_for_target_day_mph = median_speed_by_day_of_week.loc[target_day_name]
            day_of_week_factor = speed_for_target_day_mph / overall_median_speed_all_days_mph
        else:
            print(f"Cảnh báo: Ngày '{target_day_name}' không có trong median_speed_by_day_of_week (index tên).")

        # Giới hạn hệ số để tránh điều chỉnh quá mạnh
        day_of_week_factor = np.clip(day_of_week_factor, 0.7, 1.3) # Ví dụ: điều chỉnh tối đa +/- 30%
        print(f"Hệ số điều chỉnh cho {target_day_name}: {day_of_week_factor:.2f}")

    # 3. Áp dụng hệ số và chuyển đổi
    final_representative_speed_mph = speed_for_hour_mph * day_of_week_factor
    representative_speed_mps = final_representative_speed_mph * MPH_TO_MPS
    
    if representative_speed_mps <= 0:
        representative_speed_mps = 1.0 * MPH_TO_MPS # Tốc độ tối thiểu
        
    print(f"Tốc độ cơ sở theo giờ ({target_hour}h): {speed_for_hour_mph:.2f} mph")
    print(f"Tốc độ cuối cùng đại diện (đã điều chỉnh theo ngày và chuyển sang m/s): {representative_speed_mps:.2f} m/s")
    return representative_speed_mps

# ... (Hàm add_travel_times_to_graph và calculate_eta_for_route giữ nguyên) ...
# ... (Phần if __name__ == '__main__': giữ nguyên) ...
def add_travel_times_to_graph(G, base_speed_mps):
    # ... (Nội dung hàm giữ nguyên như lần sửa trước) ...
    if G is None or base_speed_mps is None:
        print("Lỗi: Đồ thị G hoặc tốc độ cơ sở không hợp lệ để thêm travel_time.")
        return G 
    if base_speed_mps <=0:
        print(f"Cảnh báo: base_speed_mps ({base_speed_mps:.2f} m/s) không dương. Sử dụng tốc độ tối thiểu thay thế.")
        base_speed_mps_safe = 0.1 * MPH_TO_MPS 
    else:
        base_speed_mps_safe = base_speed_mps
    print(f"Đang cập nhật 'travel_time'. Tốc độ cơ sở (sau khi kiểm tra an toàn): {base_speed_mps_safe:.2f} m/s")
    num_edges_missing_length = 0
    num_edges_used_maxspeed = 0
    num_edges_used_road_type_modifier = 0
    for u, v, data in G.edges(data=True):
        length_m = data.get('length')
        effective_speed_mps = None
        if length_m is not None:
            try:
                length_m = float(length_m)
                maxspeed_str = data.get('maxspeed')
                parsed_max_speed_mph = parse_maxspeed(maxspeed_str) 
                highway_attr = data.get('highway', 'default')
                if isinstance(highway_attr, list):
                    highway_type = highway_attr[0] if highway_attr else 'default'
                else:
                    highway_type = highway_attr
                modifier = ROAD_TYPE_SPEED_MODIFIERS.get(highway_type, ROAD_TYPE_SPEED_MODIFIERS['default'])
                speed_based_on_road_type_and_congestion = base_speed_mps_safe * modifier
                if parsed_max_speed_mph is not None and parsed_max_speed_mph > 0:
                    max_speed_mps_from_osm = parsed_max_speed_mph * MPH_TO_MPS
                    effective_speed_mps = min(max_speed_mps_from_osm, speed_based_on_road_type_and_congestion)
                    num_edges_used_maxspeed += 1
                else:
                    effective_speed_mps = speed_based_on_road_type_and_congestion
                    num_edges_used_road_type_modifier +=1
                if effective_speed_mps <= 0:
                    effective_speed_mps = 0.1 * MPH_TO_MPS 
                data['travel_time'] = length_m / effective_speed_mps
            except (ValueError, TypeError):
                data['travel_time'] = float('inf')
            except ZeroDivisionError:
                 data['travel_time'] = float('inf')
        else: 
            num_edges_missing_length +=1
            data['travel_time'] = float('inf')
    if num_edges_missing_length > 0:
        print(f"Cảnh báo: {num_edges_missing_length} cạnh bị thiếu 'length', travel_time được gán là vô cực.")
    print(f"Số cạnh sử dụng thông tin 'maxspeed' (sau khi xử lý): {num_edges_used_maxspeed}")
    print(f"Số cạnh sử dụng hệ số theo loại đường (do không có maxspeed hợp lệ): {num_edges_used_road_type_modifier}")
    print("Hoàn tất cập nhật 'travel_time' với tốc độ ưu tiên maxspeed và theo loại đường.")
    return G

def calculate_eta_for_route(G_with_times, origin_node, destination_node):
    # ... (Nội dung hàm giữ nguyên như lần sửa trước) ...
    if G_with_times is None or origin_node is None or destination_node is None:
        print("Lỗi: Thiếu thông tin đồ thị hoặc điểm đầu/cuối để tính ETA.")
        return None, None
    if not G_with_times.has_node(origin_node) or not G_with_times.has_node(destination_node):
        print(f"LỖI: Nút xuất phát {origin_node} hoặc nút đích {destination_node} không tồn tại trong đồ thị.")
        return None, None
    try:
        print(f"Đang tìm lộ trình từ {origin_node} đến {destination_node} (sử dụng 'travel_time' làm trọng số)...")
        route = ox.shortest_path(G_with_times, origin_node, destination_node, weight="travel_time")
        if route:
            print("Tìm thấy lộ trình!")
            route_edges_gdf = ox.routing.route_to_gdf(G_with_times, route)
            if 'travel_time' in route_edges_gdf.columns:
                if route_edges_gdf['travel_time'].isnull().any() or np.isinf(route_edges_gdf['travel_time']).any():
                    print("CẢNH BÁO: Một số đoạn đường trong lộ trình có travel_time không hợp lệ (NaN hoặc Inf).")
                total_travel_time_seconds = route_edges_gdf['travel_time'].sum()
                if np.isinf(total_travel_time_seconds):
                    print("ETA không thể tính được do có đoạn đường không thể đi qua (travel_time = vô cực).")
                    eta_minutes = float('inf')
                elif pd.isna(total_travel_time_seconds):
                    print("ETA không thể tính được do thiếu dữ liệu travel_time (NaN).")
                    eta_minutes = float('nan')
                else:
                    eta_minutes = total_travel_time_seconds / 60
                print(f"Thời gian di chuyển dự kiến (ETA): {total_travel_time_seconds:.2f} giây ({eta_minutes:.2f} phút)")
                return route, eta_minutes
            else:
                print("LỖI: Cột 'travel_time' không có trong GeoDataFrame của lộ trình (route_edges_gdf).")
                print("Các cột có trong route_edges_gdf:", route_edges_gdf.columns)
                return route, None 
        else:
            print("Không tìm thấy lộ trình giữa hai điểm đã chọn.")
            return None, None
    except Exception as e:
        print(f"Lỗi khi tìm lộ trình hoặc tính ETA: {e}")
        return None, None

if __name__ == '__main__':
    class MockMedianSpeed: 
        def __init__(self):
            self.index = range(24)
            self.values = [max(0.1, 10 + i/2 - abs(i-12)) for i in range(24)] 
        def loc(self, hour):
            if hour in self.index:
                return self.values[hour]
            return np.median(self.values) if self.values else 10.0
        def median(self):
            return np.median(self.values) if self.values else 10.0
            
    mock_median_speed_hr = MockMedianSpeed()
    mock_median_speed_dow = pd.Series([10,10,10,11,12,15,14], index=['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']) # Giả lập
    mock_overall_median_all_days = 11.5 # Giả lập

    from data_loader import load_road_network 
    G_test_routing = load_road_network() 

    if G_test_routing:
        rep_speed_mps = get_representative_speed_mps(10, 1, mock_median_speed_hr, mock_median_speed_dow, mock_overall_median_all_days) 
        if rep_speed_mps is not None and rep_speed_mps > 0 : 
            G_with_times_test = add_travel_times_to_graph(G_test_routing, rep_speed_mps) 
            nodes_list = list(G_with_times_test.nodes())
            if len(nodes_list) >= 2:
                import random
                test_orig_node = random.choice(nodes_list)
                test_dest_node = random.choice(nodes_list)
                while test_dest_node == test_orig_node: 
                    test_dest_node = random.choice(nodes_list)
                print(f"Test routing từ {test_orig_node} đến {test_dest_node}")
                _, eta_test = calculate_eta_for_route(G_with_times_test, test_orig_node, test_dest_node)
                if eta_test is not None:
                    print(f"ETA thử nghiệm: {eta_test:.2f} phút")
            else:
                print("Không đủ nút trong đồ thị thử nghiệm.")
        else:
            print("Không thể lấy tốc độ đại diện hợp lệ cho thử nghiệm.")