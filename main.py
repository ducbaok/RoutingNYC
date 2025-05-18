# main.py
import osmnx as ox
# import random # Không cần random nữa nếu người dùng nhập
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Cần cho việc kiểm tra np.isinf, np.isnan

from config import DEFAULT_TARGET_HOUR, DEFAULT_TARGET_DAY_NUMERIC # Vẫn có thể dùng làm giá trị mặc định
from data_loader import load_road_network, load_taxi_zones, load_taxi_trip_data
from data_processor import (
    filter_taxi_zones_by_borough,
    initial_trip_data_cleaning,
    filter_trips_by_location_ids,
    calculate_median_speed_by_time
)
from routing_utils import (
    get_representative_speed_mps,
    add_travel_times_to_graph,
    calculate_eta_for_route
)
def get_user_inputs(G_graph):
    # ... (Nội dung hàm giữ nguyên như lần sửa trước) ...
    while True:
        try:
            origin_address = input("Nhập địa chỉ điểm xuất phát ở Manhattan (ví dụ: 'Times Square, New York'): ")
            origin_lat, origin_lon = ox.geocode(origin_address)
            origin_node = ox.nearest_nodes(G_graph, X=origin_lon, Y=origin_lat)
            print(f"Tìm thấy Node xuất phát gần nhất: {origin_node} cho địa chỉ '{origin_address}'")
            break
        except Exception as e:
            print(f"Lỗi geocoding địa chỉ xuất phát: {e}. Vui lòng thử lại.")
    while True:
        try:
            destination_address = input("Nhập địa chỉ điểm đến ở Manhattan (ví dụ: 'Wall Street, New York'): ")
            dest_lat, dest_lon = ox.geocode(destination_address)
            destination_node = ox.nearest_nodes(G_graph, X=dest_lon, Y=dest_lat)
            print(f"Tìm thấy Node đích gần nhất: {destination_node} cho địa chỉ '{destination_address}'")
            break
        except Exception as e:
            print(f"Lỗi geocoding địa chỉ đích: {e}. Vui lòng thử lại.")
    while True:
        try:
            hour_str = input(f"Nhập giờ khởi hành (0-23, mặc định {DEFAULT_TARGET_HOUR}): ")
            if not hour_str: 
                hour = DEFAULT_TARGET_HOUR
                break
            hour = int(hour_str)
            if 0 <= hour <= 23:
                break
            else:
                print("Giờ không hợp lệ. Vui lòng nhập số từ 0 đến 23.")
        except ValueError:
            print("Vui lòng nhập một số nguyên cho giờ.")
    while True:
        try:
            day_str = input(f"Nhập ngày khởi hành (0=Thứ Hai, ..., 6=Chủ Nhật, mặc định {DEFAULT_TARGET_DAY_NUMERIC}): ")
            if not day_str:
                day = DEFAULT_TARGET_DAY_NUMERIC
                break
            day = int(day_str)
            if 0 <= day <= 6:
                break
            else:
                print("Ngày không hợp lệ. Vui lòng nhập số từ 0 đến 6.")
        except ValueError:
            print("Vui lòng nhập một số nguyên cho ngày.")
    return origin_node, destination_node, hour, day


def main():
    # ... (phần tải dữ liệu và xử lý ban đầu như cũ) ...
    print("Bắt đầu quy trình tính toán ETA...")
    G_manhattan = load_road_network()
    taxi_zones_gdf = load_taxi_zones()
    raw_taxi_trips_df = load_taxi_trip_data()
    if G_manhattan is None or taxi_zones_gdf is None or raw_taxi_trips_df is None:
        print("Lỗi tải dữ liệu đầu vào. Kết thúc chương trình.")
        return
    origin_node, destination_node, target_hour, target_day_numeric = get_user_inputs(G_manhattan)
    if origin_node is None or destination_node is None:
        print("Không thể xác định điểm đầu hoặc cuối từ địa chỉ. Kết thúc chương trình.")
        return
    _, manhattan_location_ids = filter_taxi_zones_by_borough(taxi_zones_gdf)
    if not manhattan_location_ids:
        print("Không tìm thấy LocationID nào cho Manhattan. Kết thúc chương trình.")
        return
    cleaned_taxi_trips_df = initial_trip_data_cleaning(raw_taxi_trips_df)
    if cleaned_taxi_trips_df is None or cleaned_taxi_trips_df.empty:
        print("Không có dữ liệu taxi sau khi làm sạch ban đầu. Kết thúc chương trình.")
        return
    manhattan_trips_df = filter_trips_by_location_ids(cleaned_taxi_trips_df, manhattan_location_ids)
    if manhattan_trips_df.empty:
        print("Không có chuyến đi nào hoàn toàn trong Manhattan. Kết thúc chương trình.")
        return

    median_speed_by_hour, median_speed_by_day_of_week = calculate_median_speed_by_time(manhattan_trips_df)
    
    if median_speed_by_hour.empty or median_speed_by_day_of_week.empty:
        print("Không thể tính toán median_speed_by_hour hoặc median_speed_by_day_of_week. Kết thúc chương trình.")
        return

    # TÍNH TOÁN overall_median_speed_all_days_mph
    overall_median_speed_all_days_mph = None
    if 'average_speed_mph' in manhattan_trips_df.columns and not manhattan_trips_df['average_speed_mph'].empty:
        overall_median_speed_all_days_mph = manhattan_trips_df['average_speed_mph'].median()
        print(f"Tốc độ trung vị tổng thể (all days) cho Manhattan: {overall_median_speed_all_days_mph:.2f} mph")
    else:
        print("Cảnh báo: Không thể tính overall_median_speed_all_days_mph. Sẽ không có điều chỉnh theo ngày.")
        # Nếu không tính được, hàm get_representative_speed_mps sẽ dùng day_of_week_factor = 1.0

    # --- 3. Tính toán ETA ---
    representative_speed_mps = None 
    try:
        representative_speed_mps = get_representative_speed_mps(
            target_hour, 
            target_day_numeric,
            median_speed_by_hour, 
            median_speed_by_day_of_week,
            overall_median_speed_all_days_mph # Truyền tham số mới
        )
    # ... (phần còn lại của hàm main như cũ) ...
    except Exception as e_get_speed:
        print(f"LỖI TRONG KHI GỌI get_representative_speed_mps hoặc ngay sau đó: {e_get_speed}")
    if representative_speed_mps is None: 
        print("Không thể xác định tốc độ đại diện (representative_speed_mps is None sau khi gọi hàm hoặc do lỗi). Kết thúc chương trình.")
        return
    if not isinstance(representative_speed_mps, (int, float)) or representative_speed_mps <= 0:
        print(f"Tốc độ đại diện không hợp lệ ({representative_speed_mps}). Kết thúc chương trình.")
        return
    add_travel_times_to_graph(G_manhattan, representative_speed_mps) 
    G_with_times = G_manhattan 
    print(f"\nSẽ tính ETA cho thời điểm: {target_hour} giờ, ngày thứ {target_day_numeric} trong tuần.")
    print(f"Từ Node ID: {origin_node} đến Node ID: {destination_node}")
    route, eta_minutes = calculate_eta_for_route(G_with_times, origin_node, destination_node)
    if route and eta_minutes is not None and not (isinstance(eta_minutes, float) and (pd.isna(eta_minutes) or np.isinf(eta_minutes))):
        print(f"\n--- KẾT QUẢ ETA CUỐI CÙNG ---")
        print(f"Lộ trình tìm được có {len(route)} nút.")
        print(f"ETA dự kiến: {eta_minutes:.2f} phút.")
        try:
            print("\nĐang vẽ lộ trình...")
            fig, ax = ox.plot_graph_route(
                G_with_times, route, route_color="r", route_linewidth=4,      
                node_size=0, bgcolor="w", show=True, close=False,            
                figsize=(10,10), dpi=100                 
            )
            ax.set_title(f"Lộ trình từ Node {origin_node} đến {destination_node}\nETA: {eta_minutes:.2f} phút ({target_hour}h, Ngày {target_day_numeric})", fontsize=15)
            plt.show() 
            print("Hoàn tất vẽ lộ trình.")
        except Exception as e_plot:
            print(f"Lỗi khi vẽ lộ trình: {e_plot}")
    else:
        print("Không thể tính toán ETA hợp lệ cho các điểm đã chọn.")
    print("\nQuy trình tính toán ETA kết thúc.")

if __name__ == '__main__':
    main()