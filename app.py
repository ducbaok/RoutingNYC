# app.py
import streamlit as st

from data_loader import load_taxi_trip_data

# --- DÒNG NÀY PHẢI LÀ LỆNH STREAMLIT ĐẦU TIÊN ---
st.set_page_config(page_title="Ứng dụng Dự đoán ETA Manhattan", layout="wide", initial_sidebar_state="collapsed")
# -------------------------------------------------

import pandas as pd
import numpy as np
import osmnx as ox
import joblib
import folium 
from streamlit_folium import st_folium 

try:
    from config import DEFAULT_TARGET_HOUR, DEFAULT_TARGET_DAY_NUMERIC, PLACE_NAME, YELLOW_TAXI_DATA_FILE
    from data_loader import load_road_network, load_taxi_zones 
    from data_processor import (
        filter_taxi_zones_by_borough,
        initial_trip_data_cleaning,
        filter_trips_by_location_ids,
        calculate_median_speed_by_time,
        create_ml_training_data
    )
    from routing_utils import (
        add_travel_times_to_graph,
        calculate_eta_for_route,
        parse_maxspeed # Đảm bảo hàm này được import nếu nó nằm trong routing_utils.py và cần thiết
    )
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:
    st.error(f"Lỗi import module cục bộ: {e}. Đảm bảo bạn đang chạy 'streamlit run app.py' từ thư mục gốc của dự án và tất cả các file .py cần thiết đều có mặt.")
    st.stop()


# --- Các hàm được Cache của Streamlit ---
@st.cache_resource(show_spinner="Đang tải dữ liệu bản đồ và mô hình ML...")
def load_core_data_cached():
    print("Thực thi: load_core_data_cached()")
    g_manhattan, taxi_zones, model, preprocessor = None, None, None, None
    error_messages = []
    try:
        g_manhattan = load_road_network(place_name=PLACE_NAME)
        if g_manhattan is None: error_messages.append("Lỗi: Không tải được bản đồ Manhattan.")
    except Exception as e: error_messages.append(f"Lỗi nghiêm trọng khi tải bản đồ Manhattan: {e}")
    try:
        taxi_zones = load_taxi_zones()
        if taxi_zones is None: error_messages.append("Lỗi: Không tải được dữ liệu Taxi Zones.")
    except Exception as e: error_messages.append(f"Lỗi nghiêm trọng khi tải Taxi Zones: {e}")
    if not error_messages:
        try:
            model = joblib.load('trained_rf_model.joblib')
            preprocessor = joblib.load('data_preprocessor.joblib')
            print("Tải model và preprocessor thành công.")
        except FileNotFoundError:
            error_messages.append("LƯU Ý: Không tìm thấy file model/preprocessor. Sẽ thử huấn luyện lại.")
            model, preprocessor = None, None # Để logic huấn luyện lại được kích hoạt
        except Exception as e: 
            error_messages.append(f"Lỗi khi tải model/preprocessor: {e}")
            model, preprocessor = None, None
    print("load_core_data_cached() hoàn tất.")
    return g_manhattan, taxi_zones, model, preprocessor, error_messages

@st.cache_data(show_spinner="Đang tính toán tốc độ fallback...")
def get_fallback_speeds_cached(taxi_trip_file_path_param, manhattan_loc_ids_list_param):
    print("Thực thi: get_fallback_speeds_cached()")
    fallback_speeds, error_msg = pd.Series([10.0]*24, index=range(24)), "Sử dụng tốc độ fallback mặc định ban đầu."
    if not manhattan_loc_ids_list_param:
        error_msg = "Không có LocationID của Manhattan cho fallback speeds. " + error_msg
        return fallback_speeds, error_msg
    try:
        import os
        if not os.path.exists(taxi_trip_file_path_param):
            raise FileNotFoundError(f"File dữ liệu taxi '{taxi_trip_file_path_param}' không tồn tại.")
        raw_taxi_trips = pd.read_parquet(taxi_trip_file_path_param)
        for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
            raw_taxi_trips[col] = pd.to_datetime(raw_taxi_trips[col])
        cleaned_taxi_trips = initial_trip_data_cleaning(raw_taxi_trips)
        manhattan_trips = pd.DataFrame()
        if cleaned_taxi_trips is not None and not cleaned_taxi_trips.empty:
            manhattan_trips = filter_trips_by_location_ids(cleaned_taxi_trips, manhattan_loc_ids_list_param)
        if not manhattan_trips.empty:
            median_hr, _ = calculate_median_speed_by_time(manhattan_trips)
            if not median_hr.empty: fallback_speeds, error_msg = median_hr, None # Không có lỗi nếu tính được
            else: error_msg = "Không tính được median_speed_by_hour từ dữ liệu. " + error_msg
        else: error_msg = "Không có chuyến đi Manhattan nào để tính fallback speed. " + error_msg
    except FileNotFoundError as e_fnf: error_msg = str(e_fnf)
    except Exception as e: error_msg = f"Lỗi khi tính fallback speeds: {e}. " + error_msg
    print("get_fallback_speeds_cached() hoàn tất.")
    return fallback_speeds, error_msg

# --- Khởi tạo Session State ---
default_map_center = [40.7679, -73.9822]
default_map_zoom = 12
# ... (phần khởi tạo session_state giữ nguyên) ...
for key, default_val in [
    ('origin_coords', None), ('destination_coords', None), ('route_nodes', None),
    ('map_center', default_map_center), ('map_zoom', default_map_zoom),
    ('origin_input_key_val', "Times Square, New York, NY"), 
    ('destination_input_key_val', "Wall Street, New York, NY"),
    ('hour_input_key_val', DEFAULT_TARGET_HOUR),
    ('day_input_key_val', DEFAULT_TARGET_DAY_NUMERIC)
]:
    if key not in st.session_state:
        st.session_state[key] = default_val

# --- Tải dữ liệu khởi tạo ---
G_manhattan, taxi_zones_gdf, ml_model, ml_preprocessor, initial_errors = load_core_data_cached()
if initial_errors:
    for err in initial_errors:
        if "LƯU Ý" in err: st.info(err)
        else: st.error(err)

manhattan_location_ids_for_fallback = []
if taxi_zones_gdf is not None:
    _, manhattan_location_ids_for_fallback = filter_taxi_zones_by_borough(taxi_zones_gdf)

fallback_median_speed_by_hour, fallback_errors = get_fallback_speeds_cached(
    YELLOW_TAXI_DATA_FILE, manhattan_location_ids_for_fallback
)
if fallback_errors: st.warning(fallback_errors)

# --- Logic Huấn luyện lại Model ---
if ml_model is None and G_manhattan is not None and taxi_zones_gdf is not None and not any("Lỗi nghiêm trọng" in str(err) for err in initial_errors if err is not None):
    # ... (Code huấn luyện lại giữ nguyên như phiên bản trước) ...
    st.info("Mô hình ML chưa được tải. Đang thử huấn luyện lại...")
    with st.spinner("Đang huấn luyện lại mô hình ML... (có thể mất vài phút)"):
        try:
            raw_taxi_trips_for_retrain = load_taxi_trip_data(YELLOW_TAXI_DATA_FILE)
            if raw_taxi_trips_for_retrain is None:
                st.error("Không thể tải dữ liệu taxi để huấn luyện lại mô hình.")
            else:
                cleaned_taxi_for_retrain = initial_trip_data_cleaning(raw_taxi_trips_for_retrain)
                manhattan_trips_for_retrain = pd.DataFrame()
                if cleaned_taxi_for_retrain is not None and not cleaned_taxi_for_retrain.empty and manhattan_location_ids_for_fallback:
                     manhattan_trips_for_retrain = filter_trips_by_location_ids(cleaned_taxi_for_retrain, manhattan_location_ids_for_fallback)

                if manhattan_trips_for_retrain.empty:
                    st.error("Không có dữ liệu Manhattan để huấn luyện lại mô hình.")
                else:
                    ml_training_df_retrain = create_ml_training_data(manhattan_trips_for_retrain)
                    if ml_training_df_retrain.empty:
                        st.error("Không thể tạo dữ liệu huấn luyện ML cho việc huấn luyện lại.")
                    else:
                        X_retrain = ml_training_df_retrain[['PULocationID', 'pickup_hour', 'pickup_day_of_week']]
                        y_retrain = ml_training_df_retrain['target_median_speed_mph']
                        
                        categorical_features_retrain = ['PULocationID']
                        preprocessor_retrain = ColumnTransformer(
                            transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_retrain)],
                            remainder='passthrough'
                        )
                        X_processed_retrain = preprocessor_retrain.fit_transform(X_retrain)
                        
                        ml_model_retrain = RandomForestRegressor(
                            n_estimators=100, random_state=42, n_jobs=-1, oob_score=True, 
                            max_depth=20, min_samples_split=5, min_samples_leaf=2
                        )
                        ml_model_retrain.fit(X_processed_retrain, y_retrain)
                        
                        joblib.dump(ml_model_retrain, 'trained_rf_model.joblib')
                        joblib.dump(preprocessor_retrain, 'data_preprocessor.joblib')
                        st.success("Đã huấn luyện lại và lưu mô hình, preprocessor thành công! Vui lòng làm mới trang để sử dụng.")
                        ml_model = ml_model_retrain
                        ml_preprocessor = preprocessor_retrain
        except Exception as e_retrain:
            st.error(f"Lỗi trong quá trình huấn luyện lại mô hình: {e_retrain}")


# --- Hàm tạo và cập nhật bản đồ ---
def create_and_display_map(graph_to_plot_on, origin_coords_param, dest_coords_param, route_nodes_list_param):
    map_center_from_state = st.session_state.map_center
    map_zoom_from_state = st.session_state.map_zoom

    if isinstance(map_center_from_state, dict) and 'lat' in map_center_from_state and 'lng' in map_center_from_state:
        map_location_for_folium = [map_center_from_state['lat'], map_center_from_state['lng']]
    elif isinstance(map_center_from_state, (list, tuple)) and len(map_center_from_state) == 2:
        map_location_for_folium = list(map_center_from_state)
    else:
        map_location_for_folium = default_map_center
        print(f"Cảnh báo: map_center_from_state có định dạng không mong đợi: {map_center_from_state}. Dùng default.")

    current_map = folium.Map(location=map_location_for_folium, zoom_start=map_zoom_from_state, tiles="CartoDB positron")

    if origin_coords_param and isinstance(origin_coords_param, (list, tuple)) and len(origin_coords_param) == 2:
        folium.Marker(location=origin_coords_param, popup="Điểm xuất phát", icon=folium.Icon(color='green', icon='play')).add_to(current_map)
    if dest_coords_param and isinstance(dest_coords_param, (list, tuple)) and len(dest_coords_param) == 2:
        folium.Marker(location=dest_coords_param, popup="Điểm đến", icon=folium.Icon(color='red', icon='stop')).add_to(current_map)

    if route_nodes_list_param and graph_to_plot_on and len(route_nodes_list_param) > 1: # Cần ít nhất 2 nút để tạo lộ trình
        try:
            # Lấy GeoDataFrame của các cạnh trong lộ trình
            route_edges_gdf = ox.routing.route_to_gdf(graph_to_plot_on, route_nodes_list_param)
            
            # Tạo một lớp GeoJson từ GeoDataFrame này và thêm vào bản đồ
            folium.GeoJson(
                route_edges_gdf,
                name="Lộ trình",
                style_function=lambda x: {'color': "#007bff", 'weight': 5, 'opacity': 0.7}, # Màu xanh dương
                # Bạn có thể thêm tooltip nếu muốn
                # tooltip=folium.features.GeoJsonTooltip(fields=['name', 'length', 'travel_time'], # Ví dụ
                #                                       aliases=['Đường:', 'Chiều dài (m):', 'Thời gian (s):'])
            ).add_to(current_map)
            
            # Tự động zoom vào lộ trình
            if not route_edges_gdf.empty:
                bounds = route_edges_gdf.total_bounds 
                current_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        except Exception as e_plot_route:
            st.warning(f"Lỗi khi vẽ lộ trình trên bản đồ: {e_plot_route}.")
            
    map_output = st_folium(current_map, width=700, height=500, returned_objects=["center", "zoom"])
    
    if map_output:
        if map_output.get("center"): st.session_state.map_center = map_output["center"] 
        if map_output.get("zoom"): st.session_state.map_zoom = map_output["zoom"]


# --- Giao diện người dùng Streamlit ---
st.title("Ứng dụng Dự đoán Thời gian Di chuyển (ETA) ở Manhattan")
st.markdown("Nhập địa điểm xuất phát, điểm đến và thời gian mong muốn để nhận ETA.")

map_placeholder = st.empty() 

if G_manhattan: 
    with map_placeholder.container():
        create_and_display_map(
            G_manhattan, 
            st.session_state.origin_coords, 
            st.session_state.destination_coords,
            st.session_state.route_nodes
        )

if G_manhattan is not None and ml_model is not None and ml_preprocessor is not None and fallback_median_speed_by_hour is not None and taxi_zones_gdf is not None:
    
    def geocode_address(address_str_key, coord_state_key):
        address_str = st.session_state[address_str_key]
        if address_str: 
            try:
                lat, lon = ox.geocode(address_str)
                st.session_state[coord_state_key] = (lat, lon) 
                st.session_state.route_nodes = None # Xóa lộ trình cũ khi địa chỉ thay đổi
            except Exception:
                st.session_state[coord_state_key] = None
                st.warning(f"Không tìm thấy địa chỉ: '{address_str}'")
        else:
            st.session_state[coord_state_key] = None # Xóa marker nếu input rỗng
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Điểm xuất phát:", 
                      value=st.session_state.origin_input_key_val,
                      key="origin_input_key", 
                      on_change=geocode_address, args=("origin_input_key", "origin_coords"))
        st.number_input("Giờ khởi hành (0-23):", 
                        min_value=0, max_value=23, value=st.session_state.hour_input_key_val, step=1, key="hour_input_key")
    with col2:
        st.text_input("Điểm đến:", 
                      value=st.session_state.destination_input_key_val, 
                      key="destination_input_key", 
                      on_change=geocode_address, args=("destination_input_key", "destination_coords"))
        days_options = {0: "Thứ Hai", 1: "Thứ Ba", 2: "Thứ Tư", 3: "Thứ Năm", 4: "Thứ Sáu", 5: "Thứ Bảy", 6: "Chủ Nhật"}
        st.selectbox("Ngày khởi hành:", options=list(days_options.keys()),
                     format_func=lambda x: days_options[x], 
                     index=st.session_state.day_input_key_val, key="day_input_key")

    if st.button("Tính toán ETA", key="calculate_eta_button"):
        # Lưu giá trị input hiện tại vào session_state để giữ lại sau khi rerun
        st.session_state.origin_input_key_val = st.session_state.origin_input_key
        st.session_state.destination_input_key_val = st.session_state.destination_input_key
        st.session_state.hour_input_key_val = st.session_state.hour_input_key
        st.session_state.day_input_key_val = st.session_state.day_input_key
        
        origin_address_str = st.session_state.origin_input_key
        destination_address_str = st.session_state.destination_input_key
        current_hour = st.session_state.hour_input_key
        current_day = st.session_state.day_input_key
        
        if not origin_address_str or not destination_address_str:
            st.warning("Vui lòng nhập cả điểm xuất phát và điểm đến.")
        elif st.session_state.origin_coords is None or st.session_state.destination_coords is None:
            st.warning("Vui lòng đảm bảo cả hai địa chỉ đã được geocode thành công.")
        else:
            with st.spinner("Đang tính toán ETA..."):
                origin_node = ox.nearest_nodes(G_manhattan, X=st.session_state.origin_coords[1], Y=st.session_state.origin_coords[0])
                dest_node = ox.nearest_nodes(G_manhattan, X=st.session_state.destination_coords[1], Y=st.session_state.destination_coords[0])
                
                if origin_node and dest_node:
                    G_for_eta = G_manhattan.copy() 
                    add_travel_times_to_graph(
                        G_for_eta, current_hour, current_day, ml_model,
                        ml_preprocessor, taxi_zones_gdf, fallback_median_speed_by_hour
                    )
                    route, eta_minutes = calculate_eta_for_route(G_for_eta, origin_node, dest_node)
                    st.session_state.route_nodes = route # Lưu lộ trình để vẽ lại bản đồ

                    if route and eta_minutes is not None and not (isinstance(eta_minutes, float) and (pd.isna(eta_minutes) or np.isinf(eta_minutes))):
                        st.success(f"**ETA Dự kiến: {eta_minutes:.2f} phút**")
                        st.info(f"Từ '{origin_address_str}' đến '{destination_address_str}' vào {current_hour}h, {days_options[current_day]}.")
                    elif route is None: st.error("Không tìm thấy lộ trình.")
                    else: st.error(f"Không thể tính ETA (ETA = {eta_minutes}).")
                else: st.error("Không thể tìm thấy nút gần nhất cho địa chỉ.")
            st.rerun() # Buộc chạy lại để cập nhật bản đồ với lộ trình mới
else: 
    st.header("Đang tải dữ liệu hoặc có lỗi xảy ra...")
    st.warning("Vui lòng đợi hoặc kiểm tra thông báo lỗi ở trên.")