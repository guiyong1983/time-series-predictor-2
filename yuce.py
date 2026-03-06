import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
import io
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 尝试导入必要的库，处理缺失情况
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    from prophet import Prophet
    import logging
    logging.getLogger('prophet').setLevel(logging.ERROR)
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

import streamlit as st

# ==========================================
# 1. 核心功能函数
# ==========================================

def load_data(file):
    """加载数据并提取数值列"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file)
    elif file.name.endswith('.txt'):
        # 尝试多种分隔符
        try:
            df = pd.read_csv(file, sep='\s+', header=None)
            if df.shape[1] == 1:
                df.columns = ['value']
            else:
                df = pd.read_csv(file, header=None)
        except:
            df = pd.read_csv(file, header=None)
    else:
        raise ValueError("不支持的文件格式")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("文件中未找到数值列")
    
    # 取第一个数值列
    data_series = df[numeric_cols[0]].dropna().reset_index(drop=True)
    return data_series

def detect_outliers(series, method='zscore', threshold=3):
    """异常值检测"""
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    return pd.Series([False] * len(series))

def create_features(data, lag_days=30):
    """构建 XGBoost/LightGBM 特征"""
    df = pd.DataFrame(data)
    df.columns = ['value']
    # 动态调整 lag_days，防止超过数据长度
    actual_lag = min(lag_days, len(data) - 1)
    if actual_lag < 1:
        return pd.DataFrame() # 数据太少无法构建特征
        
    for i in range(1, actual_lag + 1):
        df[f'lag_{i}'] = df['value'].shift(i)
    
    # 滚动窗口也动态调整
    for window in [7, 14, 30]:
        if window < len(data):
            df[f'rolling_mean_{window}'] = df['value'].shift(1).rolling(window=min(window, len(data)-1)).mean()
            df[f'rolling_std_{window}'] = df['value'].shift(1).rolling(window=min(window, len(data)-1)).std()
            df[f'rolling_max_{window}'] = df['value'].shift(1).rolling(window=min(window, len(data)-1)).max()
            df[f'rolling_min_{window}'] = df['value'].shift(1).rolling(window=min(window, len(data)-1)).min()
    
    # 添加趋势特征
    df['trend'] = df['value'].diff().shift(1)
    
    df = df.dropna()
    return df

def run_xgb_model(train, test, full_series):
    if not XGB_AVAILABLE:
        return None, None, None, "XGBoost 库未安装"
    
    # 动态调整滞后天数，最大不超过训练集长度的 1/3 或 30
    lag_days = min(30, max(1, len(train) // 3))
    
    df_feat = create_features(full_series, lag_days=lag_days)
    if df_feat.empty:
        return None, None, None, "训练集数据不足以构建 XGBoost 特征"
        
    feature_cols = [c for c in df_feat.columns if c != 'value']
    
    # 计算有效的训练集起始点 (减去 dropna 丢失的行)
    valid_train_count = len(train) - lag_days - 2 # 预留一点余量给 rolling window
    
    if valid_train_count <= 0:
        return None, None, None, "训练集长度不足以生成有效特征"

    X_train = df_feat.iloc[:valid_train_count][feature_cols]
    y_train = df_feat.iloc[:valid_train_count]['value']
    
    # 测试集部分
    test_start = valid_train_count
    test_end = test_start + len(test)
    
    # 确保不越界
    if test_end > len(df_feat):
        test_end = len(df_feat)
        
    if test_start >= len(df_feat):
        return None, None, None, "测试集索引超出特征范围"

    X_test = df_feat.iloc[test_start:test_end][feature_cols]
    y_test = df_feat.iloc[test_start:test_end]['value']

    if len(X_test) == 0:
        return None, None, None, "测试集为空"

    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.01, subsample=0.8, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    
    # 预测目标点 (下一个点)
    # 需要构建最后一行的特征
    last_vals = full_series.iloc[-(lag_days+5):].values # 多取一点防越界
    new_row = {}
    for i in range(1, lag_days + 1):
        if i <= len(last_vals):
            new_row[f'lag_{i}'] = last_vals[-i]
        else:
            new_row[f'lag_{i}'] = np.nan # 填充缺失
            
    for window in [7, 14, 30]:
        w = min(window, len(last_vals))
        if w > 0:
            new_row[f'rolling_mean_{window}'] = np.mean(last_vals[-w:])
            new_row[f'rolling_std_{window}'] = np.std(last_vals[-w:])
            new_row[f'rolling_max_{window}'] = np.max(last_vals[-w:])
            new_row[f'rolling_min_{window}'] = np.min(last_vals[-w:])
        else:
            new_row[f'rolling_mean_{window}'] = 0
            new_row[f'rolling_std_{window}'] = 0
            new_row[f'rolling_max_{window}'] = 0
            new_row[f'rolling_min_{window}'] = 0
    
    # 添加趋势特征
    if len(last_vals) > 1:
        new_row['trend'] = last_vals[-1] - last_vals[-2]
    else:
        new_row['trend'] = 0
            
    # 处理可能的 NaN (如果数据太短)
    X_target = pd.DataFrame([new_row])[feature_cols].fillna(method='ffill', axis=1).fillna(0)
    
    try:
        pred_target = model.predict(X_target)[0]
    except:
        pred_target = np.nan
        
    return y_pred_test, pred_target, y_test.values, None

def run_lstm_model(train, test, full_series):
    if not LSTM_AVAILABLE:
        return None, None, None, "TensorFlow 未安装"
    
    # 动态时间步长
    time_steps = min(20, len(train) // 2)
    if time_steps < 2:
        return None, None, None, "训练集太短，无法构建 LSTM 序列"

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    full_scaled = scaler.transform(full_series.values.reshape(-1, 1))

    def create_seq(data, steps):
        X, y = [], []
        for i in range(len(data) - steps):
            X.append(data[i:(i+steps), 0])
            y.append(data[i+steps, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_seq(train_scaled, time_steps)
    if len(X_train) == 0:
        return None, None, None, "训练序列生成失败"

    X_train = X_train.reshape((X_train.shape[0], time_steps, 1))

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0, callbacks=[EarlyStopping(patience=3)])

    # 评估
    X_full, y_full = create_seq(full_scaled, time_steps)
    if len(X_full) == 0:
        return None, None, None, "全量序列生成失败"
        
    X_full = X_full.reshape((X_full.shape[0], time_steps, 1))
    
    start_idx = len(train) - time_steps
    end_idx = start_idx + len(test)
    
    if end_idx > len(X_full):
        end_idx = len(X_full)
    if start_idx >= len(X_full):
        return None, None, None, "索引越界"
        
    y_pred_scaled = model.predict(X_full, verbose=0)
    y_pred_test = scaler.inverse_transform(y_pred_scaled[start_idx:end_idx]).flatten()
    y_test_actual = scaler.inverse_transform(y_full[start_idx:end_idx].reshape(-1, 1)).flatten()

    # 预测目标
    last_seq = full_scaled[-time_steps:].reshape(1, time_steps, 1)
    pred_target = scaler.inverse_transform(model.predict(last_seq, verbose=0))[0, 0]

    return y_pred_test, pred_target, y_test_actual, None

def run_prophet_model(train, test):
    if not PROPHET_AVAILABLE:
        return None, None, None, "Prophet 未安装"
    
    df_train = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=len(train), freq='D'),
        'y': train.values
    })
    
    model = Prophet(
        changepoint_prior_scale=0.05,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(df_train)
    
    future = model.make_future_dataframe(periods=len(test) + 1)
    forecast = model.predict(future)
    
    y_pred_test = forecast['yhat'].iloc[len(train):len(train)+len(test)].values
    pred_target = forecast['yhat'].iloc[-1]
    pred_lower = forecast['yhat_lower'].iloc[-1]
    pred_upper = forecast['yhat_upper'].iloc[-1]
    
    return y_pred_test, pred_target, (pred_lower, pred_upper), None

def run_arima_model(train, test):
    if not ARIMA_AVAILABLE:
        return None, None, None, "ARIMA 未安装"
    
    try:
        # 自动选择最佳参数
        best_aic = np.inf
        best_order = None
        best_model = None
        
        # 限制搜索范围以提高效率
        for p in range(0, 4):
            for d in range(0, 2):
                for q in range(0, 4):
                    try:
                        model = ARIMA(train, order=(p,d,q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p,d,q)
                            best_model = fitted_model
                    except:
                        continue
        
        if best_model is None:
            return None, None, None, "ARIMA 模型拟合失败"
        
        # 预测测试集
        forecast = best_model.forecast(steps=len(test))
        y_pred_test = forecast
        
        # 预测目标点（避免按标签索引导致 KeyError: 0）
        pred_target = best_model.forecast(steps=1).iloc[0]
        
        return y_pred_test, pred_target, None, None
    except Exception as e:
        return None, None, None, f"ARIMA 错误: {str(e)}"

def run_lightgbm_model(train, test, full_series):
    if not LGBM_AVAILABLE:
        return None, None, None, "LightGBM 未安装"
    
    # 动态调整滞后天数
    lag_days = min(30, max(1, len(train) // 3))
    
    df_feat = create_features(full_series, lag_days=lag_days)
    if df_feat.empty:
        return None, None, None, "训练集数据不足以构建 LightGBM 特征"
        
    feature_cols = [c for c in df_feat.columns if c != 'value']
    
    # 计算有效的训练集起始点
    valid_train_count = len(train) - lag_days - 2
    
    if valid_train_count <= 0:
        return None, None, None, "训练集长度不足以生成有效特征"

    X_train = df_feat.iloc[:valid_train_count][feature_cols]
    y_train = df_feat.iloc[:valid_train_count]['value']
    
    # 测试集部分
    test_start = valid_train_count
    test_end = test_start + len(test)
    
    if test_end > len(df_feat):
        test_end = len(df_feat)
        
    if test_start >= len(df_feat):
        return None, None, None, "测试集索引超出特征范围"

    X_test = df_feat.iloc[test_start:test_end][feature_cols]
    y_test = df_feat.iloc[test_start:test_end]['value']

    if len(X_test) == 0:
        return None, None, None, "测试集为空"

    model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.01, 
                             subsample=0.8, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    
    # 预测目标点
    last_vals = full_series.iloc[-(lag_days+5):].values
    new_row = {}
    for i in range(1, lag_days + 1):
        if i <= len(last_vals):
            new_row[f'lag_{i}'] = last_vals[-i]
        else:
            new_row[f'lag_{i}'] = np.nan
            
    for window in [7, 14, 30]:
        w = min(window, len(last_vals))
        if w > 0:
            new_row[f'rolling_mean_{window}'] = np.mean(last_vals[-w:])
            new_row[f'rolling_std_{window}'] = np.std(last_vals[-w:])
            new_row[f'rolling_max_{window}'] = np.max(last_vals[-w:])
            new_row[f'rolling_min_{window}'] = np.min(last_vals[-w:])
        else:
            new_row[f'rolling_mean_{window}'] = 0
            new_row[f'rolling_std_{window}'] = 0
            new_row[f'rolling_max_{window}'] = 0
            new_row[f'rolling_min_{window}'] = 0
    
    # 添加趋势特征
    if len(last_vals) > 1:
        new_row['trend'] = last_vals[-1] - last_vals[-2]
    else:
        new_row['trend'] = 0
            
    X_target = pd.DataFrame([new_row])[feature_cols].fillna(method='ffill', axis=1).fillna(0)
    
    try:
        pred_target = model.predict(X_target)[0]
    except:
        pred_target = np.nan
        
    return y_pred_test, pred_target, y_test.values, None

def cross_validate_models(train_series, n_splits=5):
    """时间序列交叉验证"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = {}
    
    models = {
        'XGBoost': run_xgb_model,
        'LSTM': run_lstm_model,
        'Prophet': run_prophet_model,
        'ARIMA': run_arima_model,
        'LightGBM': run_lightgbm_model
    }
    
    for model_name, model_func in models.items():
        if model_name == 'Prophet':
            continue  # Prophet 需要特殊处理
            
        mae_scores = []
        try:
            for train_idx, val_idx in tscv.split(train_series):
                train_fold = train_series.iloc[train_idx]
                val_fold = train_series.iloc[val_idx]
                
                # 使用完整的训练数据构建特征
                pred_test, _, y_true, _ = model_func(train_fold, val_fold, train_series)
                
                if pred_test is not None and y_true is not None:
                    # 对齐长度
                    min_len = min(len(pred_test), len(y_true))
                    mae = mean_absolute_error(y_true[:min_len], pred_test[:min_len])
                    mae_scores.append(mae)
            
            if mae_scores:
                cv_results[model_name] = {
                    'mean_mae': np.mean(mae_scores),
                    'std_mae': np.std(mae_scores)
                }
        except:
            continue
    
    return cv_results

def save_experiment_results(results, config, timestamp):
    """保存实验结果"""
    experiment_data = {
        'timestamp': timestamp,
        'config': config,
        'results': results
    }
    
    filename = f"experiment_{timestamp.replace(':', '-').replace(' ', '_')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, ensure_ascii=False, indent=2)
    
    return filename

# ==========================================
# 2. Streamlit 界面逻辑
# ==========================================

st.set_page_config(page_title="高级时间序列预测", layout="wide")
st.title("📊 高级时间序列预测平台")
st.markdown("""
**功能特点：**
- 📂 **本地上传**：支持 CSV, Excel, TXT。
- 🎚️ **完全自由**：训练集可从任意位置截取，长度可设为任意正整数。
- 🤖 **多模型融合**：XGBoost, LSTM, Prophet, ARIMA, LightGBM。
- 📉 **智能适配**：自动根据数据量调整模型参数。
- 🧪 **交叉验证**：时间序列交叉验证评估模型稳定性。
- 🚨 **异常检测**：Z-Score 和 IQR 方法检测异常值。
- 💾 **结果保存**：导出预测结果和配置参数。
""")

uploaded_file = st.file_uploader("1. 上传数据文件", type=['csv', 'xlsx', 'xls', 'txt'])

if uploaded_file:
    try:
        data_series = load_data(uploaded_file)
        total_len = len(data_series)
        st.success(f"✅ 数据加载成功！共 **{total_len}** 条记录。")
        
        # 异常值检测
        outlier_method = st.selectbox("选择异常值检测方法", ["无", "Z-Score", "IQR"])
        if outlier_method != "无":
            outliers = detect_outliers(data_series, method=outlier_method.lower())
            outlier_count = outliers.sum()
            if outlier_count > 0:
                st.warning(f"🚨 检测到 {outlier_count} 个异常值")
                if st.checkbox("查看异常值位置"):
                    outlier_indices = data_series[outliers].index.tolist()
                    st.write("异常值索引:", outlier_indices)
                    st.write("异常值:", data_series[outliers].tolist())
            else:
                st.info("✅ 未检测到异常值")
        
        # 显示前几行
        with st.expander("查看数据预览"):
            st.dataframe(data_series.head())
            st.line_chart(data_series)

        st.divider()
        st.subheader("2. 参数设置")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 训练集起始位置
            train_start_max = total_len - 10  # 至少保留10个点用于测试和预测
            train_start = st.number_input(
                "训练集起始位置", 
                min_value=0, 
                max_value=train_start_max,
                value=0,
                step=1,
                help="训练集从哪个索引位置开始（从0开始计数）"
            )
        
        with col2:
            # 训练集长度
            max_train_length = total_len - train_start - 10  # 预留空间
            default_train_length = min(3300, max_train_length) if max_train_length > 3300 else max_train_length
            
            train_length = st.number_input(
                "训练集长度", 
                min_value=10, 
                max_value=max_train_length, 
                value=int(default_train_length), 
                step=1,
                help="训练集的长度"
            )
        
        with col3:
            # 测试集长度
            test_start = train_start + train_length
            max_test_length = total_len - test_start - 1
            if max_test_length < 1:
                st.error("参数设置不当，没有剩余空间给测试集！")
                max_test_length = 1
            
            default_test_length = min(119, max_test_length) if max_test_length >= 119 else max_test_length
            
            test_length = st.number_input(
                "测试集长度", 
                min_value=1, 
                max_value=max_test_length, 
                value=int(default_test_length), 
                step=1
            )

        with col4:
            # 选择要运行的模型
            available_models = []
            if XGB_AVAILABLE:
                available_models.append("XGBoost")
            if LSTM_AVAILABLE:
                available_models.append("LSTM")
            if PROPHET_AVAILABLE:
                available_models.append("Prophet")
            if ARIMA_AVAILABLE:
                available_models.append("ARIMA")
            if LGBM_AVAILABLE:
                available_models.append("LightGBM")
            
            selected_models = st.multiselect(
                "选择要运行的模型",
                available_models,
                default=available_models,
                help="可以选择一个或多个模型进行预测"
            )

        # 计算关键参数
        train_end = train_start + train_length
        test_end = test_start + test_length
        target_index = test_end  # 预测目标点的索引
        
        # 验证参数有效性
        if train_end > total_len or test_end > total_len:
            st.error("❌ 参数设置超出数据范围，请调整参数！")
        elif train_length < 10:
            st.error("❌ 训练集长度至少需要10个点！")
        elif test_length < 1:
            st.error("❌ 测试集长度至少需要1个点！")
        else:
            remaining = total_len - target_index - 1
            st.info(f"""💡 **配置确认**:
- 训练集: [{train_start}:{train_end-1}] (长度: {train_length})
- 测试集: [{test_start}:{test_end-1}] (长度: {test_length})
- **预测目标**: 第 {target_index+1} 个数据 (索引 {target_index})
- 剩余缓冲数据: {max(0, remaining)} 条""")
            
            # 交叉验证选项
            do_cv = st.checkbox("执行交叉验证 (耗时较长)", value=False)
            
            if st.button("🚀 开始运行多模型预测"):
                # 截取数据
                train_series = data_series.iloc[train_start:train_end]
                test_series = data_series.iloc[test_start:test_end]
                # 构建用于特征工程的完整历史数据（从训练集开始到测试集结束）
                full_series = data_series.iloc[train_start:test_end]
                
                results = {}
                y_true = test_series.values
                
                st.write("### 🔄 模型训练中...")
                progress_bar = st.progress(0)
                progress_step = 100 // len(selected_models) if selected_models else 100
                
                # 交叉验证
                if do_cv and len(selected_models) > 0:
                    with st.spinner('执行交叉验证...'):
                        cv_results = cross_validate_models(train_series)
                        if cv_results:
                            st.subheader("🔬 交叉验证结果")
                            cv_df = pd.DataFrame(cv_results).T
                            st.dataframe(cv_df)
                
                # 模型映射
                model_functions = {
                    'XGBoost': run_xgb_model,
                    'LSTM': run_lstm_model,
                    'Prophet': run_prophet_model,
                    'ARIMA': run_arima_model,
                    'LightGBM': run_lightgbm_model
                }
                
                # 运行选定的模型
                for i, model_name in enumerate(selected_models):
                    with st.spinner(f'运行 {model_name}...'):
                        model_func = model_functions[model_name]
                        try:
                            if model_name in ['XGBoost', 'LSTM', 'LightGBM']:
                                pred_test, pred_target, y_test_actual, err = model_func(train_series, test_series, full_series)
                            elif model_name == 'Prophet':
                                pred_test, pred_target, conf_interval, err = model_func(train_series, test_series)
                                if conf_interval:
                                    results[model_name] = {
                                        'test': pred_test, 
                                        'target': pred_target,
                                        'conf_interval': conf_interval
                                    }
                                    continue
                            elif model_name == 'ARIMA':
                                pred_test, pred_target, y_test_actual, err = model_func(train_series, test_series)
                            
                            if err:
                                st.warning(f"⚠️ {model_name}: {err}")
                            else:
                                results[model_name] = {'test': pred_test, 'target': pred_target}
                                
                        except Exception as e:
                            st.warning(f"⚠️ {model_name} 执行出错: {str(e)}")
                    
                    progress_bar.progress((i + 1) * progress_step)

                # ==========================================
                # 3. 结果聚合与加权
                # ==========================================
                if len(results) == 0:
                    st.error("所有模型运行失败，请检查数据或库安装情况。")
                else:
                    # 统一长度：找到所有成功模型测试结果的最小长度
                    min_common_len = min([len(v['test']) for v in results.values() if 'test' in v])
                    y_true_final = test_series.values[:min_common_len]
                    
                    for name in results:
                        if 'test' in results[name]:
                            results[name]['test'] = results[name]['test'][:min_common_len]
                    
                    # 计算 MAE 和 权重
                    weights = {}
                    maes = {}
                    total_inv = 0
                    
                    st.divider()
                    st.subheader("📊 模型性能评估 (基于测试集 MAE)")
                    
                    cols = st.columns(len(results))
                    idx = 0
                    for name, data in results.items():
                        if 'test' in data:
                            mae = mean_absolute_error(y_true_final, data['test'])
                            rmse = np.sqrt(mean_squared_error(y_true_final, data['test']))
                            maes[name] = mae
                            inv_mae = 1.0 / (mae + 1e-6)
                            weights[name] = inv_mae
                            total_inv += inv_mae
                            
                            with cols[idx]:
                                st.metric(f"{name}", f"MAE: {mae:.6f}", f"RMSE: {rmse:.6f}")
                            idx += 1
                    
                    # 归一化权重
                    for name in weights:
                        weights[name] /= total_inv
                    
                    # 计算集成预测
                    ensemble_pred = 0
                    st.divider()
                    st.subheader(f"🎯 预测结果：第 {target_index+1} 个数据")
                    
                    res_cols = st.columns(len(results) + 1)
                    idx = 0
                    for name, data in results.items():
                        contribution = weights[name] * data['target']
                        ensemble_pred += contribution
                        with res_cols[idx]:
                            st.metric(f"{name}", f"{data['target']:.6f}", f"权重: {weights[name]:.4f}")
                        idx += 1
                    
                    with res_cols[-1]:
                        st.metric("**集成预测**", f"{ensemble_pred:.6f}", delta="Final Result", delta_color="normal")
                    
                    # 显示Prophet置信区间（如果有）
                    prophet_result = results.get('Prophet')
                    if prophet_result and 'conf_interval' in prophet_result:
                        lower, upper = prophet_result['conf_interval']
                        st.info(f"🔮 Prophet 置信区间 (95%): [{lower:.4f}, {upper:.4f}]")
                    
                    # 可视化
                    st.divider()
                    st.subheader("📈 预测趋势对比")
                    fig, ax = plt.subplots(figsize=(14, 7))
                    
                    # 绘制完整数据用于背景参考
                    ax.plot(range(total_len), data_series.values, 
                           color='lightgray', alpha=0.5, linewidth=1, label='完整数据')
                    
                    # 绘制训练集
                    ax.plot(range(train_start, train_end), train_series.values, 
                           color='blue', linewidth=2, alpha=0.7, label=f'训练集 [{train_start}:{train_end-1}]')
                    
                    # 绘制测试集真实值
                    test_x_axis = range(test_start, test_start + len(y_true_final))
                    ax.plot(test_x_axis, y_true_final, 
                           label='真实值 (测试集)', color='black', linewidth=2, marker='o', markersize=4)
                    
                    colors = {
                        'XGBoost': 'blue', 
                        'LSTM': 'green', 
                        'Prophet': 'orange',
                        'ARIMA': 'purple',
                        'LightGBM': 'red'
                    }
                    for name, data in results.items():
                        if 'test' in data:
                            ax.plot(test_x_axis, data['test'], 
                                  label=f'{name} (MAE={maes[name]:.4f})', 
                                  linestyle='--', 
                                  color=colors.get(name, 'gray'), 
                                  alpha=0.7)
                    
                    # 标记预测点
                    ax.scatter(target_index, ensemble_pred, 
                              color='red', s=200, marker='*', zorder=10, 
                              label=f'集成预测：{ensemble_pred:.4f}')
                    
                    # 标记分割线
                    ax.axvline(x=train_start, color='gray', linestyle=':', alpha=0.7, label='训练集开始')
                    ax.axvline(x=train_end, color='gray', linestyle='-.', alpha=0.7, label='训练集结束')
                    ax.axvline(x=test_start, color='gray', linestyle='--', alpha=0.7, label='测试集开始')
                    ax.axvline(x=test_end, color='gray', linestyle='--', alpha=0.7, label='测试集结束')
                    
                    ax.set_title(f'多模型融合预测 - 目标：第 {target_index+1} 个数据')
                    ax.set_xlabel('时间步索引')
                    ax.set_ylabel('数值')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # 保存结果
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    config = {
                        'train_start': train_start,
                        'train_length': train_length,
                        'test_length': test_length,
                        'selected_models': selected_models,
                        'do_cross_validation': do_cv
                    }
                    
                    if st.button("💾 保存实验结果"):
                        filename = save_experiment_results(results, config, timestamp)
                        st.success(f"实验结果已保存至: {filename}")
                        
                        # 提供下载链接
                        with open(filename, 'r') as f:
                            st.download_button(
                                label="📥 下载结果文件",
                                data=f,
                                file_name=filename,
                                mime="application/json"
                            )

    except Exception as e:
        st.error(f"发生错误：{str(e)}")
        st.exception(e)

else:
    st.info("👆 请上传文件以开始分析。支持 CSV, Excel, TXT 格式。")
