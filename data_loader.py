import os
import scipy.io
import numpy as np
import glob
import collections

# --- 配置参数 ---
# 请确保此路径指向您存放特征.mat文件的 "ExtractedFeatures" 文件夹
DATA_PATH = r"D:\Firfox_Download\SEED\SEED_EEG\ExtractedFeatures_4s"

# 标签：1 表示积极, 0 表示中性, -1 表示消极
# 情绪标签映射: 我们将 -1 (消极) -> 0, 0 (中性) -> 1, 1 (积极) -> 2
LABEL_MAPPING = {-1: 0, 0: 1, 1: 2} # 消极:0, 中性:1, 积极:2
LABELS_SEQUENCE = np.array([1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1])
MAPPED_LABELS_SEQUENCE = np.array([LABEL_MAPPING[l] for l in LABELS_SEQUENCE])

N_TRIALS_PER_EXPERIMENT = 15
N_CHANNELS = 62
N_FEATURES_PER_STEP = 5 # 5个频带的DE值
RANDOM_SEED = 2025 # 用于所有随机操作，确保可复现性

def get_subject_files(data_path):
    """获取每个被试对应的所有 .mat 文件列表"""
    all_files = sorted(glob.glob(os.path.join(data_path, "*.mat")))
    subject_files = collections.defaultdict(list)
    for f in all_files:
        basename = os.path.basename(f)
        if basename in ["label.mat", "readme.txt"]:
            continue
        try:
            subject_id = int(basename.split('_')[0])
            subject_files[subject_id].append(f)
        except (ValueError, IndexError):
            print(f"警告: 文件名 {basename} 不符合 'subjectID_date.mat' 格式，已跳过。")
    return subject_files

def load_and_reshape_data(data_path, selected_feature_prefix="de_LDS"):
    """
    加载所有被试的特征数据。
    每个 trial (62个通道) 被视为一个独立的样本单位，而不是将时间步展开。
    返回: 特征, 标签, 每个样本对应的被试ID, 每个样本对应的session索引, 每个样本对应的trial索引
    """
    subject_files_map = get_subject_files(data_path)
    
    all_trials_list = []
    all_labels_list = []
    all_subject_ids_list = []
    all_session_indices_list = []
    all_trial_indices_list = []

    print(f"开始加载数据，从 {len(subject_files_map)} 个被试中...")

    for subject_id in sorted(subject_files_map.keys()):
        subject_session_files = subject_files_map[subject_id]
        
        for session_idx, session_file_path in enumerate(subject_session_files):
            try:
                mat_contents = scipy.io.loadmat(session_file_path)
            except Exception as e:
                print(f"      错误: 无法加载文件 {session_file_path}. 错误: {e}")
                continue

            for trial_idx in range(1, N_TRIALS_PER_EXPERIMENT + 1):
                feature_var_name = f"{selected_feature_prefix}{trial_idx}"
                if feature_var_name not in mat_contents:
                    continue
                
                # trial_features 形状: (n_channels, n_timesteps, n_features_per_step)
                # 例如: (62, 235, 5)
                trial_features = mat_contents[feature_var_name]
                if trial_features.ndim != 3 or trial_features.shape[0] != N_CHANNELS or trial_features.shape[2] != N_FEATURES_PER_STEP:
                    continue
                
                # 直接将整个 trial 作为一个样本
                all_trials_list.append(trial_features)
                all_labels_list.append(MAPPED_LABELS_SEQUENCE[trial_idx-1])
                all_subject_ids_list.append(subject_id)
                all_session_indices_list.append(session_idx)
                all_trial_indices_list.append(trial_idx-1) # trial索引从0开始

    if not all_trials_list:
        print("错误：未能加载任何有效特征数据。请检查DATA_PATH和文件内容。")
        return None, None, None, None, None

    # 注意：此时 features_full 是一个 list of numpy arrays, 不能直接转换成一个大的 numpy array
    # 我们将在 Dataset 中处理每个 trial 的变长问题。
    # 为了能用 np.save 保存，我们将其存为 object 类型的 array
    # 使用 np.empty 和切片赋值是创建对象数组的更稳健方法，
    # 可以避免在子数组形状不同时可能出现的广播错误。
    features_full = np.empty(len(all_trials_list), dtype=object)
    features_full[:] = all_trials_list
    labels_full = np.array(all_labels_list)
    subject_ids_full = np.array(all_subject_ids_list)
    session_ids_full = np.array(all_session_indices_list)
    trial_ids_full = np.array(all_trial_indices_list)

    return features_full, labels_full, subject_ids_full, session_ids_full, trial_ids_full


if __name__ == '__main__':
    print("开始执行数据加载和预处理脚本...")
    
    features, labels, subject_ids, session_ids, trial_ids = load_and_reshape_data(DATA_PATH)

    if features is not None:
        print("\\n数据准备完成。")
        print(f"  总样本数 (trials): {features.shape[0]}")
        print(f"  每个样本的特征是一个变长的 trial, 例如第一个 trial 的形状: {features[0].shape} (Channels, Timesteps, Features per Channel)")
        print(f"  标签数据形状: {labels.shape}")
        print(f"  标签类别分布: {collections.Counter(labels)}")

        data_params = {
            "n_channels": N_CHANNELS, 
            "n_features_per_step": N_FEATURES_PER_STEP,
            "random_seed": RANDOM_SEED,
            "label_mapping": LABEL_MAPPING,
            "n_trials_per_experiment": N_TRIALS_PER_EXPERIMENT
        }
        
        print("\\n正在保存处理好的完整数据集...")
        np.save("data_params.npy", data_params)
        np.save("features_full.npy", features)
        np.save("labels_full.npy", labels)
        np.save("subject_ids_full.npy", subject_ids)
        np.save("session_ids_full.npy", session_ids)
        np.save("trial_ids_full.npy", trial_ids)
        print("所有文件 (data_params, features_full, labels_full, subject_ids_full, session_ids_full, trial_ids_full) 保存完毕。")
            
    else:
        print("\\n数据加载失败，未保存任何文件。") 