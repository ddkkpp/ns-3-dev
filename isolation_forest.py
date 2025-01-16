import sys
import json
from sklearn.ensemble import IsolationForest
import numpy as np

def path_length(tree, X):
    """
    计算样本在单棵树中的路径长度
    """
    n_samples = X.shape[0]
    path_lengths = np.zeros(n_samples)
    
    for i in range(n_samples):
        node = 0
        depth = 0
        while node != -1:
            if tree.children_left[node] == -1 and tree.children_right[node] == -1:
                break
            if X[i, tree.feature[node]] <= tree.threshold[node]:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]
            depth += 1
        path_lengths[i] = depth + _average_path_length(tree.n_node_samples[node])
    
    return path_lengths

def _average_path_length(n):
    """
    计算孤立树中节点的平均路径长度
    """
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

def main():
    # 从标准输入读取数据
    input_data = json.load(sys.stdin)
    
    # 提取特征
    features = []
    face_ids = []
    for face_id, value in input_data.items():
        face_ids.append(face_id)
        features.append([value])
    
    # 进行孤立森林检测
    clf = IsolationForest(
        n_estimators=100,        # 森林中树的数量
        max_samples='auto',      # 用于训练每棵树的样本数量
        contamination='auto',       # 数据集中异常样本的比例
        max_features=1.0,        # 用于训练每棵树的特征数量
        bootstrap=False,         # 是否在构建树时使用自举样本
        n_jobs=-1,               # 用于并行计算的作业数量
        random_state=42          # 控制随机数生成的种子
    )
    clf.fit(features)
    predictions = clf.predict(features)
    anomaly_scores = clf.decision_function(features)

    # 计算每个样本在每棵树中的路径长度
    path_lengths = np.zeros((len(features), clf.n_estimators))
    for i, tree in enumerate(clf.estimators_):
        path_lengths[:, i] = path_length(tree.tree_, np.array(features))
    
    # 计算每个样本的平均路径长度
    avg_path_lengths = np.mean(path_lengths, axis=1)
    
    # 输出检测结果
    result = {
        face_ids[i]: {
            "prediction": int(predictions[i]),
            "anomaly_score": float(anomaly_scores[i]),
            "path_lengths": path_lengths[i].tolist(),
            "avg_path_length": float(avg_path_lengths[i])
        } for i in range(len(face_ids))
    }
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()