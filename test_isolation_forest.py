import subprocess
import json

def run_isolation_forest(input_data):
    # 将输入数据写入临时文件
    with open('input.json', 'w') as input_file:
        json.dump(input_data, input_file)

    # 调用 isolation_forest.py 脚本
    command = "python3 isolation_forest.py < input.json > output.json"
    subprocess.run(command, shell=True, check=True)

    # 读取输出结果
    with open('output.json', 'r') as output_file:
        output_data = json.load(output_file)

    return output_data

def main():
    # 示例输入数据
    input_data = {
        "1": 1.67,
        "2": 1.56,
        "3": 1.84,
        "4": 1.95,
        "5": 2000
    }

    # 运行孤立森林检测
    output_data = run_isolation_forest(input_data)

    # 打印输出结果
    print("Face ID\tPrediction\tAnomaly Score\tAvg Path Length\tPath Lengths")
    for face_id, result in output_data.items():
        prediction = result["prediction"]
        anomaly_score = result["anomaly_score"]
        avg_path_length = result["avg_path_length"]
        path_lengths = result["path_lengths"]
        print(f"{face_id}\t{prediction}\t{anomaly_score}\t{avg_path_length}\t{path_lengths}")

if __name__ == "__main__":
    main()