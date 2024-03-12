import json
import csv

# 假设我们有一个名为data.jsonl的JSONL文件，其内容如下：
# {"name": "Alice", "age": 25}
# {"name": "Bob", "age": 30, "city": "New York"}

# 打开JSONL文件并逐行读取
with open('duihua.jsonl', 'r') as f:
    data = []
    for line in f:
        # 将每一行加载为字典
        item = json.loads(line)
        
        # 将字典添加到列表中
        data.append(item)

# 打开CSV文件进行写入
with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
    
    # 首先，写入CSV的标题（即JSON对象的键）
    writer.writeheader()
    
    # 然后，将每个JSON对象作为一行写入CSV
    for item in data:
        writer.writerow(item)