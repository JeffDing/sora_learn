import json

# 假设我们有一个名为data.jsonl的JSONL文件，其内容如下：
# {"name": "Alice", "age": 25}
# {"name": "Bob", "age": 30, "city": "New York"}

# 打开JSONL文件并逐行读取
with open('duihua.jsonl', 'r') as f:
    for line in f:
        # 将每一行加载为字典
        data = json.loads(line)

        # 删除"city"键及其对应的值
        if 'id' in data:
            del data['id']
            
        if 'doc' in data:
            del data['doc']

        # 将修改后的数据写入新的JSONL文件
        with open('new_data.jsonl', 'a') as new_file:
            json.dump(data, new_file,ensure_ascii=False)
            new_file.write('\n')