# -*- coding: utf-8 -*-

import csv  
import json  
  
# Step 1: Read the CSV file  
with open('data.csv', 'r', encoding='utf-8') as csv_file:  
    reader = csv.DictReader(csv_file)  
    data = [row for row in reader]  
  
# Step 2: Extract question and answer columns  
questions = [row['question'] for row in data]  
answers = [row['answer'] for row in data]  
  
# Step 3: Create the JSONL structure  
conversations = []  
for question, answer in zip(questions, answers):  
    conversation = {  
        "conversation": [  
            {  
                "system": "你是一个专业的中医医师，现在请你给患者开处方' questions.",  
                "input": question,  
                "output": answer  
            }  
        ]  
    }  
    conversations.append(conversation)  
  
# Step 4: Write the JSONL file  
with open('yiyaoduihua.jsonl', 'w', encoding='utf-8') as jsonl_file:  
    for conversation in conversations:  
        json.dump(conversation, jsonl_file, ensure_ascii=False)  
        jsonl_file.write('\n')