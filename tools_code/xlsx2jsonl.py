# -*- coding:utf-8 -*-  

import openpyxl
import json

def process_excel_to_json(input_file, output_file):
    # Load the workbook
    wb = openpyxl.load_workbook(input_file)

    # Select the "DrugQA" sheet
    sheet = wb["all_data"]

    # Initialize the output data structure
    output_data = []

    # Iterate through each row in column A and D
    for row in sheet.iter_rows(min_row=2, max_col=2, values_only=True):
        system_value = "询问一下一些党史的知识"

        # Create the conversation dictionary
        conversation = {
            "system": system_value,
            "input": row[0],
            "output": row[1]
        }

        # Append the conversation to the output data
        output_data.append({"conversation": [conversation]})

    # Write the output data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, indent=4,ensure_ascii=False)

    print(f"Conversion complete. Output written to {output_file}")

process_excel_to_json('all_data.xlsx', 'output.jsonl')

