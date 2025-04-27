import pandas as pd

def write_to_txt(df, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for index, row in df.iterrows():
            file.write(f"{index + 1}.\n")
            file.write("[结果记录]\n")
            record = row['result_record']
            # 将字符串转换为列表（假设每个记录以数字+）开始）
            if isinstance(record, str):
                items = [item.strip() for item in record.split(')') if item.strip()]
                for i, item in enumerate(items, start=1):
                    file.write(f"{i}）{item})\n") if ')' not in item else file.write(f"{item}\n")
            else:
                file.write(f"{record}\n")
            
            score = row['result_score']
            file.write(f"[符合判定] {score}\n\n")

def main():
    # 假设你的Excel文件名为'input.xlsx'，并且sheet名为'Sheet1'
    excel_file = '20241219151950_测试优化数据结果6.0.xlsx'
    sheet_name = 'report_db2_content_results_info'
    output_txt = 'output.txt'

    # 读取Excel文件
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # 写入到txt文件
    write_to_txt(df, output_txt)
    print(f"数据已成功写入到'{output_txt}'文件中。")

if __name__ == "__main__":
    main()