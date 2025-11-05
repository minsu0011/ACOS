import random

original_file = 'Appliances.jsonl'  # 원본 파일 경로
new_file = 'Appliances_trimmed.jsonl'    # 저장할 파일 경로
sample_ratio = 0.05                   # 25% 샘플링

# 파일을 한 줄씩 읽어서 처리하므로 메모리 문제가 없습니다.
try:
    with open(original_file, 'r', encoding='utf-8') as f_in, \
         open(new_file, 'w', encoding='utf-8') as f_out:
        
        processed_lines = 0
        saved_lines = 0
        
        for line in f_in:
            processed_lines += 1
            
            # 25% 확률로 현재 줄을 선택합니다.
            if random.random() < sample_ratio:
                f_out.write(line)
                saved_lines += 1
            
            # 진행 상황 표시 (선택 사항)
            if processed_lines % 100000 == 0:
                print(f"Processed {processed_lines} lines...")

    print(f"--- 작업 완료 ---")
    print(f"총 {processed_lines} 줄 중에서 {saved_lines} 줄을 저장했습니다.")
    print(f"새 파일: {new_file}")

except FileNotFoundError:
    print(f"오류: '{original_file}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"오류 발생: {e}")