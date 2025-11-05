#coding=utf-8
import codecs as cs
import os
import sys
import argparse


def main(args):
    
    print(f"Starting get_1st_pairs.py...")
    print(f"  > Step 1 Output Dir (Reading from): {args.pred_data_dir}")
    print(f"  > Step 2 Input Dir (Writing to): {args.data_dir}")
    print(f"  > Domain: {args.domain}")

    pred_pipeline_file = os.path.join(args.pred_data_dir, 'pred4pipeline.txt')

    output_file = os.path.join(args.data_dir, f'{args.domain}_pair_1st.tsv')

    
    try:
        f = cs.open(pred_pipeline_file, 'r', encoding='utf-8').readlines()
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{pred_pipeline_file}'을 찾을 수 없습니다.")
        print("  > run_step1.py가 성공적으로 완료되었는지,")
        print(f"  > {args.pred_data_dir} 경로에 pred4pipeline.txt가 있는지 확인하세요.")
        sys.exit(1) 

    wf = cs.open(output_file, 'w', encoding='utf-8')
    
    line_count = 0
    for line in f:
        asp = []; opi = []
        line = line.strip().split('\t')
        if len(line) <= 1:
            continue
        text = line[0]
        af = 0
        of = 0
        for ele in line[1:]:
            if ele.startswith('a'):
                asp.append(ele[2:])
                af = 1
            else:
                opi.append(ele[2:])
                of = 1
        if af == 0:
            asp.append('-1,-1')
        if of == 0:
            opi.append('-1,-1')
            
        if len(asp)>0 and len(opi)>0:
            pred = []

            for pa in asp:
                if ',' in pa and pa != '-1,-1': 
                    ast, aed = int(pa.split(',')[0]), int(pa.split(',')[1])
                elif pa == '-1,-1':
                    ast, aed = -1, -1
                else:
                    continue

                for po in opi:
                    if ',' in po and po != '-1,-1':
                        ost, oed = int(po.split(',')[0]), int(po.split(',')[1])
                    elif po == '-1,-1':
                        ost, oed = -1, -1
                    else:
                        continue
                        
                    pred.append([pa, po])
            
            for ele in pred:
                wf.write(text + '####' + ele[0] + ' ' + ele[1] + '\t' + '' + '\n')
                line_count += 1


    wf.close()
    print(f"File processing complete. Total {line_count} pairs generated.")



if __name__ == "__main__":
    print("Running get_1st_pairs.py directly (for testing/debugging)...")
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_data_dir', type=str, required=True, help="Directory containing pred4pipeline.txt from Step 1")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory to write output file for Step 2")
    parser.add_argument('--domain', type=str, required=True, help="Domain name (e.g., 'predict')")
    
    args = parser.parse_args()
    
    main(args)