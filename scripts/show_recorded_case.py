"""Display saved evidence; this script never calls a model or database."""
import argparse
import json
from pathlib import Path
import sys


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--turn', choices=['M1', 'M2', 'M3', 'M4', 'M5'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    case = json.loads((root / 'evidence/case-records.json').read_text(encoding='utf-8'))
    print('保存记录展示 | E05 | 实际模型运行日期：' + case['recorded_on'])
    print('当前脚本没有调用模型。\n')
    for turn in case['turns']:
        if args.turn and turn['message_id'] != args.turn:
            continue
        print(turn['message_id'] + ' · ' + turn['title'])
        print('用户：' + turn['user'])
        print('回答：' + turn['answer'])
        print('状态：实际位置=' + turn['actual_application_location']
              + '；分析范围=' + turn['analysis_mode'])
        print('来源：' + ' | '.join(dict.fromkeys(turn['source_links'])))
        print()


if __name__ == '__main__':
    main()
