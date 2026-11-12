"""Loopback-only HTTP client for the persistent learning service; no retries."""
import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

from dotenv import dotenv_values
import httpx


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('create')
    history = commands.add_parser('history')
    history.add_argument('conversation_id')
    read = commands.add_parser('read')
    read.add_argument('conversation_id')
    read.add_argument('message_id')
    send = commands.add_parser('send')
    send.add_argument('conversation_id')
    send.add_argument('--message-id', required=True)
    content = send.add_mutually_exclusive_group(required=True)
    content.add_argument('--text')
    content.add_argument('--file', type=Path, help='UTF-8 message file; indentation is preserved')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    project = Path(__file__).resolve().parents[1]
    config = {**dotenv_values(project / '.env'), **os.environ}
    token = config.get('RAG_CONVERSATION_TOKEN', '')
    if not token:
        parser.error('Set RAG_CONVERSATION_TOKEN in the local .env')
    path = '/conversations'
    if args.command != 'create':
        path += '/' + quote(args.conversation_id, safe='') + '/messages'
    if args.command == 'read':
        path += '/' + quote(args.message_id, safe='')
    payload = None
    if args.command == 'send':
        text = args.file.read_text(encoding='utf-8-sig') if args.file is not None else args.text
        payload = {'message_id': args.message_id, 'content': text}
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}', trust_env=False,
                          timeout=300, headers={'Authorization': 'Bearer ' + token}) as client:
            response = (client.post(path, json=payload) if args.command in ('create', 'send') else client.get(path))
        print(json.dumps({'http_status': response.status_code, 'body': response.json()}, ensure_ascii=False, indent=2))
        return int(response.is_error)
    except httpx.RequestError as error:
        print(json.dumps({'status': 'client_error', 'error_type': type(error).__name__,
            'message': '未收到服务响应；保留原message_id，先用read或history查询，勿自动更换标识重发。'}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
