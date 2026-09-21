"""Print release metadata from a ZIP; optionally publish to the local auth service."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import urllib.request
from hanzi_auth.updates import PRODUCTS, TARGETS, parsed_version, validate_release
from hanzi_auth.policy import server_url, uses_encrypted_channel


def main():
    parser = argparse.ArgumentParser(description='生成更新配置（不会上传安装包）')
    parser.add_argument('archive', type=Path)
    parser.add_argument('--product', required=True, choices=sorted(PRODUCTS))
    parser.add_argument('--target', required=True, choices=sorted(TARGETS))
    parser.add_argument('--version', required=True)
    parser.add_argument('--url', help='GitCode 公开发行版附件下载链接')
    parser.add_argument('--notes', default='')
    parser.add_argument('--server', help='可选：将配置写入该账号服务，例如 http://127.0.0.1:8765')
    parser.add_argument('--admin', default='admin')
    args = parser.parse_args()
    parsed_version(args.version)
    digest = hashlib.sha256()
    with args.archive.open('rb') as source:
        while chunk := source.read(1024*1024):
            digest.update(chunk)
    data = {'product':args.product, 'target':args.target, 'version':args.version,
            'size':args.archive.stat().st_size, 'sha256':digest.hexdigest(), 'notes':args.notes}
    if args.url:
        data['url'] = args.url
        validate_release(data)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    if args.server:
        if not args.url:
            parser.error('--server 需要同时填写 --url')
        url = server_url(args.server)
        if uses_encrypted_channel(url):
            parser.error('管理操作请连接 SSH 隧道的本机地址。')
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        opener = urllib.request.build_opener(NoRedirect())
        def request(path, body, token='', method='POST'):
            headers={'Content-Type':'application/json'}
            if token:
                headers['Authorization']='Bearer '+token
            req=urllib.request.Request(url+path,data=json.dumps(body).encode(),headers=headers,method=method)
            with opener.open(req,timeout=20) as response:
                return json.load(response)
        session=request('/api/login',{'username':args.admin,'password':getpass.getpass('管理员密码：'),'product':'admin'})
        try:
            body={k:v for k,v in data.items() if k not in {'product','target'}} | {'enabled':True}
            request(f'/api/admin/updates/{args.product}/{args.target}',body,session['token'],'PUT')
            print('更新配置已保存。')
        finally:
            request('/api/logout',{},session['token'])


if __name__ == '__main__':
    main()
