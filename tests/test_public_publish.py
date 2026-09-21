import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec=importlib.util.spec_from_file_location('publish_public',Path(__file__).parents[1]/'packaging/publish_public.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def make_packages(path,product='char-detection-main'):
    path.mkdir()
    for target in module.TARGETS:
        with zipfile.ZipFile(path/f'-v1.2.0-{target}.zip','w') as archive:
            archive.writestr('app/build_info.json',json.dumps(dict(product=product,version='v1.2.0',target=target)))


def test_only_checked_installers_and_distinct_product_names(tmp_path):
    source=tmp_path/'input';make_packages(source)
    (source/'secret.txt').write_text('not a release file')
    stage=tmp_path/'stage'
    manifest=module.stage_assets(source,stage,'char-detection-main','v1.2.0','a'*40)
    assert len(list(stage.iterdir()))==5
    assert len(manifest['assets'])==3
    assert all(a['name'].startswith('char-detection-main-v1.2.0-') for a in manifest['assets'])
    assert not (stage/'secret.txt').exists()
    with pytest.raises(module.PublishError,match='identity'):
        module.stage_assets(source,tmp_path/'bad','group-by-character','v1.2.0','a'*40)


def test_incomplete_assets_not_published(tmp_path):
    source=tmp_path/'input';make_packages(source)
    next(source.iterdir()).unlink()
    with pytest.raises(module.PublishError,match='three'):
        module.stage_assets(source,tmp_path/'stage','char-detection-main','v1.2.0','a'*40)


class Client:
    def __init__(self):self.data=None;self.files={};self.calls=[]
    def api(self,path):return {'private':False,'default_branch':'main'}
    def release(self,tag):return self.data
    def assets(self,release_id):return list(self.files.values())
    def call(self,*args,**kwargs):
        self.calls.append(args)
        if args[:2]==('release','create'):self.data={'id':1,'draft':True}
        if args[:2]==('release','upload'):
            file=Path(args[3])
            self.files[file.name]={'name':file.name,'size':file.stat().st_size,'digest':'sha256:'+module.digest(file)}
        if args[:2]==('release','edit'):self.data['draft']=False


def test_publish_complete_draft_and_safe_rerun(tmp_path):
    source=tmp_path/'input';make_packages(source)
    stage=tmp_path/'stage'
    manifest=module.stage_assets(source,stage,'char-detection-main','v1.2.0','a'*40)
    client=Client()
    url=module.publish(client,stage,manifest)
    assert url.endswith('/char-detection-main-v1.2.0')
    assert len(client.files)==5 and not client.data['draft']
    count=len(client.calls)
    module.publish(client,stage,manifest)
    assert len(client.calls)==count
    first=next(iter(client.files.values()));first['digest']='sha256:'+'f'*64
    with pytest.raises(module.PublishError,match='differs'):module.publish(client,stage,manifest)


def test_mac_metadata_symlink_is_not_parsed_as_json(tmp_path):
    import stat
    source=tmp_path/'input';make_packages(source)
    for file in source.glob('*macos*.zip'):
        with zipfile.ZipFile(file,'a') as archive:
            info=zipfile.ZipInfo('App.app/Contents/Frameworks/build_info.json')
            info.create_system=3
            info.external_attr=(stat.S_IFLNK|0o777)<<16
            archive.writestr(info,'../Resources/build_info.json')
    manifest=module.stage_assets(source,tmp_path/'stage','char-detection-main','v1.2.0','a'*40)
    assert len(manifest['assets'])==3


def test_tag_lookup_reuses_unpublished_draft(monkeypatch):
    client=module.GitHub()
    calls=[]
    def api(path,missing_ok=False):
        calls.append(path)
        if '/tags/' in path:return None
        return [{'id':42,'tag_name':'char-detection-main-v1.2.0','draft':True}]
    monkeypatch.setattr(client,'api',api)
    assert client.release('char-detection-main-v1.2.0')['id']==42
    assert len(calls)==2
