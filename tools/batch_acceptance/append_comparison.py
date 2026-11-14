"""Append independent traced method pairs without rewriting a frozen batch."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'src'))

from PIL import Image
import torch

from tools.batch_acceptance import run_comparison as base
from gpt_animeimage_denoise import fusion_pipeline as pipeline
from gpt_animeimage_denoise.fusion import _policy
from gpt_animeimage_denoise.fusion_models import FusionModels
from gpt_animeimage_denoise.pipeline import _json,_read_source,_sha


def snapshot_inputs(paths,extension_root,first_index):
    """Keep original bytes in durable storage; never overwrite another image."""
    folder=Path(extension_root).resolve()/'inputs';folder.mkdir(parents=True,exist_ok=True)
    items=[]
    for index,source in enumerate(map(Path,paths),first_index):
        source=source.resolve();digest=_sha(source);case_id=f'batch_{index:04d}'
        target=folder/(case_id+source.suffix.lower())
        if target.exists():
            if _sha(target)!=digest:raise ValueError('Existing attachment snapshot differs: '+str(target))
        else:shutil.copyfile(source,target)
        if _sha(target)!=digest or _sha(source)!=digest:raise ValueError('Attachment copy did not preserve bytes')
        rgb,alpha,icc=_read_source(target)
        with Image.open(target) as image:
            items.append({'id':case_id,'path':str(target),'original_temp_path':str(source),
                'sha256':digest,'title':source.stem,'width':rgb.shape[1],'height':rgb.shape[0],
                'stored_size':list(image.size),'mode':image.mode,'exif_orientation':image.getexif().get(274,1),
                'alpha_present':alpha is not None,'icc_sha256':hashlib.sha256(icc).hexdigest() if icc else None})
    return items


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--name',default='additional');parser.add_argument('--inputs',nargs='*',type=Path,default=[])
    args=parser.parse_args();parent=args.parent.resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.name):raise ValueError('Unsafe extension name')
    root=parent/'extensions'/args.name;root.mkdir(parents=True,exist_ok=True)
    parent_config_path=parent/'log/run-config.json';parent_manifest=parent/'input-manifest.json'
    old=json.loads(parent_config_path.read_text(encoding='utf-8'))
    old_items=json.loads(parent_manifest.read_text(encoding='utf-8'))['items']
    parent_files=[parent_manifest,parent_config_path,parent/'run-manifest.json',parent/'validation.json',parent/'log/events.jsonl']
    parent_snapshots={str(p):_sha(p) for p in parent_files}
    manifest_path=root/'input-manifest.json'
    if manifest_path.exists():
        document=json.loads(manifest_path.read_text(encoding='utf-8'));items=document['items']
        if document['parent_manifest_sha256']!=_sha(parent_manifest):raise ValueError('Parent input identity changed')
        if args.inputs:
            if [_sha(p) for p in args.inputs]!=[i['sha256'] for i in items]:raise ValueError('Attachment identities changed')
    else:
        if not args.inputs:raise ValueError('New extension needs attachment paths')
        # The viewer may already contain earlier append runs. Never reuse IDs.
        viewer=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
        numbers=[int(c['id'].removeprefix('batch_')) for c in viewer['cases'] if re.fullmatch(r'batch_\d+',c['id'])]
        items=snapshot_inputs(args.inputs,root,max(numbers,default=0)+1)
        document={'schema_version':1,'source':'Durable copies of user-attached local images',
                  'parent_manifest_sha256':_sha(parent_manifest),'items':items}
        _json(manifest_path,document)
    for item in items:
        if _sha(item['path'])!=item['sha256']:raise ValueError('Durable attachment changed')
    code={name:_sha(PROJECT/name) for name in base.CODE_FILES}
    if code!=old['code_hashes'] or _policy()!=old['policy']:raise ValueError('Producer code/policy differs from original batch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=old['runtime']['tf32']['matmul']
    torch.backends.cudnn.allow_tf32=old['runtime']['tf32']['cudnn']
    torch.cuda.reset_peak_memory_stats();models=FusionModels(device='cuda')
    if models.signature!=old['model_signature'] or models.identities!=old['model_hashes']:
        raise ValueError('Segmentation model identities differ from original batch')
    log=root/'log';log.mkdir(exist_ok=True);config_path=log/'run-config.json'
    config={**old,'run_id':parent.name+'--'+args.name,'created_at_utc':base.now(),
        'input_manifest_sha256':_sha(manifest_path),'parent_run_id':old['run_id'],
        'parent_run_config_sha256':_sha(parent_config_path),'append_driver_sha256':_sha(__file__)}
    if config_path.exists():
        previous=json.loads(config_path.read_text(encoding='utf-8'))
        for key in ('input_manifest_sha256','parent_run_config_sha256','append_driver_sha256','parameters','policy','model_signature','code_hashes'):
            if previous[key]!=config[key]:raise ValueError('Extension run identity changed: '+key)
        config=previous
    else:_json(config_path,config)
    parameters=config['parameters'];native=root/'native';records=[];completed=[];errors=[];context={}
    finish={key:parameters.get(key,1.) for key in ('structure_protection','restoration_gain')}
    def event(kind,**values):
        with (log/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'time':base.now(),'event':kind,**values},ensure_ascii=False)+'\n')
    try:
        with base.capture_stages(root,context):
            for item in items:
                pair={};entry={'id':item['id'],'methods':{}}
                for method in base.METHODS:
                    context.update(id=item['id'],method=method,source_sha256=item['sha256'])
                    print('START_APPEND',item['id'],method,flush=True);event('start',id=item['id'],method=method)
                    start=time.perf_counter()
                    try:
                        record=pipeline.process_image(item['path'],native,models,profile='fusion',
                            case_id=item['id']+'--'+method,denoiser=method,lam=parameters['lambda'],
                            downscale=parameters['downscale'],sigma=parameters['sigma_at_2048'],strength=parameters['strength'],
                            **finish,
                            _input_paths=[manifest_path,config_path,parent_config_path,*[i['path'] for i in items]])
                        pair[method]=record;records.append(record)
                        entry['methods'][method]={'status':record['status'],'fingerprint':record['fingerprint'],
                            'report':str((native/record['folder']/'report.json').relative_to(root))}
                        event('complete',id=item['id'],method=method,seconds=time.perf_counter()-start,
                              cache_hits=record['cache_hits'],fingerprint=record['fingerprint'],resumed=record['resumed'])
                    except Exception as error:
                        problem={'id':item['id'],'method':method,'error':type(error).__name__+': '+str(error)}
                        errors.append(problem);entry['methods'][method]={'status':'error','error':problem['error']}
                        event('error',**problem);print('ERROR_APPEND',problem['error'],flush=True)
                    pipeline.write_records(native,records,input_paths=[manifest_path,config_path,parent_config_path,*[i['path'] for i in items]])
                if len(pair)==2:entry['checks']=base.verify_pair(item['path'],native,pair)
                completed.append(entry)
                _json(root/'run-manifest.json',{'schema_version':1,'run_id':config['run_id'],'cases':completed,'errors':errors})
                print('APPEND_PAIR_DONE',item['id'],flush=True)
        if errors:raise RuntimeError('Extension has method failures; inspect logs')
        for item in items:
            if _sha(item['path'])!=item['sha256']:raise RuntimeError('Durable input changed')
        if any(_sha(p)!=digest for p,digest in parent_snapshots.items()):raise RuntimeError('Original producer files changed')
        if any(_sha(PROJECT/name)!=digest for name,digest in code.items()) or _sha(__file__)!=config['append_driver_sha256']:
            raise RuntimeError('Producer code changed during append')
        first=items[0]
        resumed=pipeline.process_image(first['path'],native,models,profile='fusion',case_id=first['id']+'--l0',
            denoiser='l0',lam=parameters['lambda'],downscale=parameters['downscale'],
            sigma=parameters['sigma_at_2048'],strength=parameters['strength'],**finish)
        if not resumed['resumed']:raise RuntimeError('Extension resume failed')
        _json(root/'validation.json',{'schema_version':1,'case_count':len(items),'method_runs':len(records),
            'cases':completed,'source_unchanged':True,'same_masks_and_weights':True,'resume_passed':True,
            'parent_run_files_unchanged':True,'parent_file_snapshots':parent_snapshots,
            'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'errors':errors,
            'quality_accepted_by_user':False,'finished_at_utc':base.now()})
        event('finished',cases=len(items),method_runs=len(records));print('ADDED_PAIRS_AND_PARENT_PRESERVATION_VERIFIED',flush=True)
    finally:models.release()


if __name__=='__main__':main()
