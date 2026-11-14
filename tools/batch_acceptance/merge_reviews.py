"""Aggregate separately verified acceptance cohorts without rewriting producers.

Only viewer metadata and aggregate records are published in the parent. Native
outputs, original inputs, stage traces and each producer configuration stay put.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import sys

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
if str(PROJECT/'src') not in sys.path:sys.path.insert(0,str(PROJECT/'src'))

from gpt_animeimage_denoise.filters import validate_unit_interval
from gpt_animeimage_denoise.pipeline import _assert_inputs_safe,_sha

METHODS=('l0','esrgan-l0')
KIND='fixed-mask-denoise-batch-acceptance'
METHOD_PATHS=('final','candidate','restored','report','changed','sr')
SHARED_PATHS=('colored','labels','source_map','source_colored','weight','weight_gray','weight_float',
    'allowed','active','zero_weight','protected','objects','unknown','conflict','white','input_original',
    'alpha','pre_sam_colored','post_sam_colored')
CHECKS=('same_masks','same_weight','source_unchanged','protected_unchanged','float_replay_verified')


def _json_bytes(value):return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
def _js_bytes(value):return b'window.BATCH_ACCEPTANCE_DATA = '+_json_bytes(value)+b';\n'


def _read(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value,dict):raise ValueError('Metadata must be a JSON object: '+str(path))
    return value


def _script_document(payload):
    match=re.fullmatch(r'\s*window\.BATCH_ACCEPTANCE_DATA\s*=\s*(\{.*\})\s*;\s*',payload.decode('utf-8'),re.DOTALL)
    if not match:raise ValueError('Viewer script is not the exported JSON assignment')
    return json.loads(match.group(1))


def _inside(root,value,*,must_exist=True):
    if (not isinstance(value,str) or not value or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',value)
            or value.startswith(('/','\\')) or PureWindowsPath(value).drive or '?' in value or '#' in value):
        raise ValueError('Expected a plain local relative resource path: '+str(value))
    candidate=(root/value.replace('\\','/')).resolve()
    if not candidate.is_relative_to(root.resolve()) or (must_exist and not candidate.is_file()):
        raise ValueError('Resource is missing or escapes its producing cohort: '+value)
    return candidate


def _path_slots(case):
    """Yield only display paths, never model paths in parameter provenance."""
    for key in ('original','thumbnail'):
        if case.get(key) is not None:yield case,key
    for value in case.get('methods',{}).values():
        for key in METHOD_PATHS:
            if value.get(key) is not None:yield value,key
    shared=case.get('shared',{})
    for key in SHARED_PATHS:
        if shared.get(key) is not None:yield shared,key
    for key in shared.get('groups',{}):yield shared['groups'],key
    for view in shared.get('evidence',{}).values():
        for model in view.values():
            if not isinstance(model,dict):continue
            for key in ('groups_colored','colored','objects','labels'):
                if model.get(key) is not None:yield model,key
            for key in model.get('groups',{}):yield model['groups'],key
    for key,value in case.get('differences',{}).items():
        if isinstance(value,str):yield case['differences'],key
        elif isinstance(value,dict) and value.get('path') is not None:yield value,'path'
        else:raise ValueError('Unsupported difference resource')
    for stage in case.get('trace',[]):
        for link in stage.get('links',[]):
            if link.get('path') is not None:yield link,'path'


def _check_passed(value):
    return value is True or (isinstance(value,dict) and value.get('passed') is True)


def _validate_parameters(parameters):
    if not isinstance(parameters,dict) or parameters.get('profile')!='fusion':
        raise ValueError('Aggregation requires the fusion comparison profile')
    if parameters.get('methods')!=list(METHODS):raise ValueError('The same two candidate methods are required')
    for key in ('lambda','downscale','sigma_at_2048','strength'):
        value=parameters.get(key)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
            raise ValueError('Invalid finite comparison parameter: '+key)
        if (key=='strength' and not 0<=value<=1) or (key!='strength' and value<=0):
            raise ValueError('Invalid positive comparison parameter: '+key)
    if parameters['downscale']>1:raise ValueError('Downscale exceeds1')
    for key in ('structure_protection','restoration_gain'):
        validate_unit_interval(parameters.get(key,1.),key)


def _fingerprints(case):
    result={}
    if set(case.get('methods',{}))!=set(METHODS):raise ValueError('A case lacks its comparison pair')
    for method in METHODS:
        value=case['methods'][method].get('report_fingerprint')
        if not isinstance(value,str) or not re.fullmatch(r'[a-fA-F0-9]{64}',value):
            raise ValueError('A full native report fingerprint is required')
        result[method]=value
    return result


def _previous_review(document):
    cases=[{'id':c['id'],'source_sha256':c['source_sha256'],'report_fingerprints':_fingerprints(c)}
           for c in document['cases']]
    return {'run_id':document['run_id'],
        'storage_key':'batch-acceptance-v1:'+document['run_id']+':'+':'.join(c['source_sha256'] for c in cases),
        'cases':cases}


def _verify_component(root,document,manifest_path):
    if (document.get('kind')!=KIND or not document.get('run_id')
            or not isinstance(document.get('cases'),list) or not document['cases']):
        raise ValueError('Only completed acceptance viewer cohorts may be combined')
    cases=document['cases'];ids=[c.get('id') for c in cases]
    if any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('Duplicate or invalid case IDs in a cohort')
    _validate_parameters(document.get('parameters'))
    paths={name:root/name for name in ('input-manifest.json','log/run-config.json','validation.json',
                                      'run-manifest.json','export-validation.json')}
    source=_read(paths['input-manifest.json']);config=_read(paths['log/run-config.json'])
    validation=_read(paths['validation.json']);export=_read(paths['export-validation.json'])
    run_record=_read(paths['run-manifest.json'])
    viewer_run_id=source.get('run_id',root.name)
    if document['run_id']!=viewer_run_id:
        raise ValueError('Viewer identity differs from its frozen exporter input/folder identity')
    if not config.get('run_id') or run_record.get('run_id')!=config['run_id']:
        raise ValueError('Frozen producer configuration and run record identities differ')
    if config.get('input_manifest_sha256')!=_sha(paths['input-manifest.json']):
        raise ValueError('Frozen producer input manifest hash differs')
    if config.get('parameters')!=document['parameters']:raise ValueError('Viewer parameters differ from producer')
    if not config.get('code_hashes') or not config.get('model_hashes') or not config.get('model_signature'):
        raise ValueError('Frozen code and model identities are required')
    items=source.get('items',[])
    if [x.get('id') for x in items]!=ids:raise ValueError('Viewer cohort differs from its frozen input order')
    if (validation.get('case_count')!=len(cases) or validation.get('method_runs')!=len(cases)*2
            or export.get('cases')!=len(cases)):
        raise ValueError('Verified producer/export case counts differ')
    if any(export.get(key) is not True for key in ('same_masks_and_weight','exact_float_replay','source_preserved')):
        raise ValueError('The component export has not passed numerical and source checks')
    verified={c['id']:c for c in validation.get('cases',[])}
    if len(verified)!=len(cases) or set(verified)!=set(ids):raise ValueError('Incomplete producer validation cases')
    inputs=[manifest_path,*paths.values()];resources=[];reports=[]
    for case,item in zip(cases,items):
        if (not isinstance(case.get('source_sha256'),str) or case['source_sha256']!=item.get('sha256')
                or not re.fullmatch(r'[a-fA-F0-9]{64}',case['source_sha256'])):
            raise ValueError('Case source identity differs from frozen input')
        original=Path(item['path']).resolve()
        if not original.is_file() or _sha(original)!=item['sha256']:raise ValueError('A persistent original source changed')
        inputs.append(original)
        for key in CHECKS:
            if not _check_passed(case.get('checks',{}).get(key)):
                raise ValueError('Viewer case has not passed '+key)
        fingerprints=_fingerprints(case)
        for holder,key in _path_slots(case):resources.append(_inside(root,holder[key]))
        for method in METHODS:
            record=verified[case['id']].get('methods',{}).get(method,{})
            expected=root/'native'/(case['id']+'--'+method)/'report.json'
            native_report=_inside(root,record.get('report'))
            if native_report!=expected.resolve():raise ValueError('Validation refers to another native case')
            report=_read(native_report);params=report.get('parameters',{})
            if (record.get('status')!='complete' or report.get('status')!='complete'
                    or report.get('profile')!='fusion' or report.get('source_sha256')!=item['sha256']
                    or report.get('fingerprint')!=fingerprints[method] or record.get('fingerprint')!=fingerprints[method]
                    or report.get('size')!=[case['width'],case['height']] or params.get('denoiser')!=method
                    or params.get('models')!=config['model_signature'] or params.get('policy')!=config.get('policy')):
                raise ValueError('Native report/source/policy/fingerprint differs from review')
            for key in ('profile','lambda','sigma_at_2048','strength'):
                if params.get(key)!=config['parameters'][key]:raise ValueError('Native parameter differs: '+key)
            for key in ('structure_protection','restoration_gain'):
                value=validate_unit_interval(params.get(key,1.),key)
                if value!=config['parameters'].get(key,1.):raise ValueError('Native parameter differs: '+key)
            scale=params.get('denoiser_config',{}).get('downscale',params.get('downscale'))
            if method=='esrgan-l0' and scale!=config['parameters']['downscale']:
                raise ValueError('Native ESRGAN scale differs')
            hashes=report.get('output_hashes')
            if not isinstance(hashes,dict) or not hashes:raise ValueError('Native output hashes are required')
            for name,digest in hashes.items():
                artifact=_inside(native_report.parent,name)
                if _sha(artifact)!=digest:raise ValueError('Native artifact changed after verification: '+name)
                inputs.append(artifact)
            copied_report=_inside(root,case['methods'][method]['report'])
            final=_inside(root,case['methods'][method]['final'])
            if _sha(copied_report)!=_sha(native_report) or _sha(final)!=hashes.get('result.png'):
                raise ValueError('Displayed final/report differs from its native producer')
            inputs.append(native_report);reports.append({'id':case['id'],'method':method,
                'report':str(native_report.relative_to(root)).replace('\\','/'),
                'sha256':_sha(native_report),'fingerprint':fingerprints[method]})
    inputs += resources
    return {'config':config,'config_path':paths['log/run-config.json'],
        'producer_run_id':config['run_id'],'viewer_run_id':document['run_id'],
        'inputs':list(dict.fromkeys(Path(p).resolve() for p in inputs)),
        'metadata_sha256':{name:_sha(path) for name,path in paths.items()},
        'review_manifest_sha256':_sha(manifest_path),'reports':reports,
        'case_ids':ids,'parameters':document['parameters'],'resources':list(dict.fromkeys(resources))}


def _rebase_case(case,extension,parent):
    result=deepcopy(case)
    for holder,key in _path_slots(result):
        holder[key]=_inside(extension,holder[key]).relative_to(parent).as_posix()
    return result


def _atomic(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(payload);temporary.replace(path)


def merge_reviews(parent_root,extension_root):
    parent=Path(parent_root).resolve();extension=Path(extension_root).resolve()
    if extension==parent or not extension.is_relative_to(parent):
        raise ValueError('The extension must be a separate directory within the parent review')
    parent_manifest=parent/'manifest.json';parent_script=parent/'manifest.js'
    backup=parent/'manifest.original.json';backup_script=parent/'manifest.original.js'
    legacy_backup=parent/'manifest.original-6.json';legacy_script=parent/'manifest.original-6.js'
    legacy=legacy_backup.exists() or legacy_script.exists()
    if legacy:
        if backup.exists() or backup_script.exists():raise ValueError('Ambiguous original review backups')
        backup,backup_script=legacy_backup,legacy_script
    current_bytes=parent_manifest.read_bytes();current_script_bytes=parent_script.read_bytes()
    current_sha=hashlib.sha256(current_bytes).hexdigest();current_script_sha=hashlib.sha256(current_script_bytes).hexdigest()
    initial_bytes=backup.read_bytes() if backup.exists() else current_bytes
    base=json.loads(initial_bytes.decode('utf-8'));current=json.loads(current_bytes.decode('utf-8'))
    if _script_document(current_script_bytes)!=current:raise ValueError('Viewer JSON and script cohorts differ')
    if backup_script.exists() and _script_document(backup_script.read_bytes())!=base:
        raise ValueError('Original JSON and script backups differ')
    extra=_read(extension/'manifest.json')
    base_count=len(base.get('cases',[]))
    if not base_count:raise ValueError('A nonempty original review snapshot is required')
    if current.get('cases',[])[:base_count]!=base['cases']:
        raise ValueError('Original review cases changed; refusing to overwrite their identity or opinions')
    if len(current.get('cases',[]))==base_count and current!=base:
        raise ValueError('Original viewer metadata changed after its snapshot')
    a=_verify_component(parent,base,backup if backup.exists() else parent_manifest)
    b=_verify_component(extension,extra,extension/'manifest.json')
    if base.get('modules')!=extra.get('modules'):raise ValueError('Module definitions differ')
    for key in ('parameters','policy','model_signature','model_hashes','code_hashes'):
        if a['config'].get(key)!=b['config'].get(key):raise ValueError('Producer configurations differ: '+key)
    parent_sha=_sha(a['config_path']);child=b['config']
    if (child.get('parent_run_config_sha256') is not None and child['parent_run_config_sha256']!=parent_sha
            or child.get('parent_run_id') is not None and child['parent_run_id']!=a['producer_run_id']):
        raise ValueError('Extension is bound to another parent producer')
    if set(a['case_ids'])&set(b['case_ids']):raise ValueError('Duplicate case IDs across producer cohorts')
    rebased=[_rebase_case(c,extension,parent) for c in extra['cases']]
    merged=deepcopy(base);merged['cases']+=rebased
    merged['summary']={**base.get('summary',{}),'cases':len(merged['cases']),
        'method_runs':len(merged['cases'])*2,'initial_cases':len(base['cases']),'added_cases':len(extra['cases']),
        'same_selection_for_both_methods':True,'engineering_checks_passed':True,'user_quality_accepted':False}
    previous=_previous_review(base)
    merged['previous_reviews']=[previous,*[p for p in base.get('previous_reviews',[])
        if p.get('storage_key')!=previous['storage_key']]]
    extension_prefix=extension.relative_to(parent).as_posix()
    merged['aggregation']={'initial_cases':len(base['cases']),'added_cases':len(extra['cases']),
        'manifest':'aggregate-manifest.json','validation':'aggregate-validation.json',
        'original_review':backup.name,'extension_prefix':extension_prefix,
        'extension_manifest':extension_prefix+'/manifest.json','extension_validation':extension_prefix+'/validation.json',
        'note':('初始6图与追加图由独立冻结的运行处理；原始validation保留各自范围，汇总核验请看aggregate-validation。'
                if legacy else '原批次和追加批次分别保留运行与核验记录，全部图片的核验见 aggregate-validation。')}
    expected_current=deepcopy(merged)
    if len(current['cases'])!=len(base['cases']) and current!=expected_current:
        raise ValueError('Existing aggregate is not this exact extension; keep it immutable')
    descriptor=[]
    for prefix,document,verification in [('',base,a),(extension.relative_to(parent).as_posix(),extra,b)]:
        descriptor.append({'run_id':verification['producer_run_id'],
            'producer_run_id':verification['producer_run_id'],'viewer_run_id':verification['viewer_run_id'],
            'prefix':prefix,'case_count':len(document['cases']),
            'case_ids':verification['case_ids'],'producer_metadata_sha256':verification['metadata_sha256'],
            'review_manifest_sha256':verification['review_manifest_sha256'],
            'native_reports':verification['reports']})
    # The initial review is the immutable backup once publication replaces the
    # main manifest. Its byte digest is identical to the original snapshot.
    descriptor[0]['review_manifest_sha256']=hashlib.sha256(initial_bytes).hexdigest()
    aggregate={'schema_version':1,'kind':'acceptance-cohort-aggregate','run_id':base['run_id'],
        'case_count':len(merged['cases']),'method_runs':len(merged['cases'])*2,
        'parameters':deepcopy(base['parameters']),'components':descriptor,
        'producer_metadata_rewritten':False,'user_quality_accepted':False}
    targets=[parent_manifest,parent_script,parent/'aggregate-manifest.json',parent/'aggregate-validation.json']
    if not backup.exists():targets.append(backup)
    if not backup_script.exists():targets.append(backup_script)
    protected=[p for p in [*a['inputs'],*b['inputs']] if p not in (parent_manifest,parent_script)]
    if backup.exists():protected.append(backup)
    if backup_script.exists():protected.append(backup_script)
    _assert_inputs_safe([p for path in targets for p in (path,path.with_suffix(path.suffix+'.tmp'))],protected)
    snapshots={str(p):_sha(p) for p in protected}
    if _sha(parent_manifest)!=current_sha or _sha(parent_script)!=current_script_sha:
        raise RuntimeError('Viewer manifest or script changed while preparing aggregate')
    if not backup.exists():_atomic(backup,initial_bytes)
    if not backup_script.exists():
        script_bytes=current_script_bytes if len(current['cases'])==base_count else _js_bytes(base)
        _atomic(backup_script,script_bytes)
    if any(_sha(path)!=digest for path,digest in snapshots.items()):raise RuntimeError('Read-only producer data changed during merge')
    _atomic(parent/'aggregate-manifest.json',_json_bytes(aggregate))
    _atomic(parent_manifest,_json_bytes(merged));_atomic(parent_script,_js_bytes(merged))
    if any(_sha(path)!=digest for path,digest in snapshots.items()):raise RuntimeError('Read-only producer data changed after publication')
    validation={'schema_version':1,'complete':True,'run_id':base['run_id'],'case_count':len(merged['cases']),
        'method_runs':len(merged['cases'])*2,'initial_cases':len(base['cases']),'added_cases':len(extra['cases']),
        'original_cases_unchanged':merged['cases'][:base_count]==base['cases'],
        'producer_metadata_unchanged':True,'source_hashes_verified':True,'native_output_hashes_verified':True,
        'same_parameters_policy_code_and_models':True,'paths_within_parent':True,
        'viewer_manifest_sha256':_sha(parent_manifest),'viewer_script_sha256':_sha(parent_script),
        'aggregate_manifest_sha256':_sha(parent/'aggregate-manifest.json'),
        'original_manifest_sha256':_sha(backup),'previous_reviews_descriptor':previous,
        'user_quality_accepted':False,'browser_runtime_verified':False}
    if legacy:
        validation.update(original_six_cases_unchanged=validation['original_cases_unchanged'],
                          original_six_manifest_sha256=validation['original_manifest_sha256'])
    _atomic(parent/'aggregate-validation.json',_json_bytes(validation))
    return merged


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--extension',type=Path,required=True)
    args=parser.parse_args(argv);result=merge_reviews(args.parent,args.extension)
    print('MERGED_ACCEPTANCE_CASES',len(result['cases']),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
