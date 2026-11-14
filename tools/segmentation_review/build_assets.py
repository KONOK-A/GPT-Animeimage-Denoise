"""Export independent model masks for a local, file:// segmentation viewer.

No inference, semantic fusion, object subtraction or morphology is performed.
See-through channels remain overlapping multilabel evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
if str(PROJECT/'src') not in sys.path:sys.path.insert(0,str(PROJECT/'src'))

from gpt_animeimage_denoise.grouping import V4_TO_GROUP
from gpt_animeimage_denoise.pipeline import _assert_inputs_safe,_case_folder,_read_source,_save_rgb,_sha

MODULES=[
    {'id':'skin','name':'皮肤','m2f':[1],'see':[9],
     'color':'#edb472','note':'See-through仅显示neck颈部，不能等同于Mask2Former的全部裸肤；手脚粗类请单独查看。'},
    {'id':'face','name':'面部','m2f':[2,8],'see':[2,7,5],
     'color':'#f28da5','note':'面部合并鼻子；See-through的耳朵并入此对照组，两个模型语义范围可能不同。'},
    {'id':'hair','name':'头发','m2f':[3,6,7],'see':[0],
     'color':'#9c8bf2','note':'Mask2Former眉毛按当前规则归头发；See-through没有独立眉毛通道。'},
    {'id':'eyes','name':'眼睛','m2f':[4,5],'see':[3],
     'color':'#5fd4bc','note':'两眼合并展示；此处没有眼睛轮廓融合或相互修正。'},
    {'id':'mouth','name':'嘴巴','m2f':[9],'see':[8],
     'color':'#ee7691','note':'嘴巴独立，不并入面部。'},
    {'id':'outfit','name':'服饰','m2f':[10,11],'see':[1,4,6,10,11,13,15],
     'color':'#72b6ee','note':'服饰含穿戴配饰；See-through的handwear/legwear语义含糊，未自动并入服饰或裸肤，原始通道可查看。'},
]
GROUP_IDS=('skin','face','hair','eyes','mouth','outfit','all')
M2F_NAMES=['背景','裸肤','面部','头发','左眼','右眼','左眉','右眉','鼻子','嘴巴','衣服','配饰']
SEE_NAMES=['头发 hair','头饰 headwear','面部 face','眼睛 eyes','眼镜 eyewear','耳朵 ears',
           '耳饰 earwear','鼻子 nose','嘴巴 mouth','颈部 neck','颈饰 neckwear','上衣 topwear',
           '手／臂粗类 handwear','下装 bottomwear','腿粗类 legwear','鞋 footwear',
           '尾巴 tail','翅膀 wings','物体 objects']
COLORS={entry['id']:entry['color'] for entry in MODULES}
M2F_COLORS=['#303b4b',COLORS['skin'],COLORS['face'],COLORS['hair'],COLORS['eyes'],COLORS['eyes'],
            COLORS['hair'],COLORS['hair'],COLORS['face'],COLORS['mouth'],COLORS['outfit'],COLORS['outfit']]
SEE_COLORS=[COLORS['hair'],COLORS['outfit'],COLORS['face'],COLORS['eyes'],COLORS['outfit'],
            COLORS['face'],COLORS['outfit'],COLORS['face'],COLORS['mouth'],COLORS['skin'],
            COLORS['outfit'],COLORS['outfit'],'#d0a0e6',COLORS['outfit'],'#b5b86e',COLORS['outfit'],
            '#e5a26e','#8fb95e','#ec674d']


def _color(value):return tuple(int(value[i:i+2],16) for i in (1,3,5))


def m2f_groups(labels):
    labels=np.asarray(labels)
    if labels.ndim!=2 or labels.dtype.kind not in 'iu' or not labels.size or np.any((labels<0)|(labels>11)):
        raise ValueError('Expected original-grid Mask2Former labels 0..11')
    grouped=V4_TO_GROUP[labels]
    result={entry['id']:grouped==index for index,entry in enumerate(MODULES,1)}
    result['all']=grouped!=0
    return result


def see_groups(logits,threshold):
    logits=np.asarray(logits)
    if logits.dtype!=np.float32 or logits.ndim!=3 or logits.shape[0]!=19 or not np.isfinite(logits).all():
        raise ValueError('Expected finite float32 raw19 See-through logits')
    if not np.isfinite(threshold):raise ValueError('Threshold must be finite')
    result={entry['id']:(logits[entry['see']]>threshold).any(axis=0) for entry in MODULES}
    result['all']=np.logical_or.reduce(list(result.values()))
    return result


def load_evidence_cache(path,*,expected_source_sha256=None,expected_shape=None):
    """Read a signed full-score cache; historical code identity is recorded.

    This inspection export does not substitute cached labels for missing scores.
    Its purpose is to show the evidence stored by that recorded implementation.
    """
    from gpt_animeimage_denoise.fusion_pipeline import _payload_hash,_unpack,_validate_evidence
    with np.load(path,allow_pickle=False) as saved:
        metadata=json.loads(str(saved['metadata'].item()));identity=metadata['identity']
        if expected_source_sha256 is not None and identity.get('source_sha256')!=expected_source_sha256:
            raise ValueError('Evidence cache belongs to another source')
        if identity.get('preprocessing')!='exif-original-grid-white-alpha-v1' or not identity.get('models'):
            raise ValueError('Unsupported evidence preprocessing/model identity')
        arrays={key:saved[key] for key in saved.files if key!='metadata'}
        if metadata['payload_hash']!=_payload_hash(metadata['tree'],arrays):
            raise ValueError('Evidence cache payload hash mismatch')
        result=_unpack(metadata['tree'],arrays)
    shape=tuple(expected_shape) if expected_shape is not None else result['alpha'].shape
    # Historical inspection caches contain dual M2F views and only white See.
    # Original See, when supplied, is verified separately below; do not claim
    # that this old payload has the new production dual-See cache identity.
    _validate_evidence(result,shape,True,require_original_see=False)
    if not identity.get('dual_view'):raise ValueError('Full original/white score evidence is required')
    result['_review_cache_identity']=identity
    return result


def load_see_original(path,source_sha256,shape,*,expected_checkpoint_sha256=None):
    """Read root-produced original-input raw19 evidence with source binding."""
    with np.load(path,allow_pickle=False) as saved:
        metadata=json.loads(str(saved['metadata'].item()))
        if metadata.get('source_sha256')!=source_sha256:
            raise ValueError('Original See evidence belongs to another source')
        if not metadata.get('checkpoint_sha256') or not metadata.get('preprocessing') or not metadata.get('precision'):
            raise ValueError('Original See evidence provenance is incomplete')
        if expected_checkpoint_sha256 is not None and metadata['checkpoint_sha256']!=expected_checkpoint_sha256:
            raise ValueError('Original/white See checkpoints differ')
        logits=saved['see_logits'].copy()
    if logits.shape!=(19,*shape) or logits.dtype!=np.float32 or not np.isfinite(logits).all():
        raise ValueError('Invalid original-input raw19 evidence')
    if metadata.get('payload_sha256')!=hashlib.sha256(logits.tobytes()).hexdigest():
        raise ValueError('Original See payload hash mismatch')
    return logits,metadata


def _mask(path,mask):
    rgba=np.full((*mask.shape,4),255,np.uint8);rgba[:,:,3]=mask.astype(np.uint8)*255
    Image.fromarray(rgba).save(path,optimize=True)


def _colored(path,masks,colors):
    shape=masks[0].shape;rgb=np.zeros((*shape,3),np.float32);count=np.zeros(shape,np.uint8)
    for mask,color in zip(masks,colors):
        rgb[mask]+=np.asarray(_color(color),np.float32);count[mask]+=1
    active=count>0;rgb[active]/=count[active,None]
    rgba=np.dstack([np.rint(rgb).astype(np.uint8),active.astype(np.uint8)*255])
    Image.fromarray(rgba).save(path,optimize=True)


def _paths_for_case(output,case_id,original_see):
    folder=_case_folder(output/'assets',case_id)
    names=['original.png','input_original.png','white.png','thumb.jpg']
    for view in ('white','original'):
        names += [f'm2f-{view}-labels.png',f'm2f-{view}-colored.png',f'm2f-{view}-groups-colored.png']
        names += [f'm2f-{view}-raw-{index:02d}.png' for index in range(12)]
        names += [f'm2f-{view}-{group}.png' for group in GROUP_IDS]
        if view=='original' and not original_see:continue
        for threshold in (0,1,2):
            prefix=f'see-{view}-t{threshold}'
            names += [prefix+'-colored.png',prefix+'-groups-colored.png']
            names += [f'{prefix}-raw-{index:02d}.png' for index in range(19)]
            names += [f'{prefix}-{group}.png' for group in GROUP_IDS]
    return folder,[folder/name for name in names]


def export_case(case,output_root,evidence,*,see_original=None,evidence_path=None,input_paths=()):
    """Export one independent model comparison, preserving source bytes."""
    output=Path(output_root).resolve();source=Path(case['path']).resolve()
    source_hash=_sha(source)
    if source_hash!=case['sha256']:raise ValueError('Source SHA256 differs from request')
    rgb,alpha,icc=_read_source(source);h,w=rgb.shape[:2]
    sky=np.asarray(evidence['alpha']);see=np.asarray(evidence['see_logits'])
    if (sky.dtype!=np.float32 or sky.shape!=(h,w) or not np.isfinite(sky).all()
            or np.any((sky<0)|(sky>1)) or see.dtype!=np.float32 or see.shape!=(19,h,w)
            or not np.isfinite(see).all()):
        raise ValueError('Evidence must match oriented source grid')
    for name in ('m2f_white','m2f_original'):
        scores=np.asarray(evidence.get(name))
        if scores.dtype!=np.float32 or scores.ndim!=3 or scores.shape[0]!=12 or not np.isfinite(scores).all():
            raise ValueError('Both native float32 Mask2Former score views are required')
    if see_original is not None and (see_original.dtype!=np.float32 or see_original.shape!=(19,h,w)
                                    or not np.isfinite(see_original).all()):
        raise ValueError('Original See score grid differs from source')
    folder,targets=_paths_for_case(output,case['id'],see_original is not None)
    inputs=[source,*map(Path,input_paths)]
    for value in (evidence_path,case.get('evidence_cache'),case.get('see_original_cache')):
        if value is not None:inputs.append(Path(value))
    _assert_inputs_safe(targets,inputs)
    snapshots={str(path):_sha(path) for path in inputs}
    folder.mkdir(parents=True,exist_ok=True)
    def relative(path):return path.relative_to(output).as_posix()
    _save_rgb(folder/'original.png',rgb,alpha,icc)
    analysis_rgb=rgb if alpha is None else np.rint(rgb*(alpha[...,None].astype(np.float32)/255)+255*(1-alpha[...,None].astype(np.float32)/255)).astype(np.uint8)
    white=np.rint(analysis_rgb.astype(np.float32)*sky[...,None]+255*(1-sky[...,None])).clip(0,255).astype(np.uint8)
    _save_rgb(folder/'input_original.png',analysis_rgb,None,icc)
    _save_rgb(folder/'white.png',white,None,icc)
    thumbnail=Image.fromarray(analysis_rgb);thumbnail.thumbnail((240,240),Image.Resampling.LANCZOS)
    thumbnail.save(folder/'thumb.jpg',quality=88)
    entry={key:case[key] for key in ('id','title','description') if key in case}
    entry.update(width=w,height=h,source_sha256=source_hash,source=str(source),
                 original=relative(folder/'original.png'),white=relative(folder/'white.png'),
                 thumbnail=relative(folder/'thumb.jpg'),views={},provenance=evidence.get('provenance',{}),
                 evidence_cache_identity=evidence.get('_review_cache_identity'),
                 m2f_postprocessing='native-argmax-then-nearest-original-grid; no semantic fusion',
                 see_postprocessing='independent raw logit > threshold; overlap retained; no object subtraction')
    if case.get('see_original_provenance'):entry['see_original_provenance']=case['see_original_provenance']
    for view in ('white','original'):
        scores=evidence['m2f_'+view]
        native=scores.argmax(axis=0).astype(np.uint8)
        labels=np.asarray(Image.fromarray(native).resize((w,h),Image.Resampling.NEAREST))
        groups=m2f_groups(labels);prefix='m2f-'+view
        label_path=folder/(prefix+'-labels.png');Image.fromarray(labels).save(label_path,optimize=True)
        masks=[];paths=[]
        for index in range(12):
            mask=labels==index;masks.append(mask);path=folder/f'{prefix}-raw-{index:02d}.png'
            _mask(path,mask);paths.append(relative(path))
        group_paths={}
        for key,mask in groups.items():
            path=folder/f'{prefix}-{key}.png';_mask(path,mask);group_paths[key]=relative(path)
        colored=folder/(prefix+'-colored.png');_colored(colored,masks[1:],M2F_COLORS[1:])
        grouped_color=folder/(prefix+'-groups-colored.png')
        _colored(grouped_color,[groups[key] for key in GROUP_IDS[:-1]],[COLORS[key] for key in GROUP_IDS[:-1]])
        m2f={'labels':relative(label_path),'masks':paths,'groups':group_paths,
              'colored':relative(colored),'groups_colored':relative(grouped_color)}
        data={'input':relative(folder/('white.png' if view=='white' else 'input_original.png')),'m2f':m2f,'see':None}
        logits=see if view=='white' else see_original
        if logits is not None:
            levels={}
            for threshold in (0,1,2):
                prefix=f'see-{view}-t{threshold}';group_masks=see_groups(logits,threshold)
                masks=[];paths=[]
                for index in range(19):
                    mask=logits[index]>threshold;masks.append(mask);path=folder/f'{prefix}-raw-{index:02d}.png'
                    _mask(path,mask);paths.append(relative(path))
                group_paths={}
                for key,mask in group_masks.items():
                    path=folder/f'{prefix}-{key}.png';_mask(path,mask);group_paths[key]=relative(path)
                colored=folder/(prefix+'-colored.png');_colored(colored,masks,SEE_COLORS)
                grouped_color=folder/(prefix+'-groups-colored.png')
                _colored(grouped_color,[group_masks[key] for key in GROUP_IDS[:-1]],[COLORS[key] for key in GROUP_IDS[:-1]])
                levels[str(threshold)]={'masks':paths,'groups':group_paths,'colored':relative(colored),
                                        'groups_colored':relative(grouped_color)}
            data['see']={'thresholds':levels}
        entry['views'][view]=data
    if any(_sha(path)!=digest for path,digest in snapshots.items()):
        raise RuntimeError('Source or evidence changed during export')
    return entry


def write_manifest(output_root,cases,*,input_paths=()):
    output=Path(output_root).resolve();targets=[output/'manifest.json',output/'manifest.js']
    inputs=[*map(Path,input_paths),*[Path(case['source']) for case in cases if case.get('source')]]
    _assert_inputs_safe(targets,inputs)
    document={'schema_version':1,'cases':cases,'modules':MODULES,
              'm2f_labels':[{'id':i,'name':name,'color':M2F_COLORS[i]} for i,name in enumerate(M2F_NAMES)],
              'see_labels':[{'id':i,'name':name,'color':SEE_COLORS[i]} for i,name in enumerate(SEE_NAMES)],
              'thresholds':[0,1,2],
              'method_info':{'m2f':'Mask2Former v4 · 768×768 · FP32 · TF32关闭',
                             'see':'See-through · raw19 · 原checkpoint精度',
                             'white':'SkyTNT 2048 FP32生成的共同白底输入',
                             'original':'EXIF方向归一后的共同原图输入；透明处按白底合成',
                             'all':'全部表示六个已映射模块的并集；See-through歧义肢体、尾巴、翅膀和物体请在原始通道查看。',
                             'purpose':'仅审阅独立模型证据；未运行SAM、降噪或更改fusion规则。'}}
    output.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False)
    targets[0].write_text(payload,encoding='utf-8')
    targets[1].write_text('window.SEG_REVIEW_DATA = '+payload+';\n',encoding='utf-8')
    return document


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv);request_path=args.request.resolve()
    request=json.loads(request_path.read_text(encoding='utf-8'));cases=request['cases'];output=args.out.resolve()
    inputs=[request_path]
    for case in cases:
        inputs += [Path(case[key]).resolve() for key in ('path','evidence_cache','see_original_cache') if case.get(key)]
    targets=[output/'manifest.json',output/'manifest.js']
    for case in cases:targets += _paths_for_case(output,case['id'],bool(case.get('see_original_cache')))[1]
    _assert_inputs_safe(targets,inputs)
    entries=[]
    for case in cases:
        print('EXPORT',case['id'],flush=True)
        rgb,_,_=_read_source(case['path'])
        ev=load_evidence_cache(case['evidence_cache'],expected_source_sha256=case['sha256'],expected_shape=rgb.shape[:2])
        raw=None
        if case.get('see_original_cache'):
            expected=ev['provenance']['model_hashes']['v2/see_through/checkpoint-18000.pt']
            raw,metadata=load_see_original(case['see_original_cache'],case['sha256'],rgb.shape[:2],
                                           expected_checkpoint_sha256=expected)
            if metadata.get('model_signature')!=ev['_review_cache_identity']['models']:
                raise ValueError('Original/white model signatures differ')
            case=dict(case,see_original_provenance=metadata)
        entries.append(export_case(case,output,ev,see_original=raw,evidence_path=case['evidence_cache'],input_paths=inputs))
        del ev,raw
    write_manifest(output,entries,input_paths=inputs)
    print('EXPORTED',len(entries),'cases',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
