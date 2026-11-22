"""Đánh giá một snapshot đã khóa; module này không gọi fit hoặc chọn ngưỡng."""
from pathlib import Path
import hashlib,json,platform,shutil
import numpy as np
import pandas as pd
import joblib,sklearn
from sklearn.metrics import (confusion_matrix,classification_report,accuracy_score,
    precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,
    balanced_accuracy_score,roc_curve,precision_recall_curve)
from threadpoolctl import threadpool_limits
from .preprocessing import FEATURE_COLUMNS,encode_target,normalize_config
from .splitting import frame_fingerprint,make_group_ids

LABELS=['<=50K','>50K']

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def _digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()
def _source_hashes(root):
    return {name:hashlib.sha256((root/'modules'/name).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
        for name in ['data_loader.py','splitting.py','preprocessing.py','training.py']}

def freeze_selection(project_dir='.'):
    """Giữ snapshot trước khi đọc kết quả test; không thay snapshot khi chạy lại."""
    root=Path(project_dir)
    selected=json.loads((root/'results/step_05/selected_model.json').read_text(encoding='utf-8'))
    if selected['validation_used_for_selection'] or selected['threshold_tuned'] or selected['refit_on_train_plus_validation']:
        raise ValueError('Lựa chọn bước 5 không khớp giao thức đã khóa.')
    cfgs=json.loads((root/'configs/experiments.json').read_text(encoding='utf-8'))
    candidate=next(c for c in cfgs['candidates'] if c['id']==selected['winner_id'])
    identity={'winner_id':selected['winner_id'],'training_data_sha256':selected['training_data_sha256'],
        'validation_data_sha256':selected['validation_data_sha256'],
        'cv_manifest_sha256':selected['cv_manifest_sha256'],'candidates_sha256':selected['candidates_sha256'],
        'candidate':candidate,'refit_rows':selected['refit_rows'],
        'source_hashes':_source_hashes(root)}
    out=root/'artifacts/final';lock_path=out/'selection_snapshot.json'
    if lock_path.exists():
        locked=json.loads(lock_path.read_text(encoding='utf-8'))
        if locked['identity']!=identity:raise ValueError('Snapshot đã khóa khác lựa chọn/dữ liệu/mã nguồn hiện tại; không tự đổi mô hình sau test.')
        if _sha(root/locked['pipeline_path'])!=locked['pipeline_sha256']:raise ValueError('Pipeline snapshot đã bị thay đổi.')
        return locked
    out.mkdir(parents=True,exist_ok=True)
    original=root/selected['selected_model_path']
    pipeline=joblib.load(original)
    if list(pipeline.named_steps['model'].classes_)!=[0,1]:raise ValueError('Sai thứ tự hai lớp.')
    if pipeline.named_steps['preprocessing'].named_steps['input'].fit_rows_!=selected['refit_rows']:
        raise ValueError('Pipeline không được fit trên số dòng train đã khóa.')
    probability=hasattr(pipeline,'predict_proba')
    target=out/'selected_pipeline.joblib';shutil.copyfile(original,target)
    locked={'schema_version':1,'identity':identity,'pipeline_path':target.relative_to(root).as_posix(),
        'pipeline_sha256':_sha(target),'pipeline_state_hash':joblib.hash(pipeline),
        'source_model_path':selected['selected_model_path'],'source_model_sha256':_sha(original),
        'threshold':0.5 if probability else 0.0,
        'score_kind':'probability' if probability else 'decision_score',
        'prediction_rule':'native estimator predict; no threshold tuning',
        'preprocessing':normalize_config(candidate['preprocessing']),
        'stage_05_selection':selected,'versions':selected['versions'],
        'created_before_test_evaluation':True}
    _json(lock_path,locked)
    return locked

def summarize_predictions(y,pred,score):
    y=np.asarray(y);pred=np.asarray(pred);score=np.asarray(score,dtype=float)
    if y.ndim!=1 or pred.shape!=y.shape or score.shape!=y.shape or len(y)==0:
        raise ValueError('Nhãn, dự đoán và score phải cùng chiều và không rỗng.')
    if set(np.unique(y))!={0,1} or not set(np.unique(pred)).issubset({0,1}) or not np.isfinite(score).all():
        raise ValueError('Cần hai lớp thật 0/1, dự đoán nhị phân và score hữu hạn.')
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
    specificity=tn/(tn+fp);fpr=fp/(tn+fp);fnr=fn/(tp+fn)
    metrics={'accuracy':float(accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,pos_label=1,zero_division=0)),
        'recall':float(recall_score(y,pred,pos_label=1,zero_division=0)),
        'f1':float(f1_score(y,pred,pos_label=1,zero_division=0)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'roc_auc':float(roc_auc_score(y,score)),
        'average_precision':float(average_precision_score(y,score)),
        'specificity':float(specificity),'false_positive_rate':float(fpr),'false_negative_rate':float(fnr),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'positive_prevalence':float(y.mean()),'error_rate':float(np.mean(y!=pred))}
    counts={'tn':tn,'fp':fp,'fn':fn,'tp':tp,'rows':len(y),'positive_rows':int(y.sum()),'negative_rows':int((y==0).sum()),'error_rows':fp+fn}
    return metrics,counts,cm

def evaluate_frozen(test,test_metadata,project_dir='.'):
    """Chỉ dự đoán test với pipeline cố định; replay kết quả nếu cùng snapshot/data."""
    root=Path(project_dir);out=root/'results/step_06';out.mkdir(parents=True,exist_ok=True)
    locked=freeze_selection(root)
    if len(test)!=16281 or len(test_metadata)!=len(test):raise ValueError('Phải giữ đủ 16.281 dòng test UCI gốc.')
    if test.columns.tolist()!=FEATURE_COLUMNS+['income']:raise ValueError('Sai schema test.')
    if not (test_metadata['source_file']=='adult.test').all() or not test_metadata['source_id'].is_unique:
        raise ValueError('Metadata không phải tập adult.test gốc hoặc trùng ID.')
    np.testing.assert_array_equal(test_metadata['source_row'].to_numpy(),np.arange(len(test)))
    if sklearn.__version__!=locked['versions']['sklearn']:
        raise ValueError('Phiên bản scikit-learn khác snapshot; không đánh giá trực tiếp một pickle khác phiên bản.')
    source_sha=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
    identity={'test_data_sha256':frame_fingerprint(test),'test_source_ids_sha256':_digest(test_metadata.source_id.astype(str).tolist()),
        'selection_snapshot_sha256':_digest(locked),'pipeline_sha256':locked['pipeline_sha256'],
        'evaluation_source_sha256':source_sha,'sklearn':sklearn.__version__,'numpy':np.__version__}
    complete=out/'evaluation_complete.json'
    if complete.exists():
        prior=json.loads(complete.read_text(encoding='utf-8'))
        if prior['identity']!=identity:raise ValueError('Đã có đánh giá test với dữ liệu/model/code khác; không tự ghi đè hoặc đánh giá lại để chọn cấu hình.')
        for rel,sha in prior['files'].items():
            if _sha(root/rel)!=sha:raise ValueError('Kết quả đánh giá đã bị thay đổi: '+rel)
        print('Đọc kết quả test đã xác minh cho cùng snapshot và dữ liệu; không dự đoán lại.')
        return json.loads((out/'test_metrics.json').read_text(encoding='utf-8'))
    pipeline=joblib.load(root/locked['pipeline_path'])
    before=joblib.hash(pipeline)
    if before!=locked['pipeline_state_hash']:raise ValueError('Trạng thái pipeline khác lúc khóa.')
    X=test[FEATURE_COLUMNS];y=encode_target(test);snapshot=test.copy(deep=True)
    with threadpool_limits(limits=2):
        pred=pipeline.predict(X)
        if locked['score_kind']=='probability':score=np.asarray(pipeline.predict_proba(X)[:,1],dtype=float)
        else:score=np.asarray(pipeline.decision_function(X),dtype=float)
    if joblib.hash(pipeline)!=before or _sha(root/locked['pipeline_path'])!=locked['pipeline_sha256']:
        raise AssertionError('Đánh giá làm thay đổi snapshot fitted.')
    pd.testing.assert_frame_equal(test,snapshot)
    metrics,counts,cm=summarize_predictions(y,pred,score)
    np.testing.assert_array_equal(pred,(score>locked['threshold']).astype(np.int8))
    frame=pd.DataFrame({'source_id':test_metadata.source_id.to_numpy(),
        'source_row':test_metadata.source_row.to_numpy(),'y_true':y,'y_pred':pred,'score':score})
    frame['outcome']=np.select([(y==0)&(pred==0),(y==0)&(pred==1),(y==1)&(pred==0)],['TN','FP','FN'],default='TP')
    frame.to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame(cm,index=LABELS,columns=LABELS).rename_axis('actual').to_csv(out/'confusion_matrix.csv')
    pd.DataFrame([metrics]).to_csv(out/'test_metrics.csv',index=False)
    report=classification_report(y,pred,labels=[0,1],target_names=LABELS,output_dict=True,zero_division=0)
    _json(out/'classification_report.json',report)
    class_table=pd.DataFrame([{**report[label],'class':label} for label in LABELS])[['class','precision','recall','f1-score','support']]
    class_table.to_csv(out/'classification_report.csv',index=False)
    fpr,tpr,roc_thresholds=roc_curve(y,score,pos_label=1)
    pd.DataFrame({'false_positive_rate':fpr,'true_positive_rate':tpr,'threshold':roc_thresholds}).to_csv(out/'roc_curve.csv',index=False)
    precision,recall,pr_thresholds=precision_recall_curve(y,score,pos_label=1)
    pd.DataFrame({'precision':precision,'recall':recall,'threshold':np.r_[pr_thresholds,np.nan]}).to_csv(out/'precision_recall_curve.csv',index=False)
    result={'winner_id':locked['identity']['winner_id'],'threshold':locked['threshold'],'score_kind':locked['score_kind'],
        'metrics':metrics,'counts':counts,'identity':identity,'pipeline_fitted_state_unchanged':True,
        'test_frame_unchanged':True,'test_evaluation_used_for_tuning':False,'classifier_refit_in_evaluation':False,
        'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__}}
    _json(out/'test_metrics.json',result)
    paths=[out/name for name in ['test_predictions.csv','confusion_matrix.csv','test_metrics.csv','test_metrics.json',
        'classification_report.csv','classification_report.json','roc_curve.csv','precision_recall_curve.csv']]
    _json(complete,{'identity':identity,'evaluation_runs':1,
        'files':{p.relative_to(root).as_posix():_sha(p) for p in paths}})
    return result

def analyze_errors(test,test_metadata,project_dir='.'):
    root=Path(project_dir);out=root/'results/step_06'
    result=json.loads((out/'test_metrics.json').read_text(encoding='utf-8'))
    if frame_fingerprint(test)!=result['identity']['test_data_sha256']:raise ValueError('Dữ liệu phân tích khác tập test đã đánh giá.')
    pred=pd.read_csv(out/'test_predictions.csv')
    if pred.source_id.tolist()!=test_metadata.source_id.tolist():raise ValueError('Dự đoán không khớp metadata.')
    detailed=pd.concat([pred.reset_index(drop=True),test[FEATURE_COLUMNS].reset_index(drop=True)],axis=1)
    errors=detailed[detailed.y_true!=detailed.y_pred].copy()
    errors.to_csv(out/'test_errors.csv',index=False)
    examples=pd.concat([
        errors[errors.outcome=='FP'].sort_values(['score','source_row'],ascending=[False,True]).head(5),
        errors[errors.outcome=='FN'].sort_values(['score','source_row'],ascending=[True,True]).head(5),
    ],ignore_index=True)
    examples.to_csv(out/'error_examples.csv',index=False)
    numeric=['age','education-num','hours-per-week','capital-gain','capital-loss']
    profile=detailed.groupby('outcome',sort=True)[numeric].median().add_suffix('_median')
    profile.insert(0,'rows',detailed.groupby('outcome').size())
    profile.to_csv(out/'outcome_numeric_profiles.csv')
    return examples,profile

def audit_test_profile_overlap(train,validation,test,project_dir='.'):
    """Chỉ chẩn đoán sau khi đã khóa/đánh giá; không bỏ dòng hay đổi mô hình."""
    root=Path(project_dir);out=root/'results/step_06'
    if not (out/'evaluation_complete.json').exists():raise ValueError('Chỉ chạy chẩn đoán sau đánh giá đã khóa.')
    groups=make_group_ids(test);tr=set(make_group_ids(train));va=set(make_group_ids(validation))
    in_train=np.array([g in tr for g in groups]);in_val=np.array([g in va for g in groups])
    result={'input_columns':FEATURE_COLUMNS,'label_in_group_key':False,
        'test_rows':len(test),'unique_test_profiles':len(set(groups)),
        'test_rows_matching_train_profile':int(in_train.sum()),
        'test_profiles_matching_train':len(set(groups[in_train])),
        'test_rows_matching_validation_profile':int(in_val.sum()),
        'test_profiles_matching_validation':len(set(groups[in_val])),
        'rows_removed':0,'used_for_model_or_threshold_selection':False,
        'interpretation':'Identical 14-feature records, not proof of same real person; original UCI test remains unchanged.'}
    _json(out/'test_profile_overlap.json',result)
    return result

def plot_evaluation(result_dir='results/step_06'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(result_dir);result=json.loads((root/'test_metrics.json').read_text(encoding='utf-8'))
    metrics=result['metrics'];cm=pd.read_csv(root/'confusion_matrix.csv',index_col=0).to_numpy()
    def save(fig,stem):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(root/f'{stem}.{ext}',dpi=165,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for ax,values,title,percent in [(axes[0],cm,'Số mẫu',False),
            (axes[1],cm/cm.sum(axis=1,keepdims=True)*100,'Tỷ lệ theo nhãn thật (%)',True)]:
        ax.imshow(values,cmap='Blues',vmin=0,vmax=100 if percent else cm.max())
        for i in range(2):
            for j in range(2):
                value=values[i,j];text=f'{value:.1f}%' if percent else f'{value:.0f}'
                ax.text(j,i,text,ha='center',va='center',fontsize=14,color='white' if value>values.max()/2 else 'black')
        ax.set_xticks([0,1],LABELS);ax.set_yticks([0,1],LABELS)
        ax.set(xlabel='Nhãn dự đoán',ylabel='Nhãn thật',title=title)
    fig.suptitle('Ma trận nhầm lẫn trên test UCI — mô hình và ngưỡng đã khóa')
    save(fig,'test_confusion_matrix')
    roc=pd.read_csv(root/'roc_curve.csv');fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(roc.false_positive_rate,roc.true_positive_rate,color='#2563EB',label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    ax.plot([0,1],[0,1],'--',color='#64748B',label='Đường tham chiếu ngẫu nhiên')
    ax.scatter([metrics['false_positive_rate']],[metrics['recall']],color='#EA580C',zorder=4,label='Ngưỡng đã khóa')
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',ylabel='True positive rate (recall)',title='ROC trên test — lớp dương >50K')
    ax.grid(alpha=.2);ax.legend(loc='lower right');save(fig,'test_roc')
    pr=pd.read_csv(root/'precision_recall_curve.csv');fig,ax=plt.subplots(figsize=(6,5))
    ax.step(pr.recall,pr.precision,where='post',color='#2563EB',label=f"Average precision = {metrics['average_precision']:.4f}")
    ax.axhline(metrics['positive_prevalence'],linestyle='--',color='#64748B',label=f"Tỷ lệ lớp dương = {metrics['positive_prevalence']:.3f}")
    ax.scatter([metrics['recall']],[metrics['precision']],color='#EA580C',zorder=4,label='Ngưỡng đã khóa')
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision',title='Precision–Recall trên test — lớp dương >50K')
    ax.grid(alpha=.2);ax.legend(loc='lower left');save(fig,'test_precision_recall')
    return ['test_confusion_matrix','test_roc','test_precision_recall']
