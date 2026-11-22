"""Chia Adult development theo nhóm; không fit đặc trưng hoặc dùng test để chọn split."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import platform

import numpy as np
import pandas as pd

from .data_loader import COLUMNS, NUMERIC_COLUMNS

DEFAULT_CONFIG = {
    'n_splits':5,
    'validation_fold':0,
    'random_state':42,
    'group_columns':COLUMNS[:-1],
}
LABELS=['<=50K','>50K']


def _canonical_rows(frame, columns):
    positions=[frame.columns.get_loc(c) for c in columns]
    numeric=set(NUMERIC_COLUMNS)
    for row in frame.itertuples(index=False,name=None):
        values=[]
        for column,position in zip(columns,positions):
            value=row[position]
            if pd.isna(value):
                values.append(None)
            elif column in numeric:
                # Không cắt phần lẻ nếu dữ liệu bị thay đổi sau bước đọc.
                numeric_value=float(value)
                values.append(int(value) if numeric_value.is_integer() else numeric_value)
            else:
                values.append(str(value))
        yield json.dumps(values,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def frame_fingerprint(frame):
    """Dấu vân tay có xét thứ tự dòng và tất cả cột của bảng development."""
    digest=hashlib.sha256()
    digest.update(json.dumps(COLUMNS).encode('utf-8'))
    for value in _canonical_rows(frame,COLUMNS):
        digest.update(value.encode('utf-8'));digest.update(b'\n')
    return digest.hexdigest()


def make_group_ids(frame, group_columns=None):
    columns=list(COLUMNS[:-1] if group_columns is None else group_columns)
    if not columns or len(columns)!=len(set(columns)) or not set(columns).issubset(COLUMNS[:-1]):
        raise ValueError('Nhóm phải dùng các thuộc tính đầu vào riêng biệt, không dùng income.')
    canonical=list(_canonical_rows(frame,columns))
    groups=np.array([hashlib.sha256(value.encode('utf-8')).hexdigest() for value in canonical])
    if len(set(groups))!=len(set(canonical)):
        raise ValueError('Phát hiện va chạm mã nhóm; cần đối chiếu lại khóa canonical.')
    return groups


def _normalize_config(config=None):
    result={**DEFAULT_CONFIG,**(config or {})}
    result['group_columns']=list(result['group_columns'])
    for key in ['n_splits','validation_fold','random_state']:
        if isinstance(result[key],bool) or not isinstance(result[key],int):
            raise ValueError(f'{key} phải là số nguyên.')
    if result['n_splits']<2 or not 0<=result['validation_fold']<result['n_splits']:
        raise ValueError('Số fold hoặc chỉ số fold validation không hợp lệ.')
    return result


def validate_indices(frame, train_indices, validation_indices, groups):
    """Kiểm tra đầy đủ dòng, chỉ số, nhãn và không chung nhóm train/validation."""
    lists=[list(train_indices),list(validation_indices)]
    if any(any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) for i in arr) for arr in lists):
        raise ValueError('Chỉ số phải là số nguyên.')
    train_indices,validation_indices=[np.array(a,dtype=np.int64) for a in lists]
    if any(len(a)==0 or len(np.unique(a))!=len(a) for a in [train_indices,validation_indices]):
        raise ValueError('Có tập rỗng hoặc chỉ số lặp trong một tập.')
    combined=np.concatenate([train_indices,validation_indices])
    if not np.array_equal(np.sort(combined),np.arange(len(frame))):
        raise ValueError('Hai tập phải chia đủ mỗi dòng development đúng một lần.')
    overlap=set(groups[train_indices])&set(groups[validation_indices])
    if overlap:
        raise ValueError('Có nhóm đầu vào đồng thời nằm trong train và validation.')
    for name,idx in [('train',train_indices),('validation',validation_indices)]:
        if set(frame.iloc[idx]['income'].unique())!=set(LABELS):
            raise ValueError(f'{name} không có đủ hai nhãn.')
    return np.sort(train_indices),np.sort(validation_indices)


def load_or_create_split(development, manifest_path, config=None):
    """Tạo một fold cố định, hoặc tái dùng đúng chỉ số đã khóa trong manifest."""
    if list(development.columns)!=COLUMNS or development['income'].isna().any():
        raise ValueError('Development không có đúng cột hoặc thiếu nhãn.')
    if set(development['income'].unique())!=set(LABELS):
        raise ValueError('Development phải có đúng hai nhãn Adult.')
    config=_normalize_config(config)
    groups=make_group_ids(development,config['group_columns'])
    fingerprint=frame_fingerprint(development)
    manifest_path=Path(manifest_path)
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('development_fingerprint')!=fingerprint:
            raise ValueError('Manifest không khớp dữ liệu hoặc thứ tự dòng development.')
        if manifest.get('config')!=config:
            raise ValueError('Manifest không khớp cấu hình. Muốn đổi cấu hình cần tạo manifest mới có chủ đích.')
        train_idx,val_idx=validate_indices(development,manifest['train_indices'],manifest['validation_indices'],groups)
        return train_idx,val_idx,groups,manifest,'frozen_manifest'

    from sklearn.model_selection import StratifiedGroupKFold
    import sklearn
    splitter=StratifiedGroupKFold(
        n_splits=config['n_splits'],shuffle=True,random_state=config['random_state']
    )
    # Cố định fold trước; không thử nhiều seed hoặc chọn fold theo điểm mô hình.
    split=None
    for fold,(tr,va) in enumerate(splitter.split(
        np.zeros((len(development),1)),development['income'].to_numpy(dtype=str),groups
    )):
        if fold==config['validation_fold']:
            split=(tr,va);break
    train_idx,val_idx=validate_indices(development,*split,groups)
    manifest={
        'schema_version':1,'method':'StratifiedGroupKFold',
        'development_fingerprint':fingerprint,'development_rows':len(development),
        'config':config,'sklearn_version_generation':sklearn.__version__,
        'train_indices':train_idx.tolist(),'validation_indices':val_idx.tolist(),
        'index_convention':'0-based data rows in adult.data, after parsing, excluding blank lines',
    }
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return train_idx,val_idx,groups,manifest,'new_split'


def export_splits(development, original_test, train_idx, validation_idx, groups,
                  manifest, project_dir='.', split_source='frozen_manifest'):
    """Xuất bảng chia, nguồn dòng và biên bản; test giữ nguyên không lập thống kê nhãn."""
    train_idx,validation_idx=validate_indices(development,train_idx,validation_idx,groups)
    if list(original_test.columns)!=COLUMNS:
        raise ValueError('Test phải giữ nguyên cấu trúc Adult gốc.')
    root=Path(project_dir); data_dir=root/'data/splits'; result=root/'results/step_03'
    data_dir.mkdir(parents=True,exist_ok=True);result.mkdir(parents=True,exist_ok=True)
    frames={
        'train':development.iloc[train_idx].copy().reset_index(drop=True),
        'validation':development.iloc[validation_idx].copy().reset_index(drop=True),
        'test':original_test.copy(deep=True).reset_index(drop=True),
    }
    metadata={}
    for name,idx in [('train',train_idx),('validation',validation_idx)]:
        metadata[name]=pd.DataFrame({
            'source_file':['adult.data']*len(idx),'source_row':idx,
            'source_id':[f'adult.data:{i}' for i in idx],
            'group_id':groups[idx],
        })
    metadata['test']=pd.DataFrame({
        'source_file':['adult.test']*len(original_test),
        'source_row':np.arange(len(original_test),dtype=np.int64),
        'source_id':[f'adult.test:{i}' for i in range(len(original_test))],
        'group_id':pd.Series([pd.NA]*len(original_test),dtype='string'),
    })
    for name,frame in frames.items():
        frame.to_csv(data_dir/f'{name}.csv',index=False,encoding='utf-8')
        metadata[name].to_csv(data_dir/f'{name}_metadata.csv',index=False,encoding='utf-8')
    assignments=pd.concat([m.assign(split=name) for name,m in metadata.items()],ignore_index=True)
    assignments.to_csv(result/'split_assignments.csv',index=False,encoding='utf-8')

    class_rows=[];summary_rows=[]
    for name in ['train','validation']:
        frame=frames[name]
        counts=frame['income'].value_counts().reindex(LABELS,fill_value=0)
        for label in LABELS:
            class_rows.append({'split':name,'income':label,'count':int(counts[label]),
                'percent':float(counts[label]/len(frame)*100)})
        summary_rows.append({'split':name,'source_file':'adult.data','rows':len(frame),
            'share_of_development_percent':len(frame)/len(development)*100,
            'unique_groups':metadata[name]['group_id'].nunique(),
            'positive_count':int(counts['>50K']),
            'positive_percent':float(counts['>50K']/len(frame)*100)})
    summary_rows.append({'split':'test','source_file':'adult.test','rows':len(original_test),
        'share_of_development_percent':None,'unique_groups':None,'positive_count':None,'positive_percent':None})
    summary_table=pd.DataFrame(summary_rows)
    summary_table.to_csv(result/'split_summary.csv',index=False,encoding='utf-8')
    balance=pd.DataFrame(class_rows)
    balance.to_csv(result/'train_validation_class_balance.csv',index=False,encoding='utf-8')

    labels_by_group=defaultdict(set)
    for key,label in zip(groups,development['income']):labels_by_group[key].add(str(label))
    conflicting={key for key,labels in labels_by_group.items() if len(labels)>1}
    membership={key:'train' for key in groups[train_idx]}
    membership.update({key:'validation' for key in groups[validation_idx]})
    conflicts=pd.DataFrame([
        {'group_id':key,'split':membership[key],'rows':int((groups==key).sum()),'n_labels':len(labels_by_group[key])}
        for key in sorted(conflicting)
    ],columns=['group_id','split','rows','n_labels'])
    conflicts.to_csv(result/'conflicting_groups_assignment.csv',index=False,encoding='utf-8')

    checks={
        'train_validation_source_index_overlap':len(set(train_idx)&set(validation_idx)),
        'train_validation_group_overlap':len(set(groups[train_idx])&set(groups[validation_idx])),
        'all_development_rows_assigned_once':bool(np.array_equal(np.sort(np.r_[train_idx,validation_idx]),np.arange(len(development)))),
        'all_source_ids_unique':bool(assignments['source_id'].is_unique),
        'test_rows_preserved':len(frames['test'])==len(original_test),
        'test_values_and_order_preserved':bool(frames['test'].equals(original_test.reset_index(drop=True))),
        'test_used_to_choose_split':False,
        'test_label_statistics_computed':False,
        'train_test_feature_overlap_audit':'not_performed_at_step_03',
        'preprocessing_fitted':False,'rows_removed':0,
    }
    if checks['train_validation_group_overlap'] or checks['train_validation_source_index_overlap']:
        raise ValueError('Kiểm tra rò rỉ train/validation không đạt.')
    if not all(checks[k] for k in ['all_development_rows_assigned_once','all_source_ids_unique','test_rows_preserved','test_values_and_order_preserved']):
        raise ValueError('Kiểm tra bảo toàn dữ liệu không đạt.')
    (result/'leakage_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={
        'step':3,'method':'StratifiedGroupKFold','split_source':split_source,
        'config':manifest['config'],'development_rows':len(development),
        'train_rows':len(frames['train']),'validation_rows':len(frames['validation']),'test_rows':len(frames['test']),
        'development_unique_groups':len(set(groups)),
        'train_unique_groups':int(metadata['train']['group_id'].nunique()),
        'validation_unique_groups':int(metadata['validation']['group_id'].nunique()),
        'train_positive_percent':float(summary_rows[0]['positive_percent']),
        'validation_positive_percent':float(summary_rows[1]['positive_percent']),
        'class_rate_difference_percentage_points':abs(summary_rows[0]['positive_percent']-summary_rows[1]['positive_percent']),
        'conflicting_groups':len(conflicting),
        'conflicting_group_splits':conflicts[['split','rows','n_labels']].to_dict(orient='records'),
        'development_fingerprint':manifest['development_fingerprint'],
        'sklearn_version_generation':manifest['sklearn_version_generation'],
        'rows_removed':0,'model_features_exported':False,
        'test_label_statistics_computed':False,
    }
    (result/'split_protocol.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return frames,metadata,summary


def plot_split_balance(result_dir='results/step_03'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(result_dir)
    balance=pd.read_csv(root/'train_validation_class_balance.csv')
    pivot=balance.pivot(index='split',columns='income',values='percent').reindex(['train','validation'])
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    x=np.arange(2);bottom=np.zeros(2)
    for label,color in zip(LABELS,['#2563EB','#EA8A23']):
        values=pivot[label].to_numpy()
        ax.bar(x,values,bottom=bottom,label=label,color=color,width=.5)
        for i,value in enumerate(values):
            ax.text(x[i],bottom[i]+value/2,f'{value:.2f}%',ha='center',va='center',color='white',fontsize=11)
        bottom+=values
    ax.set_xticks(x,['Train','Validation']);ax.set_ylim(0,105)
    ax.set_ylabel('Tỷ lệ trong từng tập (%)')
    ax.set_title('Phân phối nhãn sau khi chia theo nhóm')
    ax.legend(title='income',bbox_to_anchor=(1.02,1),loc='upper left')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(root/f'train_validation_balance.{ext}',bbox_inches='tight',dpi=170)
    plt.close(fig)


def load_saved_splits(project_dir='.'):
    """Đọc lại CSV phân chia với đúng kiểu và giá trị thiếu cho bước 4."""
    root=Path(project_dir)
    types={c:'Int64' for c in NUMERIC_COLUMNS}
    types.update({c:'string' for c in COLUMNS if c not in NUMERIC_COLUMNS})
    frames={}
    metadata={}
    for name in ['train','validation','test']:
        frame=pd.read_csv(root/'data/splits'/f'{name}.csv',dtype=types,
            keep_default_na=False,na_values=[''])
        if list(frame.columns)!=COLUMNS:
            raise ValueError(f'{name}: tên hoặc thứ tự cột không đúng.')
        info=pd.read_csv(root/'data/splits'/f'{name}_metadata.csv',
            dtype={'source_file':'string','source_row':'int64','source_id':'string','group_id':'string'},
            keep_default_na=False,na_values=[''])
        if len(frame)!=len(info) or not info['source_id'].is_unique:
            raise ValueError(f'{name}: metadata không khớp số dòng hoặc trùng source_id.')
        frames[name]=frame;metadata[name]=info
    return frames,metadata


def update_latex(result_dir='results/step_03',report_root='reports/latex'):
    import shutil
    result=Path(result_dir);report=Path(report_root)
    s=json.loads((result/'split_protocol.json').read_text(encoding='utf-8'))
    def num(x):return f'{int(x):,}'.replace(',','.')
    def dec(x,digits=2):return f'{float(x):.{digits}f}'.replace('.',',')
    (report/'chapters/chapter_4/4_2_tai_du_lieu_va_phan_chia_cac_tap.tex').write_text(r'''\section{Tải dữ liệu và phân chia các tập}
\label{sec:4_2}
Quy trình chuẩn bị bắt đầu từ các tệp UCI gốc, thống nhất cách biểu diễn dữ liệu và khóa vai trò train, validation, test trước khi fit bất kỳ bộ biến đổi hoặc mô hình nào. Các chỉ số dòng cùng mã nhóm được lưu để đối chiếu và tái lập ở các bước tiếp theo.

\input{chapters/chapter_4/4_2_1_tai_du_lieu_tu_nguon_cong_khai}
\input{chapters/chapter_4/4_2_2_chia_tap_huan_luyen_kiem_dinh_va_kiem_tra}
\input{chapters/chapter_4/4_2_3_ngan_ngua_ro_ri_du_lieu}
''',encoding='utf-8')
    target=report/'Images/chapter_4/4_2/4_2_2';target.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:shutil.copyfile(result/f'train_validation_balance.{ext}',target/f'train_validation_balance.{ext}')
    text=r'''\subsection{Chia tập huấn luyện, kiểm định và kiểm tra}
\label{subsec:4_2_2}
Hai tệp nguồn được giữ đúng vai trò ban đầu. Tệp \texttt{adult.data} là dữ liệu phát triển và được chia tiếp thành train và validation; \texttt{adult.test} được dành cho đánh giá cuối. Bước chia tập không điền thiếu, mã hóa, chuẩn hóa hoặc loại mẫu.

Các dòng có cùng toàn bộ 14 thuộc tính đầu vào được gán cùng một nhóm, trong đó giá trị thiếu được biểu diễn nhất quán. Nhãn \texttt{income} không nằm trong khóa nhóm. Cách chia dùng \texttt{StratifiedGroupKFold} với năm fold, bật trộn nhóm, seed 42 và lấy cố định fold số 0 làm validation~\cite{sklearn-sgkf}. Bốn fold còn lại làm train. Đây là một lần chia đã định trước, không lựa chọn fold hoặc seed theo điểm mô hình. Tỷ lệ số dòng chỉ xấp xỉ 80/20 vì nhóm được giữ nguyên.
'''
    text+=r'''
\begin{table}[htbp]
\centering\small
\caption{Phân chia dữ liệu đã khóa ở bước 3}\label{tab:split-stage3}
\begin{tabular}{lrrp{.32\textwidth}}
\toprule
Tập & Số mẫu & Tỷ lệ trong development & Vai trò \\
\midrule
'''
    text+=f"Train & {num(s['train_rows'])} & {dec(s['train_rows']/s['development_rows']*100)}\\% & Fit tiền xử lý và huấn luyện \\\\\n"
    text+=f"Validation & {num(s['validation_rows'])} & {dec(s['validation_rows']/s['development_rows']*100)}\\% & Đối chiếu cấu hình đã chọn \\\\\n"
    text+=f"Test UCI & {num(s['test_rows'])} & --- & Đánh giá cuối \\\\\n"
    text+=r'''\bottomrule
\end{tabular}
\end{table}
'''
    text+=f"\nTỷ lệ lớp \\texttt{{>50K}} là {dec(s['train_positive_percent'])}\\% trong train và {dec(s['validation_positive_percent'])}\\% trong validation, chênh lệch {dec(s['class_rate_difference_percentage_points'],4)} điểm phần trăm. Phần kiểm tra này chỉ mô tả train và validation, không lập phân phối nhãn test.\n"
    text+=r'''
\begin{figure}[htbp]
\centering
\includegraphics[width=.85\textwidth]{Images/chapter_4/4_2/4_2_2/train_validation_balance.pdf}
\caption{Phân phối nhãn của train và validation sau khi chia theo nhóm}
\label{fig:split-balance}
\end{figure}

Hai tập được xuất dưới dạng CSV cùng các tệp metadata chứa tệp nguồn, chỉ số dòng gốc và mã nhóm. Các chỉ số train/validation được khóa trong \texttt{results/step\_03/split\_indices.json}, kèm dấu vân tay của dữ liệu phát triển và phiên bản thư viện tạo split. Notebook nhúng manifest này để tái dùng chính xác tập chia giữa các môi trường; trước khi tái dùng, chương trình kiểm tra lại dữ liệu và cấu hình. Thứ tự dòng trong mỗi tập theo thứ tự nguồn.
'''
    (report/'chapters/chapter_4/4_2_2_chia_tap_huan_luyen_kiem_dinh_va_kiem_tra.tex').write_text(text,encoding='utf-8')

    text=r'''\subsection{Ngăn ngừa rò rỉ dữ liệu}
\label{subsec:4_2_3}
Kiểm tra tự động xác nhận train và validation không chung chỉ số dòng và không chung nhóm đầu vào. Hai tập hợp lại chứa đủ mọi dòng của \texttt{adult.data}, mỗi dòng đúng một lần; các hồ sơ trùng hoặc khác nhãn trong cùng nhóm được giữ cùng một phía. Không xóa hoặc gán lại nhãn những hồ sơ này.
'''
    text+=f"\nCó {num(s['development_unique_groups'])} nhóm đầu vào trong dữ liệu phát triển, với {num(s['train_unique_groups'])} nhóm ở train và {num(s['validation_unique_groups'])} nhóm ở validation. Số nhóm giao nhau là 0. Nhóm cùng đầu vào nhưng khác nhãn đã ghi nhận ở EDA cũng được kiểm tra để không bị tách sang hai phía.\n"
    text+=r'''
Tập test giữ nguyên số dòng, giá trị và thứ tự từ tệp UCI sau bước đọc dữ liệu. Test không được dùng để chọn seed, fold, quy tắc loại mẫu hoặc cấu hình tiền xử lý. Kiểm tra nhóm trong bước này tập trung vào ranh giới train/validation; chưa đối chiếu các hồ sơ phát triển với test để lọc dữ liệu.

Ở bước tiếp theo, imputer, encoder, scaler và PCA chỉ được fit trên train; validation và test được biến đổi bằng các đối tượng đã fit. Nếu thực hiện cross-validation bên trong train, các nhóm phải được truyền vào bộ chia và toàn bộ tiền xử lý phải được fit lại bên trong từng fold. Khi đã chốt cấu hình cuối, có thể fit lại trên train cùng validation trước lần đánh giá test, đồng thời lưu rõ phiên bản pipeline và các đặc trưng tương ứng.

EDA ở chương 3 mô tả toàn bộ dữ liệu phát triển trước khi chia nội bộ. Các thống kê dùng để fit tiền xử lý hoặc lựa chọn cấu hình phải được tính lại trên phần train tương ứng; các bảng EDA đó không được dùng làm trực tiếp các tham số biến đổi cho validation hoặc test.
'''
    (report/'chapters/chapter_4/4_2_3_ngan_ngua_ro_ri_du_lieu.tex').write_text(text,encoding='utf-8')

    protocol=r'''\section{Giao thức đánh giá và chỉ số lựa chọn}
\label{sec:5_2}
Giao thức chia dữ liệu được khóa ở bước 3: train dùng cho fit tiền xử lý và huấn luyện, validation dùng để đối chiếu cấu hình, test UCI dành cho đánh giá cuối. Nếu thử nhiều siêu tham số bằng cross-validation trên train, các fold phải giữ nguyên nhóm hồ sơ và fit riêng pipeline trong từng fold. Cấu hình cụ thể của các thí nghiệm sẽ được bổ sung sau khi xây dựng pipeline.

Chỉ số lựa chọn chính dự kiến là F1 của lớp \texttt{>50K}; precision và recall của cùng lớp được báo cáo để giải thích sự đánh đổi. Accuracy, macro-F1 và ROC-AUC được dùng làm chỉ số bổ sung. Lựa chọn này hướng đến việc theo dõi lớp thu nhập cao có ít mẫu hơn trong dữ liệu phát triển. Tại bước 3 chưa huấn luyện nên chưa có giá trị của các chỉ số mô hình.

Các lựa chọn về mô hình, siêu tham số và ngưỡng dự đoán phải được chốt bằng dữ liệu phát triển trước khi đọc kết quả đánh giá test. Báo cáo cuối sẽ phân biệt rõ giai đoạn chọn cấu hình và giai đoạn đánh giá test.
'''
    (report/'chapters/chapter_5/5_2_giao_thuc_danh_gia_va_chi_so_lua_chon.tex').write_text(protocol,encoding='utf-8')

    reference=report/'backmatter/references.tex'
    refs=reference.read_text(encoding='utf-8')
    if r'\bibitem{sklearn-sgkf}' not in refs:
        refs=refs.replace(r'\end{thebibliography}',r'''\bibitem{sklearn-sgkf}
Scikit-learn developers, \textit{StratifiedGroupKFold}, tài liệu chính thức.
\url{https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html}.
\end{thebibliography}''')
    reference.write_text(refs,encoding='utf-8')
    appendix=report/'backmatter/appendix_a/A_1_chay_toan_bo_notebook_tren_google_colab.tex'
    appendix.write_text(r'''\section{Chạy toàn bộ notebook trên Google Colab}
\label{sec:A_1}
Notebook chính là \texttt{notebooks/Adult\_BTL.ipynb}. Tải notebook lên Google Colab và chọn \texttt{Runtime > Run all}. Notebook tự chuẩn bị các module, tải dữ liệu công khai, chạy EDA trên dữ liệu phát triển, tái dùng manifest chia tập đã khóa và xuất các sản phẩm cùng ZIP LaTeX. Không cần mount dịch vụ cloud cá nhân hoặc tải module thủ công.

Bảng EDA nằm trong \texttt{results/step\_02/}; các tập chia và metadata nằm trong \texttt{data/splits/}; manifest cùng kiểm tra rò rỉ nằm trong \texttt{results/step\_03/}. Notebook nhúng cả nguồn LaTeX và manifest. Các sửa đổi tự thực hiện trong nguồn báo cáo cần được sao lưu trước khi chạy lại vì notebook dựng lại từ khung nhúng.

Các ô Python đã chạy cục bộ trên ZIP UCI gốc. Chưa xác nhận chạy trực tiếp trên dịch vụ Colab hoặc biên dịch toàn bộ tài liệu tại đây. Bước 3 chưa fit tiền xử lý, huấn luyện hoặc xuất ma trận đặc trưng mô hình.
''',encoding='utf-8')
    (report/'TIEN_DO.md').write_text('# Tiến độ sau bước 3\n\nGiữ chương 1 và chương 3. Đã thêm 4.2.2, 4.2.3, giao thức 5.2 và cập nhật hướng dẫn A.1. Chưa fit tiền xử lý, huấn luyện hoặc có kết quả mô hình. Chưa biên dịch toàn bộ PDF tại đây.\n',encoding='utf-8')
    return report
