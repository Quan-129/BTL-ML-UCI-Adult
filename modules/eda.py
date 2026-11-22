"""EDA mô tả trên adult.data gốc; không chỉnh sửa dữ liệu đầu vào."""
from pathlib import Path
import json
import textwrap
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .data_loader import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, COLUMNS

LABELS=['<=50K','>50K']
BLUE='#2563EB'
ORANGE='#EA8A23'
MISSING='(Thiếu)'


def _save_table(frame, path, index=True):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=index,encoding='utf-8-sig')


def _save_figure(fig, stem):
    fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'),bbox_inches='tight',dpi=170)
    plt.close(fig)


def _categories_from_names(names_path):
    """Đọc các miền phân loại theo adult.names thay vì suy ra từ test."""
    domains={}
    for line in Path(names_path).read_text(encoding='utf-8').splitlines():
        if line.startswith('|') or ':' not in line:
            continue
        name,spec=line.split(':',1)
        if name.strip() in CATEGORICAL_COLUMNS:
            domains[name.strip()]={x.strip() for x in spec.strip().rstrip('.').split(',')}
    return domains


def analyze_train(df_train, output_dir='results/step_02', names_path='data/adult.names'):
    """Xuất bảng và biểu đồ EDA chỉ từ DataFrame train được truyền vào."""
    if list(df_train.columns)!=COLUMNS or df_train.empty:
        raise ValueError('Cần df_train có đúng các cột Adult và ít nhất một dòng.')
    if df_train['income'].isna().any() or set(df_train['income'].unique())!=set(LABELS):
        raise ValueError('Nhãn train chưa chuẩn hóa hoặc bị thiếu.')
    original=df_train.copy(deep=True)
    out=Path(output_dir)
    table_dir=out/'tables'; figure_dir=out/'figures'
    table_dir.mkdir(parents=True,exist_ok=True)
    figure_dir.mkdir(parents=True,exist_ok=True)
    n=len(df_train)
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'savefig.facecolor':'white',
    })

    # 1. Kiểu dữ liệu, số mức giá trị, missing và nhãn.
    schema=pd.DataFrame({
        'dtype':df_train.dtypes.astype(str),
        'group':['numeric' if c in NUMERIC_COLUMNS else 'target' if c=='income' else 'categorical' for c in COLUMNS],
        'n_unique_non_missing':df_train.nunique(dropna=True),
        'missing_count':df_train.isna().sum(),
    })
    # pandas 2.x có thể dùng chung Index với df_train.columns; tạo Index mới khi đổi tên.
    schema.index = schema.index.set_names('column', inplace=False)
    _save_table(schema,table_dir/'schema.csv')
    counts=df_train['income'].value_counts().reindex(LABELS,fill_value=0)
    label_table=counts.rename('count').to_frame()
    label_table['percent']=label_table['count']/n*100
    label_table.index = label_table.index.set_names('income', inplace=False)
    _save_table(label_table,table_dir/'label_distribution.csv')

    missing=df_train.isna().sum().rename('missing_count').to_frame()
    missing['missing_percent']=missing['missing_count']/n*100
    missing.index = missing.index.set_names('column', inplace=False)
    _save_table(missing,table_dir/'missing_all_columns.csv')
    nonzero_missing=missing[missing['missing_count']>0].sort_values('missing_count',ascending=False)
    _save_table(nonzero_missing,table_dir/'missing_nonzero.csv')
    per_row=df_train.isna().sum(axis=1)
    missing_pattern=per_row.value_counts().sort_index().rename('row_count').to_frame()
    missing_pattern.index = missing_pattern.index.set_names('n_missing_columns', inplace=False)
    _save_table(missing_pattern,table_dir/'missing_per_row.csv')

    # 2. Tách rõ số dòng dư, số dòng tham gia và số nhóm trùng.
    feature_columns=COLUMNS[:-1]
    exact_extra=int(df_train.duplicated(keep='first').sum())
    exact_involved=int(df_train.duplicated(keep=False).sum())
    exact_group_count=int(df_train.loc[df_train.duplicated(keep=False)].drop_duplicates().shape[0])
    groups=df_train.groupby(feature_columns,dropna=False,sort=False,observed=True)['income'].agg(n_rows='size',n_labels='nunique')
    duplicate_groups=groups.loc[groups['n_rows']>1].reset_index()
    conflicts=groups.loc[groups['n_labels']>1].reset_index()
    feature_extra=int(df_train.duplicated(subset=feature_columns,keep='first').sum())
    feature_involved=int(df_train.duplicated(subset=feature_columns,keep=False).sum())
    duplicate_summary=pd.DataFrame([
        {'definition':'all_15_columns','extra_rows':exact_extra,'involved_rows':exact_involved,'groups':exact_group_count},
        {'definition':'14_features_only','extra_rows':feature_extra,'involved_rows':feature_involved,'groups':len(duplicate_groups)},
    ])
    _save_table(duplicate_summary,table_dir/'duplicate_summary.csv',index=False)
    _save_table(duplicate_groups,table_dir/'duplicate_feature_groups.csv',index=False)
    _save_table(conflicts,table_dir/'conflicting_label_groups.csv',index=False)

    # 3. Các miền số được khai báo rõ; không tự sửa các trường hợp không hợp lệ.
    numeric_rules={
        'age':('> 16',lambda x:x>16),
        'fnlwgt':('> 1',lambda x:x>1),
        'education-num':('1 to 16',lambda x:x.between(1,16)),
        'capital-gain':('>= 0',lambda x:x>=0),
        'capital-loss':('>= 0',lambda x:x>=0),
        'hours-per-week':('> 0 and <= 168',lambda x:x.between(1,168)),
    }
    validity=[]
    for column,(rule,predicate) in numeric_rules.items():
        series=df_train[column]
        invalid=(~predicate(series)).fillna(False)&series.notna()
        validity.append({'column':column,'rule':rule,'invalid_count':int(invalid.sum()),'missing_count':int(series.isna().sum())})
    domains=_categories_from_names(names_path)
    if set(domains)!=set(CATEGORICAL_COLUMNS):
        raise ValueError('Không đọc được đầy đủ miền phân loại từ adult.names.')
    for column in CATEGORICAL_COLUMNS:
        invalid=df_train[column].notna()&~df_train[column].isin(domains[column])
        validity.append({'column':column,'rule':'listed in adult.names','invalid_count':int(invalid.sum()),'missing_count':int(df_train[column].isna().sum())})
    validity=pd.DataFrame(validity)
    _save_table(validity,table_dir/'domain_checks.csv',index=False)

    # 4. Thống kê số và các cờ IQR chỉ phục vụ mô tả.
    numeric_stats=df_train[NUMERIC_COLUMNS].astype(float).describe().T
    numeric_stats['zero_count']=(df_train[NUMERIC_COLUMNS]==0).sum()
    numeric_stats['zero_percent']=numeric_stats['zero_count']/n*100
    numeric_stats['skewness']=df_train[NUMERIC_COLUMNS].astype(float).skew()
    numeric_stats.index = numeric_stats.index.set_names('column', inplace=False)
    _save_table(numeric_stats,table_dir/'numeric_summary.csv')
    iqr_rows=[]
    for column in NUMERIC_COLUMNS:
        series=df_train[column].dropna().astype(float)
        q1=float(series.quantile(.25)); q3=float(series.quantile(.75)); iqr=q3-q1
        lower=q1-1.5*iqr; upper=q3+1.5*iqr
        flagged=(series<lower)|(series>upper)
        iqr_rows.append({'column':column,'q1':q1,'q3':q3,'iqr':iqr,'lower':lower,'upper':upper,
            'flagged_count':int(flagged.sum()),'flagged_percent':float(flagged.mean()*100)})
    iqr_table=pd.DataFrame(iqr_rows).set_index('column')
    _save_table(iqr_table,table_dir/'iqr_flags.csv')

    # 5. Thống kê đủ mọi mức phân loại, gồm một mức riêng khi thiếu.
    category_tables={}
    for column in CATEGORICAL_COLUMNS:
        values=df_train[column].fillna(MISSING)
        freq=values.value_counts().rename('count').to_frame()
        freq['percent']=freq['count']/n*100
        positives=values[df_train['income'].eq('>50K')].value_counts()
        freq['positive_count']=positives.reindex(freq.index,fill_value=0)
        freq['positive_rate_percent']=freq['positive_count']/freq['count']*100
        freq.index = freq.index.set_names('category', inplace=False)
        category_tables[column]=freq
        _save_table(freq,table_dir/f'categorical_{column}.csv')

    # 6. Biểu đồ nhãn và missing.
    fig,ax=plt.subplots(figsize=(7,4.2))
    bars=ax.bar(LABELS,counts.to_numpy(),color=[BLUE,ORANGE],width=.55)
    for bar,value in zip(bars,counts):
        ax.annotate(f'{value:,}\n{value/n*100:.2f}%',(bar.get_x()+bar.get_width()/2,value),xytext=(0,5),textcoords='offset points',ha='center',va='bottom')
    ax.set(title='Phân phối nhãn trên train gốc',xlabel='Nhãn income',ylabel='Số mẫu',ylim=(0,float(counts.max())*1.25))
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); fig.tight_layout()
    _save_figure(fig,figure_dir/'label_distribution')

    fig,ax=plt.subplots(figsize=(7,4.2))
    if len(nonzero_missing):
        bars=ax.bar(nonzero_missing.index,nonzero_missing['missing_percent'],color=BLUE,width=.55)
        for bar,value in zip(bars,nonzero_missing['missing_count']):
            ax.annotate(f'{bar.get_height():.2f}%\n(n={int(value):,})',(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,5),textcoords='offset points',ha='center',va='bottom')
        ax.set_ylim(0,float(nonzero_missing['missing_percent'].max())*1.3)
    else: ax.text(.5,.5,'Không có giá trị thiếu',ha='center',transform=ax.transAxes)
    ax.set(title='Giá trị thiếu trên train gốc',xlabel='Thuộc tính',ylabel='Tỷ lệ thiếu (%)')
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); fig.tight_layout()
    _save_figure(fig,figure_dir/'missing_values')

    fig,axes=plt.subplots(3,2,figsize=(11,9.2))
    for ax,column in zip(axes.flat,NUMERIC_COLUMNS):
        values=df_train[column].dropna().astype(float)
        ax.hist(values,bins=30,color=BLUE,alpha=.85)
        ax.set_title(column); ax.set_ylabel('Số mẫu'); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    fig.suptitle('Phân phối 6 thuộc tính số trên train gốc',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.97]); _save_figure(fig,figure_dir/'numeric_histograms')

    fig,axes=plt.subplots(3,2,figsize=(11,8.5))
    for ax,column in zip(axes.flat,NUMERIC_COLUMNS):
        ax.boxplot(df_train[column].dropna().astype(float).to_numpy(),vert=False,
            flierprops={'markersize':2,'alpha':.25},medianprops={'color':ORANGE})
        ax.set_title(column); ax.set_yticks([]); ax.grid(axis='x',alpha=.15)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    fig.suptitle('Boxplot trên thang đo gốc; chưa loại điểm ngoài IQR',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.97]); _save_figure(fig,figure_dir/'numeric_boxplots')

    for page,columns in enumerate([CATEGORICAL_COLUMNS[:4],CATEGORICAL_COLUMNS[4:]],1):
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        for ax,column in zip(axes.flat,columns):
            full=category_tables[column]['count']
            plotted=full.head(10).copy()
            if len(full)>10:plotted.loc['Các mức còn lại']=full.iloc[10:].sum()
            plotted=plotted.iloc[::-1]
            ax.barh([textwrap.fill(str(x),23) for x in plotted.index],plotted.to_numpy(),color=BLUE)
            ax.set_title(column+(' (10 mức phổ biến + phần còn lại)' if len(full)>10 else ''))
            ax.set_xlabel('Số mẫu'); ax.tick_params(axis='y',labelsize=8)
            ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
        fig.suptitle(f'Phân phối biến phân loại trên train — nhóm {page}',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.97]); _save_figure(fig,figure_dir/f'categorical_counts_{page}')

    edu=category_tables['education'].copy()
    edu_map=df_train.groupby('education',observed=True)['education-num'].median()
    edu['order']=edu_map.reindex(edu.index)
    edu=edu.sort_values('order')
    fig,ax=plt.subplots(figsize=(9.5,6.5))
    bars=ax.barh(edu.index,edu['positive_rate_percent'],color=BLUE)
    for bar,support in zip(bars,edu['count']):
        ax.annotate(f'{bar.get_width():.1f}%  (n={int(support):,})',(bar.get_width(),bar.get_y()+bar.get_height()/2),xytext=(4,0),textcoords='offset points',va='center',fontsize=8)
    ax.set(title='Tỷ lệ nhãn >50K theo học vấn trên train',xlabel='Tỷ lệ >50K trong từng nhóm (%)',xlim=(0,100))
    ax.xaxis.set_major_formatter(PercentFormatter(100)); ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout(); _save_figure(fig,figure_dir/'education_income_rate')

    summary={
        'stage':2,'eda_population':'adult.data original training split only',
        'n_rows':n,'n_features':14,'n_numeric':6,'n_categorical':8,
        'label_counts':{k:int(v) for k,v in counts.items()},
        'label_percent':{k:float(v) for k,v in label_table['percent'].items()},
        'majority_to_minority_ratio':float(counts.max()/counts.min()),
        'missing_counts':{k:int(v) for k,v in nonzero_missing['missing_count'].items()},
        'rows_with_missing':int(per_row.gt(0).sum()),
        'rows_with_missing_percent':float(per_row.gt(0).mean()*100),
        'exact_duplicate_extra':exact_extra,'exact_duplicate_involved':exact_involved,'exact_duplicate_groups':exact_group_count,
        'feature_duplicate_extra':feature_extra,'feature_duplicate_involved':feature_involved,'feature_duplicate_groups':len(duplicate_groups),
        'conflicting_label_groups':len(conflicts),
        'conflicting_label_rows':int(conflicts['n_rows'].sum()) if len(conflicts) else 0,
        'invalid_domain_total':int(validity['invalid_count'].sum()),
        'iqr_flagged_counts':{k:int(v) for k,v in iqr_table['flagged_count'].items()},
        'zero_percent':{k:float(v) for k,v in numeric_stats['zero_percent'].items()},
        'figures':[p.stem for p in sorted(figure_dir.glob('*.pdf'))],
        'input_mutated':False,'test_used_for_eda':False,'rows_removed':0,
    }
    pd.testing.assert_frame_equal(df_train,original)
    (out/'eda_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def _escape(value):
    value=str(value)
    for old,new in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:
        value=value.replace(old,new)
    return value


def update_latex(eda_dir='results/step_02', report_root='reports/latex'):
    """Điền các mục EDA và chèn hình từ chính kết quả đã xuất."""
    eda=Path(eda_dir); report=Path(report_root)
    s=json.loads((eda/'eda_summary.json').read_text(encoding='utf-8'))
    tables=eda/'tables'
    numeric=pd.read_csv(tables/'numeric_summary.csv',index_col=0)
    iqr=pd.read_csv(tables/'iqr_flags.csv',index_col=0)
    missing=pd.read_csv(tables/'missing_nonzero.csv',index_col=0)
    categories={c:pd.read_csv(tables/f'categorical_{c}.csv',index_col=0) for c in CATEGORICAL_COLUMNS}

    def write(filename,text):
        path=report/'chapters/chapter_3'/filename
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    def num(x):return f'{int(x):,}'.replace(',','.')
    def decimal(x,digits=2):return f'{float(x):.{digits}f}'.replace('.',',')
    def figure(name,folder,caption,label,width='.92'):
        target=report/'Images/chapter_3'/folder;target.mkdir(parents=True,exist_ok=True)
        for ext in ['.pdf','.png']:shutil.copyfile(eda/'figures'/(name+ext),target/(name+ext))
        return ('\n\\begin{figure}[htbp]\n\\centering\n'
            f'\\includegraphics[width={width}\\textwidth,height=.72\\textheight,keepaspectratio]{{Images/chapter_3/{folder}/{name}.pdf}}\n'
            f'\\caption{{{caption}}}\n\\label{{fig:{label}}}\n\\end{{figure}}\n')
    def table(headers,rows,caption,label,align):
        body='\n'.join(' & '.join(_escape(v) for v in row)+r' \\' for row in rows)
        return ('\n\\begin{table}[htbp]\n\\centering\n\\small\n'
            f'\\caption{{{caption}}}\\label{{tab:{label}}}\n\\begin{{tabular}}{{{align}}}\n\\toprule\n'
            +' & '.join(headers)+r' \\'+'\n\\midrule\n'+body+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    write('3_3_kiem_tra_chat_luong_du_lieu.tex',r'''\section{Kiểm tra chất lượng dữ liệu}
\label{sec:3_3}
Phần khảo sát này chỉ sử dụng toàn bộ tệp \texttt{adult.data} gốc, được xem là tập dữ liệu phát triển trước khi chia train và validation ở bước tiếp theo. Tập \texttt{adult.test} được giữ riêng và không dùng để tính các bảng hoặc biểu đồ EDA. Các thống kê là mô tả dữ liệu; bước này chưa quyết định siêu tham số, chưa điền thiếu và chưa loại bản ghi.

Các quy tắc kiểm tra bao gồm tuổi lớn hơn 16, trọng số khảo sát lớn hơn 1, mã học vấn từ 1 đến 16, các khoản vốn không âm và số giờ làm việc từ 1 đến 168 mỗi tuần. Hai điều kiện đầu và điều kiện số giờ dương xuất phát từ mô tả trích xuất dữ liệu; cận 168 là kiểm tra tính hợp lý theo số giờ trong tuần. Miền phân loại được đọc từ \texttt{adult.names}. Dữ liệu thiếu được thống kê riêng, không bị gộp thành giá trị ngoài miền.
'''+f"\nTheo các quy tắc đã khai báo, tổng số ô ngoài miền phát hiện được là {num(s['invalid_domain_total'])}. Kết quả này chỉ phản ánh các kiểm tra trên, không chứng minh dữ liệu hoàn toàn không có lỗi.\n"+r'''
\input{chapters/chapter_3/3_3_1_gia_tri_thieu_va_gia_tri_khong_hop_le}
\input{chapters/chapter_3/3_3_2_ban_ghi_trung_lap}
\input{chapters/chapter_3/3_3_3_gia_tri_ngoai_lai}
''')
    text=r'''\subsection{Giá trị thiếu và giá trị không hợp lệ}
\label{subsec:3_3_1}
'''
    text+=f"Tập phát triển gốc có {num(s['rows_with_missing'])} dòng thiếu ít nhất một thuộc tính, chiếm {decimal(s['rows_with_missing_percent'])}\\%. Giá trị thiếu chỉ xuất hiện ở các cột trong Bảng~\\ref{{tab:missing-train}}. Một dòng có thể thiếu nhiều thuộc tính, nên tổng số ô thiếu không phải số dòng thiếu.\n"
    text+=table(['Thuộc tính','Số ô thiếu','Tỷ lệ (\\%)'],[(c,num(r['missing_count']),decimal(r['missing_percent'])) for c,r in missing.iterrows()], 'Giá trị thiếu trên adult.data gốc','missing-train','lrr')
    text+=figure('missing_values','3_3/3_3_1','Tỷ lệ thiếu tính trên toàn bộ adult.data gốc','missing-train')
    text+='\nCác ký hiệu thiếu đã được nhận diện khi đọc dữ liệu. Việc lựa chọn cách điền thiếu sẽ được thực hiện trong pipeline và các giá trị điền được học từ train của từng lần chia, thay vì tính trước trên toàn bộ dữ liệu phát triển.\n'
    write('3_3_1_gia_tri_thieu_va_gia_tri_khong_hop_le.tex',text)

    text=r'''\subsection{Bản ghi trùng lặp}
\label{subsec:3_3_2}
Hai định nghĩa được khảo sát riêng: trùng toàn bộ 15 cột, gồm cả nhãn, và trùng 14 thuộc tính đầu vào. Số dòng dư là số dòng lặp ngoài lần xuất hiện đầu tiên; số dòng tham gia bao gồm tất cả các thành viên của những nhóm trùng. Giá trị thiếu được xem là cùng một trạng thái khi đối chiếu các hồ sơ.
'''
    text+=table(['Định nghĩa','Dòng dư','Dòng tham gia','Số nhóm'],[
        ('Toàn bộ 15 cột',num(s['exact_duplicate_extra']),num(s['exact_duplicate_involved']),num(s['exact_duplicate_groups'])),
        ('Chỉ 14 đầu vào',num(s['feature_duplicate_extra']),num(s['feature_duplicate_involved']),num(s['feature_duplicate_groups']))],
        'Các dạng trùng lặp trong adult.data gốc','duplicates-train','lrrr')
    text+=f"\nCó {num(s['conflicting_label_groups'])} nhóm đầu vào giống nhau nhưng có nhãn khác nhau, gồm {num(s['conflicting_label_rows'])} dòng. "
    text+='Các trường hợp này được lưu để kiểm tra, chưa bị xóa hoặc gán lại nhãn. Ở bước chia dữ liệu, cần cân nhắc giữ những hồ sơ có cùng đầu vào trong cùng một phía để hạn chế việc các bản sao xuất hiện đồng thời ở train và validation. Phép kiểm tra ở đây chỉ đối chiếu nội bộ dữ liệu phát triển, chưa so sánh với test.\n'
    write('3_3_2_ban_ghi_trung_lap.tex',text)

    text=r'''\subsection{Giá trị ngoại lai}
\label{subsec:3_3_3}
Các điểm ngoài khoảng $[Q_1-1{,}5\,IQR,\ Q_3+1{,}5\,IQR]$, với $IQR=Q_3-Q_1$, được đánh dấu để khảo sát. Đây là quy tắc mô tả theo phân vị, không phải kết luận một giá trị nhập sai và không phải quy tắc loại dòng trong pipeline.
'''
    text+=table(['Thuộc tính','$Q_1$','$Q_3$','Số điểm gắn cờ','Tỷ lệ (\\%)'],[(c,decimal(r['q1'],1),decimal(r['q3'],1),num(r['flagged_count']),decimal(r['flagged_percent'])) for c,r in iqr.iterrows()], 'Các cờ IQR trên sáu thuộc tính số','iqr-train','lrrrr')
    text+='\nĐặc biệt, các biến khoản vốn có nhiều giá trị bằng 0. Khi cả hai tứ phân vị bằng 0, IQR cũng bằng 0 và mọi giá trị dương sẽ nằm ngoài hàng rào trên. Vì vậy, không dùng riêng tiêu chuẩn này để loại những hồ sơ có khoản vốn dương.\n'
    text+=figure('numeric_boxplots','3_3/3_3_3','Boxplot riêng cho từng thuộc tính số trên thang đo gốc','numeric-boxplots')
    write('3_3_3_gia_tri_ngoai_lai.tex',text)

    text=r'''\section{Phân phối nhãn và mức độ mất cân bằng}
\label{sec:3_4}
'''
    text+=table(['Nhãn','Số mẫu','Tỷ lệ (\\%)'],[(c,num(s['label_counts'][c]),decimal(s['label_percent'][c])) for c in LABELS], 'Phân phối nhãn trên adult.data gốc','label-train','lrr')
    text+=f"\nLớp \\texttt{{<=50K}} có nhiều mẫu hơn lớp \\texttt{{>50K}}, với tỷ lệ số lượng hai lớp là khoảng {decimal(s['majority_to_minority_ratio'])}:1. "
    text+='Mức chênh lệch này cho thấy cần theo dõi khả năng nhận diện lớp thu nhập cao khi đánh giá mô hình; accuracy cần được đọc cùng precision, recall và F1 của lớp quan tâm. Các chỉ số mô hình chưa được tính trong bước EDA.\n'
    text+=figure('label_distribution','3_4','Số lượng và tỷ lệ hai nhãn trên adult.data gốc','label-distribution')
    write('3_4_phan_phoi_nhan_va_muc_do_mat_can_bang.tex',text)

    text=r'''\section{Phân tích các biến số}
\label{sec:3_5}
Sáu thuộc tính số được khảo sát trên thang đo gốc. Bảng thống kê đầy đủ, gồm độ lệch chuẩn và các phân vị, được lưu trong \texttt{results/step\_02/tables/numeric\_summary.csv}.
'''
    text+=table(['Thuộc tính','Trung bình','Trung vị','Nhỏ nhất','Lớn nhất'],[(c,decimal(r['mean']),decimal(r['50%'],1),num(r['min']),num(r['max'])) for c,r in numeric.iterrows()], 'Thống kê mô tả thuộc tính số trên adult.data gốc','numeric-summary','lrrrr')
    text+=f"\nTỷ lệ bằng 0 của \\texttt{{capital-gain}} là {decimal(s['zero_percent']['capital-gain'])}\\%, còn \\texttt{{capital-loss}} là {decimal(s['zero_percent']['capital-loss'])}\\%. "
    text+='Do đó, trung vị và trung bình cần được đọc cùng biểu đồ, đặc biệt đối với hai phân phối lệch phải này. Các thuộc tính có đơn vị và miền giá trị khác nhau; việc chuẩn hóa sẽ được thử trong pipeline. Thống kê EDA ở đây không được dùng làm các tham số chuẩn hóa cho dữ liệu validation hoặc test.\n'
    text+=figure('numeric_histograms','3_5','Histogram của sáu thuộc tính số; không loại điểm cực trị','numeric-histograms')
    write('3_5_phan_tich_cac_bien_so.tex',text)

    text=r'''\section{Phân tích các biến phân loại}
\label{sec:3_6}
Mỗi thuộc tính được thống kê đủ các mức quan sát trên dữ liệu phát triển, với một mức riêng cho giá trị thiếu. Với biến có trên mười mức, biểu đồ gộp phần còn lại để dễ đọc; các bảng CSV vẫn giữ đầy đủ mọi mức.
'''
    dominant=[]
    for c in CATEGORICAL_COLUMNS:
        freq=categories[c]; dominant.append((c,len(freq),str(freq.index[0]),decimal(freq.iloc[0]['percent'])))
    text+=table(['Thuộc tính','Số mức kể cả thiếu','Mức phổ biến nhất','Tỷ lệ (\\%)'],dominant,'Tóm tắt các thuộc tính phân loại trên adult.data gốc','categorical-summary','lp{.16\\textwidth}p{.28\\textwidth}r')
    text+=figure('categorical_counts_1','3_6','Phân phối workclass, education, marital-status và occupation','categorical-1')
    text+=figure('categorical_counts_2','3_6','Phân phối relationship, race, sex và native-country','categorical-2')
    text+='\nBiểu đồ theo học vấn bổ sung tỷ lệ nhãn \\texttt{>50K} trong từng nhóm và số mẫu làm mẫu số. Đây là mối liên hệ mô tả trên dữ liệu quan sát, không được diễn giải thành tác động nhân quả. Khi so sánh tỷ lệ cần chú ý các nhóm có ít mẫu; các tỷ lệ này chưa được dùng để chọn siêu tham số hoặc xếp hạng mô hình.\n'
    text+=figure('education_income_rate','3_6','Tỷ lệ lớp thu nhập cao theo học vấn, kèm số mẫu của từng nhóm','education-income-rate')
    write('3_6_phan_tich_cac_bien_phan_loai.tex',text)

    text=r'''\section{Nhận xét EDA và định hướng xử lý}
\label{sec:3_7}
Kết quả khảo sát định hướng các bước thực hiện tiếp theo như sau:
\begin{itemize}
  \item Giá trị thiếu nằm ở các biến phân loại. Pipeline sẽ cần cách điền thiếu phù hợp và chỉ fit cách điền trên train của từng lần chia.
  \item Các nhóm hồ sơ trùng đầu vào cần được xét khi tổ chức train và validation. Bước EDA chưa loại trùng hoặc giải quyết các nhóm khác nhãn.
  \item Sự chênh lệch số mẫu giữa hai nhãn cần được phản ánh trong cách chia tập và lựa chọn chỉ số đánh giá; bước này chưa áp dụng tái lấy mẫu hay trọng số lớp.
  \item Các biến số có miền giá trị khác nhau và hai khoản vốn có nhiều giá trị bằng 0. Sẽ so sánh các lựa chọn chuẩn hóa; không tự loại điểm chỉ vì bị gắn cờ IQR.
  \item Các biến phân loại có số mức khác nhau. Việc mã hóa và xử lý mức chưa thấy sẽ được cấu hình trong pipeline.
\end{itemize}

EDA này mô tả \texttt{adult.data} trước khi chia nội bộ. Sau khi chốt tập validation, các thống kê dùng để fit tiền xử lý và các kiểm tra phục vụ lựa chọn cấu hình phải được tính lại trên phần train tương ứng. Tập test vẫn dành cho đánh giá cuối, sau khi đã chốt các lựa chọn. Toàn bộ dòng dữ liệu gốc được giữ nguyên ở bước 2.
'''
    write('3_7_nhan_xet_eda_va_dinh_huong_xu_ly.tex',text)
    appendix=report/'backmatter/appendix_a/A_1_chay_toan_bo_notebook_tren_google_colab.tex'
    appendix.parent.mkdir(parents=True,exist_ok=True)
    appendix.write_text(r'''\section{Chạy toàn bộ notebook trên Google Colab}
\label{sec:A_1}
Notebook chính là \texttt{notebooks/Adult\_BTL.ipynb}. Người dùng tải notebook lên Google Colab và chọn \texttt{Runtime > Run all}. Notebook tự chuẩn bị hai module hỗ trợ, tải dữ liệu UCI, chạy EDA trên tập phát triển gốc và xuất bảng, hình cùng nguồn LaTeX cập nhật. Không cần tải module thủ công hoặc mount dịch vụ cloud cá nhân.

Các bảng EDA được lưu ở \texttt{results/step\_02/tables/}, biểu đồ được lưu dưới cả định dạng PDF và PNG ở \texttt{results/step\_02/figures/}. Các hình dùng trong báo cáo được sao chép vào các thư mục ảnh tương ứng của chương 3. File ZIP LaTeX được tạo trong \texttt{reports/} để tải lên Overleaf.

Các ô Python đã được chạy cục bộ trên ZIP gốc UCI, sử dụng dữ liệu đã tải hợp lệ. Chưa xác nhận chạy trực tiếp trên dịch vụ Colab hoặc biên dịch toàn bộ tài liệu trong môi trường này. Bước 2 chưa chia lại dữ liệu, chưa fit bộ tiền xử lý, chưa huấn luyện hoặc đánh giá mô hình.
''',encoding='utf-8')
    (report/'TIEN_DO.md').write_text(
        '# Tiến độ sau bước 2\n\nĐã có chương 1; toàn bộ chương 3 gồm dữ liệu, EDA và các biểu đồ từ dữ liệu train thật; mục 4.2.1; phụ lục A.1; tài liệu tham khảo. Các phần lý thuyết, pipeline đầy đủ, thiết kế thí nghiệm và kết quả mô hình tiếp tục được bổ sung ở các bước sau.\n\nChưa biên dịch toàn bộ LaTeX tại đây.\n',encoding='utf-8')
    return report
