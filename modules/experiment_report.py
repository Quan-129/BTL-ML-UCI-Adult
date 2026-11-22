"""Cập nhật các mục LaTeX từ bảng thực nghiệm bước 5 đã chạy."""
from pathlib import Path
import json,shutil
import pandas as pd

def update_experiment_report(result_dir='results/step_05',report_root='reports/latex'):
    result=Path(result_dir);report=Path(report_root)
    summary=pd.read_csv(result/'cv_summary.csv')
    validation=pd.read_csv(result/'validation_summary.csv')
    chosen=json.loads((result/'selected_model.json').read_text(encoding='utf-8'))
    def write(folder,prefix,text):
        matches=list((report/folder).glob(prefix+'*.tex'))
        if len(matches)!=1:raise ValueError((folder,prefix,matches))
        matches[0].write_text(text,encoding='utf-8')
    def esc(value):return str(value).replace('_',r'\_').replace('%',r'\%')
    def num(x):return f'{x:.4f}'.replace('.',',')
    def table(data,caption,label):
        lines=[r'\begin{table}[htbp]',r'\centering\scriptsize',r'\caption{'+caption+r'}\label{'+label+'}',
               r'\begin{tabular}{lrrrr}',r'\toprule',r'Cấu hình & F1 CV & SD & Precision & Recall \\',r'\midrule']
        for _,r in data.iterrows():
            lines.append(esc(r.candidate_id)+' & '+' & '.join(num(r[c]) for c in ['f1_mean','f1_std','precision_mean','recall_mean'])+r' \\')
        return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    def figure(stem,section,caption):
        dest=report/'Images/chapter_6'/section;dest.mkdir(parents=True,exist_ok=True)
        for ext in ['png','pdf']:shutil.copyfile(result/f'{stem}.{ext}',dest/f'{stem}.{ext}')
        return '\n'.join([r'\begin{figure}[htbp]',r'\centering',r'\includegraphics[width=.95\textwidth]{Images/chapter_6/'+section+'/'+stem+'.pdf}',
            r'\caption{'+caption+'}',r'\label{fig:'+stem.replace('_','-')+'}',r'\end{figure}'])
    c2='chapters/chapter_2';c4='chapters/chapter_4';c5='chapters/chapter_5';c6='chapters/chapter_6'
    write(c2,'2_4_cac_',r'''\section{Các mô hình học máy truyền thống}
\label{sec:2_4}
Ba họ mô hình được sử dụng gồm Logistic Regression, SVM tuyến tính và Random Forest. Chúng đại diện cho mô hình xác suất tuyến tính, mô hình phân tách biên và tổ hợp cây quyết định. Cùng một quy trình điền thiếu, mã hóa và chuẩn hóa được đặt trước mỗi bộ phân loại để so sánh trên các fold giống nhau.
\input{chapters/chapter_2/2_4_1_logistic_regression}
\input{chapters/chapter_2/2_4_2_support_vector_machine}
\input{chapters/chapter_2/2_4_3_random_forest}
''')
    write(c2,'2_4_1_',r'''\subsection{Logistic Regression}
\label{subsec:2_4_1}
Với vector đặc trưng $\mathbf{x}$, mô hình ước lượng xác suất lớp dương bằng
\[
p(y=1\mid\mathbf{x})=\sigma(\mathbf{w}^{T}\mathbf{x}+b),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
\]
Tham số được học bằng cách giảm hàm mất mát logarit có phạt L2. Tham số $C$ điều khiển nghịch đảo mức regularization: giá trị nhỏ tương ứng với phạt mạnh hơn trong cùng cách triển khai. Thí nghiệm sử dụng bộ giải lbfgs và thử $C\in\{0{,}1;1;10\}$~\cite{sklearn-logistic}. Nhãn dự đoán dùng ngưỡng xác suất 0,5; không tối ưu ngưỡng trên validation hoặc test.

Mô hình có thời gian huấn luyện tương đối thấp và tạo mốc so sánh dễ giải thích. Tuy nhiên, một biên tuyến tính trên đặc trưng đã mã hóa có thể chưa biểu diễn đầy đủ các quan hệ phi tuyến giữa các thuộc tính của Adult.
''')
    write(c2,'2_4_2_',r'''\subsection{Support Vector Machine}
\label{subsec:2_4_2}
SVM tuyến tính tìm một siêu phẳng phân tách các lớp. Với nhãn quy đổi $t_i\in\{-1,+1\}$ và $f(\mathbf{x})=\mathbf{w}^{T}\mathbf{x}+b$, mất mát squared hinge có dạng $\max(0,1-t_i f(\mathbf{x}_i))^2$. Thành phần phạt L2 kiểm soát độ lớn tham số, còn $C$ điều chỉnh sự đánh đổi giữa phạt và sai số trên dữ liệu huấn luyện.

Bài làm sử dụng \texttt{LinearSVC}, không thử kernel RBF. Vì số mẫu lớn hơn số đặc trưng, bộ giải được cấu hình \texttt{dual=False}. Khảo sát $C\in\{0{,}1;1;10\}$ trên cùng các fold. Dự đoán nhãn dùng dấu của decision score, còn ROC-AUC dùng trực tiếp decision score; các điểm này không được gọi là xác suất~\cite{sklearn-linearsvc}.
''')
    write(c2,'2_4_3_',r'''\subsection{Random Forest}
\label{subsec:2_4_3}
Random Forest kết hợp nhiều cây quyết định được học trên các mẫu bootstrap và các tập thuộc tính ứng viên ngẫu nhiên. Trong triển khai này, xác suất lớp dương là trung bình xác suất từ các cây: $\hat p(1\mid\mathbf{x})=B^{-1}\sum_{b=1}^{B}p_b(1\mid\mathbf{x})$. Mô hình có thể biểu diễn tương tác phi tuyến mà không giả định một biên phân tách tuyến tính~\cite{sklearn-rf}.

Thí nghiệm dùng 150 cây, \texttt{max\_features=sqrt} và ba cấu hình: độ sâu không giới hạn với lá tối thiểu 1 mẫu; độ sâu không giới hạn với lá tối thiểu 3 mẫu; độ sâu tối đa 20 với lá tối thiểu 3 mẫu. Các thiết lập khác được giữ cố định. Lá nhỏ và cây sâu có thể tăng khả năng khớp train nhưng cũng tăng chi phí bộ nhớ và nguy cơ quá khớp; kết quả CV được dùng để lựa chọn trong phạm vi đã khảo sát.
''')
    write(c2,'2_5_',r'''\section{Mạng nơ-ron nhiều lớp MLP}
\label{sec:2_5}
MLP truyền đầu vào qua các lớp kết nối đầy đủ. Với $\mathbf{h}^{(0)}=\mathbf{x}$, một lớp ẩn thực hiện $\mathbf{h}^{(l)}=\operatorname{ReLU}(W^{(l)}\mathbf{h}^{(l-1)}+\mathbf{b}^{(l)})$. Đầu ra nhị phân sử dụng hàm logistic để ước lượng xác suất lớp dương. Các tham số được cập nhật bằng Adam dựa trên mất mát logarit có regularization L2~\cite{sklearn-mlp}.

Hai cấu hình được so sánh là một lớp ẩn 64 nút với $\alpha=10^{-4}$ và hai lớp ẩn 64--32 nút với $\alpha=10^{-3}$. Đây là hai cấu hình kết hợp; vì kiến trúc và mức phạt cùng thay đổi, không quy mọi khác biệt kết quả chỉ cho số lớp. Batch size bằng 256, learning rate ban đầu 0,001 và tối đa 300 epoch. Không bật early stopping với một tập nội bộ ngẫu nhiên để tránh phát sinh cách chia không theo nhóm; tiêu chí dừng theo training loss được áp dụng trong phạm vi phần train của mỗi fold.

Sau khi hai cấu hình trên phát sinh cảnh báo đạt giới hạn epoch trong pilot, bổ sung cấu hình 32 nút với $\alpha=0{,}01$ và 64 nút với $\alpha=0{,}1$. Hai cấu hình bổ sung dùng tol $10^{-3}$, chờ 10 epoch, thay cho tol $10^{-4}$ và chờ 20 epoch ban đầu. Cả bốn cấu hình cùng được báo cáo; không dùng validation hoặc test để quyết định mở rộng này.

MLP là phần mở rộng để đối chiếu với pipeline truyền thống. Mạng vẫn dùng các đặc trưng đã điền thiếu, one-hot và chuẩn hóa; không được mô tả như một mô hình tự xử lý dữ liệu phân loại thô đầu cuối.
''')
    write(c2,'2_6_',r'''\section{Các chỉ số đánh giá mô hình}
\label{sec:2_6}
Chọn \texttt{>50K} làm lớp dương. TP là số mẫu dương dự đoán đúng, FP là số mẫu âm dự đoán nhầm thành dương, FN là số mẫu dương dự đoán nhầm thành âm, TN là số mẫu âm dự đoán đúng. Các chỉ số được tính như sau:
\[
\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\quad
\mathrm{Precision}=\frac{TP}{TP+FP},\quad
\mathrm{Recall}=\frac{TP}{TP+FN},
\]
\[
F_1=\frac{2TP}{2TP+FP+FN},\qquad
\mathrm{Macro\text{-}}F_1=\frac{F_{1,0}+F_{1,1}}{2}.
\]
Khi không có dự đoán dương, precision và F1 được quy ước bằng 0. ROC-AUC đánh giá thứ hạng điểm dự đoán qua các ngưỡng thay vì chỉ dùng nhãn tại một ngưỡng. Accuracy dễ tạo ấn tượng cao khi lớp âm chiếm đa số; vì vậy F1 lớp dương được chọn làm tiêu chí chính, đi kèm precision, recall, macro-F1 và ROC-AUC~\cite{sklearn-f1}.

Mỗi cấu hình được báo cáo trung bình và độ lệch chuẩn mẫu (ddof=1) qua ba fold. Độ lệch chuẩn mô tả biến động giữa các fold, không phải khoảng tin cậy hoặc kiểm định ý nghĩa thống kê. Điểm CV cao nhất sau khi thử nhiều cấu hình còn chịu ảnh hưởng của quá trình lựa chọn và không thay thế đánh giá test độc lập.
''')
    write(c4,'4_5_',r'''\section{Huấn luyện các mô hình truyền thống}
\label{sec:4_5}
Module \texttt{modules/training.py} tạo một pipeline gồm bộ tiền xử lý chưa fit và bộ phân loại. Logistic Regression, LinearSVC và Random Forest được huấn luyện trên raw train trong ba fold theo nhóm. Các file X đã xuất ở bước 4 không được dùng trực tiếp để tính CV.

Trong mỗi fold, imputer, encoder, scaler và PCA chỉ fit trên phần train của fold, sau đó mới transform phần đánh giá. Các thống kê fitted được đối chiếu với phần train tương ứng. Số mẫu fit, số đặc trưng, thời gian, số vòng lặp và cảnh báo hội tụ được lưu lại. Mọi cấu hình dùng cùng các chỉ số fold đã khóa trong \texttt{results/step\_05/cv\_indices.json}.

Sau khi hoàn tất CV, cấu hình tốt nhất trong từng họ được fit lại trên 26.049 dòng train và lưu cùng bộ tiền xử lý dưới dạng pipeline \texttt{.joblib}. Tập validation chỉ được dùng để đối chiếu các mô hình đã khóa; không huấn luyện lại bằng train cộng validation ở bước này.
''')
    write(c4,'4_6_',r'''\section{Xây dựng mô hình MLP để so sánh}
\label{sec:4_6}
MLP được đặt sau đúng quy trình tiền xử lý cơ sở và sử dụng cùng nhãn, fold và tiêu chí đánh giá với các mô hình truyền thống. Hai cấu hình ban đầu là 64 và 64--32 nút; hai cấu hình 32 và 64 nút với regularization mạnh hơn được bổ sung sau pilot để xử lý cảnh báo tối ưu. Seed bằng 42; số luồng BLAS được giới hạn ở 2 để kiểm soát tài nguyên khi chạy tuần tự.

Không dùng validation bên ngoài để dừng học hoặc chọn số epoch. Early stopping nội bộ của \texttt{MLPClassifier} được tắt; thuật toán có thể dừng khi training loss không cải thiện đủ theo tol và số epoch chờ đã cấu hình. Cảnh báo không hội tụ được ghi vào audit, không bị bỏ qua. Đường training loss của bản MLP fit lại được lưu ở \texttt{mlp\_training\_loss.csv}; đây là training loss, không phải đường đánh giá test.
''')
    versions=chosen['versions']
    write(c5,'5_1_',r'''\section{Môi trường thực nghiệm và khả năng tái lập}
\label{sec:5_1}
Notebook tự tải ZIP UCI, tạo các module và chuẩn bị thư viện, không mount cloud cá nhân. Các fold được lưu theo chỉ số dòng với fingerprint dữ liệu và mã nhóm; khi chạy lại, dữ liệu hoặc cấu hình không khớp sẽ bị từ chối thay vì tự chia lại.

Các kết quả trong báo cáo được tạo bằng '''+
        'Python '+esc(versions['python'])+', scikit-learn '+esc(versions['sklearn'])+', NumPy '+esc(versions['numpy'])+' và pandas '+esc(versions['pandas'])+r'''. Seed bằng 42. Các cấu hình được chạy tuần tự, Random Forest dùng một worker, BLAS giới hạn 2 luồng. Thời gian huấn luyện phụ thuộc phần cứng và được dùng để tham khảo, không xem là kết quả benchmark giữa các máy.

Cache CV chỉ được tái sử dụng khi dữ liệu, metadata, fold, cấu hình, phiên bản, mã nguồn và SHA-256 của các tệp kết quả đều khớp. Notebook độc lập trên môi trường Colab mới không có cache sẽ thực hiện đầy đủ thí nghiệm. Quy trình notebook đã được thực thi từ đầu đến cuối trên Google Colab. Các số liệu trong báo cáo tương ứng với môi trường thực nghiệm được ghi ở trên.
''')
    write(c5,'5_2_',r'''\section{Giao thức đánh giá và chỉ số lựa chọn}
\label{sec:5_2}
Giữ nguyên phân chia bước 3: train 26.049 mẫu, validation 6.512 mẫu và test UCI 16.281 mẫu. Bên trong train, áp dụng StratifiedGroupKFold với 3 fold, shuffle bật và random state bằng 42. Các bản ghi có cùng 14 thuộc tính đầu vào được giữ chung nhóm, kể cả trường hợp nhãn khác nhau~\cite{sklearn-sgkf}.

F1 của lớp \texttt{>50K} là tiêu chí lựa chọn. Các chỉ số bổ sung gồm precision, recall, accuracy, macro-F1 và ROC-AUC. Chọn cấu hình có trung bình F1 CV cao nhất; khi bằng nhau ưu tiên độ lệch chuẩn nhỏ hơn, sau đó ID theo thứ tự chữ. Cấu hình có cảnh báo không hội tụ trong bất kỳ fold nào vẫn được báo cáo nhưng không được khóa làm mô hình cuối của họ.

Tất cả bộ biến đổi được fit lại bên trong fold bằng raw train. Không dùng X đã fit tiền xử lý trên toàn bộ train để tính CV~\cite{sklearn-pitfalls}. Không resampling, không class weighting, không tối ưu ngưỡng. Dùng ngưỡng dự đoán mặc định: xác suất 0,5 hoặc decision score 0 cho LinearSVC.

Lựa chọn chung và lựa chọn trong từng họ được khóa bằng CV trước khi tính validation. Validation là đối chiếu mô tả; không chọn lại theo F1 validation. Điểm CV của cấu hình thắng là điểm phục vụ lựa chọn trong một tìm kiếm giới hạn, không phải ước lượng không thiên lệch cho hiệu quả cuối cùng.
''')
    write(c5,'5_3_',r'''\section{Mô hình cơ sở và các siêu tham số khảo sát}
\label{sec:5_3}
Vòng thí nghiệm cuối gồm 20 cấu hình, mỗi cấu hình được đánh giá trên 3 fold, tương ứng 60 lượt fit CV:
\begin{itemize}
\item DummyClassifier luôn dự đoán lớp đa số: 1 cấu hình tham chiếu, không tham gia chọn mô hình thắng.
\item Logistic Regression: $C=0{,}1;1;10$, lbfgs, tối đa 2.000 vòng lặp, tol $10^{-5}$: 3 cấu hình.
\item LinearSVC: $C=0{,}1;1;10$, squared hinge, L2, dual=False, tối đa 10.000 vòng lặp, tol $10^{-5}$: 3 cấu hình.
\item Random Forest: 150 cây, max\_features=sqrt, bootstrap; ba lựa chọn depth/leaf đã mô tả: 3 cấu hình.
\item MLP: hai cấu hình ban đầu 64 và 64--32, Adam, 300 epoch tối đa, batch 256, tol $10^{-4}$, chờ 20 epoch; hai cấu hình bổ sung 32 và 64 nút, regularization mạnh hơn, tol $10^{-3}$, chờ 10 epoch: tổng 4 cấu hình.
\item Logistic Regression $C=1$ với bốn biến thể điền thiếu/mã hóa/scaling và hai biến thể PCA: 6 cấu hình.
\end{itemize}
Toàn bộ cấu hình đầy đủ được lưu trong \texttt{configs/experiments.json}; tham số mô hình mặc định còn lại được lưu trong các pipeline và được diễn giải theo phiên bản thư viện. Không gọi đây là tìm kiếm tối ưu toàn cục: số lựa chọn của các họ không bằng nhau và phạm vi khảo sát có giới hạn.

Pilot 18 cấu hình đầu tiên tạo 54 lượt fit, trong đó hai cấu hình MLP đều đạt giới hạn epoch ở cả ba fold. Sau khi kiểm tra cảnh báo tối ưu, bổ sung hai cấu hình MLP và chạy lại vòng 20 cấu hình. Đây là mở rộng dựa trên thông tin train/CV, không phải toàn bộ tìm kiếm được ấn định trước từ đầu. Kết quả pilot được giữ trong \texttt{results/step\_05/pilot\_18/}; lý do thay đổi nằm trong \texttt{experiment\_revision.json}. Chưa xem validation hoặc test trước khi thay đổi.
''')
    write(c5,'5_4_',r'''\section{Thí nghiệm so sánh các mô hình truyền thống}
\label{sec:5_4}
So sánh ba họ Logistic Regression, LinearSVC và Random Forest trên cùng tiền xử lý cơ sở: median, mode, one-hot, StandardScaler cho sáu biến số và không PCA. Trong mỗi họ, chọn siêu tham số tốt nhất theo F1 CV trong các cấu hình scope=model. Dummy đa số cho biết khả năng của một mô hình không học được cách nhận diện lớp dương.

Các kết quả được ghi theo từng fold và tổng hợp mean/SD. So sánh này đánh giá hiệu quả trong các cấu hình đã chạy; không suy rộng rằng một họ luôn tốt hơn họ khác trên mọi tập dữ liệu hoặc mọi lưới siêu tham số.
''')
    write(c5,'5_5_',r'''\section{Thí nghiệm so sánh các cách tiền xử lý}
\label{sec:5_5}
Giữ Logistic Regression $C=1$ và cùng các fold, thay đúng một lựa chọn so với cấu hình cơ sở: điền biến phân loại bằng nhóm \texttt{\_\_MISSING\_\_}; dùng ordinal thay one-hot; dùng MinMax [0,1] thay StandardScaler; hoặc dùng MinMax [-1,1]. Các thành phần còn lại giữ nguyên. Hai khoảng MinMax được ghi rõ để thể hiện khả năng cấu hình feature\_range.

Không so sánh mean với median bằng điểm mô hình trong bước này vì các cột số gốc không thiếu. Chức năng chọn mean vẫn được hỗ trợ nhưng không gán cho nó một hiệu quả xử lý thiếu chưa quan sát. Các lựa chọn như không scaling, class weighting và resampling chưa được thực nghiệm.
''')
    write(c5,'5_6_',r'''\section{Thí nghiệm ảnh hưởng của PCA}
\label{sec:5_6}
Giữ Logistic Regression $C=1$ và tiền xử lý cơ sở, so sánh không PCA, PCA 90\% và PCA 95\%. PCA được fit riêng trong mỗi fold, sau khi ghép nhánh số và nhánh one-hot. Không sử dụng các hướng chiếu đã học từ toàn bộ train để đánh giá fold.

Số thành phần của một fold phụ thuộc chính dữ liệu fit của fold đó và có thể khác con số 22/31 trong các file đặc trưng fit từ toàn bộ train ở bước 4. Tỷ lệ phương sai được giữ lại không trực tiếp đảm bảo chất lượng phân loại; tác động được nhận xét bằng F1 CV trong cùng thiết kế kiểm soát.
''')
    write(c5,'5_7_',r'''\section{Thí nghiệm so sánh pipeline truyền thống và MLP}
\label{sec:5_7}
MLP dùng cùng tiền xử lý cơ sở với ba họ truyền thống và được chọn giữa các cấu hình đủ điều kiện hội tụ bằng F1 CV. Bốn cấu hình đã chạy, bao gồm hai cấu hình bổ sung sau pilot, đều có kết quả và cảnh báo được lưu. So sánh bổ sung precision, recall, ROC-AUC và thời gian fit. Chi phí và độ nhạy cấu hình của MLP được xem cùng mức tăng hoặc giảm chất lượng, không chỉ nhìn một chỉ số duy nhất.

Các cấu hình được chạy với một seed. Chưa có thí nghiệm nhiều seed hoặc kiểm định ý nghĩa thống kê; chênh lệch nhỏ không được diễn giải thành ưu thế chắc chắn. Đường training loss chỉ dùng kiểm tra quá trình tối ưu, không dùng thay cho điểm trên fold chưa thấy.
''')
    winner=summary.set_index('candidate_id').loc[chosen['winner_id']]
    write(c5,'5_8_',r'''\section{Nguyên tắc chọn mô hình và đánh giá cuối cùng}
\label{sec:5_8}
Lựa chọn được thực hiện theo thứ tự: hoàn tất toàn bộ CV, lọc cấu hình không hội tụ theo quy tắc đã nêu, khóa mô hình thắng chung và từng họ, fit lại trên train, sau đó mới đối chiếu validation. Không sửa lựa chọn dựa trên validation. Cấu hình đã khóa là \texttt{'''+esc(chosen['winner_id'])+r'''} với F1 CV trung bình '''+num(winner.f1_mean)+r''' và SD '''+num(winner.f1_std)+r'''.

Pipeline đã fit được lưu tại \texttt{'''+esc(chosen['selected_model_path'])+r'''}. Tệp \texttt{selected\_model.json} ghi cấu hình lựa chọn, fingerprint dữ liệu, phiên bản và đường dẫn mô hình. Bước 6 sẽ dùng đúng mô hình và ngưỡng đã khóa để đánh giá một lần trên test. Bước 5 không tính chỉ số hoặc dự đoán test và không fit lại bằng train cộng validation.
''')
    base=summary[summary.scope.isin(['model','reference'])].sort_values(['f1_mean','f1_std','candidate_id'],ascending=[False,True,True]).groupby('family',sort=True).head(1).sort_values('family')
    traditional=base[base.family.isin(['Dummy','LogisticRegression','LinearSVC','RandomForest'])]
    write(c6,'6_1_',r'''\section{Kết quả so sánh các mô hình truyền thống}
\label{sec:6_1}
Các bảng trong mục 6.1--6.4 là kết quả CV trên train, chưa phải kết quả test. Chọn cấu hình tốt nhất của mỗi họ trong phạm vi scope=model với tiền xử lý cơ sở. F1, precision và recall đều tính cho lớp \texttt{>50K}.
'''+table(traditional,'Kết quả CV của các mô hình truyền thống và mốc đa số','tab:cv-traditional')+'\n'+
        figure('cv_models','6_1','F1 CV của cấu hình tốt nhất mỗi họ trên tiền xử lý cơ sở; thanh sai số là SD qua ba fold')+r'''
Mốc đa số có F1 lớp dương bằng 0 dù accuracy có thể cao, cho thấy chỉ dùng accuracy là chưa đủ. Các họ mô hình học từ thuộc tính cần được đánh giá cùng precision và recall để thấy mức đánh đổi khi phát hiện nhóm thu nhập cao.
''')
    indexed=summary.set_index('candidate_id')
    prep=indexed.loc[['lr_c1','lr_missing_constant','lr_ordinal','lr_minmax01','lr_minmax11']].reset_index()
    best=prep.sort_values('f1_mean',ascending=False).iloc[0]
    write(c6,'6_2_',r'''\section{Ảnh hưởng của các cách tiền xử lý}
\label{sec:6_2}
Giữ nguyên Logistic Regression $C=1$ và thay từng thành phần riêng lẻ. Kết quả mean/SD được tính trên cùng ba fold:
'''+table(prep,'Đối chiếu các lựa chọn tiền xử lý với Logistic Regression C=1','tab:cv-preprocessing')+'\n'+
        figure('cv_preprocessing','6_2','Ảnh hưởng của các lựa chọn tiền xử lý trong thí nghiệm kiểm soát')+
        '\nTrong nhóm này, cấu hình có F1 trung bình cao nhất là \\texttt{'+esc(best.candidate_id)+'}, đạt '+num(best.f1_mean)+r'''. Kết quả chỉ phản ánh các lựa chọn đã khảo sát. Ordinal tạo thứ tự số cho thuộc tính vốn có thể không có thứ tự tự nhiên; thay đổi này ảnh hưởng trực tiếp đến biên tuyến tính. Tác động của scaling còn phụ thuộc regularization đang giữ cố định, nên không kết luận một scaler luôn tốt hơn trên mọi mô hình.
''')
    pca=indexed.loc[['lr_c1','lr_pca90','lr_pca95']].reset_index()
    write(c6,'6_3_',r'''\section{Ảnh hưởng của giảm chiều bằng PCA}
\label{sec:6_3}
So sánh ba cấu hình trên cùng các fold, với PCA chỉ fit từ phần train của từng fold:
'''+table(pca,'So sánh không PCA và hai ngưỡng giữ phương sai','tab:cv-pca')+'\n'+
        figure('cv_pca','6_3','F1 CV của Logistic Regression C=1 khi thay lựa chọn PCA')+r'''
PCA giảm số chiều nhưng tối ưu phương sai của đầu vào, không sử dụng nhãn để chọn hướng. Vì vậy, giảm chiều có thể làm mất thông tin hữu ích cho phân loại. Các kết quả này được dùng để nhận xét riêng thí nghiệm PCA trên Logistic Regression; không suy rộng tác động sang Random Forest hoặc MLP chưa chạy với PCA.
''')
    fam_ids=list(chosen['family_winners'].values())
    compare=indexed.loc[fam_ids].reset_index()
    val_lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Đối chiếu validation sau khi khóa bằng CV}\label{tab:validation-locked}',
        r'\begin{tabular}{lrrr}',r'\toprule',r'Cấu hình & F1 & Precision & Recall \\',r'\midrule']
    for _,r in validation.iterrows():val_lines.append(esc(r.candidate_id)+' & '+' & '.join(num(r[k]) for k in ['f1','precision','recall'])+r' \\')
    val_lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    write(c6,'6_4_',r'''\section{So sánh pipeline truyền thống và MLP}
\label{sec:6_4}
Bảng sau tổng hợp cấu hình tốt nhất của từng họ trên toàn bộ các cấu hình đã khảo sát của họ đó. Với Logistic Regression, tập ứng viên này bao gồm cả các biến thể tiền xử lý/PCA; vì vậy bảng có thể khác mốc chỉ dùng tiền xử lý cơ sở ở mục 6.1.
'''+table(compare,'Cấu hình đã khóa trong từng họ theo F1 CV','tab:cv-family-winners')+'\n'+
        '\n'.join(val_lines)+r'''
Validation được tính sau khi khóa lựa chọn và không dùng để đổi cấu hình thắng. Hai nguồn điểm CV/validation phải được phân biệt; không chọn số lớn hơn giữa chúng để trình bày như điểm test. MLP được khảo sát trong phạm vi bốn cấu hình và một seed nên chưa thể khẳng định ưu thế tổng quát của mạng nơ-ron so với học máy truyền thống.
''')
    warnings_total=int(summary.convergence_warning_count.sum())
    write(c6,'6_7_',r'''\section{Nhận xét tổng hợp về các kết quả}
\label{sec:6_7}
Trong 20 cấu hình đã thử, quy tắc lựa chọn khóa \texttt{'''+esc(chosen['winner_id'])+r'''} bằng F1 CV, trước khi xem validation. Thí nghiệm đã đối chiếu mô hình truyền thống, MLP, bốn lựa chọn tiền xử lý và hai mức PCA, với cùng cách chia theo nhóm. Tổng số cảnh báo hội tụ trong các lượt CV là '''+str(warnings_total)+r'''; các cảnh báo và số vòng lặp được lưu để kiểm tra.

Giới hạn hiện tại gồm lưới siêu tham số nhỏ, số cấu hình không đều giữa các họ, ba fold và một seed, chưa có kiểm định thống kê và chưa có kết quả test. Các khác biệt phản ánh thiết kế đã chạy, không chứng minh một thuật toán tốt nhất nói chung. Kết luận hiệu quả cuối cùng sẽ được bổ sung sau đánh giá test ở bước 6, còn chương kết luận và các phần hoàn thiện báo cáo thuộc bước 7.
''')
    write('backmatter/appendix_a','A_1_',r'''\section{Chạy toàn bộ notebook trên Google Colab}
\label{sec:A_1}
Tải \texttt{notebooks/Adult\_BTL.ipynb} lên Colab rồi chọn Runtime $\rightarrow$ Run all. Notebook tự tải ZIP UCI và chuẩn bị module, không mount Drive. Bản bước 5 chạy các bước dữ liệu, EDA, chia tập, đặc trưng, CV theo nhóm, fit lại và đối chiếu validation. Việc huấn luyện cần nhiều thời gian hơn các bước thống kê.

Kết quả nằm trong \texttt{results/step\_05/}, pipeline trong \texttt{artifacts/models/}, đặc trưng trong \texttt{features/}, nguồn LaTeX trong \texttt{reports/latex/}. Notebook tạo ZIP LaTeX mới ở cuối. Cache chỉ được dùng khi vượt qua kiểm tra fingerprint và SHA-256; trên Colab mới, notebook tự huấn luyện. Các chỉnh sửa thủ công trong nguồn báo cáo cần được sao lưu vì Run all dựng lại nguồn.
''')
    write('backmatter/appendix_b','B_2_',r'''\section{Tiến độ thực hiện của nhóm}
\label{sec:B_2}
Đến bước 5, đã chuẩn bị dữ liệu, EDA, chia tập theo nhóm, xuất đặc trưng, thực hiện CV và khóa mô hình để đánh giá cuối. Các tệp kết quả và audit được lưu cùng notebook. Đánh giá test và hoàn thiện báo cáo còn thuộc bước 6--7. Bảng phân công, tỷ lệ đóng góp và minh chứng hoạt động phải được nhóm điền theo công việc thực tế; không suy ra các thông tin này từ việc chạy mã.
''')
    refs=report/'backmatter/references.tex';text=refs.read_text(encoding='utf-8')
    new=[]
    for key,title,url in [
        ('sklearn-logistic','LogisticRegression','https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'),
        ('sklearn-linearsvc','LinearSVC','https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html'),
        ('sklearn-rf','RandomForestClassifier','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'),
        ('sklearn-mlp','MLPClassifier','https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html'),
        ('sklearn-f1','F1 score','https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html')]:
        if '\\bibitem{'+key+'}' not in text:new.append('\\bibitem{'+key+'} scikit-learn developers. '+title+'. \\url{'+url+'}.')
    text=text.replace(r'\end{thebibliography}','\n'.join(new)+'\n'+r'\end{thebibliography}')
    refs.write_text(text,encoding='utf-8')
