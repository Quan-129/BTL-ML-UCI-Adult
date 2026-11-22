"""Dàn PDF từ cùng nguồn LaTeX bằng ReportLab; không mô phỏng kết quả biên dịch TeX."""
from pathlib import Path
import hashlib,html,json,re
from xml.sax.saxutils import escape
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY,TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,PageBreak,
    NextPageTemplate,Image,Table,TableStyle,KeepTogether,Flowable)
from reportlab.platypus.tableofcontents import TableOfContents

PAGE_W,PAGE_H=A4
LEFT=25*72/25.4;RIGHT=22*72/25.4;WIDTH=PAGE_W-LEFT-RIGHT

def _group(text,pos):
    while pos<len(text) and text[pos].isspace():pos+=1
    if pos>=len(text) or text[pos]!='{':raise ValueError('Expected braced group at '+text[pos:pos+80])
    start=pos+1;depth=1;pos+=1
    while pos<len(text):
        if text[pos]=='\\':pos+=2;continue
        if text[pos]=='{':depth+=1
        elif text[pos]=='}':
            depth-=1
            if depth==0:return text[start:pos],pos+1
        pos+=1
    raise ValueError('Unclosed group')

def _remove_commands(text,names):
    for name,nargs in names.items():
        pattern=re.compile(r'\\'+name+r'\b')
        while True:
            m=pattern.search(text)
            if not m:break
            end=m.end()
            for _ in range(nargs):_,end=_group(text,end)
            text=text[:m.start()]+text[end:]
    return text

def _env(text,pos):
    begin=re.match(r'\\begin\{([^}]+)\}',text[pos:]);name=begin.group(1);start=pos+begin.end()
    pattern=re.compile(r'\\(begin|end)\{'+re.escape(name)+r'\}')
    depth=1
    for m in pattern.finditer(text,start):
        depth+=1 if m.group(1)=='begin' else -1
        if depth==0:return name,text[start:m.start()],m.end()
    raise ValueError('Unclosed environment '+name)

class TaggedParagraph(Paragraph):
    def __init__(self,text,style,kind=None,key=None,level=0,plain=None):
        super().__init__(text,style);self.kind=kind;self.key=key;self.level=level;self.plain=plain or text

class Marker(Flowable):
    def __init__(self,kind):super().__init__();self.kind=kind
    def wrap(self,a,b):return 0,0
    def draw(self):pass

class FilteredTOC(TableOfContents):
    def __init__(self,kind,styles):
        super().__init__();self.kind=kind;self.levelStyles=styles;self.dotsMinLevel=0
        self.formatter=lambda n:_roman(-n) if n<0 else str(n)
    def notify(self,kind,stuff):
        if kind==self.kind:super().notify('TOCEntry',stuff)

def _roman(n):
    value=''
    for number,letter in [(1000,'m'),(900,'cm'),(500,'d'),(400,'cd'),(100,'c'),(90,'xc'),(50,'l'),(40,'xl'),(10,'x'),(9,'ix'),(5,'v'),(4,'iv'),(1,'i')]:
        while n>=number:value+=letter;n-=number
    return value

class ReportDoc(BaseDocTemplate):
    def __init__(self,path,styles,logo):
        super().__init__(str(path),pagesize=A4,leftMargin=LEFT,rightMargin=RIGHT,topMargin=88,bottomMargin=71,
            title='Bài tập lớn Học máy - Dự đoán thu nhập với UCI Adult',author='Nhóm A01',
            subject='Báo cáo thực nghiệm; PDF dàn trang từ nguồn LaTeX kèm theo')
        self.styles=styles;self.logo=logo;self.body_start=None;self.heading_pages=[]
        frame=Frame(LEFT,71,WIDTH,PAGE_H-88-71,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates([PageTemplate(id='cover',frames=[frame]),
            PageTemplate(id='body',frames=[frame],onPageEnd=self.header_footer)])
    def beforeDocument(self):self.body_start=None;self.heading_pages=[]
    def header_footer(self,canvas,doc):
        canvas.saveState();blue=colors.HexColor('#092D82')
        if self.logo.is_file():canvas.drawImage(str(self.logo),LEFT,PAGE_H-63,width=26,height=26,mask='auto')
        canvas.setFillColor(colors.HexColor('#333333'));canvas.setFont('ReportSans',9)
        canvas.drawString(LEFT+34,PAGE_H-43,'Trường Đại học Bách khoa, ĐHQG-HCM')
        canvas.drawString(LEFT+34,PAGE_H-55,'Khoa Khoa học và Kỹ thuật Máy tính')
        canvas.setLineWidth(.4);canvas.setStrokeColor(blue);canvas.line(LEFT,PAGE_H-70,PAGE_W-RIGHT,PAGE_H-70)
        canvas.line(LEFT,49,PAGE_W-RIGHT,49);canvas.setFillColor(colors.HexColor('#333333'));canvas.setFont('ReportSans',8)
        canvas.drawString(LEFT,36,'Bài tập lớn Học máy - Năm học 2026–2027')
        page=str(doc.page-self.body_start+1) if self.body_start is not None else _roman(max(1,doc.page-1))
        canvas.drawRightString(PAGE_W-RIGHT,36,'Trang '+page);canvas.restoreState()
    def afterFlowable(self,f):
        if isinstance(f,Marker) and f.kind=='arabic':self.body_start=self.page
        if isinstance(f,TaggedParagraph) and f.kind:
            page=str(self.page-self.body_start+1) if self.body_start is not None else _roman(max(1,self.page-1))
            self.canv.bookmarkPage(f.key)
            numeric_page=self.page-self.body_start+1 if self.body_start is not None else -(max(1,self.page-1))
            self.notify(f.kind,(f.level,f.plain,numeric_page,f.key))
            if f.kind=='TOCEntry':
                self.heading_pages.append({'title':f.plain,'pdf_page':self.page,'display_page':page})

class Renderer:
    def __init__(self,root,work):
        self.root=Path(root);self.work=Path(work);self.work.mkdir(parents=True,exist_ok=True)
        self.blocks=[];self.labels={};self.chapter=0;self.section=0;self.subsection=0;self.figure=0;self.table=0;self.equation=0
        self.appendix=False;self.last_number='';self.bib={};self.math_failures=[];self.math_images={};self.parsed_files=[]
        self.styles=self.make_styles()
    def make_styles(self):
        import matplotlib.font_manager as fm
        def font(name,win,family,weight='normal',style='normal'):
            path=Path('C:/Windows/Fonts')/win
            if not path.is_file():path=Path(fm.findfont(fm.FontProperties(family=family,weight=weight,style=style)))
            pdfmetrics.registerFont(TTFont(name,str(path)))
        font('Report','times.ttf','DejaVu Serif');font('ReportBold','timesbd.ttf','DejaVu Serif','bold')
        font('ReportItalic','timesi.ttf','DejaVu Serif',style='italic');font('ReportBoldItalic','timesbi.ttf','DejaVu Serif','bold','italic')
        font('ReportSans','arial.ttf','DejaVu Sans');font('ReportMono','consola.ttf','DejaVu Sans Mono')
        pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold',italic='ReportItalic',boldItalic='ReportBoldItalic')
        return {
            'body':ParagraphStyle('body',fontName='Report',fontSize=12,leading=16.5,alignment=TA_JUSTIFY,spaceAfter=6,firstLineIndent=13),
            'small':ParagraphStyle('small',fontName='Report',fontSize=10,leading=13,spaceAfter=5),
            'chapter':ParagraphStyle('chapter',fontName='ReportBold',fontSize=18,leading=23,spaceAfter=17,keepWithNext=True),
            'section':ParagraphStyle('section',fontName='ReportBold',fontSize=14,leading=18,spaceBefore=12,spaceAfter=6,keepWithNext=True),
            'subsection':ParagraphStyle('subsection',fontName='ReportBold',fontSize=12,leading=16,spaceBefore=9,spaceAfter=5,keepWithNext=True),
            'caption':ParagraphStyle('caption',fontName='Report',fontSize=10,leading=13,alignment=TA_CENTER,spaceBefore=5,spaceAfter=8,keepWithNext=False),
            'cell':ParagraphStyle('cell',fontName='Report',fontSize=9.5,leading=12,spaceAfter=0,wordWrap='LTR',splitLongWords=True),
            'cover':ParagraphStyle('cover',fontName='Report',fontSize=12,leading=18,alignment=TA_CENTER,spaceAfter=5),
            'cover_title':ParagraphStyle('cover_title',fontName='ReportBold',fontSize=18,leading=25,alignment=TA_CENTER,spaceAfter=9),
        }
    def expand(self,path):
        path=Path(path);self.parsed_files.append(path.relative_to(self.root).as_posix())
        text=re.sub(r'(?<!\\)%[^\n]*','',path.read_text(encoding='utf-8'))
        return re.sub(r'\\input\{([^}]+)\}',lambda m:self.expand(self.root/(m.group(1)+'.tex')),text)
    def number(self):return chr(64+self.chapter) if self.appendix else str(self.chapter)
    def math(self,expr,inline=True):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.mathtext import math_to_image
        from matplotlib.font_manager import FontProperties
        expr=' '.join(expr.strip().split())
        if expr not in self.math_images:
            path=self.work/('math_'+hashlib.sha256(expr.encode()).hexdigest()[:15]+'.png')
            try:
                with matplotlib.rc_context({'mathtext.fontset':'stix'}):math_to_image('$'+expr+'$',str(path),prop=FontProperties(size=12),dpi=220,format='png',color='black')
                with PILImage.open(path) as im:w,h=im.size
                self.math_images[expr]=(path,w*72/220,h*72/220)
            except Exception as e:
                self.math_failures.append({'expression':expr,'error':str(e)})
                self.math_images[expr]=None
        info=self.math_images[expr]
        if info is None:return escape(expr)
        path,w,h=info
        if inline:
            scale=min(1,(WIDTH*.92)/max(w,1));w*=scale;h*=scale
            return f'<img src="{path.as_posix()}" width="{w:.2f}" height="{h:.2f}" valign="-3"/>'
        scale=min(1,(WIDTH*.95)/max(w,1))
        return Image(str(path),width=w*scale,height=h*scale)
    def inline(self,text):
        text=text.replace('~',' ');parts=[];i=0
        while i<len(text):
            ch=text[i]
            if ch=='$':
                end=text.find('$',i+1)
                if end!=-1:parts.append(self.math(text[i+1:end]));i=end+1;continue
            if ch=='\\':
                if i+1<len(text) and text[i+1] in '%_&#{}$':parts.append(escape(text[i+1]));i+=2;continue
                if i+1<len(text) and text[i+1]=='\\':parts.append('<br/>');i+=2;continue
                m=re.match(r'\\([A-Za-z]+)\*?',text[i:])
                if m:
                    name=m.group(1);i+=m.end()
                    if name in ['textbf','textit','emph','texttt','textrm','textsf','mathrm','mbox','MakeUppercase']:
                        arg,i=_group(text,i);value=self.inline(arg)
                        if name=='textbf':value='<b>'+value+'</b>'
                        elif name in ['textit','emph']:value='<i>'+value+'</i>'
                        elif name=='texttt':value='<font name="ReportMono" size="9">'+value+'</font>'
                        parts.append(value);continue
                    if name in ['url','href']:
                        arg,i=_group(text,i)
                        if name=='href':label,i=_group(text,i)
                        else:label=arg
                        parts.append('<link href="'+escape(arg,{'"':'&quot;'})+'" color="#092D82">'+escape(label)+'</link>');continue
                    if name=='cite':
                        arg,i=_group(text,i);parts.append('['+', '.join(str(self.bib.get(k.strip(),'?')) for k in arg.split(','))+']');continue
                    if name=='ref':
                        arg,i=_group(text,i);parts.append(escape(self.labels.get(arg,'?')));continue
                    if name in ['label','addcontentsline']:
                        for _ in range(3 if name=='addcontentsline' else 1):_,i=_group(text,i)
                        continue
                    if name in ['noindent','centering','small','scriptsize','normalsize','footnotesize','raggedright','bfseries','itshape','phantomsection','hfill','vfill','toprule','midrule','bottomrule','endhead','endfirsthead','endfoot','endlastfoot']:continue
                    if name in ['textwidth','linewidth']:continue
                    if name in ['vspace','hspace']:
                        _,i=_group(text,i);parts.append(' ');continue
                    if name in ['NamHoc','TenNhom','MaMonHoc','GiangVien']:
                        parts.append({'NamHoc':'2026–2027','TenNhom':'A01','MaMonHoc':'CO3117','GiangVien':'Trương Vĩnh Lân'}[name]);continue
                    parts.append(escape(name));continue
                i+=1;continue
            if ch in '{}':i+=1;continue
            parts.append(escape(ch));i+=1
        return ''.join(parts).replace('---','—').replace('--','–')
    def p(self,text,style='body'):return Paragraph(self.inline(' '.join(text.strip().split())),self.styles[style])
    def heading(self,title,level,number=''):
        key=f'heading-{len(self.blocks)}';prefix=(number+' ') if number else ''
        if level==0:
            self.blocks.append(('pagebreak',None))
        self.blocks.append(('heading',{'text':prefix+title,'level':level,'key':key}))
        self.last_number=number or title
    def parse(self,text):
        special=re.compile(r'\\(?:chapter\*?|section\*?|subsection\*?|paragraph\*?|begin\{[^}]+\}|label|appendix\b|pagenumbering|tableofcontents\b|listoffigures\b|listoftables\b|clearpage\b|newpage\b|phantomsection\b|addcontentsline|\[)')
        i=0
        while i<len(text):
            m=special.search(text,i)
            if not m:
                self.text_blocks(text[i:]);break
            self.text_blocks(text[i:m.start()]);pos=m.start();command=m.group(0);end=m.end()
            if re.match(r'\\(chapter|section|subsection|paragraph)',command):
                title,end=_group(text,end);kind=command[1:].rstrip('*');star=command.endswith('*')
                if kind=='chapter':
                    if not star:self.chapter+=1;self.section=self.subsection=self.figure=self.table=self.equation=0
                    number=('Phụ lục ' if self.appendix else 'Chương ')+self.number()+'.' if not star else ''
                    self.heading(title,0,number)
                elif kind=='section':
                    if not star:self.section+=1;self.subsection=0
                    self.heading(title,1,f'{self.number()}.{self.section}' if not star else '')
                elif kind=='subsection':
                    if not star:self.subsection+=1
                    self.heading(title,2,f'{self.number()}.{self.section}.{self.subsection}' if not star else '')
                else:self.blocks.append(('paragraph',r'\textbf{'+title+'}'))
            elif command.startswith('\\begin'):
                name,body,end=_env(text,pos)
                if name=='titlepage':self.blocks.append(('cover',None))
                elif name in ['itemize','enumerate']:
                    for number,item in enumerate(re.split(r'\\item(?:\[[^]]*\])?',body)[1:],1):
                        self.blocks.append(('bullet',{'text':item,'bullet':f'{number}.' if name=='enumerate' else '•'}))
                elif name in ['table','longtable']:self.parse_table(name,body)
                elif name=='figure':self.parse_figure(body)
                elif name=='equation':
                    self.equation+=1;self.blocks.append(('equation',{'text':body,'number':f'{self.number()}.{self.equation}'}))
                elif name=='thebibliography':
                    _,body_start=_group(body,0);body=body[body_start:]
                    self.heading('Tài liệu tham khảo',0)
                    for bm in re.finditer(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\Z)',body,re.S):
                        self.blocks.append(('bib',f'[{self.bib[bm.group(1)]}] '+bm.group(2).strip()))
                else:self.text_blocks(body)
            elif command=='\\[':
                close=text.find(r'\]',end)
                self.blocks.append(('equation',{'text':text[end:close],'number':None}));end=close+2
            elif command=='\\label':
                label,end=_group(text,end);self.labels[label]=self.last_number
            elif command=='\\appendix':self.appendix=True;self.chapter=0
            elif command=='\\pagenumbering':
                arg,end=_group(text,end)
                if arg=='arabic':self.blocks.append(('arabic',None))
            elif command in ['\\tableofcontents','\\listoffigures','\\listoftables']:
                kind,title={'\\tableofcontents':('TOCEntry','Mục lục'),'\\listoffigures':('LOFEntry','Danh mục hình'),'\\listoftables':('LOTEntry','Danh mục bảng')}[command]
                self.heading(title,0);self.blocks.append(('toc',kind))
            elif command in ['\\clearpage','\\newpage']:self.blocks.append(('pagebreak',None))
            elif command=='\\addcontentsline':
                for _ in range(3):_,end=_group(text,end)
            i=end
    def text_blocks(self,text):
        for part in re.split(r'\n\s*\n',text):
            part=part.strip()
            if part and part!='}':self.blocks.append(('paragraph',part))
    def caption(self,body,kind):
        m=re.search(r'\\caption',body);caption=''
        if m:
            caption,end=_group(body,m.end());body=body[:m.start()]+body[end:]
        number=f'{self.number()}.{self.figure if kind=="figure" else self.table}'
        for key in re.findall(r'\\label\{([^}]+)\}',body):self.labels[key]=number
        body=_remove_commands(body,{'label':1})
        self.last_number=number
        return caption,number,body
    def parse_figure(self,body):
        self.figure+=1;caption,number,body=self.caption(body,'figure')
        paths=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',body)
        self.blocks.append(('figure',{'caption':caption,'number':number,'paths':paths,'key':f'fig-{self.number()}-{self.figure}'}))
    def parse_table(self,name,body):
        if name=='table':
            self.table+=1;caption,number,body=self.caption(body,'table')
            m=re.search(r'\\begin\{(?:tabular|longtable)\}',body)
            if not m:raise ValueError('No tabular in table')
            _,body,_=_env(body,m.start())
        else:
            self.table+=1;caption,number,body=self.caption(body,'table')
        spec,start=_group(body,0);body=body[start:]
        body=re.sub(r'\\(?:toprule|midrule|bottomrule|hline|endhead|endfirsthead|endfoot|endlastfoot)\b','',body)
        rows=[];row=[];cell='';depth=0;math=False;i=0
        while i<len(body):
            c=body[i]
            if c=='$':math=not math
            if c=='\\' and i+1<len(body):
                if body[i+1]=='\\' and depth==0 and not math:
                    row.append(cell.strip());cell=''
                    if any(row):rows.append(row)
                    row=[];i+=2;continue
                if body[i+1] in '&%_{}':cell+=body[i:i+2];i+=2;continue
            if c=='{':depth+=1
            elif c=='}':depth-=1
            if c=='&' and depth==0 and not math:row.append(cell.strip());cell=''
            else:cell+=c
            i+=1
        if cell.strip():row.append(cell.strip())
        if row and any(row):rows.append(row)
        rows=[r for r in rows if any(x.strip() for x in r)]
        if not rows:return
        count=max(map(len,rows));rows=[r+['']*(count-len(r)) for r in rows]
        self.blocks.append(('table',{'caption':caption,'number':number,'rows':rows,'spec':spec,'key':f'tab-{self.number()}-{self.table}'}))
    def cover(self):
        s=[]
        for text in ['ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH','<b>TRƯỜNG ĐẠI HỌC BÁCH KHOA</b>','<b>KHOA KHOA HỌC VÀ KỸ THUẬT MÁY TÍNH</b>']:
            s.append(Paragraph(text,self.styles['cover']))
        s += [Spacer(1,20),Image(str(self.root/'Images/frontmatter/logo_hcmut.png'),width=87,height=87),Spacer(1,25),
            Paragraph('Bài tập lớn Học máy (CO3117)',self.styles['cover_title']),Spacer(1,12),
            Paragraph('Dự đoán thu nhập năm trên 50.000 USD<br/>với dữ liệu UCI Adult',self.styles['cover_title']),
            Spacer(1,8),Paragraph('Học kỳ 261 · Nhóm A01',self.styles['cover']),Spacer(1,22),
            Paragraph('<b>Giảng viên học phần:</b> Trương Vĩnh Lân',self.styles['cover']),Spacer(1,16),
            Paragraph('<b>Sinh viên thực hiện</b>',self.styles['cover'])]
        for name,id in [('Hồng Chấn Phước','2352963'),('Trần Bá Minh Quân','2353015'),('Đinh Nguy Nguyệt Hà','2352286')]:
            s.append(Paragraph(name+' — '+id,self.styles['cover']))
        s += [Spacer(1,56),Paragraph('THÀNH PHỐ HỒ CHÍ MINH, THÁNG 11 NĂM 2026',self.styles['cover']),NextPageTemplate('body')]
        return s
    def caption_p(self,text,number,kind,key):
        prefix='Hình' if kind=='LOFEntry' else 'Bảng'
        return TaggedParagraph(prefix+' '+number+'. '+self.inline(text),self.styles['caption'],kind=kind,key=key,plain=prefix+' '+number+'. '+html.unescape(re.sub('<[^>]+>','',self.inline(text))))
    def table_flowable(self,b):
        rows=[[self.p(x,'cell') for x in r] for r in b['rows']];count=len(rows[0]);widths=None
        specs=re.findall(r'p\{([^}]+)\}|([lrcX])',b['spec'])
        if len(specs)==count and any(a for a,_ in specs):
            widths=[]
            for value,align in specs:
                m=re.search(r'([.\d]+)\\textwidth',value)
                widths.append(float(m.group(1))*WIDTH if m else None)
            remaining=WIDTH-sum(w for w in widths if w is not None)
            if any(w is None for w in widths):widths=[remaining/sum(x is None for x in widths) if w is None else w for w in widths]
        if widths is None:
            weights=[]
            for j in range(count):
                length=max(len(re.sub(r'\\[a-zA-Z]+|[{}$]','',r[j])) for r in b['rows'])
                weights.append(min(48,max(7,length))* (1.4 if j==0 else 1))
            widths=[WIDTH*w/sum(weights) for w in weights]
        if sum(widths)>WIDTH:widths=[w*WIDTH/sum(widths) for w in widths]
        t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EDF2F9')),
            ('LINEABOVE',(0,0),(-1,0),.7,colors.HexColor('#092D82')),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#75849A')),
            ('LINEBELOW',(0,-1),(-1,-1),.7,colors.HexColor('#092D82'))]))
        return t
    def build(self,path):
        main=(self.root/'main.tex').read_text(encoding='utf-8');body=main.split(r'\begin{document}',1)[1].split(r'\end{document}',1)[0]
        text=re.sub(r'\\input\{([^}]+)\}',lambda m:self.expand(self.root/(m.group(1)+'.tex')),body)
        self.bib={k:i+1 for i,k in enumerate(re.findall(r'\\bibitem\{([^}]+)\}',text))}
        self.parse(text);story=[]
        def pagebreak():
            if story and not isinstance(story[-1],(PageBreak,Marker)):story.append(PageBreak())
        for kind,b in self.blocks:
            if kind=='cover':story+=self.cover()
            elif kind=='pagebreak':pagebreak()
            elif kind=='arabic':
                pagebreak();story.append(Marker('arabic'))
            elif kind=='heading':
                style=['chapter','section','subsection'][b['level']]
                story.append(TaggedParagraph(self.inline(b['text']),self.styles[style],kind='TOCEntry',key=b['key'],level=b['level'],plain=b['text']))
            elif kind in ['paragraph','bib']:
                p=self.p(b,'small' if kind=='bib' else 'body')
                if p.getPlainText().strip():story.append(p)
            elif kind=='bullet':
                style=ParagraphStyle('bullet',parent=self.styles['body'],leftIndent=14,firstLineIndent=0,bulletIndent=0)
                story.append(Paragraph(self.inline(' '.join(b['text'].split())),style,bulletText=b['bullet']))
            elif kind=='toc':
                styles=[ParagraphStyle(f'{kind}-{i}',fontName='ReportBold' if i==0 else 'Report',fontSize=11 if i<2 else 10,
                    leading=14,spaceBefore=5 if i==0 else 1,leftIndent=12*i,firstLineIndent=0,rightIndent=24) for i in range(3)]
                story.append(FilteredTOC(b,styles))
            elif kind=='equation':
                image=self.math(b['text'],False)
                if isinstance(image,str):story.append(self.p(image))
                elif b['number']:
                    max_width=WIDTH-45
                    if image.drawWidth>max_width:
                        scale=max_width/image.drawWidth;image.drawWidth*=scale;image.drawHeight*=scale
                    story.append(Table([[image,Paragraph('('+b['number']+')',self.styles['small'])]],colWidths=[WIDTH-40,40],style=[('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
                else:story.append(image)
                story.append(Spacer(1,8))
            elif kind=='figure':
                images=[];cell_width=WIDTH/(len(b['paths']) or 1)-8
                for rel in b['paths']:
                    p=self.root/rel
                    if p.suffix.lower()=='.pdf':p=p.with_suffix('.png')
                    if not p.is_file():raise FileNotFoundError(p)
                    with PILImage.open(p) as im:w,h=im.size
                    scale=min(cell_width/w,390/h);images.append(Image(str(p),width=w*scale,height=h*scale))
                group=[Table([images],colWidths=[cell_width+8]*len(images))] if len(images)>1 else images
                group.append(self.caption_p(b['caption'],b['number'],'LOFEntry',b['key']))
                story.append(KeepTogether(group));story.append(Spacer(1,5))
            elif kind=='table':
                if b['caption']:
                    cap=self.caption_p(b['caption'],b['number'],'LOTEntry',b['key']);cap.keepWithNext=True;story.append(cap)
                story.append(self.table_flowable(b));story.append(Spacer(1,9))
        doc=ReportDoc(path,self.styles,self.root/'Images/frontmatter/logo_hcmut.png')
        doc.multiBuild(story,maxPasses=8)
        return {'engine':'ReportLab from LaTeX source','latex_compiled':False,'pages':doc.page,
            'source_files':sorted(set(self.parsed_files)),'math_image_count':len(self.math_images),
            'math_failures':self.math_failures,'headings':doc.heading_pages,
            'block_counts':{kind:sum(k==kind for k,_ in self.blocks) for kind in sorted({k for k,_ in self.blocks})}}

def render_report(project_dir='.'):
    root=Path(project_dir);out=root/'results/step_07';out.mkdir(parents=True,exist_ok=True)
    path=root/'reports/BTL_ML_Bao_Cao.pdf'
    renderer=Renderer(root/'reports/latex',out/'render_assets')
    result=renderer.build(path)
    result['pdf_path']=path.relative_to(root).as_posix()
    result['pdf_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'pdf_render_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if result['math_failures']:raise RuntimeError('Một số công thức chưa dàn trang đúng; xem pdf_render_manifest.json.')
    return result
