#%%
import os
import latex2markdown
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema.document import Document
import copy
import shutil
import json
import fitz  # pip install PyMuPDF (also used by ChatGPT)
from collections import defaultdict

# Load OPENAI_API_KEY from a .env file
#from dotenv import load_dotenv
#load_dotenv()
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv(), override=True)
#os.environ["OPENAI_API_KEY"] = "sk-REDACTED"


headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4")
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


#%%
skip = ['GDR3_master_include.tex', 'rvs_cep.tex']
exclude = ['_skip', 'acknow', 'authors', 'acronyms', 'notations', 'table', 'rejected', 'psfig', 'binhex', 'natbib']

chunk_dir = "latex_data_chunks/" 
if os.path.exists(chunk_dir):
    shutil.rmtree(chunk_dir)
os.mkdir(chunk_dir)   
all_docs = []
latex_dir = "plato_latex/"

latex_processed = []

with open('OneDrive_1_11-03-2024/PLATOChat_papers.json', "r") as file:
    data = json.load(file)
    for i, key in enumerate(data):
        if not data[key]["arXiv ID"] is None:
            ArXivID = data[key]["arXiv ID"]
            title = data[key]["title"]
            link = 'https://arxiv.org/pdf/'+ArXivID+'.pdf'
            folder = latex_dir+ArXivID+'_source'
            if os.path.isdir(folder):
                print(folder)
                for fn in os.listdir(folder):
                    if fn.endswith('.tex') and not fn in skip and not any([t in fn for t in exclude]):
                        print('\t'+fn)
                        latex_processed.append(ArXivID)
                        outname = ArXivID+'-'+fn[:-4]+'.json'
                        outfile = open(chunk_dir+outname, 'w')
                        path = os.path.join(folder, fn)
                        try:
                            f = open(path, "r")
                            latex_string = f.read()
                        except UnicodeDecodeError:
                            f = open(path, "r", encoding='ISO 8859-1')
                            latex_string = f.read()
                        l2m = latex2markdown.LaTeX2Markdown(latex_string)
                        md = l2m.to_markdown()
                        #print(md)
                        sections = markdown_splitter.split_text(md)
                        paragraphs = []
                        for s in sections:
                            #print(s)
                            s.metadata['source'] = path
                            s.metadata['title'] = title
                            s.metadata['link'] = link
                            for field in s.metadata:
                                if field.startswith('Header'):
                                    if '\\label' in s.metadata[field]:
                                        s.metadata[field] = s.metadata[field].split('\\label')[0]
                            pars = s.page_content.split(' \n')
                            for n, p in enumerate(pars):
                                if p and not p.startswith('%'):
                                    
                                    metadata_par = copy.deepcopy(s.metadata)
                                    metadata_par["paragraph"] = str(n+1)
                                    par_doc = Document(page_content=p, metadata=metadata_par)
                                    paragraphs.append(par_doc)
                                    print(type(par_doc))
                                    print(par_doc)
                        for p in paragraphs:
                            outfile.write(p.json()+'\n')
                        outfile.close()
                        all_docs.extend(paragraphs)

                    
print('\n')

#%%
chunk_dir_pdf = "pdf_data_chunks/"  
if os.path.exists(chunk_dir_pdf):
    shutil.rmtree(chunk_dir_pdf)
os.mkdir(chunk_dir_pdf)   
#all_docs = []
pdf_dir = "plato_pdf/"

pfd_processed = []


def get_header_sizes(mainsize, sizes):
    large = []
    for size in sizes:
        if size > mainsize:
            large.append(size)
    header_sizes = {}
    for n, s in enumerate(sorted(large, reverse=True)[:4]):
        header_sizes[s] = n+1
    return header_sizes

def get_main_size(block):
    sizes = defaultdict(int)
    for l in block["lines"]:
        for s in l["spans"]:
            fontsize =  s.get("size")
            if fontsize:
                sizes[fontsize] += 1
    mainsize = max(sizes, key=sizes.get)
    return mainsize

def get_text(block, headers, mainfontsize):
    text = []
    for l in block["lines"]:
        spans = l["spans"]
        if text == []:
            firstspan = spans[0]
            sizefirst = firstspan["size"]
            firsttext = firstspan["text"]
            if firsttext.strip() == '':
                text.append(firsttext+'\n')
            elif sizefirst > mainfontsize:
                firsttok = firsttext.split()[0]
                numbers = firsttok.split('.')
                if all([x.isdigit() for x in numbers]):
                    hashes = len(numbers)*'#'
                    text.append('\n'+hashes+' ')
                elif sizefirst in headers:
                    hashes = headers[sizefirst]*'#'
                    text.append('\n'+hashes+' ')
        for  s in l["spans"]:
            text.append(s["text"])
        textstr = ' '.join(text)
        textstr.replace('  ', ' ')
        if textstr.strip().endswith('.'):
            textstr = textstr + ' \n'
    return textstr


with open('OneDrive_1_11-03-2024/PLATOChat_papers.json', "r") as file:
    data = json.load(file)
    for i, key in enumerate(data):
        if not data[key]["arXiv ID"] is None:
            ArXivID = data[key]["arXiv ID"]
            title = data[key]["title"]
            link = 'https://arxiv.org/pdf/'+ArXivID+'.pdf'
            if not ArXivID in latex_processed: #ArXivID == '1310.0696',2012.06672', 1411.7511':
                path = pdf_dir+ArXivID+'.pdf'
                if os.path.exists(path):
                    doc = fitz.open(path)
                    md_lines = []
                    textsizes = defaultdict(int)
                    lowest_text = []
                    highest_text =[]
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        blocks = page.get_text("dict", flags=11)["blocks"]
                        #blocks = page.get_text("dict")["blocks"]
                        lowest_on_page = 0
                        highest_on_page = 100
                        for n, block in enumerate(blocks):
                            print(n)
                            print(block)
                            #nlines = len(block["lines"])
                            #print(nlines)
                            # if nlines > 3:
                            #     lowest = block["bbox"][3]
                            #     if lowest > lowest_on_page:
                            #         lowest_on_page = lowest
                            for l in block["lines"]:
                                for s in l["spans"]:
                                    fontsize =  s.get("size")
                                    if fontsize:
                                        textsizes[fontsize] += 1
                            bltext = get_text(block, {}, 10)
                            if len(bltext) > 70:
                                lowest = block["bbox"][3]
                                highest = block["bbox"][1]
                                if lowest > lowest_on_page:
                                    lowest_on_page = lowest
                                if highest < highest_on_page:
                                    highest_on_page = highest            
                        lowest_text.append(lowest_on_page)
                        highest_text.append(highest_on_page)
                    print(textsizes)
                    mainfontsize = max(textsizes, key=textsizes.get)
                    lower_margin = max(lowest_text)
                    upper_margin = min(highest_text)
                    print(mainfontsize)
                    headers = get_header_sizes(mainfontsize, textsizes)
                    for s in headers:
                        print(s, headers[s])
                    print(lower_margin)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        blocks = page.get_text("dict", flags=11)["blocks"]
                        for bl in blocks:
                            bl_size = get_main_size(bl)
                            # if bl_size in headers:
                            #     text = get_text(bl)
                            #     md_lines.append('#'*headers[bl_size]+' '+text)
                            if bl_size >= mainfontsize:
                                orig = bl["bbox"][1]#bl["origin"]
                                if orig < lower_margin and orig > upper_margin:
                                    text = get_text(bl, headers, mainfontsize)
                                    md_lines.append(text)
                    outname = ArXivID+'.json'
                    outfile = open(chunk_dir_pdf+outname, 'w')
                    for l in md_lines:
                        print(l)
                    md = '\n'.join(md_lines)
                    sections = markdown_splitter.split_text(md)
                    paragraphs = []
                    for s in sections:
                        #print(s)
                        s.metadata['source'] = path
                        s.metadata['title'] = title
                        s.metadata['link'] = link
                        for field in s.metadata:
                            if field.startswith('Header'):
                                if '\\label' in s.metadata[field]:
                                    s.metadata[field] = s.metadata[field].split('\\label')[0]
                        #if len(s.page_content.split(' ')) > 150:
                        pars = s.page_content.split(' \n')
                            #print(s.page_content)
                            #print(len(s.page_content.split(' ')))
                        #else:
                            #pars = [s.page_content]
                        for n, p in enumerate(pars):
                            if p and not p.startswith('%'):
                                
                                metadata_par = copy.deepcopy(s.metadata)
                                metadata_par["paragraph"] = str(n+1)
                                par_doc = Document(page_content=p, metadata=metadata_par)
                                paragraphs.append(par_doc)
                                #print(type(par_doc))
                                #print(par_doc)
                    for p in paragraphs:
                        outfile.write(p.json()+'\n')
                    outfile.close()
                    all_docs.extend(paragraphs)
                    

#%%
def extract_headings(pdf_path):
    doc = fitz.open(pdf_path)
    headings = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")

        for block in blocks:
            # block format: (x0, y0, x1, y1, text, block_no, block_type, span_count)
            try:
                for span in page.get_text("dict")["blocks"][block[5]]["lines"][0]["spans"]:
                    # Assuming span contains font size and text
                    text = span["text"]
                    font_size = span["size"]

                    # Example heuristic: if font size is above a threshold, consider it a heading
                    if font_size > 10:  # Adjust this threshold based on your documents
                        headings.append(text.strip())
            except KeyError:
                continue

    return headings
