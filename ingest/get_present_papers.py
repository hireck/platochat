
#%%
import json
import os

latex_dir = '/Users/hilke/data/plato_data/plato_latex'
pdf_dir = '/Users/hilke/data/plato_data/plato_pdf'


present_papers = {}

#def find_downloaded_papers():
with open('OneDrive_1_11-03-2024/PLATOChat_papers.json', "r") as file:
    data = json.load(file)
    print(data)
    for key in data:
        if not data[key]["arXiv ID"] is None:
            ArXivID = data[key]["arXiv ID"]
            latex_name = ArXivID+'_source'
            latexpath = os.path.join(latex_dir, latex_name)
            if os.path.exists(latexpath):
                present_papers[key] = (latex_dir, latex_name)
            else:
                pdf_name = ArXivID+'.pdf'
                pdfpath = os.path.join(pdf_dir, pdf_name)
                if os.path.exists(pdfpath):
                    present_papers[key] = (pdf_dir, pdf_name)

with open('present_papers.json', 'w') as f:
    json.dump(present_papers, f, indent=2)

            
# %%
import importlib, latexml_to_markdown
importlib.reload(latexml_to_markdown)
from latexml_to_markdown import latex_to_markdown
import shutil
import json
import os

latex_dir = '/Users/hilke/data/plato_data/plato_latex'
pdf_dir = '/Users/hilke/data/plato_data/plato_pdf'

skip = ['GDR3_master_include.tex', 'rvs_cep.tex']
exclude = ['_skip', 'acknow', 'authors', 'acronyms', 'notations', 'table', 'rejected', 'psfig', 'binhex', 'natbib']

md_dir = "/Users/hilke/data/plato_data/plato_markdown/" 
if os.path.exists(md_dir):
    shutil.rmtree(md_dir)
os.mkdir(md_dir)   

with open('present_papers.json', 'r') as pp_file:
    present_papers = json.load(pp_file)
    for key in present_papers:
        parent, sourcedir = present_papers[key]
        if parent == latex_dir:
            latexpath = os.path.join(parent, sourcedir)
            print(latexpath)
            for fn in os.listdir(latexpath):
                if fn.endswith('.tex') and not fn in skip and not any([t in fn for t in exclude]):
                    print('\t'+fn)
                    image_dir = os.path.join(md_dir, key+'_images')
                    md = latex_to_markdown(os.path.join(latexpath, fn), image_dir=image_dir)
                    texfile = os.path.join(latexpath, fn)
                    md = latex_to_markdown(texfile)
                    outpath = os.path.join(md_dir, key+'.md')
                    with open(outpath, 'w') as f:
                        f.write(md)
# %%


import re
import shutil
import json
import os
from pdf_to_markdown import pdf_to_markdown_mineru

md_dir = '/Users/hilke/data/plato_data/mineru_test'
os.mkdir(md_dir)
#md_dir = '/Users/hilke/data/plato_data/marker_test'

with open('present_papers.json', 'r') as pp_file:
    present_papers = json.load(pp_file)
    for key in present_papers:
        parent, sourcedir = present_papers[key]
        if parent == pdf_dir:
            pdfpath = os.path.join(parent, sourcedir)
            print(pdfpath)
            image_dir = os.path.join(md_dir, key+'_images')
            md = pdf_to_markdown_mineru(pdfpath, image_dir=image_dir)
            # The converter returns bare-filename image refs (![](name.jpg))
            # that resolve relative to image_dir. We write the .md one level up
            # in md_dir, so prefix the refs with the images subfolder name to
            # keep them resolvable from beside the .md.
            subdir = os.path.basename(image_dir)
            md = re.sub(r'(!\[[^\]]*\]\()(?!https?://|data:|/)',
                        r'\g<1>' + subdir + '/', md)
            outpath = os.path.join(md_dir, key+'.md')
            with open(outpath, 'w') as f:
                f.write(md)

#%%
import os
import json
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5")
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


chunk_dir = '/Users/hilke/data/plato_data/plato_chunks'
#os.mkdir(chunk_dir)

md_dir = '/Users/hilke/data/plato_data/plato_markdown/'
#md_dir = "../../data/ecotalk_data/pdf_slides2markdown/"
#for folder in os.listdir(md_dir):
    #if not folder == '.DS_Store':

metafields = ['title', 'authors', 'journal', "arXiv ID"]    

with open('OneDrive_1_11-03-2024/PLATOChat_papers.json', "r") as file:
    data = json.load(file)
    sectnum = 1
    for key in data:
        mdfile = key+'.md'
        if mdfile in os.listdir(md_dir):
            outname = key+'.json'
            outpath = os.path.join(chunk_dir, outname)
            outfile = open(outpath, 'w') 
            path = os.path.join(md_dir, mdfile)
            with open(path, "r") as f:
                print(path)
                md = f.read()
                sections = markdown_splitter.split_text(md)
                #sectnum = 1
                for s in sections:
                    s.metadata["parent_doc"] = key
                    s.metadata["chunk_number"] = str(sectnum)
                    for field in data[key]:
                        if field in metafields:
                            s.metadata[field] = data[key][field]
                    print(s)
                    outfile.write(s.json()+'\n')
                    sectnum += 1
                outfile.close()

#%%
from sentence_transformers import SentenceTransformer

embedding_model =  SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")#, encode_kwargs={"normalize_embeddings": True},)
#%%
import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.config import Configure
import json

client = weaviate.connect_to_local()
#client = weaviate.connect_to_local(host="localhost", port=8080)
print(client.is_ready())


# #Here we create the collection named "DocumentChunk"

client.collections.delete("DocumentChunk") #Uncomment if you want to delete everything and redo this step

client.collections.create(
    "PLATO",
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="authors", data_type=DataType.TEXT),
        Property(name="journal", data_type=DataType.TEXT),
        #Property(name="year", data_type=DataType.INT),
        Property(name="page_content", data_type=DataType.TEXT),
        Property(name="section_headers", data_type=DataType.TEXT_ARRAY),
        Property(name="parent_doc", data_type=DataType.TEXT),
        Property(name="chunk_number", data_type=DataType.INT),
        Property(name="link", data_type=DataType.TEXT),
    ],
   vectorizer_config=Configure.Vectorizer.none(),
)
client.close()

#Here we load the actual data
#%%
client = weaviate.connect_to_local()
print(client.is_ready())

chunks = client.collections.get("PLATO")

chunk_dir = '/Users/hilke/data/plato_data/plato_chunks'

for fn in os.listdir(chunk_dir):
    print(fn)
    path = os.path.join(chunk_dir, fn)
    with open(path, 'r') as f: #opening the file created with embed_chunks.py
        #lines = f.reeadlines()
        data = [json.loads(l) for l in f]
        texts = [d["page_content"] for d in data]
        embeddings = embedding_model.encode(texts)
        for ch, v in zip(data, embeddings):
            ch["embedding"] = v.tolist()
        for ch in data:
            m = ch["metadata"]
            
            #print(d["page_content"])
            #print(d["bge_dense_vector"])

            # Build the object payload
            chunk_obj = {
                "title": m["title"],
                #"year": int(d["year"]),
                "authors": m["authors"],
                "journal": m["journal"],
                "page_content": ch["page_content"],
                "section_headers": m.get("section_headers"),
                "parent_doc": m["DocID"],
                "chunk_number": int(m["chunk_number"]),
                "link": m.get("arXiv ID")
            }
            print(chunk_obj)
            # Get the vector
            vector = ch["embedding"]

            # Add object (including vector) 
            uuid = chunks.data.insert(
                properties=chunk_obj,
                vector=vector  # Add the custom vector
                # references=reference_obj  # You can add references here
            )
client.close()

# %%
