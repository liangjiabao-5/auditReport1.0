from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os
import sys

if __name__ == "__main__":
    inputText = llm = sys.argv[1]

    with open("./script/output.txt") as f:
        real_estate_sales = f.read()

    # 使用 CharacterTextSplitter 进行文本分割
    text_splitter = CharacterTextSplitter(
        separator = r'\d+\.',
        chunk_size = 160,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = True,
    )
    docs = text_splitter.create_documents([real_estate_sales])

    # 提取文本内容
    texts = [doc.page_content for doc in docs]

    # 使用SentenceTransformer模型
    EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', '/mnt/workspace/m3')
    embedding_model = SentenceTransformer(EMBEDDING_PATH, device="cuda")

    # 使用模型生成文本的嵌入向量
    embeddings = embedding_model.encode(texts, convert_to_tensor=True)

    # 创建Chroma数据库并嵌入文档
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_PATH)  # 移除device参数
    db = Chroma.from_documents(documents=docs, embedding=embedding_function, collection_name="embed")


    # query = "小区吵不吵"
    # answer_list = db.similarity_search(query)
    # for ans in answer_list:
    #     print(ans.page_content + "\n")


    #-----------------------------------------

    # 使用向量数据库进行查询，提取前三个
    topK_retriever = db.as_retriever(search_kwargs={"k": 3})
    results = topK_retriever.get_relevant_documents(inputText)
    # print("使用向量数据库进行查询，提取前三个相近的：-------------------------------------")
    page_contents = ""
    for result in results:
        page_contents += result.page_content + "\n\n"
        # print(result.page_content + "\n")
    # 打印或返回response
    print(page_contents)
