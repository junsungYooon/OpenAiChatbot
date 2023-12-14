import os
import pymysql
from flask import Flask, render_template, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = "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"

app = Flask(__name__)

class Document:
    def __init__(self, page_content='', metadata=None):
        if metadata is None:
            metadata = {}
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

# 초기 메모리 및 모델 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
embeddings = OpenAIEmbeddings()
vectorstore = None
qa = None  # 모델을 전역 변수로 설정

# MySQL 연결 설정
connection = pymysql.connect(
    host="127.0.0.1",
    port=3306,
    user="root",
    password= ,   # 실행 전 변경
    database="openai_sql",
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# Already closed 오류 대안으로, sql 서버가 닫혀 있어도 접속하여 연결이 가능
def insertSQL(pdf_path):
    connection = pymysql.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password= NO,
        database="openai_sql",
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    file_name = os.path.basename(pdf_path)
    pages = repr(pages)
    try:
        with connection.cursor() as cursor:
            file_data = {
                'fileName': file_name,
                'fileContent': pages
            }
            sql = "INSERT INTO filetable (fileName, fileContent) VALUES (%s, %s)"
            cursor.execute(sql, (file_data['fileName'], file_data['fileContent']))
        connection.commit()

    finally:
        connection.close()

try:
    i = 0
    new_list = []
    while 1:
        with connection.cursor() as cursor:
            # SELECT 쿼리문
            select_sql = "SELECT fileContent FROM filetable LIMIT 1 OFFSET %s"

            # 데이터베이스에서 데이터 가져오기
            cursor.execute(select_sql, (i,))
            result = cursor.fetchone()

            if result:
                # fileContent가 존재하면 변수에 저장
                retrieved_file_content = result['fileContent']
                new_list.append(eval(retrieved_file_content))
                i = int(i) + 1
            else:
                break

finally:
    # 연결 종료
    connection.close()

flat_list = [item for sublist in new_list for item in sublist]

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(flat_list, embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(
        temperature=0.2,
        model_name="gpt-3.5-turbo"
    ),
    vectorstore.as_retriever(),

    memory=memory
)

# 메인 페이지 라우트
@app.route('/')
def index():
    return render_template('index.html')

# process file 엔드포인트 라우트
@app.route('/process_file', methods=['POST'])
def process_file():
    global vectorstore
    global qa

    file = request.files['file']
    file_path = os.path.join("/Users/kms/Documents/ohey/Flask/filedir", file.filename)

    file.save(file_path)

    # 데이터베이스에 파일 저장
    insertSQL(file_path)

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    for i in range(len(pages)):
        pages[i].page_content = pages[i].page_content

    # 모든 페이지에 대해 줄바꿈 문자 제거
    for page in pages:
        page.page_content = page.page_content.replace("\n", "")

    vectorstore = Chroma.from_documents(pages, embeddings)
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(
            temperature=0.8,
            model_name="gpt-3.5-turbo"
        ),
        vectorstore.as_retriever(),
        memory=memory
    )

    return jsonify({'message': '파일이 성공적으로 업로드되었습니다.'})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.form.get('question')

    result = qa({"question": question})
    answer = (result["answer"])

    return jsonify({'answer': answer})

# 애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)
