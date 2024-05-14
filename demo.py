import streamlit as st

#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader

load_dotenv()

# handle streaming conversation
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# function to extract text from an HWP file
import olefile
import zlib
import struct

def get_hwp_text(filename):
    f = olefile.OleFileIO(filename)
    dirs = f.listdir()

    # HWP 파일 검증
    if ["FileHeader"] not in dirs or \
       ["\x05HwpSummaryInformation"] not in dirs:
        raise Exception("Not Valid HWP.")

    # 문서 포맷 압축 여부 확인
    header = f.openstream("FileHeader")
    header_data = header.read()
    is_compressed = (header_data[36] & 1) == 1

    # Body Sections 불러오기
    nums = []
    for d in dirs:
        if d[0] == "BodyText":
            nums.append(int(d[1][len("Section"):]))
    sections = ["BodyText/Section"+str(x) for x in sorted(nums)]

    # 전체 text 추출
    text = ""
    for section in sections:
        bodytext = f.openstream(section)
        data = bodytext.read()
        if is_compressed:
            unpacked_data = zlib.decompress(data, -15)
        else:
            unpacked_data = data
    
        # 각 Section 내 text 추출    
        section_text = ""
        i = 0
        size = len(unpacked_data)
        while i < size:
            header = struct.unpack_from("<I", unpacked_data, i)[0]
            rec_type = header & 0x3ff
            rec_len = (header >> 20) & 0xfff

            if rec_type in [67]:
                rec_data = unpacked_data[i+4:i+4+rec_len]
                section_text += rec_data.decode('utf-16')
                section_text += "\n"

            i += 4 + rec_len

        text += section_text
        text += "\n"

    return text

# Function to extract text from an PDF file
from pdfminer.high_level import extract_text

def get_pdf_text(filename):
    raw_text = extract_text(filename)
    return raw_text

# document preprocess
#def process_uploaded_file(uploaded_file):
#    # Load document if file is upload노트
#    if uploaded_file is not None:
#        # loader
#        # pdf파일을 처리하려면?
#        if uploaded_file.type == 'application/pdf':
#            raw_text = get_pdf_text(uploaded_file)
#                    
#        # splitter
#        text_splitter = CharacterTextSplitter(
#            separator = "\n\n",
#            chunk_size = 1000,
#            chunk_overlap  = 200,
#            length_function = len,
#            is_separator_regex = False,
#        )
#        all_splits = text_splitter.create_documents([raw_text])
#        print("총 " + str(len(all_splits)) + "개의 passage")
#        
#        # storage
#        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
#                
#        return vectorstore, raw_text
#    return None

def process_uploaded_file(uploaded_files):
    vectorstores = {}
    raw_texts = {}
    try:
        for i, uploaded_file in enumerate(uploaded_files):
            if uploaded_file.type == 'application/pdf':
                raw_text = get_pdf_text(uploaded_file)
            else:
                st.error(f"Unsupported file type: {uploaded_file.type}")
                continue

            raw_text = raw_text.replace('\n\n', '\n').strip()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            all_splits = text_splitter.create_documents([raw_text])
            st.write(f"Lecture {i+1} 총 " + str(len(all_splits)) + "개의 passage")
            
            vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
            vectorstores[f'Lecture {i+1}'] = vectorstore
            raw_texts[f'Lecture {i+1}'] = raw_text

        return vectorstores, raw_texts
    except Exception as e:
        st.error(f"Error processing files: {e}")
        return None, None

def get_script(url, language="en", add_video_info=True):
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=add_video_info,
        language=language,
    )
    return loader.load()

def load_youtube_scripts(urls):
    scripts = []
    for url in urls:
        try:
            script = get_script(url)
            for doc in script:
                scripts.append(doc.page_content)  # Ensure we extract the correct content
        except Exception as e:
            st.error(f"Error loading script for {url}: {e}")
    return scripts

# generate response using RAG technic
def generate_response(query_text, pdf_vectorstore, youtube_vectorstore, callback):
    pdf_docs = pdf_vectorstore.similarity_search(query_text, k=5)
    pdf_text = "".join([f"'문서{i+1}': {doc.page_content}\n" for i, doc in enumerate(pdf_docs)])
    
    youtube_docs = youtube_vectorstore.similarity_search(query_text, k=5)
    youtube_text = "".join([f"'비디오{i+1}': {doc.page_content}\n" for i, doc in enumerate(youtube_docs)])

    examples = [
        {
            "role": "user",
            "content": "강의노트에서 '전기 회로'에 대한 설명을 알려줘."
        },
        {
            "role": "assistant",
            "content": "전기 회로는 전기가 흐르는 통로를 말합니다. 전기 회로는 배터리, 전선, 저항, 전구 등으로 구성됩니다. 전기가 흐르기 위해서는 회로가 닫혀 있어야 합니다. 예를 들어, 배터리에서 나온 전기가 전선을 통해 저항과 전구를 거쳐 다시 배터리로 돌아가는 경로가 완성될 때 전기 회로가 됩니다. 🔋➡️💡"
        },
        {
            "role": "user",
            "content": "강의노트에서 '진동'에 대한 설명을 알려줘."
        },
        {
            "role": "assistant",
            "content": "진동은 물체가 일정한 간격으로 반복해서 움직이는 현상을 말합니다. 예를 들어, 시계의 추나 기타 줄의 움직임이 진동의 예입니다. 진동의 주기와 진폭은 각각 진동이 한 번 완료되는 데 걸리는 시간과 진동의 최대 변위입니다. 🎸↔️🎶"
        }
    ]
    
    # generator
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, streaming=True, callbacks=[callback])
    
    # chaining
    rag_prompt = [
        SystemMessage(
            content="너는 강의노트와 YouTube 강의 링크에 대해 질의응답을 하는 '교수'야. 주어진 강의노트와 YouTube 강의 링크를 참고하여 사용자의 질문에 답변을 해줘. 노트에 내용이 정확하게 나와있지 않으면 너의 지식 선에서 잘 얘기해줘. 답변은 논리적으로 이해 하기 쉽게 예를 들어서 설명해줘. 이모티콘을 적절히 추가하여 이해를 도와줘! 답변을 잘하면 200달러 팁을 줄게"
        ),
        HumanMessage(
            content=f"질문:{query_text}\n\n강의노트:\n{pdf_text}\n\nYouTube 강의 내용:\n{youtube_text}"
        ),
        *examples
    ]

    response = llm(rag_prompt)
    
    return response.content


def generate_summarize(raw_text, callback):
    # generator 
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, streaming=True, callbacks=[callback])
    
    # prompt formatting
    rag_prompt = [
        SystemMessage(
            content="다음 나올 문서를 'Notion style'로 요약해줘. 중요한 내용만."
        ),
        HumanMessage(
            content=raw_text
        ),
    ]
    
    response = llm(rag_prompt)
    return response.content


# page title
st.set_page_config(page_title='🎓 CS182 강의봇 🤖')
st.title('🎓 CS182 강의봇 🤖')

# enter token
import os
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
save_button = st.sidebar.button("Save Key")
if save_button and len(api_key)>10:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API Key saved successfully!")

# 초기화 코드 추가
if 'vectorstores' not in st.session_state:
    st.session_state['vectorstores'] = {}

if 'youtube_vectorstores' not in st.session_state:
    st.session_state['youtube_vectorstores'] = {}

if 'youtube_scripts' not in st.session_state:
    st.session_state['youtube_scripts'] = {}

if 'raw_texts' not in st.session_state:
    st.session_state['raw_texts'] = {}

# file upload
uploaded_file = st.file_uploader('Upload lecture PDFs', type=['pdf'], accept_multiple_files=True)

# file upload logic
if uploaded_file:
    vectorstore, raw_text = process_uploaded_file(uploaded_file)
    if vectorstore:
        st.session_state['vectorstore'] = vectorstore
        st.session_state['raw_text'] = raw_text

lecture_titles = [
    "Lecture 1: Introduction.",
    "Lecture 2: ML Basics 1.",
    "Lecture 3: ML Basics 2.",
    "Lecture 4: Optimization.",
    "Lecture 5: Backpropagation.",
    "Lecture 6: Convolutional Nets.",
    "Lecture 7: Getting Neural Nets to Train.",
    "Lecture 8: Computer Vision.",
    "Lecture 9: Generating Images from CNNs.",
    "Lecture 10: Recurrent Neural Networks.",
    "Lecture 11: Sequence To Sequence Models."
]
lecture_urls = {
    "Lecture 1": [
        "https://youtu.be/rSY1pVGdZ4I?si=HJ0w04z57oSg3l2T",
        "https://youtu.be/FHsGHxQYxvc?si=swR3M09Xdjk2SJWP",
        "https://youtu.be/s2B0c_o_rbw?si=eZckP8Pyg4fM_Fks"
    ],
    "Lecture 2": [
        "https://youtu.be/aUNnGCxvAg0?si=3J7nzJWWkXpRCBLA",
        "https://youtu.be/oLc822BT-K4?si=SiefY6V5gHgrxsSg",
        "https://youtu.be/zY2QgvPfSm8?si=1f6G4Lplz1KU3CSe",
        "https://youtu.be/voJ4qSH-uqw?si=78149mesZb3s7Q_w"
    ],
    "Lecture 3": [
        "https://youtu.be/PBYWWM9We-0?si=h4r-KtBupFJg3keO",
        "https://youtu.be/U_cpdaJ-adk?si=NtW8t0ePyBVb21Jj",
        "https://youtu.be/-BKfF-odbSQ?si=Xhc9iuIoNLEymFon"
    ],
    "Lecture 4": [
        "https://youtu.be/RdoZWcXmXhk?si=N5LuULe3QIgSMRIm",
        "https://youtu.be/fg3GyrfcclY?si=eWTtemPPOTP2rZrj",
        "https://youtu.be/CO3-sFmADfI?si=lQA0eQOcm4VuRVKd"
    ],
    "Lecture 5": [
        "https://youtu.be/lKRatcD9hEg?si=r5jvTTkdEkJfLroF",
        "https://youtu.be/hpS8oIEQzcs?si=49QCQNknyEEwnqFE",
        "https://youtu.be/JXX5Ea0TXTM?si=J4ZnAEfw0bWxzQDN"
    ],
    "Lecture 6": [
        "https://youtu.be/jNW1Hi7Yi4c?si=M9WySH60Fu_ACYSQ",
        "https://youtu.be/xAcAWaeUxYs?si=U00725uJOmw3Kc68",
        "https://youtu.be/HlJ8rpwKH5c?si=wZxQ54C3rqWt3IOV"
    ],
    "Lecture 7": [
        "https://youtu.be/0dNAhN4ypFc?si=fhPcIy_ZOLBZaYsL",
        "https://youtu.be/k5uLipr49zQ?si=3UJobqmLZ2dvdsfR",
        "https://youtu.be/Nx48Idc0_68?si=2YCsgZdxRbaKqTCi"
    ],
    "Lecture 8": [
        "https://youtu.be/MgabSQ93IE8?si=jnehT9qW1AAnKRMS",
        "https://youtu.be/XHrSobup-vU?si=NKbwf4jeyeXljS68",
        "https://youtu.be/gAH5dH2uTc0?si=BsZS33KsXDPi7996",
        "https://youtu.be/LACVGqw29J0?si=6zCKYMdH7z5FEuQl"
    ],
    "Lecture 9": [
        "https://youtu.be/VKPkM6jt_P0?si=yskt4ZNb7ZOpkvUy",
        "https://youtu.be/AsGaxH7vizk?si=wK3z9cjugbYRNd3I",
        "https://youtu.be/vyfq3SgXQyU?si=NALHovpFSjvxXQYg"
    ],
    "Lecture 10": [
        "https://youtu.be/PyZvbaC5oQY?si=rBY8s8ZYVXFqry4G",
        "https://youtu.be/BOyQQbQzKG4?si=xDzo7xyhQ7_PuR-x",
        "https://youtu.be/EFbKmZdB61g?si=zOyt0b5u1mnJ0DcO"
    ],
    "Lecture 11": [
        "https://youtu.be/36RjPbbcA28?si=KT2pgr0-cEHiMuNm",
        "https://youtu.be/VX5_uKOUliE?si=5hpggIGAOg8n6_fK",
        "https://youtu.be/49wn_m7JE-c?si=4ip0JWDjyvCGmiRE"
    ],
}

st.sidebar.header("강의 목록")
selected_lecture = st.sidebar.selectbox("강의를 선택하세요", lecture_titles)

if selected_lecture:
    lecture_key = selected_lecture.split(":")[0]
    #if "youtube_scripts" not in st.session_state:
    #    st.session_state["youtube_scripts"] = {}
    
    if lecture_key not in st.session_state["youtube_scripts"]:
        st.session_state["youtube_scripts"][lecture_key] = load_youtube_scripts(lecture_urls[lecture_key])

    #if "youtube_vectorstores" not in st.session_state:
    #    st.session_state["youtube_vectorstores"] = {}
    
    if lecture_key not in st.session_state["youtube_vectorstores"]:
        scripts = st.session_state["youtube_scripts"][lecture_key]
        all_splits = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        for script in scripts:
            splits = text_splitter.create_documents([script])
            all_splits.extend(splits)
        try:
            embeddings = OpenAIEmbeddings().embed_documents([doc.page_content for doc in all_splits])
            if not embeddings:
                st.error("Failed to generate embeddings. Check your API key and internet connection.")
            vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
            st.session_state["youtube_vectorstores"][lecture_key] = vectorstore
        except Exception as e:
            st.error(f"Error creating FAISS vectorstore for YouTube scripts of {lecture_key}: {e}")

# chatbot greatings
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant", content="안녕하세요✋! 저는 CS182 강의에 대한 이해를 도와주는 챗봇입니다. 어떤게 궁금하신가요?"
        )
    ]

# conversation history print 
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)
    
# message interaction
if prompt := st.chat_input(f"'{selected_lecture}'에 대한 질문을 입력해보세요!"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    stream_handler = StreamHandler(st.empty())
    if "요약" in prompt.lower():
        if lecture_key in st.session_state['raw_texts']:
            response = generate_summarize(st.session_state['raw_texts'][lecture_key], stream_handler)
        else:
            response = "선택된 강의에 대한 원본 텍스트가 없습니다."
    else:
        if lecture_key in st.session_state['vectorstores'] and lecture_key in st.session_state['youtube_vectorstores']:
            response = generate_response(
                prompt,
                st.session_state['vectorstores'][lecture_key],
                st.session_state['youtube_vectorstores'][lecture_key],
                stream_handler
            )
        else:
            response = "선택된 강의에 대한 데이터가 없습니다."

    st.session_state["messages"].append(ChatMessage(role="assistant", content=response))
    st.chat_message("assistant").write(response)
