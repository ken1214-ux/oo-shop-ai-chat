# rag_system_logic.py

import os
from docx2pdf import convert
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.document_loaders import Docx2txtLoader
import streamlit as st # st.warning, st.error を使うため、streamlitもインポート
from langchain.prompts import PromptTemplate


# 環境変数をロード
load_dotenv()

@st.cache_resource
def load_rag_components():
    """マニュアルデータを読み込み、ベクトルDBを準備し、RAGチェーンを返す"""

    # LLMの準備 (RAGロジック内で必要なのでここで定義)
    # ★★★ 修正点1: temperatureを0.2に変更 ★★★
    rag_llm_instance = OpenAI(model_name="gpt-4o-mini", temperature=0.2)


    manual_files = [
        "data/charge_manual.docx"
    ]

    for file_path in manual_files:
        pdf_path = file_path.replace(".docx", ".pdf")
        # docx2pdfでPDFが古いか存在しなければ再生成
        if not os.path.exists(pdf_path) or os.path.getmtime(pdf_path) < os.path.getmtime(file_path):
            try:
                convert(file_path, pdf_path)  # ➕ docx2pdf の convert を呼ぶ
            except Exception as e:
                st.error(f"エラー: {file_path} のPDF変換中に問題が発生しました: {e}")
                return None, None

    manual_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file_path in manual_files:
        if not os.path.exists(file_path):
            st.warning(f"警告: マニュアルファイル '{file_path}' が見つかりませんでした。スキップします。")
            continue
        try:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            manual_documents.extend(text_splitter.split_documents(docs))
            # ★★★ 修正点2: 読み込みとチャンク分割のst.infoメッセージを削除 ★★★
            # st.info(f"マニュアルファイル {file_path} を読み込み、チャンク分割しました。")
        except Exception as e:
            st.error(f"エラー: {file_path} の読み込み中に問題が発生しました: {e}")

    embeddings = OpenAIEmbeddings()
    persist_directory = 'db'
    
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        # ★★★ 修正点3: 既存DBロードのst.infoメッセージを削除 ★★★
        # st.info("既存のベクトルデータベースをロードしました。")
    else:
        if not manual_documents:
            st.error("エラー: 読み込むマニュアルドキュメントが見つからず、ベクトルデータベースを作成できませんでした。")
            st.stop()
        # ★★★ 修正点4: 新規DB作成のst.infoメッセージを削除 ★★★
        # st.info("ベクトルデータベースを新規作成し、マニュアルを保存します。初回のみ時間がかかります...")
        vectordb = Chroma.from_documents(documents=manual_documents, embedding=embeddings, persist_directory=persist_directory)
        vectordb.persist()
        # ★★★ 修正点5: 成功メッセージのst.successを削除 ★★★
        # st.success("ベクトルデータベースの作成と保存が完了しました。")
    
    rag_memory = ConversationSummaryBufferMemory(
        llm=rag_llm_instance,
        max_token_limit=1000,
        memory_key="chat_history",
        input_key="question",      # ← 質問のキー
        output_key="answer",       # ← 応答のキー
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=rag_llm_instance,
        retriever=vectordb.as_retriever(),
        memory=rag_memory,
        return_source_documents=True,
        output_key="answer",
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate.from_template(
                """あなたは店舗のマニュアルアシスタントです。
                以下の「コンテキスト」のみを参考にして、店長からの質問に明確かつ簡潔に答えてください。
                もしコンテキストに質問の答えが含まれていない場合は、「マニュアルにはその情報が見つかりませんでした。」と回答してください。
                コンテキスト外の情報は一切使用しないでください。

                チャット履歴:
                {chat_history}

                コンテキスト:
                {context}

                質問: {question}
                回答:"""
            )
        }
    )
    return qa_chain, rag_memory

qa_chain, rag_memory_for_qa = load_rag_components()
