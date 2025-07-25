import streamlit as st
import os
import random
import datetime
from dotenv import load_dotenv

st.set_page_config(layout="wide")
st.markdown("""
<!-- PWA マニフェスト -->
<link rel="manifest" href="static/manifest.json">
<meta name="theme-color" content="#ffffff">

<!-- Service Worker 登録 -->
<script>
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('static/service-worker.js', { scope: './' })
        .then(reg => console.log('ServiceWorker 登録 succeeded:', reg))
        .catch(err => console.log('ServiceWorker 登録 failed:', err));
    });
  }
</script>
""", unsafe_allow_html=True)


# ロジックファイルから必要なものをインポート
# Import necessary components from logic files
from rag_system_logic import qa_chain, rag_memory_for_qa
from sales_analysis_logic import sales_agent_executor, df_sales_for_sales, sales_memory_for_agent
from langchain_core.messages import HumanMessage, AIMessage

# 環境変数をロードし、APIキーの存在を確認
# Load environment variables and check for API key
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    st.error("エラー: OPENAI_API_KEY が設定されていません。環境変数をご確認ください。")
    st.stop()

# ページ設定とモバイル対応メタタグ
st.set_page_config(layout="wide")
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1">',
    unsafe_allow_html=True
)

# カスタムCSSを追加
# Add custom CSS
# カスタムCSS（メディアクエリ付き）
st.markdown(
    """
    <style>
    /* 全体のコンテナ */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* チャットバブル */
    .user-bubble {
        background-color: #dcf8c6;
        color: #000;
        padding: 10px;
        border-radius: 15px;
        max-width: 60%;
        margin-left: auto;
        margin-bottom: 8px;
        word-wrap: break-word;
    }
    .assistant-bubble {
        background-color: #fff;
        color: #000;
        padding: 10px;
        border-radius: 15px;
        max-width: 60%;
        margin-right: auto;
        margin-bottom: 8px;
        border: 1px solid #ddd;
        word-wrap: break-word;
    }
    /* タイムスタンプ */
    .timestamp {
        font-size: 0.7rem;
        color: #888;
        text-align: right;
        margin-top: 4px;
    }
    /* 入力フォーム固定 */
    .fixed-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f2f6;
        padding: 10px 1rem;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    /* Streamlit デフォルト干渉除去 */
    div.stChatMessage { background-color: transparent; }
    div.stChatMessage > div:first-child { display: none; }
    .stChatInputContainer { padding: 0; }

    /* —— モバイル／タブレット対応 —— */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0 !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        .user-bubble, .assistant-bubble {
            max-width: 90% !important;
            padding: 8px !important;
            border-radius: 10px !important;
        }
        .fixed-input-container {
            padding: 8px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.set_page_config(layout="wide")
st.title("OO店AIチャット")  # タイトルをより自然な日本語に修正

# セッションステート初期化（タイムスタンプ付き）
# Initialize session state (with timestamp)
if "manual_messages" not in st.session_state:
    st.session_state.manual_messages = [
        {
            "role": "assistant",
            "content": "マニュアルに関して",  # 初期メッセージを調整
            "timestamp": datetime.datetime.now()
        }
    ]
if "sales_messages" not in st.session_state:
    st.session_state.sales_messages = [
        {
            "role": "assistant",
            "content": "売上データに関して",  # 初期メッセージを調整
            "timestamp": datetime.datetime.now()
        }
    ]

# 各入力フィールドの現在の値を保持するためのstate
# State to hold current values of input fields
if "manual_input_text" not in st.session_state:
    st.session_state.manual_input_text = ""
if "sales_input_text" not in st.session_state:
    st.session_state.sales_input_text = ""

# タブ定義
# Tab definition
tab1, tab2 = st.tabs(["マニュアル検索", "売上分析"])

# ——————————
# マニュアル検索タブ
# ——————————
with tab1:
    st.header("マニュアル検索")

    # メッセージ表示
    # Display messages
    for msg in st.session_state.manual_messages:
        content = msg["content"].replace("\n", "<br>")
        ts = msg.get("timestamp")
        ts_str = ts.strftime("%Y/%m/%d %H:%M:%S") if ts else ""
        if msg["role"] == "user":
            st.markdown(f"""
                <div class='user-bubble'>
                  👤 {content}
                  <div class='timestamp'>{ts_str}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='assistant-bubble'>
                  🤖 {content}
                  <div class='timestamp'>{ts_str}</div>
                </div>
            """, unsafe_allow_html=True)

    # 入力フォーム
    # Input form
    with st.container():
        st.markdown("<div class='fixed-input-container'>", unsafe_allow_html=True)
        with st.form(key="manual_form", clear_on_submit=True):
            user_input = st.text_area(
                "マニュアル質問入力",  # ★★★ 修正点1: labelに空ではない文字列を設定 ★★★
                height=80,
                placeholder="マニュアルに関する質問を入力してください。",
                key="manual_user_input",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("送信")
        st.markdown("</div>", unsafe_allow_html=True)

    # フォームが送信された場合の処理
    # Process when form is submitted
    if submitted and user_input:
        # ユーザーメッセージ追加（タイムスタンプ付き）
        st.session_state.manual_messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.datetime.now()
        })
        # LLM 呼び出し
        with st.spinner("マニュアル検索中…"):
            try:
                res = qa_chain.invoke({"question": user_input})
                answer = res.get("answer", "回答が見つかりませんでした。")
                rag_memory_for_qa.save_context({"question": user_input}, {"answer": answer})
                source_docs = res.get("source_documents", [])

                # 回答と参照箇所表示フォーマット
                full_response = answer
                # 参照ドキュメントごとに番号付きでスニペットとリンクを改行表示
                for idx, doc in enumerate(source_docs, start=1):
                    src_doc = doc.metadata.get('source', '')
                    pdf_file = os.path.abspath(src_doc.replace('.docx', '.pdf'))
                    snippet = doc.page_content.strip().replace('\n', ' ')[:100]
                    page_num = doc.metadata.get('page', 1)
                    link = f"file://{pdf_file}#page={page_num}"
                    # 番号は⑴、⑵などで表示したい場合、以下のように直接文字を指定できます
                    num_mark = f"({idx})"
                    full_response += f"\n\n{num_mark} {snippet}\n　[参照元PDF]({link})"

                # チャットに表示
                st.session_state.manual_messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.datetime.now()
                })
            except Exception as e:
                st.error(f"例外が発生しました: {e}")
                import traceback; st.text(traceback.format_exc())
                st.session_state.manual_messages.append({
                    "role": "assistant",
                    "content": f"【例外】 {e}",
                    "timestamp": datetime.datetime.now()
                })
                st.stop()
        st.rerun()


# ——————————
# 売上分析タブ
# ——————————
with tab2:
    st.header("売上分析")

    # メッセージ表示
    # Display messages
    for msg in st.session_state.sales_messages:
        content = msg["content"].replace("\n", "<br>")
        ts = msg.get("timestamp")
        ts_str = ts.strftime("%Y/%m/%d %H:%M:%S") if ts else ""
        if msg["role"] == "user":
            st.markdown(f"""
                <div class='user-bubble'>
                  👤 {content}
                  <div class='timestamp'>{ts_str}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='assistant-bubble'>
                  🤖 {content}
                  <div class='timestamp'>{ts_str}</div>
                </div>
            """, unsafe_allow_html=True)

    # 入力フォーム
    # Input form
    with st.container():
        st.markdown("<div class='fixed-input-container'>", unsafe_allow_html=True)
        with st.form(key="sales_form", clear_on_submit=True):
            user_input_sales = st.text_area(
                "売上質問入力",  # ★★★ 修正点2: labelに空ではない文字列を設定 ★★★
                height=80,
                placeholder="売上データに関する質問を入力してください。",
                key="sales_user_input",
                label_visibility="collapsed"
            )
            submitted_sales = st.form_submit_button("送信")
        st.markdown("</div>", unsafe_allow_html=True)

    # フォームが送信された場合の処理
    if submitted_sales and user_input_sales:
        # ユーザーメッセージ追加
        st.session_state.sales_messages.append({
            "role": "user",
            "content": user_input_sales,
            "timestamp": datetime.datetime.now()
        })
        with st.spinner("売上分析中…"):
            try:
                hist = sales_memory_for_agent.chat_memory.messages
                resp = sales_agent_executor.invoke({
                    "input": user_input_sales,
                    "chat_history": hist
                })
                out = resp.get("output", "回答が見つかりませんでした。")

                # もしAIがグラフ作成を指示するような内容を生成したら、実際にグラフを表示
                if "data:image/png;base64," in out:  # out変数を使用
                    st.image(out, caption="売上トレンドグラフ")
                    out = out.replace("data:image/png;base64,", "") 
                    st.markdown(f"上記が売上トレンドのグラフです。\n\n{out}")
                else:
                    st.markdown(out)

                st.session_state.sales_messages.append({
                    "role": "assistant",
                    "content": out,
                    "timestamp": datetime.datetime.now()
                })

                sales_memory_for_agent.save_context(
                    {"input": user_input_sales},
                    {"output": out}
                )
            except Exception as e:
                friendly_error_message = "申し訳ございません、現在システムに問題が発生しているため、回答できません。時間をおいて再度お試しください。"
                if "insufficient_quota" in str(e):
                    friendly_error_message = "申し訳ございません、AIサービスの利用上限に達しているため、現在回答できません。管理者にご連絡ください。"

                st.error(f"エラーが発生しました。詳細はログをご確認ください。")
                st.session_state.sales_messages.append({
                    "role": "assistant",
                    "content": friendly_error_message,
                    "timestamp": datetime.datetime.now()
                })
        st.rerun()
