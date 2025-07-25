# sales_analysis_logic.py

import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import OpenAI # ここは変更なし
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain # このインポートはAgentを使うので直接は使わないが、残しておく
from langchain.memory import ConversationSummaryBufferMemory

# --- Function Calling (Tools) のための追加インポート ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st # st.warning, st.error, st.info を使うため、streamlitもインポート

# データ分析用関数のインポート
from data_analyzer import create_sales_trend_chart, get_top_selling_products, get_sales_data_for_period, compare_sales_periods, get_product_sales_details

# 環境変数をロード (ファイルの先頭でOK)
load_dotenv()

# sales_llm の定義を load_sales_data_and_agent 関数の中に移動します
# sales_llm = OpenAI(model_name="gpt-4o-mini", temperature=0.5) # この行を削除またはコメントアウト

@st.cache_resource
def load_sales_data_and_agent(): # ★★★ ここを修正：引数 llm を削除 ★★★
    """売上データを読み込み、ツールとAgentExecutorを準備する"""

    # LLMの準備 (Agent用) - この関数内でインスタンスを作成
    sales_llm_instance = OpenAI(model_name="gpt-4o-mini", temperature=0.5) # ★★★ ここに移動 ★★★

    # --- 売上データの読み込み ---
    sales_data_path = "data/sales_data.csv" # あなたの売上ファイル名に修正
    df_sales = pd.DataFrame()
    try:
        df_sales = pd.read_csv(sales_data_path)
        if '日付' in df_sales.columns:
            df_sales['日付'] = pd.to_datetime(df_sales['日付'])
    except FileNotFoundError:
        st.error(f"エラー: 売上データファイル '{sales_data_path}' が見つかりません。")
        st.stop()
    except Exception as e:
        st.error(f"エラー: 売上データの読み込みまたは処理中に問題が発生しました: {e}")
        st.stop()

    # @tool デコレータを使って、LLMに呼び出させる関数を定義します
    # ツール関数内でdf_salesを使えるように、ラッパー関数を定義
    @tool
    def get_top_selling_products_tool(top_n: int = 5, metric: str = "amount", period: str = None) -> str:
        """
        現在の売上データから最も売れている商品を抽出します。
        引数:
            top_n (int): 上位何件の商品を抽出するか。デフォルトは5。
            metric (str): 'amount' (売上金額) または 'quantity' (販売点数) で指定。デフォルトは 'amount'。
            period (str): 'today', 'yesterday', 'last_week', 'this_month', 'last_month', 'all' から選択（省略時は'all'）。
        例: get_top_selling_products_tool(top_n=3, metric='quantity', period='last_week')
        """
        return get_top_selling_products(df_sales, top_n=top_n, metric=metric, period=period) # period引数も渡す

    @tool
    def create_sales_chart_tool(period: str = "daily") -> str:
        """
        売上データのトレンドグラフ（線グラフ）を作成し、Base64エンコードされた画像として返します。
        期間は 'daily' (日次) または 'monthly' (月次) を指定できます。
        例: create_sales_chart_tool(period='monthly')
        """
        return create_sales_trend_chart(df_sales, period=period)

    @tool
    def compare_sales_periods_tool(
        start_date1: str, end_date1: str,
        start_date2: str, end_date2: str,
        metric: str = "amount"
    ) -> str:
        """
        指定された2つの期間の売上を比較し、その結果をテキストで返します。
        日付フォーマットは 'YYYY-MM-DD' を使用してください。
        例: compare_sales_periods_tool(start_date1='2023-01-01', end_date1='2023-01-07', start_date2='2023-01-08', end_date2='2023-01-14', metric='amount')
        """
        return compare_sales_periods(df_sales, start_date1, end_date1, start_date2, end_date2, metric)

    @tool
    def get_product_sales_details_tool(product_name: str, period: str = None) -> str:
        """
        指定された商品の詳細な売上データを返します。商品名の一部でも検索可能です。
        periodは 'today', 'yesterday', 'last_week', 'this_month', 'last_month', 'all' から選択（省略時は'all'）。
        例: get_product_sales_details_tool(product_name='アウター', period='this_month')
        """
        return get_product_sales_details(df_sales, product_name, period=period) # period引数も渡す

    # ツールをリストにまとめる
    tools = [
        get_top_selling_products_tool,
        create_sales_chart_tool,
        compare_sales_periods_tool,
        get_product_sales_details_tool
    ]

    # Agentのプロンプトをハブからロード
    prompt = hub.pull("hwchase17/react-chat")

    # Agentの作成
    sales_agent = create_react_agent(sales_llm_instance, tools, prompt) # ★★★ llm引数を sales_llm_instance に修正 ★★★
    sales_agent_executor = AgentExecutor(agent=sales_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # AgentExecutor用のメモリをここで初期化
    sales_memory = ConversationSummaryBufferMemory(llm=sales_llm_instance, max_token_limit=1000, memory_key="chat_history", return_messages=True) # ★★★ llm引数を sales_llm_instance に修正 ★★★

    return sales_agent_executor, df_sales, sales_memory # AgentExecutor, df_sales, memoryを返す

# AgentExecutor, df_sales, sales_memoryをロードする関数を呼び出し、変数に格納
# ★★★ ここも修正：引数なしで呼び出す ★★★
sales_agent_executor, df_sales_for_sales, sales_memory_for_agent = load_sales_data_and_agent()

# このファイルがインポートされたときに、sales_agent_executor, df_sales_for_sales, sales_memory_for_agent が利用可能になる
