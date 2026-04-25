import os
import operator
import pandas as pd
from ddgs import DDGS
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_core.tools import Tool
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from classification import NewsPredictor
from langgraph.graph import StateGraph, END
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


class DisinfoAgent:
    def __init__(self,input_path: str,):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.serp_api_key = os.environ.get("SERP_API_KEY")

        self.predictor = NewsPredictor(input_path)

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key = self.openai_api_key)
        self.tools = self._build_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def duck_search(self, query, max_results=5):
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r["title"] + " - " + r["href"])
        except Exception as e:
            print("DuckDuckGo error:", e)
        return results

    def serp_search(self, query, max_results=5):
        results = []
        try:
            params = {
                "q": query,
                "hl": "uk",
                "api_key": self.serp_api_key,
                "tbs": "qdr:m"
            }
            search = GoogleSearch(params)
            data = search.get_dict()

            for r in data.get("organic_results", [])[:max_results]:
                results.append(r["title"] + " - " + r["link"])
        except Exception as e:
            print("SerpAPI error:", e)
        return results

    def smart_search(self, query: str) -> list:
        today = datetime.now().strftime("%Y-%m-%d")

        results = self.serp_search(query)

        if len(results) < 3:
            query = f"{query} новини {today}"
            duck_results = self.duck_search(query)
            results += duck_results

        return results

    def _build_tools(self) -> list:
        return [
            Tool(
                name="web_search",
                description="Шукає інформацію в інтернеті для перевірки новин. Використовує DuckDuckGo і fallback на Google.",
                func=self.smart_search,
            )
        ]

    def _should_continue(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    def _call_llm(self, state: AgentState) -> AgentState:
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _build_graph(self) -> StateGraph:
        tool_node = ToolNode(self.tools)

        graph = StateGraph(AgentState)
        graph.add_node("llm", self._call_llm)
        graph.add_node("tools", tool_node)

        graph.set_entry_point("llm")
        graph.add_conditional_edges("llm", self._should_continue)
        graph.add_edge("tools", "llm")

        return graph.compile()

    def _verify_one(self, text: str, roberta_label: str, roberta_chance: float) -> str:
        system_prompt = SystemMessage(content=f"""Сьогоднішня дата: {datetime.now().strftime("%Y-%m-%d")}
                                      Ти — експерт з верифікації новин та виявлення дезінформації.
                                      Твоє завдання:
                                      1. Прийми до уваги попередню оцінку класифікатора XLM-RoBERTa
                                      2. Використай інструмент пошуку щоб знайти підтвердження або спростування
                                      3. Сформуй чіткий висновок: ПРАВДА / ФЕЙК / НЕДОСТАТНЬО ДАНИХ
                                      4. Поясни своє рішення з посиланням на знайдені джерела
                                      Відповідай українською мовою.""")

        user_message = HumanMessage(content=f"""Перевір цю новину:
                                    {text}
                                    Попередня оцінка класифікатора XLM-RoBERTa:
                                    - Фейк: {roberta_label}
                                    - Шанс фейку: {roberta_chance}%""")

        result = self.graph.invoke({
            "messages": [system_prompt, user_message]
        })

        return result["messages"][-1].content

    def run(self) -> pd.DataFrame:
        df = self.predictor.run()

        df["Висновок агента"] = df.apply(
            lambda row: self._verify_one(
                text=row["Text"],
                roberta_label=row["Class"],
                roberta_chance=row["Fake Chance"],
            ),
            axis=1,
        )

        return df


if __name__ == "__main__":
    agent = DisinfoAgent(
        input_path="uploaded_data.csv",
    )

    result = agent.run()
    print(result)