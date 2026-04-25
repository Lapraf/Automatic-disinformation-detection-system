import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from agent import DisinfoAgent

SAVE_PATH = "uploaded_data.csv"

BG = "#1e1e2e"
SURFACE = "#2a2a3e"
ACCENT = "#534AB7"
ACCENT_HOVER = "#6a5ae0"
GREEN_BG = "#1a3a2a"
GREEN_FG = "#4caf87"
RED_BG = "#3a1a1a"
RED_FG = "#e05c6a"
ORANGE_BG = "#3a2a1a"
ORANGE_FG = "#f0a04b"
TEXT = "#e0e0f0"
MUTED = "#8888aa"
BORDER = "#3a3a5a"
FONT = ("Segoe UI", 11)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_RESULT = ("Segoe UI", 10)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Disinformation Checker")
        self.root.geometry("780x700")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.dataset_path = None
        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg=ACCENT, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="Disinformation Checker",
                 font=FONT_TITLE, bg=ACCENT, fg="white").pack()
        tk.Label(header, text="XLM-RoBERTa + GPT-4o-mini",
                 font=FONT_SMALL, bg=ACCENT, fg="#ccc").pack()

        main = tk.Frame(self.root, bg=BG, padx=24, pady=16)
        main.pack(fill="both", expand=True)

        tk.Label(main, text="Текст новини для перевірки",
                 font=FONT_BOLD, bg=BG, fg=TEXT).pack(anchor="w", pady=(0, 4))

        text_frame = tk.Frame(main, bg=BORDER, padx=1, pady=1)
        text_frame.pack(fill="x")
        self.text_input = tk.Text(
            text_frame, height=7, font=FONT,
            bg=SURFACE, fg=TEXT, insertbackground=TEXT,
            relief="flat", padx=10, pady=8, wrap="word",
            selectbackground=ACCENT
        )
        self.text_input.pack(fill="x")

        btn_frame = tk.Frame(main, bg=BG)
        btn_frame.pack(fill="x", pady=12)

        self.upload_btn = tk.Button(
            btn_frame, text="Завантажити CSV",
            command=self.load_file, font=FONT,
            bg=SURFACE, fg=TEXT, relief="flat",
            padx=14, pady=7, cursor="hand2",
            activebackground=BORDER, activeforeground=TEXT,
            bd=0
        )
        self.upload_btn.pack(side="left")

        self.file_label = tk.Label(
            btn_frame, text="Файл не обрано",
            font=FONT_SMALL, bg=BG, fg=MUTED
        )
        self.file_label.pack(side="left", padx=10)

        self.check_btn = tk.Button(
            btn_frame, text="Перевірити",
            command=self.run_agent, font=FONT_BOLD,
            bg=ACCENT, fg="white", relief="flat",
            padx=20, pady=7, cursor="hand2",
            activebackground=ACCENT_HOVER, activeforeground="white",
            bd=0
        )
        self.check_btn.pack(side="right")

        self.progress = ttk.Progressbar(main, mode="indeterminate")
        self.status_label = tk.Label(
            main, text="", font=FONT_SMALL, bg=BG, fg=MUTED
        )

        tk.Label(main, text="Результат перевірки",
                 font=FONT_BOLD, bg=BG, fg=TEXT).pack(anchor="w", pady=(4, 6))

        cards_frame = tk.Frame(main, bg=BG)
        cards_frame.pack(fill="x", pady=(0, 10))

        self.verdict_card = self._metric_card(cards_frame, "Вердикт", "—")
        self.verdict_card["frame"].pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.chance_card = self._metric_card(cards_frame, "Шанс фейку", "—")
        self.chance_card["frame"].pack(side="left", expand=True, fill="x", padx=3)

        self.class_card = self._metric_card(cards_frame, "Клас RoBERTa", "—")
        self.class_card["frame"].pack(side="left", expand=True, fill="x", padx=(6, 0))

        explanation_frame = tk.Frame(main, bg=SURFACE, padx=14, pady=12)
        explanation_frame.pack(fill="both", expand=True)

        tk.Label(explanation_frame, text="Пояснення агента",
                 font=FONT_SMALL, bg=SURFACE, fg=MUTED).pack(anchor="w")

        self.result_text = tk.Text(
            explanation_frame, font=FONT_RESULT,
            bg=SURFACE, fg=TEXT, relief="flat",
            wrap="word", padx=4, pady=6,
            state="disabled", cursor="arrow",
            selectbackground=ACCENT
        )
        self.result_text.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(explanation_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

    def _metric_card(self, parent, label, value):
        frame = tk.Frame(parent, bg=SURFACE, padx=12, pady=10)
        lbl = tk.Label(frame, text=label, font=FONT_SMALL, bg=SURFACE, fg=MUTED)
        lbl.pack(anchor="w")
        val = tk.Label(frame, text=value, font=FONT_BOLD, bg=SURFACE, fg=TEXT)
        val.pack(anchor="w", pady=(2, 0))
        return {"frame": frame, "label": lbl, "value": val}

    def _update_card(self, card, value, color):
        card["value"].config(text=value, fg=color)
        card["frame"].config(bg=color.replace("e0", "1a").replace("c6", "1a") if "4c" in color else SURFACE)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            import os
            df = pd.read_csv(file_path)
            df.to_csv(SAVE_PATH, index=False, sep="|")
            self.dataset_path = SAVE_PATH
            self.text_input.delete("1.0", tk.END)
            self.file_label.config(
                text=f"✓ {os.path.basename(file_path)}", fg=GREEN_FG
            )

    def _set_loading(self, loading: bool):
        if loading:
            self.check_btn.config(state="disabled", text="Перевірка...")
            self.progress.pack(fill="x", pady=(0, 4))
            self.progress.start(10)
            self.status_label.config(text="Агент аналізує новину...")
            self.status_label.pack()
        else:
            self.check_btn.config(state="normal", text="Перевірити")
            self.progress.stop()
            self.progress.pack_forget()
            self.status_label.pack_forget()

    def _show_result(self, result_df: pd.DataFrame):
        row = result_df.iloc[0]
        verdict_text = row.get("Висновок агента", "")
        fake_label = row.get("Class", "—")
        fake_chance = row.get("Fake Chance", "—")

        upper = verdict_text.upper()
        if "ФЕЙК" in upper:
            color, verdict = RED_FG, "ФЕЙК"
        elif "ПРАВДА" in upper:
            color, verdict = GREEN_FG, "ПРАВДА"
        else:
            color, verdict = ORANGE_FG, "НЕДОСТАТНЬО ДАНИХ"

        self.verdict_card["value"].config(text=verdict, fg=color)
        self.chance_card["value"].config(text=f"{fake_chance}%", fg=TEXT)
        self.class_card["value"].config(text=str(fake_label), fg=TEXT)

        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", verdict_text)
        self.result_text.config(state="disabled")

    def run_agent(self):
        text = self.text_input.get("1.0", tk.END).strip()

        if not text and not self.dataset_path:
            messagebox.showerror("Помилка", "Введіть текст або завантажте файл")
            return

        if text:
            df = pd.DataFrame({"Text": [text]})
            df.to_csv(SAVE_PATH, index=False, sep="|")
            self.dataset_path = SAVE_PATH

        self._set_loading(True)
        self.root.update()

        try:
            agent = DisinfoAgent(input_path=self.dataset_path)
            result_df = agent.run()
            self._show_result(result_df)
        except Exception as e:
            messagebox.showerror("Помилка", str(e))
        finally:
            self._set_loading(False)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()