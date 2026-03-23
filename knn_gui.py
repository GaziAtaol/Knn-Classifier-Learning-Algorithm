import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os

# ── Project imports ──
from data_loader import load_data, get_num_attributes
from knn import classify
from evaluator import evaluate

# Try matplotlib (optional for chart)
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class KNNApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("k-NN Classifier")
        self.geometry("960x720")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)

        # ── State ──
        self.training_data = None
        self.test_data = None
        self.num_attributes = 0
        self.has_label_var = tk.BooleanVar(value=True)
        self.k_var = tk.IntVar(value=3)
        self.new_sample_entries = []

        self._build_styles()
        self._build_ui()

    # ────────────────────────── STYLES ──────────────────────────
    def _build_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        BG      = "#1e1e2e"
        CARD    = "#2a2a3d"
        FG      = "#cdd6f4"
        ACCENT  = "#89b4fa"
        BTN_BG  = "#45475a"
        BTN_FG  = "#cdd6f4"

        self.style.configure("TFrame", background=BG)
        self.style.configure("Card.TFrame", background=CARD, relief="flat")
        self.style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        self.style.configure("Card.TLabel", background=CARD, foreground=FG, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI", 16, "bold"))
        self.style.configure("Sub.TLabel", background=CARD, foreground="#a6adc8", font=("Segoe UI", 9))
        self.style.configure("Accent.TButton", background=ACCENT, foreground="#1e1e2e", font=("Segoe UI", 10, "bold"), padding=(12, 6))
        self.style.map("Accent.TButton", background=[("active", "#74c7ec")])
        self.style.configure("TButton", background=BTN_BG, foreground=BTN_FG, font=("Segoe UI", 10), padding=(10, 5))
        self.style.map("TButton", background=[("active", "#585b70")])
        self.style.configure("TCheckbutton", background=CARD, foreground=FG, font=("Segoe UI", 10))
        self.style.configure("TSpinbox", fieldbackground=CARD, foreground=FG)

        self.colors = {"bg": BG, "card": CARD, "fg": FG, "accent": ACCENT}

    # ────────────────────────── UI BUILD ──────────────────────────
    def _build_ui(self):
        # Header
        ttk.Label(self, text="k-NN Classifier", style="Header.TLabel").pack(pady=(16, 4))

        # Main container with two columns
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=16, pady=8)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # ── LEFT PANEL ──
        left = ttk.Frame(main, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # File loading section
        self._section_label(left, "📂  Data Loading")

        frm_train = ttk.Frame(left, style="Card.TFrame")
        frm_train.pack(fill="x", padx=12, pady=(0, 4))
        ttk.Label(frm_train, text="Training file:", style="Card.TLabel").pack(anchor="w")
        self.train_path_var = tk.StringVar()
        ttk.Entry(frm_train, textvariable=self.train_path_var, width=30).pack(side="left", fill="x", expand=True)
        ttk.Button(frm_train, text="Browse", command=self._browse_train).pack(side="right", padx=(4, 0))

        frm_test = ttk.Frame(left, style="Card.TFrame")
        frm_test.pack(fill="x", padx=12, pady=(0, 4))
        ttk.Label(frm_test, text="Test file:", style="Card.TLabel").pack(anchor="w")
        self.test_path_var = tk.StringVar()
        ttk.Entry(frm_test, textvariable=self.test_path_var, width=30).pack(side="left", fill="x", expand=True)
        ttk.Button(frm_test, text="Browse", command=self._browse_test).pack(side="right", padx=(4, 0))

        ttk.Checkbutton(left, text="Test file has labels", variable=self.has_label_var, style="TCheckbutton").pack(anchor="w", padx=12, pady=4)

        ttk.Button(left, text="Load Data", style="Accent.TButton", command=self._load_data).pack(padx=12, pady=6, fill="x")

        self.info_label = ttk.Label(left, text="No data loaded yet.", style="Sub.TLabel", wraplength=280)
        self.info_label.pack(padx=12, pady=(0, 8), anchor="w")

        # k selection & evaluate
        self._section_label(left, "⚙️  Classification")

        frm_k = ttk.Frame(left, style="Card.TFrame")
        frm_k.pack(fill="x", padx=12, pady=4)
        ttk.Label(frm_k, text="k =", style="Card.TLabel").pack(side="left")
        self.k_spin = tk.Spinbox(frm_k, from_=1, to=999, textvariable=self.k_var, width=6,
                                  bg=self.colors["card"], fg=self.colors["fg"],
                                  buttonbackground=self.colors["card"], font=("Segoe UI", 11))
        self.k_spin.pack(side="left", padx=6)

        ttk.Button(left, text="Evaluate Test Set", style="Accent.TButton", command=self._evaluate).pack(padx=12, pady=6, fill="x")

        if HAS_MATPLOTLIB:
            ttk.Button(left, text="Show Accuracy Chart", command=self._show_chart).pack(padx=12, pady=(0, 6), fill="x")

        # New sample classification
        self._section_label(left, "🔍  Classify New Sample")

        self.new_sample_frame = ttk.Frame(left, style="Card.TFrame")
        self.new_sample_frame.pack(fill="x", padx=12, pady=4)
        self.new_sample_hint = ttk.Label(self.new_sample_frame, text="Load data first.", style="Sub.TLabel")
        self.new_sample_hint.pack(anchor="w")

        ttk.Button(left, text="Classify", style="Accent.TButton", command=self._classify_new).pack(padx=12, pady=6, fill="x")

        self.predict_label = ttk.Label(left, text="", style="Card.TLabel", font=("Segoe UI", 11, "bold"))
        self.predict_label.pack(padx=12, pady=(0, 12))

        # ── RIGHT PANEL ──
        right = ttk.Frame(main, style="Card.TFrame")
        right.grid(row=0, column=1, sticky="nsew")

        ttk.Label(right, text="Output", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=12, pady=(12, 4))

        self.output_text = scrolledtext.ScrolledText(
            right, wrap="word", font=("Consolas", 10),
            bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
            relief="flat", borderwidth=0, state="disabled"
        )
        self.output_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    # ────────────────────────── HELPERS ──────────────────────────
    def _section_label(self, parent, text):
        ttk.Label(parent, text=text, style="Card.TLabel",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 2))

    def _log(self, msg):
        self.output_text.configure(state="normal")
        self.output_text.insert("end", msg + "\n")
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def _clear_log(self):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.configure(state="disabled")

    def _browse_train(self):
        p = filedialog.askopenfilename(filetypes=[("Text/CSV", "*.txt *.csv *.data"), ("All", "*.*")])
        if p:
            self.train_path_var.set(p)

    def _browse_test(self):
        p = filedialog.askopenfilename(filetypes=[("Text/CSV", "*.txt *.csv *.data"), ("All", "*.*")])
        if p:
            self.test_path_var.set(p)

    # ────────────────────────── ACTIONS ──────────────────────────
    def _load_data(self):
        train_path = self.train_path_var.get().strip()
        test_path = self.test_path_var.get().strip()

        if not train_path or not test_path:
            messagebox.showwarning("Missing files", "Please select both training and test files.")
            return

        try:
            self.training_data = load_data(train_path)
            self.test_data = load_data(test_path, has_label=self.has_label_var.get())
            self.num_attributes = get_num_attributes(self.training_data)
            self.k_spin.configure(to=len(self.training_data))

            info = (f"✓ Training: {len(self.training_data)} samples\n"
                    f"✓ Test: {len(self.test_data)} samples\n"
                    f"✓ Attributes: {self.num_attributes}")
            self.info_label.configure(text=info)

            self._clear_log()
            self._log("Data loaded successfully.")
            self._log(f"  Training samples : {len(self.training_data)}")
            self._log(f"  Test samples     : {len(self.test_data)}")
            self._log(f"  Attributes       : {self.num_attributes}")

            # Build new-sample input fields
            self._build_sample_inputs()

        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _build_sample_inputs(self):
        for w in self.new_sample_frame.winfo_children():
            w.destroy()
        self.new_sample_entries.clear()

        cols = 4
        for i in range(self.num_attributes):
            r, c = divmod(i, cols)
            frm = ttk.Frame(self.new_sample_frame, style="Card.TFrame")
            frm.grid(row=r, column=c, padx=4, pady=2, sticky="w")
            ttk.Label(frm, text=f"x{i+1}:", style="Card.TLabel").pack(side="left")
            ent = tk.Entry(frm, width=8, bg="#313244", fg="#cdd6f4",
                           insertbackground="#cdd6f4", font=("Segoe UI", 10), relief="flat")
            ent.pack(side="left", padx=2)
            self.new_sample_entries.append(ent)

    def _evaluate(self):
        if not self.training_data or not self.test_data:
            messagebox.showwarning("No data", "Load data first.")
            return

        k = self.k_var.get()
        self._clear_log()
        self._log(f"Evaluating with k={k} ...\n")

        # Redirect print output to our text widget
        import io, sys
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer

        try:
            accuracy = evaluate(self.training_data, self.test_data, k)
        finally:
            sys.stdout = old_stdout

        self._log(buffer.getvalue())
        if accuracy is not None:
            self._log(f"\n→ Accuracy: {accuracy:.2f}%")

    def _classify_new(self):
        if not self.training_data:
            messagebox.showwarning("No data", "Load data first.")
            return

        try:
            attrs = [float(e.get().replace(",", ".")) for e in self.new_sample_entries]
        except ValueError:
            messagebox.showwarning("Invalid input", "Please enter valid numbers for all attributes.")
            return

        sample = {"attributes": attrs, "label": None}
        k = self.k_var.get()
        result = classify(self.training_data, sample, k)

        self.predict_label.configure(text=f"→ Predicted: {result}")
        self._log(f"\nNew sample {attrs}  →  {result}  (k={k})")

    def _show_chart(self):
        if not self.training_data or not self.test_data:
            messagebox.showwarning("No data", "Load data first.")
            return
        if self.test_data[0]["label"] is None:
            messagebox.showinfo("No labels", "Test data has no labels — cannot plot accuracy.")
            return

        max_k = len(self.training_data)
        current_k = self.k_var.get()

        # Run in thread to keep UI responsive
        def compute():
            ks = list(range(1, max_k + 1))
            accs = []
            for k in ks:
                correct = sum(1 for s in self.test_data if classify(self.training_data, s, k) == s["label"])
                accs.append((correct / len(self.test_data)) * 100)

            # Plot in main thread
            self.after(0, lambda: self._draw_chart(ks, accs, current_k))

        self._log("\nCalculating accuracy for all k values...")
        threading.Thread(target=compute, daemon=True).start()

    def _draw_chart(self, ks, accs, current_k):
        win = tk.Toplevel(self)
        win.title("Accuracy vs k")
        win.configure(bg=self.colors["bg"])

        fig, ax = plt.subplots(figsize=(10, 4), facecolor=self.colors["bg"])
        ax.set_facecolor("#313244")
        ax.plot(ks, accs, color="#89b4fa", linewidth=2, marker="o", markersize=3)

        if current_k <= len(ks):
            ax.plot(current_k, accs[current_k - 1], "o", color="#f38ba8", markersize=10,
                    label=f"k={current_k} ({accs[current_k-1]:.1f}%)")
            ax.legend(facecolor="#2a2a3d", edgecolor="#45475a", labelcolor="#cdd6f4")

        ax.set_title("Accuracy vs k", color="#cdd6f4")
        ax.set_xlabel("k", color="#a6adc8")
        ax.set_ylabel("Accuracy (%)", color="#a6adc8")
        ax.tick_params(colors="#a6adc8")
        for spine in ax.spines.values():
            spine.set_color("#45475a")
        ax.grid(True, linestyle="--", alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self._log("Chart displayed.")


if __name__ == "__main__":
    app = KNNApp()
    app.mainloop()
