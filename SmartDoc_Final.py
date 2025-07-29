import tkinter as tk
from tkinter import filedialog, messagebox, Checkbutton, IntVar, DoubleVar, Scale, Frame, Canvas, Label
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import cv2
import os
import threading
import time
from datetime import datetime
import tkinter.ttk as ttk

model_path = r"C:\Users\tomra\OneDrive\Documents\Final_Project\\pneumothorax_xception_cbam_256_2.keras"
model = tf.keras.models.load_model(model_path, compile=False)

IMG_SIZE = (256, 256)
FEEDBACK_DIR = "feedback_images"
PERFORMANCE_LOG = "performance_log.txt"

os.makedirs(FEEDBACK_DIR, exist_ok=True)


def preprocess_image_classification(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img


def preprocess_image_segmentation(path):
    # Load as RGB (3 channels), uint8, no normalization
    img = Image.open(path).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.uint8)  # (256, 256, 3)
    return np.expand_dims(img_array, axis=0), img  # (1, 256, 256, 3), PIL image


def make_gradcam_heatmap(img_array, model, layer_name='block14_sepconv2_act'):
    grad_model = tf.keras.models.Model([
        model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_score = predictions[:, 0]

    grads = tape.gradient(predicted_score, conv_outputs)
    grads_power_2 = tf.square(grads)
    grads_power_3 = grads_power_2 * grads
    sum_grads = tf.reduce_sum(grads, axis=(1, 2), keepdims=True)

    alpha_num = grads_power_2
    alpha_den = grads_power_2 * 2.0 + sum_grads * grads_power_3
    alpha_den = tf.where(alpha_den != 0.0, alpha_den, tf.ones_like(alpha_den))
    alphas = alpha_num / alpha_den

    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(1, 2))
    cam = tf.reduce_sum(tf.multiply(weights[:, tf.newaxis, tf.newaxis, :], conv_outputs), axis=-1)
    heatmap = tf.squeeze(cam)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else heatmap
    heatmap = heatmap.numpy() if hasattr(heatmap, 'numpy') else heatmap
    return heatmap


def overlay_heatmap(heatmap, image, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, image.size)
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    image = np.array(image)
    if image.dtype != np.uint8:
        image = np.uint8(image * 255)
    if image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, image, 1 - alpha, 0)
    return Image.fromarray(superimposed_img)


def spo2_adjustment(spo2: float) -> float:
    """
    f(SpO₂) from the provided piece-wise function.
    """
    if spo2 < 90:
        return +0.20
    elif 90 <= spo2 < 94:
        return +0.10
    elif 94 <= spo2 <= 98:
        return 0.0
    else:  # SpO₂ > 98
        return -0.05


# Load segmentation model
SEGMENTATION_MODEL_PATH = r"C:\Users\tomra\OneDrive\Documents\Final_Project\\segmentation_model.keras"  # Change if needed
if os.path.exists(SEGMENTATION_MODEL_PATH):
    segmentation_model = tf.keras.models.load_model(SEGMENTATION_MODEL_PATH, compile=False)
else:
    segmentation_model = None

class PneumothoraxApp:
    def __init__(self, master):
        self.master = master
        self.master.title("SmartDoc")
        self.master.geometry("1240x1260")
        self.master.configure(bg="#f0f0f0")

        self.heatmap_var = IntVar()
        self.segmentation_var = IntVar()
        self.mask_var = IntVar() 
        self.current_index = 0
        self.results = []
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.thumbnails = []

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Main Menu (was Load Images)
        self.tab_load = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.tab_load, text="Main Menu")
        # Center frame for buttons and suggestions
        self.menu_center = tk.Frame(self.tab_load, bg="#f0f0f0")
        self.menu_center.place(relx=0.5, rely=0.4, anchor="center")
        # Logo
        try:
            logo_img = Image.open("Logo.png").resize((512, 512), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            tk.Label(self.menu_center, image=self.logo_photo, bg="#f0f0f0").pack(pady=(0, 10))
        except Exception:
            tk.Label(self.menu_center, text="[Logo]", font=("Arial", 24, "bold"), bg="#f0f0f0").pack(pady=(0, 10))
        # Welcome message
        tk.Label(self.menu_center, text="Welcome to SmartDoc Pneumothorax Detection!", font=("Arial", 32, "bold"), bg="#f0f0f0").pack(pady=(0, 5))
        tk.Label(self.menu_center, text="Instructions:\n Click 'Select Images' to load chest X-ray images.\n Review predictions and provide feedback.", bg="#f0f0f0", font=("Arial", 20)).pack(pady=(0, 15))
        # Last retrain time
        self.last_retrain_label = tk.Label(self.menu_center, text=" ", bg="#f0f0f0", font=("Arial", 20, "italic"))
        self.last_retrain_label.pack(pady=(0, 15))
        self.update_last_retrain_time()
        tk.Button(self.menu_center, text="Select Images", command=self.load_images, bg="#4CAF50", fg="white", width=40, height=4).pack(pady=10)
        retrain_btn = tk.Button(self.menu_center, text="Retrain Now", command=lambda: [self.animate_button(retrain_btn), threading.Thread(target=self.retrain_with_feedback).start()], width=20, height=2)
        retrain_btn.pack(pady=10)
        # Credits
        tk.Label(self.menu_center, text="Created by Tom Rapoport and Omri Kaplan\nfor the TAU Electrical Engineering Final Project", bg="#f0f0f0", font=("Arial", 15, "italic")).pack(pady=(30, 0))

        # Tab 2: Predictions
        self.tab_pred = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.tab_pred, text="Predictions")

        # Parent frame for predictions content (canvas + thumbnails)
        self.pred_content = tk.Frame(self.tab_pred, bg="#f0f0f0")
        self.pred_content.pack(fill="both", expand=True)

        self.canvas = Canvas(self.pred_content, bg="#ffffff")
        self.canvas.pack(pady=10, fill="both", expand=True, side="left")

        self.result_label = Label(self.tab_pred, text="", font=("Arial", 14), bg="#f0f0f0")
        self.result_label.pack(pady=10, side="top")

        # Right frame for thumbnails (initially hidden)
        self.right_frame = tk.Frame(self.pred_content, width=200, bg="#e0e0e0")
        self.thumbnail_canvas = tk.Canvas(self.right_frame, bg="#e0e0e0", width=180)
        self.thumbnail_scrollbar = tk.Scrollbar(self.right_frame, orient="vertical", command=self.thumbnail_canvas.yview)
        self.thumbnail_frame = tk.Frame(self.thumbnail_canvas, bg="#e0e0e0")
        self.thumbnail_frame.bind(
            "<Configure>",
            lambda e: self.thumbnail_canvas.configure(scrollregion=self.thumbnail_canvas.bbox("all"))
        )
        self.thumbnail_canvas.create_window((0, 0), window=self.thumbnail_frame, anchor="nw")
        self.thumbnail_canvas.configure(yscrollcommand=self.thumbnail_scrollbar.set)
        self.thumbnail_canvas.bind("<Enter>", lambda e: self._bind_mousewheel())
        self.thumbnail_canvas.bind("<Leave>", lambda e: self._unbind_mousewheel())
        # Don't pack right_frame yet

        Checkbutton(self.tab_pred, text="Show Grad-CAM++", variable=self.heatmap_var, bg="#f0f0f0",
                    command=self.display_image).pack(pady=5)
        Checkbutton(self.tab_pred, text="Show Segmentation", variable=self.segmentation_var, bg="#f0f0f0",
                    command=self.display_image).pack(pady=5)
        Checkbutton(self.tab_pred, text="Show Mask", variable=self.mask_var, bg="#f0f0f0",
                    command=self.display_image).pack(pady=5)
        # Remove Show Performance Log button
        # Add Correct/Incorrect buttons for online learning
        self.feedback_frame = tk.Frame(self.tab_pred, bg="#f0f0f0")
        self.feedback_frame.pack(pady=10)
        correct_btn = tk.Button(self.feedback_frame, text="Correct", bg="#4CAF50", fg="white", width=12, command=lambda: [self.animate_button(correct_btn), self.mark_correct()])
        correct_btn.pack(side="left", padx=10)
        incorrect_btn = tk.Button(self.feedback_frame, text="Incorrect", bg="#F44336", fg="white", width=12, command=lambda: [self.animate_button(incorrect_btn), self.mark_incorrect()])
        incorrect_btn.pack(side="left", padx=10)

        self.canvas.bind("<MouseWheel>", self.mouse_zoom)
        # Remove mouse drag bindings
        # self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        # self.canvas.bind("<B1-Motion>", self.on_mouse_drag)

        # Tab 3: Performance Log
        self.tab_log = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.tab_log, text="Performance Log")
        # Title/Header
        self.log_title = tk.Label(self.tab_log, text="Performance Log", font=("Arial", 18, "bold"), bg="#f0f0f0")
        self.log_title.pack(pady=(18, 5))
        # Only one main frame for charts/stats, centered
        self.stats_main_frame = tk.Frame(self.tab_log, bg="#f0f0f0")
        self.stats_main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        # Stats and charts with period tabs
        self.stats_notebook = ttk.Notebook(self.stats_main_frame)
        self.stats_notebook.pack(pady=(0,0), fill="both", expand=True)
        self.period_frames = {}
        for period in ["Daily", "Weekly", "Monthly"]:
            frame = tk.Frame(self.stats_notebook, bg="#f0f0f0")
            self.stats_notebook.add(frame, text=period)
            self.period_frames[period] = frame
        self.stats_label = tk.Label(self.period_frames["Daily"], text="", font=("Arial", 12), bg="#f0f0f0")
        self.stats_label.pack(pady=(10,0))
        self.stats_canvas = None
        self.trend_canvas = None
        self.update_statistics_chart()
        self.stats_notebook.bind("<<NotebookTabChanged>>", lambda e: self.update_statistics_chart())


    def _on_mousewheel(self, event):
        self.thumbnail_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _bind_mousewheel(self):
        self.thumbnail_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self):
        self.thumbnail_canvas.unbind_all("<MouseWheel>")

    def fade_in_tab(self, tab):
        # 5: Fade in tab content
        for alpha in range(0, 11):
            tab.update()
            tab.attributes('-alpha', alpha/10)
            self.master.after(20)
        tab.attributes('-alpha', 1.0)

    def load_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[["Image Files", "*.png;*.jpg;*.jpeg"]])
        if not file_paths:
            return
        self.results.clear()
        self.thumbnails.clear()
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()
        for idx, file_path in enumerate(file_paths):
            # Prompt for SpO2 value for each image
            spo2 = self.prompt_spo2(file_path)
            if spo2 is None:
                continue  # Skip if user cancels
            img_array, img_raw = preprocess_image_classification(file_path)
            prediction = model.predict(img_array)[0][0]
            adjusted_prediction = prediction + spo2_adjustment(spo2)
            adjusted_prediction = min(max(adjusted_prediction, 0.0), 1.0)  # Clamp to [0,1]
            original_img = Image.open(file_path).convert("RGB")
            # Store (file_path, adjusted_prediction, img_raw, spo2, original_prediction)
            self.results.append((file_path, adjusted_prediction, original_img, spo2, prediction))
        self.results.sort(key=lambda x: x[1], reverse=True)
        self.current_index = 0
        self.display_image()
        self.display_thumbnails()
        # Show the thumbnails panel only after images are loaded
        self.right_frame.pack(side="right", fill="y")
        self.thumbnail_canvas.pack(side="left", fill="both", expand=True)
        self.thumbnail_scrollbar.pack(side="right", fill="y")
        # Automatically switch to Predictions tab after all SpO2 values are entered
        self.notebook.select(self.tab_pred)
        # 5: Fade in predictions tab (simulate with a quick flash for Tkinter)
        self.tab_pred.after(0, lambda: self.tab_pred.config(bg="#e3f2fd"))
        self.tab_pred.after(200, lambda: self.tab_pred.config(bg="#f0f0f0"))

    def prompt_spo2(self, file_path):
        # Modal dialog to get SpO2 value from user, always on top and focused
        import tkinter.simpledialog
        while True:
            temp_root = tk.Toplevel(self.master)
            temp_root.title("SpO₂ Input")
            temp_root.attributes('-topmost', True)
            temp_root.grab_set()  # Make modal
            temp_root.focus_force()
            temp_root.resizable(False, False)
            temp_root.geometry("300x120")
            # Center the dialog over the main window
            self.master.update_idletasks()
            x = self.master.winfo_x() + (self.master.winfo_width() // 2) - 150
            y = self.master.winfo_y() + (self.master.winfo_height() // 2) - 60
            temp_root.geometry(f"300x120+{x}+{y}")

            label = tk.Label(temp_root, text=f"Enter SpO₂ value for {os.path.basename(file_path)} :", font=("Arial", 10))
            label.pack(pady=(15, 5))
            entry = tk.Entry(temp_root, font=("Arial", 12), justify="center")
            entry.pack(pady=5)
            entry.focus_set()

            result = {'value': None}
            def on_ok():
                try:
                    val = float(entry.get())
                    if 50 <= val <= 100:
                        result['value'] = val
                        temp_root.destroy()
                    else:
                        messagebox.showerror("Invalid Input", "Please enter a value between 50 and 100.", parent=temp_root)
                except Exception:
                    messagebox.showerror("Invalid Input", "Please enter a valid number.", parent=temp_root)
            def on_cancel():
                temp_root.destroy()
            btn_frame = tk.Frame(temp_root)
            btn_frame.pack(pady=5)
            ok_btn = tk.Button(btn_frame, text="OK", width=8, command=on_ok)
            ok_btn.pack(side="left", padx=10)
            cancel_btn = tk.Button(btn_frame, text="Cancel", width=8, command=on_cancel)
            cancel_btn.pack(side="left", padx=10)
            temp_root.bind('<Return>', lambda e: on_ok())
            temp_root.bind('<Escape>', lambda e: on_cancel())
            temp_root.wait_window()  # Modal: block until closed
            if result['value'] is None:
                # User cancelled
                return None
            else:
                return result['value']

    def display_thumbnails(self):
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()
        self.thumbnails = []
        for idx, (file_path, prediction, img_raw, spo2, orig_pred) in enumerate(self.results):
            thumb = img_raw.copy()
            thumb.thumbnail((64, 64))
            photo = ImageTk.PhotoImage(thumb)
            self.thumbnails.append(photo)
            # Highlight the first (top) thumbnail as urgent
            if idx == 0:
                btn_bg = "#ffcccc"  # Light red background for urgency
                btn_bd = 3
                btn_relief = "solid"
            else:
                btn_bg = "#e0e0e0"
                btn_bd = 0
                btn_relief = "flat"
            btn = tk.Button(self.thumbnail_frame, image=photo, command=lambda i=idx: self.on_thumbnail_click(i), bd=btn_bd, relief=btn_relief, highlightthickness=0, bg=btn_bg, activebackground=btn_bg)
            btn.grid(row=idx, column=0, pady=2, padx=2, sticky="w")
            def on_enter(e, b=btn): b.config(bg="#b3e5fc")
            def on_leave(e, b=btn, bg=btn_bg): b.config(bg=bg)
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            label_text = f"{os.path.basename(file_path)}\n{prediction:.2f}"
            lbl = tk.Label(self.thumbnail_frame, text=label_text, bg=btn_bg, font=("Arial", 8))
            lbl.grid(row=idx, column=1, sticky="w")
        # Remove progress bar for image queue
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.destroy()
            self.progress_bar = None
        # Update scrollregion and disable scrolling if not needed
        self.thumbnail_canvas.update_idletasks()
        frame_height = self.thumbnail_canvas.winfo_height()
        content_height = self.thumbnail_frame.winfo_height()
        if content_height <= frame_height:
            self.thumbnail_canvas.unbind_all("<MouseWheel>")
            self.thumbnail_scrollbar.pack_forget()
            self.thumbnail_canvas.yview_moveto(0)
        else:
            self.thumbnail_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            self.thumbnail_scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(self, event):
        # Clamp scrolling so you can't scroll past the first or last image
        self.thumbnail_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.thumbnail_canvas.update_idletasks()
        # Clamp to top
        if self.thumbnail_canvas.yview()[0] <= 0:
            self.thumbnail_canvas.yview_moveto(0)
        # Clamp to bottom
        elif self.thumbnail_canvas.yview()[1] >= 1:
            self.thumbnail_canvas.yview_moveto(1)

    def on_thumbnail_click(self, idx):
        self.current_index = idx
        self.display_image()

    def display_image(self):
        if not self.results:
            return
        file_path, prediction, img_raw, spo2, orig_pred = self.results[self.current_index]
        patient_name = os.path.basename(file_path)
        label = "Pneumothorax" if prediction >= 0.5 else "Normal"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        display_img = img_raw.copy()
        # Segmentation overlay
        if getattr(self, 'segmentation_var', None) and self.segmentation_var.get() and segmentation_model is not None:
            img_array, _ = preprocess_image_segmentation(file_path)
            mask = segmentation_model.predict(img_array)[0]
            print(f"[DEBUG] Raw mask: min={mask.min()}, max={mask.max()}, unique={np.unique(mask)}")
            # Squeeze to 2D if needed
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)
            elif mask.ndim == 3:
                mask = mask.squeeze()
            # Try a lower threshold for debugging
            mask_thresh = 0.5
            mask_bin = (mask > mask_thresh).astype(np.uint8) * 255
            mask_bin = np.array(mask_bin, dtype=np.uint8)
            if np.all(mask_bin == 0):
                print("[DEBUG] Mask is all zeros after thresholding. Displaying raw mask as overlay.")
                # Scale raw mask to [0,255] for visualization
                mask_vis = (mask / (mask.max() if mask.max() > 0 else 1) * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_vis).resize(display_img.size, Image.NEAREST)
                mask_rgba = Image.new("RGBA", display_img.size, (255,0,0,0))
                mask_rgba.paste((255,0,0,120), mask=mask_img)
                display_img = display_img.convert("RGBA")
                display_img = Image.alpha_composite(display_img, mask_rgba).convert("RGB")
                # Optionally, show a warning to the user
                messagebox.showwarning("Segmentation Mask", "Segmentation mask is nearly empty. Model may need retraining or input preprocessing may be mismatched.")
            else:
                mask_img = Image.fromarray(mask_bin).resize(display_img.size, Image.NEAREST)
                mask_rgba = Image.new("RGBA", display_img.size, (255,0,0,0))
                mask_rgba.paste((255,0,0,180), mask=mask_img)
                display_img = display_img.convert("RGBA")
                display_img = Image.alpha_composite(display_img, mask_rgba).convert("RGB")
        # Mask overlay
        if getattr(self, 'mask_var', None) and self.mask_var.get():
            mask_path = os.path.splitext(file_path)[0] + "_mask.png"
            if os.path.exists(mask_path):
                try:
                    mask_img = Image.open(mask_path).convert("L").resize(display_img.size, Image.NEAREST)
                    mask_rgba = Image.new("RGBA", display_img.size, (255,0,0,0))
                    mask_rgba.paste((144, 238, 144, 150), mask=mask_img)
                    display_img = display_img.convert("RGBA")
                    display_img = Image.alpha_composite(display_img, mask_rgba).convert("RGB")
                except Exception as e:
                    messagebox.showwarning("Mask Overlay", f"Could not load mask image: {e}")
        # Grad-CAM++ overlay
        if self.heatmap_var.get():
            img_array, _ = preprocess_image_classification(file_path)
            try:
                heatmap = make_gradcam_heatmap(img_array, model)
                display_img = overlay_heatmap(heatmap, display_img)
            except Exception as e:
                messagebox.showwarning("Grad-CAM++", f"Could not generate heatmap: {e}")
        w, h = display_img.size
        new_size = (int(w * self.zoom), int(h * self.zoom))
        img_display = display_img.resize(new_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(img_display)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=photo, tags="img")
        self.canvas.image = photo
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        # Main result line
        prediction_text = f"Patient: {patient_name}\nSpO₂: {spo2}\nPrediction: {label} ({confidence * 100:.2f}%)"

        # Add colored Positive/Negative status
        if prediction >= 0.5:
            status_text = "Positive"
            status_color = "green"
        else:
            status_text = "Negative"
            status_color = "red"

        # Update the label (first set text, then set color using HTML-like formatting)
        self.result_label.config(text=prediction_text + f"\n", fg="black")
        # Append the color-coded label as a second label widget just below
        self.result_label.config(text=self.result_label.cget("text") + f"{status_text}", fg=status_color)

        #self.result_label.config(
         #   text=f"Patient: {patient_name}\nSpO₂: {spo2} \nPrediction: {label} ({confidence * 100:.2f}%)"
        #)

    def update_statistics_chart(self):
        import datetime
        log_path = 'feedback_log.csv'
        # Remove old charts if present
        for period in ["Daily", "Weekly", "Monthly"]:
            frame = self.period_frames[period]
            for widget in frame.winfo_children():
                if widget not in [self.stats_label]:
                    widget.destroy()
        if hasattr(self, 'stats_canvas') and self.stats_canvas is not None:
            self.stats_canvas.get_tk_widget().pack_forget()
        if hasattr(self, 'trend_canvas') and self.trend_canvas is not None:
            self.trend_canvas.get_tk_widget().pack_forget()
        # Empty state
        if not os.path.exists(log_path):
            self.stats_label.config(text="No feedback data yet.")
            return
        df = pd.read_csv(log_path, names=['timestamp', 'result'])
        # Parse timestamp
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # Determine selected period
        period = self.stats_notebook.tab(self.stats_notebook.select(), "text")
        now = pd.Timestamp.now()
        if period == "Daily":
            mask = df['timestamp'].dt.date == now.date()
            groupby = None
            time_label = None
        elif period == "Weekly":
            mask = df['timestamp'] >= (now - pd.Timedelta(days=7))
            groupby = df['timestamp'].dt.date
            time_label = 'Date'
        elif period == "Monthly":
            mask = df['timestamp'] >= (now - pd.Timedelta(days=30))
            groupby = df['timestamp'].dt.isocalendar().week
            time_label = 'Week'
        else:
            mask = slice(None)
            groupby = None
            time_label = None
        dff = df[mask]
        total = len(dff)
        correct = (dff['result'] == 'correct').sum()
        incorrect = (dff['result'] == 'incorrect').sum()
        accuracy = correct / total if total > 0 else 0
        self.stats_label.config(text=f"Total: {total}  Correct: {correct}  Incorrect: {incorrect}  Accuracy: {accuracy*100:.2f}%")
        frame = self.period_frames[period]
        if total == 0:
            # Show message instead of chart
            msg = tk.Label(frame, text=f"No feedback for this period.", font=("Arial", 12, "italic"), fg="#888888", bg="#f0f0f0")
            msg.pack(pady=30)
            return
        # Pie chart
        fig, ax = plt.subplots(figsize=(3,3))
        ax.pie([correct, incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        ax.set_title(f'Prediction Feedback ({period})')
        plt.tight_layout()
        self.stats_canvas = FigureCanvasTkAgg(fig, master=frame)
        self.stats_canvas.draw()
        self.stats_canvas.get_tk_widget().pack(pady=10)
        plt.close(fig)
        # Trend/history chart
        self._show_trend_chart(dff, frame)
        # Bar chart: only for Weekly and Monthly
        if period == "Weekly" and groupby is not None:
            self._show_feedback_bar_chart(dff, frame, groupby, time_label, period)
        elif period == "Monthly" and groupby is not None:
            self._show_feedback_bar_chart(dff, frame, groupby, time_label, period)

    def _show_trend_chart(self, df, frame):
        if len(df) < 2:
            return
        accs = []
        correct = 0
        for i, res in enumerate(df['result']):
            if res == 'correct':
                correct += 1
            accs.append(correct / (i+1))
        fig, ax = plt.subplots(figsize=(3,1.2))
        ax.plot(range(1, len(accs)+1), [a*100 for a in accs], color="#1976D2", marker="o", linewidth=2)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Feedback #")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy Trend", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        self.trend_canvas = FigureCanvasTkAgg(fig, master=frame)
        self.trend_canvas.draw()
        self.trend_canvas.get_tk_widget().pack(pady=(0,10))
        plt.close(fig)

    def _show_feedback_bar_chart(self, df, frame, groupby, time_label, period):
        import pandas as pd
        if len(df) == 0:
            return
        # Group by time and result
        grouped = df.groupby([groupby, 'result']).size().unstack(fill_value=0)
        grouped = grouped.sort_index()
        if period == "Weekly":
            # Show all days in the last 7 days
            now = pd.Timestamp.now()
            all_days = pd.date_range(end=now, periods=7).date
            grouped = grouped.reindex(all_days, fill_value=0)
            x = [d.strftime('%Y-%m-%d') for d in all_days]
        elif period == "Monthly":
            # Show all weeks in the last 4-5 weeks
            now = pd.Timestamp.now()
            week_info = df['timestamp'].dt.isocalendar()
            min_week = (now - pd.Timedelta(days=30)).isocalendar().week
            max_week = now.isocalendar().week
            all_weeks = list(range(min_week, max_week+1))
            grouped = grouped.reindex(all_weeks, fill_value=0)
            year = now.year
            x = [f"{year}-W{int(week):02d}" for week in all_weeks]
        else:
            x = grouped.index.astype(str)
        fig, ax = plt.subplots(figsize=(6, 2.5))
        if 'correct' in grouped:
            ax.bar(x, grouped['correct'], label='Correct', color='#4CAF50', width=0.4)
        if 'incorrect' in grouped:
            ax.bar(x, grouped['incorrect'], bottom=grouped['correct'] if 'correct' in grouped else None, label='Incorrect', color='#F44336', width=0.4)
        ax.set_xlabel(time_label)
        ax.set_ylabel('Feedback Count')
        ax.set_title(f'Feedback Volume by {time_label}')
        ax.legend()
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
            label.set_fontsize(8)
        from matplotlib.ticker import MaxNLocator
        if len(x) > 10:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        fig.tight_layout()
        fig.autofmt_xdate()
        bar_canvas = FigureCanvasTkAgg(fig, master=frame)
        bar_canvas.draw()
        bar_canvas.get_tk_widget().pack(pady=(0,10))
        plt.close(fig)

    def mouse_zoom(self, event):
        # Disable zoom if overlays are active
        if (getattr(self, 'segmentation_var', None) and self.segmentation_var.get()) or (getattr(self, 'heatmap_var', None) and self.heatmap_var.get() or getattr(self, 'mask_var', None) and self.mask_var.get()):
            self.show_toast("Zoom disabled while overlay is active.")
            return
        # Only allow zooming in, or zooming out to original size (not below 1.0)
        old_zoom = self.zoom
        factor = 1.1 if event.delta > 0 else 0.9
        new_zoom = self.zoom * factor
        # Prevent zooming out below original size (1.0)
        if new_zoom < 1.0:
            new_zoom = 1.0
        self.zoom = max(1.0, min(new_zoom, 10.0))
        if self.results:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            rel_x = (x - self.offset_x) / old_zoom
            rel_y = (y - self.offset_y) / old_zoom
            self.offset_x = x - rel_x * self.zoom
            self.offset_y = y - rel_y * self.zoom
        self.display_image()

    def mark_correct(self):
        self._flash_feedback('green')
        self.save_feedback_sample(is_positive=True)
        self.log_feedback_event(True)
        self.remove_current_image_from_queue()
        self.update_statistics_chart()

    def mark_incorrect(self):
        self._flash_feedback('red')
        self.save_feedback_sample(is_positive=False)
        self.log_feedback_event(False)
        self.remove_current_image_from_queue()
        self.update_statistics_chart()

    def _flash_feedback(self, color):
        # 3: Show a quick border flash on feedback
        orig = self.canvas['highlightbackground'] if 'highlightbackground' in self.canvas.keys() else None
        self.canvas.config(highlightthickness=4, highlightbackground=color)
        self.master.after(200, lambda: self.canvas.config(highlightthickness=0, highlightbackground=orig if orig else '#ffffff'))

    def show_toast(self, message, duration=1500):
        # 6: Show a temporary toast notification
        toast = tk.Toplevel(self.master)
        toast.overrideredirect(True)
        toast.attributes('-topmost', True)
        width = 320  # Increased width for longer messages
        x = self.master.winfo_x() + self.master.winfo_width()//2 - width//2
        y = self.master.winfo_y() + 100
        toast.geometry(f"{width}x40+{x}+{y}")
        tk.Label(toast, text=message, bg="#323232", fg="white", font=("Arial", 11), wraplength=width-20, justify="center").pack(expand=True, fill="both")
        toast.after(duration, toast.destroy)

    def retrain_with_feedback(self):
        # Gather feedback samples
        pos_dir = os.path.join(FEEDBACK_DIR, "positive")
        neg_dir = os.path.join(FEEDBACK_DIR, "negative")
        pos_samples = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))] if os.path.exists(pos_dir) else []
        neg_samples = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))] if os.path.exists(neg_dir) else []
        # Log retraining start
        with open(PERFORMANCE_LOG, 'a') as f:
            f.write(f"Retraining with feedback started at {datetime.now()}\n")
            f.write(f"Positive samples: {len(pos_samples)}, Negative samples: {len(neg_samples)}\n")
        
        # --- NEW: Actual retraining logic using sorted folder and CSV ---
        import random
        sorted_csv = "sorted_labels.csv"
        if not os.path.exists(sorted_csv):
            messagebox.showwarning("Retrain", "No labeled data found in sorted_labels.csv. Please provide feedback first.")
            return
        # Load image paths and labels
        df = pd.read_csv(sorted_csv, names=["path", "label"])
        if len(df) < 2:
            messagebox.showwarning("Retrain", "Not enough labeled data for retraining.")
            return
        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        # Load and preprocess images
        X = []
        y = []
        for idx, row in df.iterrows():
            try:
                img = Image.open(row["path"]).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
                arr = np.array(img) / 255.0
                X.append(arr)
                y.append(row["label"])
            except Exception as e:
                print(f"Error loading {row['path']}: {e}")
        X = np.array(X)
        y = np.array(y, dtype=np.float32)
        # Simple train/val split
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        # Compile model for fine-tuning
        global model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        # Fine-tune for a few epochs
        history = model.fit(X_train, y_train, epochs=3, batch_size=8, validation_data=(X_val, y_val), verbose=2)
        # Save updated model
        model.save(model_path)
        # Reload model
        model = tf.keras.models.load_model(model_path, compile=False)
        # --- END NEW ---

        with open(PERFORMANCE_LOG, 'a') as f:
            f.write(f"Retraining with feedback finished at {datetime.now()}\n")
        # Update last retrain time
        with open("last_retrain.txt", "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.update_last_retrain_time()
        self.update_performance_log()
        self.show_toast("Model retrained with feedback samples.")
        # messagebox.showinfo("Retrain", "Model retrained with feedback samples.")

    def update_last_retrain_time(self):
        try:
            with open("last_retrain.txt", "r") as f:
                last_time = f.read().strip()
            self.last_retrain_label.config(text=f"Last retrain: {last_time}")
        except Exception:
            self.last_retrain_label.config(text="Last retrain: Never")

    def remove_current_image_from_queue(self):
        if not self.results:
            return
        del self.results[self.current_index]
        if self.current_index >= len(self.results):
            self.current_index = max(0, len(self.results) - 1)
        self.display_thumbnails()
        self.display_image()

    def animate_button(self, button):
        # 4: Button press/release animation
        orig_bg = button.cget('bg')
        button.config(bg='#bdbdbd')
        self.master.after(100, lambda: button.config(bg=orig_bg))


if __name__ == "__main__":
    root = tk.Tk()
    app = PneumothoraxApp(root)
    root.mainloop()