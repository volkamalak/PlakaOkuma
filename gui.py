import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import time
import os
from src.detector import PlateDetector

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detector")
        self.root.geometry("1200x700")

        # Layout frames
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Widgets - Top
        self.btn_load = tk.Button(self.top_frame, text="Resim Yükle", command=self.load_image, font=("Arial", 12))
        self.btn_load.pack(side=tk.LEFT, padx=10)

        self.btn_detect = tk.Button(self.top_frame, text="Programı Çalıştır (Tespit Et)", command=self.detect_plate, state=tk.DISABLED, font=("Arial", 12), bg="#dddddd")
        self.btn_detect.pack(side=tk.LEFT, padx=10)

        # Widgets - Canvases
        self.lbl_original = tk.Label(self.left_frame, text="Orijinal Resim", font=("Arial", 10, "bold"))
        self.lbl_original.pack()
        self.canvas_original = tk.Canvas(self.left_frame, bg="#f0f0f0", relief="sunken", bd=1)
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        self.lbl_processed = tk.Label(self.right_frame, text="İşlenmiş Resim (Çerçeve)", font=("Arial", 10, "bold"))
        self.lbl_processed.pack()
        self.canvas_processed = tk.Canvas(self.right_frame, bg="#f0f0f0", relief="sunken", bd=1)
        self.canvas_processed.pack(fill=tk.BOTH, expand=True)

        # Widgets - Bottom (Info)
        self.lbl_info = tk.Label(self.bottom_frame, text="Hazır", font=("Arial", 12), justify=tk.LEFT)
        self.lbl_info.pack(pady=5)

        # State variables
        self.original_image_cv = None
        self.original_image_pil = None
        self.processed_image_pil = None
        self.detector = None

        # Initialize detector
        self.init_detector()

    def init_detector(self):
        try:
            self.lbl_info.config(text="Model Yükleniyor...")
            self.root.update()
            # Attempt to use best.pt logic from detector.py
            self.detector = PlateDetector()
            self.lbl_info.config(text="Model Yüklendi. Resim Seçiniz.")
        except Exception as e:
            messagebox.showerror("Hata", f"Model yüklenirken hata oluştu: {e}")
            self.lbl_info.config(text="Model Hatası")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        try:
            self.original_image_cv = cv2.imread(file_path)
            if self.original_image_cv is None:
                raise ValueError("Resim okunamadı")

            # Convert for display (BGR to RGB)
            image_rgb = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2RGB)
            self.original_image_pil = Image.fromarray(image_rgb)

            self.display_image(self.original_image_pil, self.canvas_original)

            # Reset processed view
            self.canvas_processed.delete("all")
            self.btn_detect.config(state=tk.NORMAL, bg="#aaffaa") # Greenish for active
            self.lbl_info.config(text="Resim yüklendi. Tespit için butona basınız.")
        except Exception as e:
            messagebox.showerror("Hata", f"Resim yüklenirken hata: {e}")

    def display_image(self, pil_image, canvas):
        # Resize to fit canvas
        canvas.update() # Ensure dimensions are known
        w = canvas.winfo_width()
        h = canvas.winfo_height()

        if w <= 1 or h <= 1:
             w, h = 400, 300 # Default fallback

        # Calculate aspect ratio
        img_w, img_h = pil_image.size
        ratio = min(w/img_w, h/img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)

        if new_w <= 0: new_w = 1
        if new_h <= 0: new_h = 1

        resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)

        # Keep reference to avoid garbage collection
        canvas.image = tk_img
        canvas.create_image(w//2, h//2, image=tk_img, anchor=tk.CENTER)

    def detect_plate(self):
        if self.original_image_cv is None or self.detector is None:
            return

        self.lbl_info.config(text="Tespit yapılıyor...")
        self.root.update()

        start_time = time.time()
        detections = self.detector.detect(self.original_image_cv)
        end_time = time.time()
        duration = end_time - start_time

        # Draw on copy
        processed_cv = self.original_image_cv.copy()

        info_lines = [f"Süre: {duration:.3f} saniye"]

        if not detections:
            info_lines.append("Plaka bulunamadı.")
        else:
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['box']
                conf = det['conf']

                # Draw Box
                cv2.rectangle(processed_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Add text label above box
                label = f"Plaka {conf:.2f}"
                cv2.putText(processed_cv, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Append to info text
                info_lines.append(f"Plaka {i+1}: ({x1}, {y1}) - ({x2}, {y2})")

        # Display processed
        image_rgb = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB)
        self.processed_image_pil = Image.fromarray(image_rgb)
        self.display_image(self.processed_image_pil, self.canvas_processed)

        self.lbl_info.config(text="\n".join(info_lines))

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()
