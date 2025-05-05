import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import random
import Image_Detector  # Ensure this module is available


class AIDetectorApp:
    def __init__(self, root):
        # Set up main window
        self.root = root
        self.root.title("Hệ Thống Nhận Diện Ảnh")
        self.root.geometry("1080x640+100+50")
        self.root.configure(bg="#2C3E50")
        self.root.resizable(False, False)

        # Store selected image path
        self.selected_img_path = None

        # Load logo (using the original logo)
        try:
            self.logo_image = tk.PhotoImage(file="logo.png")
            root.iconphoto(False, self.logo_image)
        except Exception as e:
            print("Lỗi khi tải logo:", e)
            self.logo_image = None

        # Create main frames with gradient effect
        self.create_gradient_background()
        self.setup_interface()

    def create_gradient_background(self):
        # Create background with layered panels for modern design
        self.header_frame = tk.Frame(self.root, bg="#1ABC9C", height=80)
        self.header_frame.pack(fill=tk.X)

        # Create a decorative line
        self.separator = ttk.Separator(self.root, orient='horizontal')
        self.separator.pack(fill=tk.X, pady=2)

        # Main content area
        self.main_frame = tk.Frame(self.root, bg="#34495E")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def setup_interface(self):
        # Add logo and title to header
        header_content = tk.Frame(self.header_frame, bg="#1ABC9C")
        header_content.pack(pady=10)

        # Add logo if available
        if self.logo_image:
            logo_label = tk.Label(header_content, image=self.logo_image, bg="#1ABC9C")
            logo_label.pack(side=tk.LEFT, padx=(0, 15))

        # Add title
        title_label = tk.Label(header_content,
                               text="HỆ THỐNG NHẬN DIỆN ẢNH THẬT - ẢNH AI",
                               font=("Helvetica", 22, "bold"),
                               bg="#1ABC9C",
                               fg="white")
        title_label.pack(side=tk.LEFT, padx=10)

        # Create two main panels side by side
        self.left_panel = tk.Frame(self.main_frame, bg="#ECF0F1", width=450, height=500)
        self.left_panel.pack(side=tk.LEFT, padx=15, pady=10)
        self.left_panel.pack_propagate(False)

        self.right_panel = tk.Frame(self.main_frame, bg="#3498DB", width=550, height=500)
        self.right_panel.pack(side=tk.RIGHT, padx=15, pady=10)
        self.right_panel.pack_propagate(False)

        # Setup left panel (image selection area)
        self.setup_image_panel()

        # Setup right panel (results area)
        self.setup_results_panel()

    def setup_image_panel(self):
        # Image panel title
        tk.Label(self.left_panel,
                 text="Chọn Ảnh Để Phân Tích",
                 font=("Arial", 16, "bold"),
                 bg="#ECF0F1").pack(pady=10)

        # Frame for image preview with border
        self.img_frame = tk.Frame(self.left_panel,
                                  bg="#2C3E50",
                                  width=400,
                                  height=350,
                                  bd=2,
                                  relief=tk.RIDGE)
        self.img_frame.pack(pady=15)
        self.img_frame.pack_propagate(False)

        # Label to display selected image
        self.img_label = tk.Label(self.img_frame, bg="#2C3E50")
        self.img_label.pack(expand=True, fill=tk.BOTH)

        # Button frame
        btn_frame = tk.Frame(self.left_panel, bg="#ECF0F1")
        btn_frame.pack(fill=tk.X, pady=15)

        # Buttons for file selection and analysis
        browse_btn = tk.Button(btn_frame,
                               text="Chọn Ảnh",
                               font=("Arial", 12),
                               bg="#2980B9",
                               fg="white",
                               command=self.select_image,
                               width=15,
                               cursor="hand2")
        browse_btn.pack(side=tk.LEFT, padx=20)

        analyze_btn = tk.Button(btn_frame,
                                text="Phân Tích",
                                font=("Arial", 12),
                                bg="#E74C3C",
                                fg="white",
                                command=self.analyze_image,
                                width=15,
                                cursor="hand2")
        analyze_btn.pack(side=tk.RIGHT, padx=20)

    def setup_results_panel(self):
        # Results panel title
        tk.Label(self.right_panel,
                 text="Kết Quả Phân Tích",
                 font=("Arial", 18, "bold"),
                 bg="#3498DB",
                 fg="white").pack(pady=20)

        # Classification section
        tk.Label(self.right_panel,
                 text="Phân Loại:",
                 font=("Arial", 14, "bold"),
                 bg="#3498DB",
                 fg="white").pack(anchor="w", padx=40, pady=(30, 5))

        # Frame for classification result
        self.result_frame1 = tk.Frame(self.right_panel,
                                      bg="#F8F9F9",
                                      width=400,
                                      height=100,
                                      bd=3,
                                      relief=tk.GROOVE)
        self.result_frame1.pack(pady=10, padx=40)
        self.result_frame1.pack_propagate(False)

        # Default classification label (will be updated on analysis)
        self.class_label = tk.Label(self.result_frame1,
                                    text="Đang chờ phân tích...",
                                    font=("Arial", 26, "bold"),
                                    bg="#F8F9F9",
                                    fg="#7F8C8D")
        self.class_label.pack(expand=True)

        # Reliability section
        tk.Label(self.right_panel,
                 text="Độ Tin Cậy:",
                 font=("Arial", 14, "bold"),
                 bg="#3498DB",
                 fg="white").pack(anchor="w", padx=40, pady=(30, 5))

        # Frame for reliability result
        self.result_frame2 = tk.Frame(self.right_panel,
                                      bg="#F8F9F9",
                                      width=400,
                                      height=100,
                                      bd=3,
                                      relief=tk.GROOVE)
        self.result_frame2.pack(pady=10, padx=40)
        self.result_frame2.pack_propagate(False)

        # Default reliability label (will be updated on analysis)
        self.reliability_label = tk.Label(self.result_frame2,
                                          text="--",
                                          font=("Arial", 26, "bold"),
                                          bg="#F8F9F9",
                                          fg="#7F8C8D")
        self.reliability_label.pack(expand=True)

    def select_image(self):
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title='Chọn Tệp Ảnh',
            filetypes=(
                ('Tệp PNG', '*.png'),
                ('Tệp JPG', '*.jpg'),
                ('Tệp JPEG', '*.jpeg'),
                ('Tất cả tệp ảnh', '*.png;*.jpg;*.jpeg')
            )
        )

        if filename:
            # Store path and display image
            self.selected_img_path = filename
            self.display_preview(filename)

    def display_preview(self, path):
        # Display selected image in preview area
        try:
            img = Image.open(path)
            aspect_ratio = img.width / img.height

            # Resize image while maintaining aspect ratio
            new_width = 380
            new_height = int(new_width / aspect_ratio)

            if new_height > 330:  # Adjust if too tall
                new_height = 330
                new_width = int(new_height * aspect_ratio)

            img = img.resize((new_width, new_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.img_label.configure(image=photo)
            self.img_label.image = photo  # Keep reference

            # Reset result labels
            self.class_label.config(text="Sẵn sàng phân tích", fg="#7F8C8D")
            self.reliability_label.config(text="--", fg="#7F8C8D")

        except Exception as e:
            print("Lỗi khi tải ảnh:", e)

    def analyze_image(self):
        if not self.selected_img_path:
            # Show error if no image selected
            self.class_label.config(text="Chưa chọn ảnh", fg="#E74C3C")
            return

        try:
            # Use the same Image_Detector module as in original code
            model_path = 'trained_model/model ver 3/normal/trained_model_after_normal.keras'
            output1, output2 = Image_Detector.classify_image(self.selected_img_path, model_path)

            # Translate results if needed and update UI
            if output1 == "AI":
                result_text = "AI"  # or "Ảnh AI" if you prefer
                self.class_label.config(text=result_text, fg="#E74C3C")
                self.result_frame1.config(bg="#FADBD8")
                self.class_label.config(bg="#FADBD8")
            else:
                result_text = "THẬT"  # or "Ảnh Thật" if you prefer
                self.class_label.config(text=result_text, fg="#27AE60")
                self.result_frame1.config(bg="#D5F5E3")
                self.class_label.config(bg="#D5F5E3")

            # Show confidence score
            confidence = round(output2, 4)
            self.reliability_label.config(text=str(confidence), fg="#2C3E50")

            # Change reliability background based on confidence
            if confidence > 0.8:
                self.result_frame2.config(bg="#D5F5E3")
                self.reliability_label.config(bg="#D5F5E3")
            elif confidence > 0.5:
                self.result_frame2.config(bg="#FCF3CF")
                self.reliability_label.config(bg="#FCF3CF")
            else:
                self.result_frame2.config(bg="#FADBD8")
                self.reliability_label.config(bg="#FADBD8")

        except Exception as e:
            print("Lỗi trong quá trình phân loại:", e)
            self.class_label.config(text="Lỗi Phân Tích", fg="#E74C3C")
            self.reliability_label.config(text="N/A", fg="#E74C3C")


# Create and run application
if __name__ == "__main__":
    root = tk.Tk()
    app = AIDetectorApp(root)
    root.mainloop()