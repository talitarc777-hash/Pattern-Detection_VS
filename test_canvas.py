import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ReviewWindow(ctk.CTkToplevel):
    def __init__(self, parent, page_bgr, candidates):
        super().__init__(parent)
        self.title("Review Detections")
        self.geometry("1000x800")
        
        self.page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
        self.candidates = candidates
        
        # Add control frame
        self.control_frame = ctk.CTkFrame(self, height=50)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.mode_var = tk.StringVar(value="delete")
        ctk.CTkRadioButton(self.control_frame, text="Delete mode (Click)", variable=self.mode_var, value="delete").pack(side=tk.LEFT, padx=10)
        ctk.CTkRadioButton(self.control_frame, text="Add mode (Drag rect)", variable=self.mode_var, value="add").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(self.control_frame, text="Save & Next", command=self.save).pack(side=tk.RIGHT, padx=10)
        
        self.canvas = tk.Canvas(self, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Fit image to canvas initially
        self.bind("<Configure>", self.on_resize)
        
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)
        
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.pan_start = None
        self.draw_start_img = None
        self.pending_rect = None
        
        self.first_resize = True

    def on_resize(self, event):
        if self.first_resize and event.width > 100:
            self.first_resize = False
            img_h, img_w = self.page_rgb.shape[:2]
            scale_w = event.width / img_w
            scale_h = (event.height - 50) / img_h
            self.scale = min(scale_w, scale_h, 1.0)
            self.offset_x = (event.width - img_w * self.scale) / 2
            self.offset_y = ((event.height - 50) - img_h * self.scale) / 2
            self.redraw()

    def canvas_to_img(self, cx, cy):
        ix = (cx - self.offset_x) / self.scale
        iy = (cy - self.offset_y) / self.scale
        return int(ix), int(iy)

    def img_to_canvas(self, ix, iy):
        cx = ix * self.scale + self.offset_x
        cy = iy * self.scale + self.offset_y
        return cx, cy

    def on_mousewheel(self, event):
        x, y = event.x, event.y
        ix, iy = self.canvas_to_img(x, y)
        
        if event.delta > 0:
            self.scale *= 1.2
        else:
            self.scale /= 1.2
            
        self.scale = max(0.05, min(self.scale, 10.0))
        
        # Adjust offset so the point under mouse stays in same place
        new_cx = ix * self.scale + self.offset_x
        new_cy = iy * self.scale + self.offset_y
        
        self.offset_x += (x - new_cx)
        self.offset_y += (y - new_cy)
        
        self.redraw()

    def start_pan(self, event):
        self.pan_start = (event.x, event.y)

    def do_pan(self, event):
        if self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start = (event.x, event.y)
            self.redraw()

    def on_click(self, event):
        ix, iy = self.canvas_to_img(event.x, event.y)
        if self.mode_var.get() == "delete":
            # Find closest candidate to delete
            for i in range(len(self.candidates)-1, -1, -1):
                c = self.candidates[i]
                if c[0] <= ix <= c[0]+c[2] and c[1] <= iy <= c[1]+c[3]:
                    del self.candidates[i]
                    self.redraw()
                    return
        else:
            self.draw_start_img = (ix, ly)

    def on_drag(self, event):
        if self.mode_var.get() == "add" and self.draw_start_img:
            ix, iy = self.canvas_to_img(event.x, event.y)
            self.pending_rect = (self.draw_start_img[0], self.draw_start_img[1], ix, iy)
            self.redraw()

    def on_release(self, event):
        if self.mode_var.get() == "add" and self.pending_rect:
            x0, y0, x1, y1 = self.pending_rect
            left = min(x0, x1)
            right = max(x0, x1)
            top = min(y0, y1)
            bottom = max(y0, y1)
            if right - left > 5 and bottom - top > 5:
                self.candidates.append((left, top, right-left, bottom-top))
            self.pending_rect = None
            self.draw_start_img = None
            self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        
        # Fast resize
        img_h, img_w = self.page_rgb.shape[:2]
        new_w = int(img_w * self.scale)
        new_h = int(img_h * self.scale)
        
        if new_w <= 0 or new_h <= 0:
            return
            
        # Draw image
        resized = cv2.resize(self.page_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        img = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(img) # Keep ref
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor="nw")
        
        # Draw boxes
        for x, y, w, h in self.candidates:
            cx0, cy0 = self.img_to_canvas(x, y)
            cx1, cy1 = self.img_to_canvas(x+w, y+h)
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="#00ff00", width=2)
            
        # Draw pending
        if self.pending_rect:
            x0, y0, x1, y1 = self.pending_rect
            cx0, cy0 = self.img_to_canvas(x0, y0)
            cx1, cy1 = self.img_to_canvas(x1, y1)
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="#ff7700", width=2)

    def save(self):
        self.destroy()

if __name__ == "__main__":
    app = ctk.CTk()
    app.withdraw()
    
    # Dummy data
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 40
    cands = [(100, 100, 50, 50), (400, 500, 100, 20)]
    
    win = ReviewWindow(app, img, cands)
    win.wait_window()
