import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Callable, Optional

from app.detector import Candidate, DetectionSummary, PageResult, _draw_candidates

class ReviewWindow(ctk.CTkToplevel):
    def __init__(
        self,
        parent: ctk.CTk,
        pages: list[np.ndarray],
        summary: DetectionSummary,
        normalized_scope: Optional[tuple[tuple[float, float], ...]],
        on_save: Callable[[DetectionSummary], None],
    ):
        super().__init__(parent)
        self.title("Review Detections")
        self.geometry("1100x800")
        
        # Bring window to front
        self.lift()
        self.attributes("-topmost", True)
        self.after(100, lambda: self.attributes("-topmost", False))
        
        self.pages = pages
        self.summary = summary
        self.normalized_scope = normalized_scope
        self.on_save = on_save
        
        self.page_index = 0
        # Deep copy candidates to allow editing without affecting the original unless saved
        self.page_candidates = [list(page.candidates) for page in summary.page_results]
        
        self.mode_var = tk.StringVar(value="delete")
        
        self._build_ui()
        
        # State
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_start = None
        self.draw_start_img = None
        self.pending_rect = None
        self.photo = None  # to keep reference
        self.canvas_image_id = None
        self.base_page_rgbs: list[np.ndarray | None] = [None] * len(self.pages)
        self.render_cache_key = None
        self.first_resize = True
        
        self.bind("<Configure>", self.on_resize)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)
        
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Keyboard shortcuts
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("a", lambda e: self.mode_var.set("add"))
        self.bind("A", lambda e: self.mode_var.set("add"))
        self.bind("d", lambda e: self.mode_var.set("delete"))
        self.bind("D", lambda e: self.mode_var.set("delete"))
        self.bind("n", lambda e: self.next_page())
        self.bind("N", lambda e: self.next_page())
        self.bind("p", lambda e: self.prev_page())
        self.bind("P", lambda e: self.prev_page())
        self.bind("s", lambda e: self.save_all())
        self.bind("S", lambda e: self.save_all())

        # Fit first page soon
        self.after(50, lambda: self._fit_to_screen(force=True))

    def _build_ui(self):
        # Top toolbar
        toolbar = ctk.CTkFrame(self, height=50, corner_radius=0)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        
        self.lbl_info = ctk.CTkLabel(toolbar, text="", font=ctk.CTkFont(weight="bold"))
        self.lbl_info.pack(side=tk.LEFT, padx=16, pady=8)
        
        ctk.CTkButton(toolbar, text="Prev Page", command=self.prev_page, width=80).pack(side=tk.LEFT, padx=8)
        ctk.CTkButton(toolbar, text="Next Page", command=self.next_page, width=80).pack(side=tk.LEFT, padx=8)
        
        ctk.CTkRadioButton(toolbar, text="Delete mode (Click)", variable=self.mode_var, value="delete").pack(side=tk.LEFT, padx=(30, 10))
        ctk.CTkRadioButton(toolbar, text="Add mode (Drag)", variable=self.mode_var, value="add").pack(side=tk.LEFT, padx=10)
        
        ctk.CTkButton(toolbar, text="Save & Close", command=self.save_all, fg_color="#2B8C67", hover_color="#1E654A").pack(side=tk.RIGHT, padx=16)
        ctk.CTkButton(toolbar, text="Cancel", command=self.destroy, fg_color="transparent", border_width=1, text_color=("black", "white")).pack(side=tk.RIGHT)
        
        # Scrollable Canvas
        self.canvas = tk.Canvas(self, bg="#212121", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _fit_to_screen(self, force=False):
        if not hasattr(self, "canvas"): return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 100 or h < 100:
            if force: self.after(50, lambda: self._fit_to_screen(True))
            return
            
        page = self.pages[self.page_index]
        img_h, img_w = page.shape[:2]
        scale_w = w / float(img_w)
        scale_h = h / float(img_h)
        self.scale = min(scale_w, scale_h, 1.0) * 0.95
        self.offset_x = (w - img_w * self.scale) / 2
        self.offset_y = (h - img_h * self.scale) / 2
        self._update_info()
        self.redraw(full=True)

    def on_resize(self, event):
        if self.first_resize and event.widget == self and event.width > 200:
            self.first_resize = False
            self.after(50, lambda: self._fit_to_screen(True))

    def _update_info(self):
        count = len(self.page_candidates[self.page_index])
        self.lbl_info.configure(text=f"Page {self.page_index + 1}/{len(self.pages)}  |  Count: {count}")

    def prev_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self._fit_to_screen(force=True)

    def next_page(self):
        if self.page_index < len(self.pages) - 1:
            self.page_index += 1
            self._fit_to_screen(force=True)

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
            
        self.scale = max(0.05, min(self.scale, 20.0))
        
        new_cx = ix * self.scale + self.offset_x
        new_cy = iy * self.scale + self.offset_y
        
        self.offset_x += (x - new_cx)
        self.offset_y += (y - new_cy)
        self.redraw(full=True)

    def start_pan(self, event):
        self.pan_start = (event.x, event.y)

    def do_pan(self, event):
        if self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start = (event.x, event.y)
            self.redraw(full=False)

    def on_click(self, event):
        ix, iy = self.canvas_to_img(event.x, event.y)
        candidates = self.page_candidates[self.page_index]
        
        if self.mode_var.get() == "delete":
            # Find closest candidate to delete
            for i in range(len(candidates)-1, -1, -1):
                c = candidates[i]
                if c.x <= ix <= c.x + c.w and c.y <= iy <= c.y + c.h:
                    del candidates[i]
                    self._update_info()
                    self.redraw(full=False)
                    return
        else:
            self.draw_start_img = (ix, iy)

    def on_drag(self, event):
        if self.mode_var.get() == "add" and self.draw_start_img:
            ix, iy = self.canvas_to_img(event.x, event.y)
            self.pending_rect = (self.draw_start_img[0], self.draw_start_img[1], ix, iy)
            self.redraw(full=False)

    def on_release(self, event):
        if self.mode_var.get() == "add" and self.pending_rect:
            x0, y0, x1, y1 = self.pending_rect
            left = min(x0, x1)
            right = max(x0, x1)
            top = min(y0, y1)
            bottom = max(y0, y1)
            
            page = self.pages[self.page_index]
            max_h, max_w = page.shape[:2]
            left = max(0, min(left, max_w - 1))
            top = max(0, min(top, max_h - 1))
            right = max(0, min(right, max_w - 1))
            bottom = max(0, min(bottom, max_h - 1))
            
            w = right - left
            h = bottom - top
            
            if w > 4 and h > 4:
                self.page_candidates[self.page_index].append(
                    Candidate(x=left, y=top, w=w, h=h, score=1.0, angle=0.0, scale=1.0)
                )
                
            self.pending_rect = None
            self.draw_start_img = None
            self._update_info()
            self.redraw(full=False)

    def _get_base_page_rgb(self) -> np.ndarray:
        cached = self.base_page_rgbs[self.page_index]
        if cached is not None:
            return cached

        page_rgb = cv2.cvtColor(self.pages[self.page_index], cv2.COLOR_BGR2RGB)
        img_h, img_w = page_rgb.shape[:2]

        if self.normalized_scope:
            pts = [(int(x * img_w), int(y * img_h)) for x, y in self.normalized_scope]
            pts_array = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(page_rgb, [pts_array], isClosed=False, color=(0, 255, 0), thickness=2)
            cv2.circle(page_rgb, pts[0], 6, (0, 255, 0), -1)
            cv2.circle(page_rgb, pts[-1], 6, (0, 255, 0), -1)

        self.base_page_rgbs[self.page_index] = page_rgb
        return page_rgb

    def _redraw_overlays(self):
        self.canvas.delete("overlay")

        for cand in self.page_candidates[self.page_index]:
            cx0, cy0 = self.img_to_canvas(cand.x, cand.y)
            cx1, cy1 = self.img_to_canvas(cand.x + cand.w, cand.y + cand.h)
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="#00ff00", width=2, tags="overlay")

        if self.pending_rect:
            x0, y0, x1, y1 = self.pending_rect
            cx0, cy0 = self.img_to_canvas(x0, y0)
            cx1, cy1 = self.img_to_canvas(x1, y1)
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="#ff7700", width=2, tags="overlay")

    def redraw(self, full: bool = False):
        if not hasattr(self, "canvas"):
            return

        page_rgb = self._get_base_page_rgb()
        img_h, img_w = page_rgb.shape[:2]
        new_w = int(img_w * self.scale)
        new_h = int(img_h * self.scale)
        
        if new_w <= 0 or new_h <= 0:
            return

        cache_key = (self.page_index, new_w, new_h)
        if full or self.render_cache_key != cache_key or self.canvas_image_id is None:
            resized = cv2.resize(page_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            self.photo = ImageTk.PhotoImage(Image.fromarray(resized))
            if self.canvas_image_id is None:
                self.canvas_image_id = self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor="nw")
            else:
                self.canvas.itemconfigure(self.canvas_image_id, image=self.photo)
                self.canvas.coords(self.canvas_image_id, self.offset_x, self.offset_y)
            self.render_cache_key = cache_key
        else:
            self.canvas.coords(self.canvas_image_id, self.offset_x, self.offset_y)

        self._redraw_overlays()

    def save_all(self):
        # Apply the edited candidates back to a new DetectionSummary and save annotated images
        updated_results = []
        total = 0
        import pathlib
        
        for idx, (page, page_result, candidates) in enumerate(zip(self.pages, self.summary.page_results, self.page_candidates)):
            # Sort like in original
            candidates = sorted(candidates, key=lambda c: (c.y, c.x))
            total += len(candidates)
            
            # Draw final image
            scope_polygon = None
            if self.normalized_scope:
                h, w = page.shape[:2]
                scope_polygon = tuple((int(x * w), int(y * h)) for x, y in self.normalized_scope)
                
            annotated = _draw_candidates(page, candidates, scope_polygon)
            
            # Write annotated image
            ext = pathlib.Path(page_result.annotated_path).suffix.lower() or ".png"
            ok, encoded = cv2.imencode(ext, annotated)
            if ok:
                encoded.tofile(str(page_result.annotated_path))
                
            updated_results.append(
                PageResult(
                    page_number=idx + 1,
                    count=len(candidates),
                    annotated_path=page_result.annotated_path,
                    candidates=tuple(candidates),
                )
            )

        new_summary = DetectionSummary(
            total_count=total,
            page_results=tuple(updated_results),
            output_dir=self.summary.output_dir,
        )
        
        self.on_save(new_summary)
        self.destroy()

def run_review(root: ctk.CTk, pages: list[np.ndarray], summary: DetectionSummary, scope: Optional[tuple[tuple[float, float], ...]], on_save: Callable[[DetectionSummary], None]):
    ReviewWindow(root, pages, summary, scope, on_save)
