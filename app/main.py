from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

import cv2
import numpy as np

from app.detector import Candidate, DetectionConfig, DetectionSummary, PageResult, _draw_candidates, detect_document, load_document_pages
from app.review_ui import run_review


class PatternDetectionApp:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title("Pattern Markup Counter")
        self.root.geometry("860x620")

        self.input_var = tk.StringVar()
        self.template_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str((Path.cwd() / "outputs").resolve()))
        self.scope_var = tk.StringVar()

        self.threshold_var = tk.StringVar(value="0.45")
        self.dpi_var = tk.StringVar(value="220")
        self.min_scale_var = tk.StringVar(value="0.18")
        self.max_scale_var = tk.StringVar(value="1.0")
        self.dark_threshold_var = tk.StringVar(value="0")
        self.nms_var = tk.StringVar(value="0.35")
        self.num_scales_var = tk.StringVar(value="12")
        self.mode_var = tk.StringVar(value="Balanced (Recommended)")
        self.detail_var = tk.StringVar(value="Normal")
        self.show_advanced_var = tk.BooleanVar(value=False)
        self.last_summary: DetectionSummary | None = None
        self.last_pages: list[np.ndarray] = []
        self.last_scope = None

        self._build_ui()

    def _build_ui(self) -> None:
        main = ctk.CTkScrollableFrame(self.root, fg_color="transparent")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self._add_picker_row(
            main,
            "Input file (.png/.jpg/.jpeg/.pdf)",
            self.input_var,
            self._browse_input,
        )
        self._add_picker_row(
            main,
            "Markup template image",
            self.template_var,
            self._browse_template,
        )
        self._add_picker_row(
            main,
            "Output folder",
            self.output_var,
            self._browse_output,
            button_text="Browse...",
        )
        self._add_scope_row(main)

        ctk.CTkLabel(main, text="Detection settings", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(20, 4))
        simple = ctk.CTkFrame(main)
        simple.pack(fill="x", pady=4)
        self._add_dropdown_setting(
            simple,
            "Detection mode:",
            self.mode_var,
            ["Balanced (Recommended)", "Find More (Sensitive)", "Avoid False Detections (Strict)"],
            row=0,
            col=0,
        )
        self._add_dropdown_setting(
            simple,
            "Detail level:",
            self.detail_var,
            ["Normal", "Faster", "Tiny marks (Slower)"],
            row=0,
            col=1,
        )
        ctk.CTkLabel(
            simple,
            text="Tip: Start with Balanced + Normal. Change only if results are too many/few.",
            text_color=("gray35", "gray75"),
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 8))

        toggle = ctk.CTkFrame(main, fg_color="transparent")
        toggle.pack(fill="x", pady=(2, 2))
        ctk.CTkCheckBox(
            toggle,
            text="Show advanced settings",
            variable=self.show_advanced_var,
            command=self._toggle_advanced_settings,
        ).pack(anchor="w")

        self.advanced_settings = ctk.CTkFrame(main)
        self._add_setting(self.advanced_settings, "Match threshold (0.45-0.9):", self.threshold_var, row=0, col=0)
        self._add_setting(self.advanced_settings, "PDF render DPI:", self.dpi_var, row=0, col=1)
        self._add_setting(self.advanced_settings, "Min scale:", self.min_scale_var, row=1, col=0)
        self._add_setting(self.advanced_settings, "Max scale:", self.max_scale_var, row=1, col=1)
        self._add_setting(self.advanced_settings, "Dark threshold (0=auto):", self.dark_threshold_var, row=2, col=0)
        self._add_setting(self.advanced_settings, "NMS IoU threshold:", self.nms_var, row=2, col=1)
        self._add_setting(self.advanced_settings, "Num scale steps:", self.num_scales_var, row=3, col=0)

        action = ctk.CTkFrame(main, fg_color="transparent")
        action.pack(fill="x", pady=20)
        ctk.CTkButton(action, text="Run Detection", command=self._run_detection, height=40, font=ctk.CTkFont(weight="bold")).pack(side=tk.LEFT)
        ctk.CTkButton(action, text="Review Results", command=self._review_results, height=40, fg_color="#2B8C67", hover_color="#1E654A", font=ctk.CTkFont(weight="bold")).pack(side=tk.LEFT, padx=(12, 0))
        ctk.CTkButton(action, text="Clear Log", command=self._clear_log, fg_color="transparent", border_width=1, text_color=("black", "white")).pack(side=tk.RIGHT)

        ctk.CTkLabel(main, text="Run log", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(0, 4))
        self.log = ctk.CTkTextbox(main, height=200)
        self.log.pack(fill=tk.BOTH, expand=True)
        self._append_log("Ready. Pick input file + template image, then click Run Detection.")

    def _add_picker_row(
        self,
        parent: ctk.CTkFrame,
        label: str,
        variable: tk.StringVar,
        browse_handler,
        button_text: str = "Select",
    ) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text=label, width=220, anchor="w").pack(side=tk.LEFT)
        ctk.CTkEntry(row, textvariable=variable).pack(side=tk.LEFT, fill="x", expand=True, padx=12)
        ctk.CTkButton(row, text=button_text, command=browse_handler, width=80).pack(side=tk.RIGHT)

    def _add_setting(self, parent: ctk.CTkFrame, text: str, variable: tk.StringVar, row: int, col: int) -> None:
        holder = ctk.CTkFrame(parent, fg_color="transparent")
        holder.grid(row=row, column=col, padx=12, pady=6, sticky="ew")
        ctk.CTkLabel(holder, text=text, width=180, anchor="w").pack(side=tk.LEFT)
        entry = ctk.CTkEntry(holder, textvariable=variable, width=80)
        self._enable_decimal_input(entry)
        entry.pack(side=tk.LEFT)
        parent.grid_columnconfigure(col, weight=1)

    def _add_dropdown_setting(
        self,
        parent: ctk.CTkFrame,
        text: str,
        variable: tk.StringVar,
        values: list[str],
        row: int,
        col: int,
    ) -> None:
        holder = ctk.CTkFrame(parent, fg_color="transparent")
        holder.grid(row=row, column=col, padx=12, pady=6, sticky="ew")
        ctk.CTkLabel(holder, text=text, width=180, anchor="w").pack(side=tk.LEFT)
        ctk.CTkOptionMenu(holder, variable=variable, values=values, width=220).pack(side=tk.LEFT)
        parent.grid_columnconfigure(col, weight=1)

    def _enable_decimal_input(self, entry: ctk.CTkEntry) -> None:
        # Some keyboard/layout combinations do not emit "." into CTkEntry.
        # Map decimal-like keys explicitly so numeric fields always accept decimals.
        def insert_dot(event):
            entry.insert(tk.INSERT, ".")
            return "break"

        def map_decimal_keys(event):
            if event.char in {".", ",", "。", "．"} or event.keysym in {"period", "KP_Decimal", "decimal", "comma"}:
                return insert_dot(event)
            return None

        entry.bind("<KeyPress>", map_decimal_keys)
        entry.bind("<KeyPress-period>", insert_dot)
        entry.bind("<KeyPress-KP_Decimal>", insert_dot)
        entry.bind("<KeyPress-comma>", insert_dot)

    def _toggle_advanced_settings(self) -> None:
        if self.show_advanced_var.get():
            self.advanced_settings.pack(fill="x", pady=(4, 8))
        else:
            self.advanced_settings.pack_forget()

    def _apply_easy_settings(self) -> None:
        mode = self.mode_var.get()
        detail = self.detail_var.get()

        mode_cfg = {
            "Balanced (Recommended)": ("0.45", "0.18", "1.0", "0.35"),
            "Find More (Sensitive)": ("0.38", "0.14", "1.05", "0.40"),
            "Avoid False Detections (Strict)": ("0.56", "0.20", "0.95", "0.30"),
        }
        detail_cfg = {
            "Faster": ("180", "10"),
            "Normal": ("220", "12"),
            "Tiny marks (Slower)": ("300", "16"),
        }

        threshold, min_scale, max_scale, nms = mode_cfg.get(mode, mode_cfg["Balanced (Recommended)"])
        dpi, num_scales = detail_cfg.get(detail, detail_cfg["Normal"])

        self.threshold_var.set(threshold)
        self.min_scale_var.set(min_scale)
        self.max_scale_var.set(max_scale)
        self.nms_var.set(nms)
        self.dpi_var.set(dpi)
        self.num_scales_var.set(num_scales)
        self.dark_threshold_var.set("0")

    def _browse_input(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Choose input file",
            filetypes=[
                ("Supported files", "*.png *.jpg *.jpeg *.pdf"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.input_var.set(file_path)

    def _browse_template(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Choose markup template image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.template_var.set(file_path)

    def _browse_output(self) -> None:
        folder = filedialog.askdirectory(title="Choose output folder")
        if folder:
            self.output_var.set(folder)

    def _add_scope_row(self, parent: ctk.CTkFrame) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text="Detection scope (optional)", width=220, anchor="w").pack(side=tk.LEFT)
        ctk.CTkEntry(row, textvariable=self.scope_var).pack(side=tk.LEFT, fill="x", expand=True, padx=12)
        ctk.CTkButton(row, text="Draw Scope", command=self._pick_scope, width=100).pack(side=tk.RIGHT)
        ctk.CTkButton(row, text="Clear", command=lambda: self.scope_var.set(""), fg_color="transparent", border_width=1, text_color=("black", "white"), width=60).pack(side=tk.RIGHT, padx=(0, 12))

    def _pick_scope(self) -> None:
        try:
            input_path = Path(self.input_var.get().strip())
            if not input_path.exists():
                raise ValueError("Choose an input file before picking scope.")

            pages = load_document_pages(input_path, dpi=int(self.dpi_var.get()))
            if not pages:
                raise ValueError("Cannot load preview image for scope selection.")

            image = pages[0]
            preview, scale = _resize_preview(image, max_dim=1400)
            polygon = _collect_freehand_polygon(preview)
            if not polygon:
                self._append_log("Scope selection cancelled.")
                return

            orig_h, orig_w = image.shape[:2]
            normalized = []
            for x, y in polygon:
                normalized.append(((x / scale) / orig_w, (y / scale) / orig_h))
            scope_text = "poly:" + ";".join(f"{x:.6f},{y:.6f}" for x, y in normalized)
            self.scope_var.set(scope_text)
            self._append_log(f"Scope set: {scope_text}")
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"ERROR picking scope: {exc}")
            messagebox.showerror(title="Scope Selection Failed", message=str(exc))

    def _run_detection(self) -> None:
        try:
            input_path = Path(self.input_var.get().strip())
            template_path = Path(self.template_var.get().strip())
            output_dir = Path(self.output_var.get().strip())

            if not input_path.exists():
                raise ValueError(f"Input file does not exist: {input_path}")
            if not template_path.exists():
                raise ValueError(f"Template file does not exist: {template_path}")
            output_dir.mkdir(parents=True, exist_ok=True)

            if not self.show_advanced_var.get():
                self._apply_easy_settings()

            config = DetectionConfig(
                dpi=int(self.dpi_var.get()),
                match_threshold=float(self.threshold_var.get()),
                min_scale=float(self.min_scale_var.get()),
                max_scale=float(self.max_scale_var.get()),
                dark_threshold=int(self.dark_threshold_var.get()),
                nms_iou_threshold=float(self.nms_var.get()),
                num_scales=int(self.num_scales_var.get()),
                scope=_parse_scope(self.scope_var.get()),
            )

            self._append_log(f"Input: {input_path}")
            self._append_log(f"Template: {template_path}")
            self._append_log(f"Output: {output_dir}")
            self._append_log("Running detection...")
            self.root.update_idletasks()

            summary = detect_document(
                input_path=input_path,
                template_path=template_path,
                output_dir=output_dir,
                config=config,
            )
            self.last_summary = summary
            self.last_pages = load_document_pages(input_path, dpi=int(self.dpi_var.get()))
            self.last_scope = config.scope

            self._append_log(f"Total markups found: {summary.total_count}")
            for page in summary.page_results:
                self._append_log(
                    f"Page {page.page_number}: {page.count} markups -> {page.annotated_path}"
                )

            messagebox.showinfo(
                title="Detection Complete",
                message=f"Total markups found: {summary.total_count}\nOutput: {summary.output_dir}",
            )
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"ERROR: {exc}")
            self._append_log(traceback.format_exc())
            messagebox.showerror(title="Detection Failed", message=str(exc))

    def _review_results(self) -> None:
        try:
            if self.last_summary is None or not self.last_pages:
                raise ValueError("Run detection first, then review the results.")
                
            def on_review_save(updated_summary: DetectionSummary):
                self.last_summary = updated_summary
                self._append_log(f"Reviewed total markups: {updated_summary.total_count}")
                for page in updated_summary.page_results:
                    self._append_log(f"Reviewed page {page.page_number}: {page.count} markups -> {page.annotated_path}")
                messagebox.showinfo(
                    title="Review Saved",
                    message=f"Updated total markups: {updated_summary.total_count}\nOutput: {updated_summary.output_dir}",
                )
                
            run_review(self.root, self.last_pages, self.last_summary, self.last_scope, on_review_save)
            
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"ERROR during review: {exc}")
            self._append_log(traceback.format_exc())
            messagebox.showerror(title="Review Failed", message=str(exc))

    def _clear_log(self) -> None:
        self.log.delete("0.0", tk.END)

    def _append_log(self, text: str) -> None:
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pattern markup counter")
    parser.add_argument("--input", type=Path, help="Input file (.png/.jpg/.jpeg/.pdf)")
    parser.add_argument("--template", type=Path, help="Template image of the markup symbol")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output folder")
    parser.add_argument("--threshold", type=float, default=0.45, help="Match threshold")
    parser.add_argument("--dpi", type=int, default=220, help="PDF rendering DPI")
    parser.add_argument("--min-scale", type=float, default=0.18, help="Minimum template scale")
    parser.add_argument("--max-scale", type=float, default=1.0, help="Maximum template scale")
    parser.add_argument("--dark-threshold", type=int, default=0, help="Dark pixel threshold (0=auto)")
    parser.add_argument("--nms-threshold", type=float, default=0.35, help="NMS IoU threshold")
    parser.add_argument("--num-scales", type=int, default=12, help="Number of scale steps for template matching sweep")
    parser.add_argument(
        "--color-sensitivity",
        type=str,
        choices=("auto", "soft", "strict"),
        default="auto",
        help="Template color sensitivity: auto, soft, or strict",
    )
    parser.add_argument("--scope", type=str, default="", help="Normalized polygon scope: poly:x0,y0;x1,y1;...")
    return parser.parse_args(argv)


def _run_cli(args: argparse.Namespace) -> int:
    if not args.input or not args.template:
        raise ValueError("CLI mode requires --input and --template")

    cfg = DetectionConfig(
        dpi=args.dpi,
        match_threshold=args.threshold,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        dark_threshold=args.dark_threshold,
        nms_iou_threshold=args.nms_threshold,
        num_scales=args.num_scales,
        scope=_parse_scope(args.scope),
        color_sensitivity=args.color_sensitivity,
    )
    summary = detect_document(
        input_path=args.input,
        template_path=args.template,
        output_dir=args.output,
        config=cfg,
    )
    print(f"Total markups found: {summary.total_count}")
    for page in summary.page_results:
        print(f"Page {page.page_number}: {page.count} -> {page.annotated_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.input and args.template:
        return _run_cli(args)

    root = ctk.CTk()
    PatternDetectionApp(root)
    root.mainloop()
    return 0


def _parse_scope(value: str) -> tuple[tuple[float, float], ...] | None:
    text = value.strip()
    if not text:
        return None
    if not text.lower().startswith("poly:"):
        raise ValueError("Scope must use polygon format: poly:x0,y0;x1,y1;...")
    points: list[tuple[float, float]] = []
    for chunk in text[5:].split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [item.strip() for item in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError("Each scope point must be x,y using normalized values from 0 to 1.")
        x = float(parts[0])
        y = float(parts[1])
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            raise ValueError("Scope values must stay between 0 and 1.")
        points.append((x, y))
    if len(points) < 3:
        raise ValueError("Free-drawn scope needs at least 3 points.")
    return tuple(points)


def _resize_preview(image, max_dim: int):
    height, width = image.shape[:2]
    current_max = max(height, width)
    if current_max <= max_dim:
        return image, 1.0
    scale = float(max_dim) / float(current_max)
    preview = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return preview, scale


def _collect_freehand_polygon(image):
    window = "Draw detection scope"
    points = []
    drawing = {"active": False}

    base = image.copy()
    canvas = image.copy()

    def redraw():
        nonlocal canvas
        canvas = base.copy()
        if len(points) >= 2:
            pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 180, 0), thickness=3)
        if points:
            cv2.circle(canvas, points[0], 4, (0, 180, 0), -1)
            cv2.circle(canvas, points[-1], 4, (0, 255, 0), -1)
        cv2.putText(
            canvas,
            "Hold left mouse to draw. Enter=finish, C=clear, Esc=cancel",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 100, 0),
            2,
            cv2.LINE_AA,
        )

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["active"] = True
            points.append((x, y))
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and drawing["active"]:
            if not points or abs(x - points[-1][0]) + abs(y - points[-1][1]) >= 4:
                points.append((x, y))
                redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            drawing["active"] = False
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            break
        if key in (27,):
            points = []
            break
        if key in (ord("c"), ord("C")):
            points = []
            redraw()

    cv2.destroyWindow(window)
    if len(points) < 3:
        return []
    return _simplify_points(points)


def _simplify_points(points, min_gap: int = 8):
    simplified = [points[0]]
    for point in points[1:]:
        if abs(point[0] - simplified[-1][0]) + abs(point[1] - simplified[-1][1]) >= min_gap:
            simplified.append(point)
    if len(simplified) >= 3 and simplified[-1] != simplified[0]:
        simplified.append(simplified[0])
    return simplified


# (OpenCV review functions removed)



if __name__ == "__main__":
    raise SystemExit(main())
