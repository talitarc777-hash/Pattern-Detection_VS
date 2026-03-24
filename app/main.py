from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import customtkinter as ctk

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

import cv2
import numpy as np

from app.detector import (
    Candidate,
    DetectionConfig,
    DetectionSummary,
    MarkupClass,
    PageResult,
    _draw_candidates,
    _imwrite_unicode,
    detect_document,
    detect_document_multi,
    load_document_pages,
)
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
        self.markup_source_path: Path | None = None
        self.markup_items: list[dict[str, object]] = []

        self.threshold_var = tk.StringVar(value="0.45")
        self.dpi_var = tk.StringVar(value="220")
        self.min_scale_var = tk.StringVar(value="0.18")
        self.max_scale_var = tk.StringVar(value="1.0")
        self.dark_threshold_var = tk.StringVar(value="0")
        self.nms_var = tk.StringVar(value="0.35")
        self.num_scales_var = tk.StringVar(value="12")
        self.mode_var = tk.StringVar(value="Balanced (Recommended)")
        self.detail_var = tk.StringVar(value="Normal")
        self.color_mode_var = tk.StringVar(value="Auto (Recommended)")
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
            "1. Input file (.png/.jpg/.jpeg/.pdf)",
            self.input_var,
            self._browse_input,
        )
        self._add_markup_row(main)
        self._add_picker_row(
            main,
            "4. Output folder",
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
        self._add_dropdown_setting(
            simple,
            "Color matching:",
            self.color_mode_var,
            ["Auto (Recommended)", "Allow More Variation", "Same Color Only (Stricter)"],
            row=1,
            col=0,
        )
        ctk.CTkLabel(
            simple,
            text="Tip: Start with Balanced + Normal + Auto. Use 'Same Color Only' when real markups share one color.",
            text_color=("gray35", "gray75"),
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 8))

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
        self._append_log("Ready. Choose the input file, pick one markup from that file, draw scope if needed, then run detection.")
        self._append_log("You can add multiple markups (classes) from the same file before running.")

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

    def _add_markup_row(self, parent: ctk.CTkFrame) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text="2. Markup classes from this file", width=220, anchor="w").pack(side=tk.LEFT)
        list_wrap = ctk.CTkFrame(row)
        list_wrap.pack(side=tk.LEFT, fill="x", expand=True, padx=12)
        self.markup_listbox = tk.Listbox(list_wrap, height=4, exportselection=False)
        self.markup_listbox.pack(fill="x", expand=True, padx=6, pady=6)

        btns = ctk.CTkFrame(row, fg_color="transparent")
        btns.pack(side=tk.RIGHT)
        ctk.CTkButton(btns, text="Add Markup", command=self._pick_markup, width=110).pack(anchor="e", pady=(0, 4))
        ctk.CTkButton(btns, text="Rename", command=self._rename_markup, width=110).pack(anchor="e", pady=(0, 4))
        ctk.CTkButton(btns, text="Remove", command=self._remove_markup, width=110).pack(anchor="e", pady=(0, 4))
        ctk.CTkButton(
            btns,
            text="Clear",
            command=self._clear_markup,
            fg_color="transparent",
            border_width=1,
            text_color=("black", "white"),
            width=110,
        ).pack(anchor="e")

    def _legacy_enable_decimal_input(self, entry: ctk.CTkEntry) -> None:
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

    def _selected_color_sensitivity(self) -> str:
        mapping = {
            "Auto (Recommended)": "auto",
            "Allow More Variation": "soft",
            "Same Color Only (Stricter)": "strict",
        }
        return mapping.get(self.color_mode_var.get(), "auto")

    def _effective_scale_bounds(self, input_path: Path, min_scale: float, max_scale: float) -> tuple[float, float]:
        # If markup is cropped from the same file, symbols are usually close in size.
        # Tightening the scale sweep improves precision against text/lookalikes.
        if self.markup_source_path is None or self.markup_source_path != input_path:
            return min_scale, max_scale

        auto_min = 0.60
        auto_max = 1.50
        tuned_min = max(min_scale, auto_min)
        tuned_max = min(max_scale, auto_max)
        if tuned_max < tuned_min:
            return min_scale, max_scale
        return tuned_min, tuned_max

    def _enable_decimal_input(self, entry: ctk.CTkEntry) -> None:
        # Some keyboard/layout combinations do not emit "." into CTkEntry.
        # Map decimal-like keys explicitly so numeric fields always accept decimals.
        def insert_dot(event):
            entry.insert(tk.INSERT, ".")
            return "break"

        def map_decimal_keys(event):
            if event.char in {".", ",", "\u3002", "\uff0e"} or event.keysym in {"period", "KP_Decimal", "decimal", "comma"}:
                return insert_dot(event)
            return None

        entry.bind("<KeyPress>", map_decimal_keys)
        entry.bind("<KeyPress-period>", insert_dot)
        entry.bind("<KeyPress-KP_Decimal>", insert_dot)
        entry.bind("<KeyPress-comma>", insert_dot)

    def _browse_input(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Choose input file",
            filetypes=[
                ("Supported files", "*.png *.jpg *.jpeg *.pdf"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            previous = self.input_var.get().strip()
            self.input_var.set(file_path)
            if previous and Path(previous).resolve() != Path(file_path).resolve():
                self._clear_markup()
                self.scope_var.set("")
                self._append_log("Input file changed. Markup selection and scope were cleared.")

    def _browse_output(self) -> None:
        folder = filedialog.askdirectory(title="Choose output folder")
        if folder:
            self.output_var.set(folder)

    def _clear_markup(self) -> None:
        self.template_var.set("")
        self.markup_source_path = None
        self.markup_items = []
        self._refresh_markup_list()

    def _refresh_markup_list(self) -> None:
        if not hasattr(self, "markup_listbox"):
            return
        self.markup_listbox.delete(0, tk.END)
        for idx, item in enumerate(self.markup_items, start=1):
            name = str(item["name"])
            path = Path(str(item["path"]))
            color_hint = str(item.get("color_hint", "unknown-color"))
            shape_hint = str(item.get("shape_hint", "unknown-shape"))
            self.markup_listbox.insert(tk.END, f"{idx}. {name} [{color_hint}, {shape_hint}] ({path.name})")

    def _rename_markup(self) -> None:
        if not self.markup_items:
            return
        sel = self.markup_listbox.curselection()
        if not sel:
            messagebox.showinfo(title="Rename Markup", message="Select one markup class to rename.")
            return
        idx = int(sel[0])
        current = str(self.markup_items[idx]["name"])
        new_name = simpledialog.askstring("Rename Markup", "Markup class name:", initialvalue=current)
        if new_name and new_name.strip():
            self.markup_items[idx]["name"] = new_name.strip()
            self._refresh_markup_list()

    def _remove_markup(self) -> None:
        if not self.markup_items:
            return
        sel = self.markup_listbox.curselection()
        if not sel:
            messagebox.showinfo(title="Remove Markup", message="Select one markup class to remove.")
            return
        idx = int(sel[0])
        del self.markup_items[idx]
        if not self.markup_items:
            self.markup_source_path = None
        self._refresh_markup_list()

    def _add_scope_row(self, parent: ctk.CTkFrame) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text="3. Detection scope (optional)", width=220, anchor="w").pack(side=tk.LEFT)
        ctk.CTkEntry(row, textvariable=self.scope_var).pack(side=tk.LEFT, fill="x", expand=True, padx=12)
        ctk.CTkButton(row, text="Draw Scope", command=self._pick_scope, width=100).pack(side=tk.RIGHT)
        ctk.CTkButton(row, text="Clear", command=lambda: self.scope_var.set(""), fg_color="transparent", border_width=1, text_color=("black", "white"), width=60).pack(side=tk.RIGHT, padx=(0, 12))

    def _pick_markup(self) -> None:
        try:
            input_path = Path(self.input_var.get().strip())
            if not input_path.exists():
                raise ValueError("Choose an input file before picking markup.")

            pages = load_document_pages(input_path, dpi=int(self.dpi_var.get()))
            if not pages:
                raise ValueError("Cannot load preview image for markup selection.")

            image = pages[0]
            preview, scale = _resize_preview(image, max_dim=1400)
            selection = _collect_markup_region(preview)
            if selection is None:
                self._append_log("Markup selection cancelled.")
                return

            crop = _extract_markup_crop_from_selection(image, selection, scale)
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                raise ValueError("Markup crop is too small. Please select a slightly larger region.")
            color_hint, shape_hint = _infer_markup_hint(crop)
            cache_dir = (Path.cwd() / ".pattern_detection_cache").resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            markup_path = cache_dir / f"selected_markup_{len(self.markup_items)+1:02d}.png"
            _imwrite_unicode(markup_path, crop)

            self.markup_source_path = input_path.resolve()
            self.markup_items.append(
                {
                    "name": f"Markup {len(self.markup_items)+1}",
                    "path": str(markup_path),
                    "source": str(self.markup_source_path),
                    "color_hint": color_hint,
                    "shape_hint": shape_hint,
                }
            )
            self.template_var.set(str(markup_path))
            self._refresh_markup_list()
            self._append_log(
                f"Markup class added from page 1 -> {markup_path} "
                f"(hint: {color_hint}, {shape_hint})"
            )
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"ERROR picking markup: {exc}")
            messagebox.showerror(title="Markup Selection Failed", message=str(exc))

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
            output_dir = Path(self.output_var.get().strip())

            if not input_path.exists():
                raise ValueError(f"Input file does not exist: {input_path}")
            if not self.markup_items:
                legacy_path = Path(self.template_var.get().strip())
                if legacy_path.exists():
                    self.markup_items = [{"name": legacy_path.stem or "Markup 1", "path": str(legacy_path), "source": str(input_path.resolve())}]
                else:
                    raise ValueError("Pick at least one markup from the input file before running detection.")
            if self.markup_source_path is None or self.markup_source_path != input_path.resolve():
                raise ValueError("Pick markups again after changing the input file.")
            output_dir.mkdir(parents=True, exist_ok=True)

            if not self.show_advanced_var.get():
                self._apply_easy_settings()

            user_min_scale = float(self.min_scale_var.get())
            user_max_scale = float(self.max_scale_var.get())
            resolved_input = input_path.resolve()
            min_scale, max_scale = self._effective_scale_bounds(resolved_input, user_min_scale, user_max_scale)

            config = DetectionConfig(
                dpi=int(self.dpi_var.get()),
                match_threshold=float(self.threshold_var.get()),
                min_scale=min_scale,
                max_scale=max_scale,
                dark_threshold=int(self.dark_threshold_var.get()),
                nms_iou_threshold=float(self.nms_var.get()),
                num_scales=int(self.num_scales_var.get()),
                scope=_parse_scope(self.scope_var.get()),
                color_sensitivity=self._selected_color_sensitivity(),
                uniform_size_assist=self.markup_source_path == resolved_input,
                debug_artifacts=True,
            )

            self._append_log(f"Input: {input_path}")
            self._append_log(f"Output: {output_dir}")
            self._append_log(f"Markup classes: {len(self.markup_items)}")
            self._append_log(f"Color matching: {self.color_mode_var.get()}")
            if (min_scale, max_scale) != (user_min_scale, user_max_scale):
                self._append_log(
                    f"Auto size assist: scale narrowed from [{user_min_scale:.2f}, {user_max_scale:.2f}]"
                    f" to [{min_scale:.2f}, {max_scale:.2f}] (same-file markup)"
                )
            if config.uniform_size_assist:
                self._append_log("Uniform box assist: final detections will be normalized to similar size and re-deduped.")
            self._append_log("Running detection...")
            self._append_log(f"Debug artifacts will be saved to: {output_dir / '_debug'}")
            self.root.update_idletasks()

            markups = tuple(
                MarkupClass(name=str(item["name"]), template_path=Path(str(item["path"])))
                for item in self.markup_items
            )
            summary = detect_document_multi(input_path=input_path, markups=markups, output_dir=output_dir, config=config)
            self.last_summary = summary
            self.last_pages = load_document_pages(input_path, dpi=int(self.dpi_var.get()))
            self.last_scope = config.scope

            self._append_log(f"Total markups found: {summary.total_count}")
            for class_name, class_count in summary.class_totals:
                self._append_log(f"  - {class_name}: {class_count}")
            if summary.unclassified_count > 0:
                self._append_log(f"  - Unclassified (excluded): {summary.unclassified_count}")
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
    parser.add_argument(
        "--template",
        type=Path,
        action="append",
        help="Template image of the markup symbol. Repeat for multiple classes.",
    )
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
        debug_artifacts=True,
    )
    templates = [path.resolve() for path in args.template]
    if len(templates) == 1:
        summary = detect_document(
            input_path=args.input,
            template_path=templates[0],
            output_dir=args.output,
            config=cfg,
        )
    else:
        markups = tuple(MarkupClass(name=path.stem or f"markup_{idx+1}", template_path=path) for idx, path in enumerate(templates))
        summary = detect_document_multi(
            input_path=args.input,
            markups=markups,
            output_dir=args.output,
            config=cfg,
        )
    print(f"Total markups found: {summary.total_count}")
    if summary.class_totals:
        for class_name, class_count in summary.class_totals:
            print(f"  - {class_name}: {class_count}")
    if summary.unclassified_count:
        print(f"  - unclassified(excluded): {summary.unclassified_count}")
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


def _collect_markup_region(image):
    window = "Pick markup"
    base = image.copy()
    canvas = image.copy()
    selection = {"start": None, "end": None, "dragging": False, "rect": None}
    freehand = {"drawing": False, "points": []}
    mode = {"value": "rect"}
    zoom = {"value": 1.0}
    max_zoom = 6.0
    min_zoom = 0.5
    view = {"x0": 0, "y0": 0, "w": base.shape[1], "h": base.shape[0]}
    center = {"x": base.shape[1] / 2.0, "y": base.shape[0] / 2.0}
    pan = {"active": False, "last": None}

    def _clamp_center(vw: int, vh: int) -> None:
        bw = base.shape[1]
        bh = base.shape[0]
        min_cx = vw / 2.0
        max_cx = max(min_cx, bw - vw / 2.0)
        min_cy = vh / 2.0
        max_cy = max(min_cy, bh - vh / 2.0)
        center["x"] = float(np.clip(center["x"], min_cx, max_cx))
        center["y"] = float(np.clip(center["y"], min_cy, max_cy))

    def _update_view():
        z = max(zoom["value"], 1e-6)
        bh, bw = base.shape[:2]
        vw = max(1, int(round(bw / z)))
        vh = max(1, int(round(bh / z)))
        _clamp_center(vw, vh)
        x0 = int(np.clip(round(center["x"] - vw / 2.0), 0, max(0, bw - vw)))
        y0 = int(np.clip(round(center["y"] - vh / 2.0), 0, max(0, bh - vh)))
        view["x0"] = x0
        view["y0"] = y0
        view["w"] = vw
        view["h"] = vh

    def _to_base(point):
        bh, bw = base.shape[:2]
        x0 = view["x0"]
        y0 = view["y0"]
        vw = max(1, view["w"])
        vh = max(1, view["h"])
        px = int(round(x0 + (point[0] * vw / float(max(1, bw)))))
        py = int(round(y0 + (point[1] * vh / float(max(1, bh)))))
        px = int(np.clip(px, 0, bw - 1))
        py = int(np.clip(py, 0, bh - 1))
        return px, py

    def _to_view(point):
        bh, bw = base.shape[:2]
        x0 = view["x0"]
        y0 = view["y0"]
        vw = max(1, view["w"])
        vh = max(1, view["h"])
        vx = int(round((point[0] - x0) * bw / float(vw)))
        vy = int(round((point[1] - y0) * bh / float(vh)))
        return vx, vy

    def _change_zoom(factor):
        prev = zoom["value"]
        zoom["value"] = float(np.clip(prev * factor, min_zoom, max_zoom))
        redraw()

    def _pan_by_canvas_delta(dx: int, dy: int):
        bw = max(1, base.shape[1])
        bh = max(1, base.shape[0])
        vw = max(1, view["w"])
        vh = max(1, view["h"])
        # Drag right/down moves view to the left/up (content follows cursor).
        center["x"] -= dx * (vw / float(bw))
        center["y"] -= dy * (vh / float(bh))
        redraw()

    def redraw():
        nonlocal canvas
        _update_view()
        bh, bw = base.shape[:2]
        x0 = view["x0"]
        y0 = view["y0"]
        vw = view["w"]
        vh = view["h"]
        cropped = base[y0 : y0 + vh, x0 : x0 + vw]
        canvas = cv2.resize(cropped, (bw, bh), interpolation=cv2.INTER_LINEAR)
        start = selection["start"]
        end = selection["end"]
        rect = selection["rect"]
        points = freehand["points"]

        if mode["value"] == "rect" and start is not None and end is not None:
            x0 = min(start[0], end[0])
            y0 = min(start[1], end[1])
            x1 = max(start[0], end[0])
            y1 = max(start[1], end[1])
            sx0, sy0 = _to_view((x0, y0))
            sx1, sy1 = _to_view((x1, y1))
            cv2.rectangle(canvas, (sx0, sy0), (sx1, sy1), (0, 180, 0), 2)
        elif mode["value"] == "rect" and rect is not None:
            x0, y0, x1, y1 = rect
            sx0, sy0 = _to_view((x0, y0))
            sx1, sy1 = _to_view((x1, y1))
            cv2.rectangle(canvas, (sx0, sy0), (sx1, sy1), (0, 180, 0), 2)
        elif mode["value"] == "freehand" and len(points) >= 2:
            pts_view = np.asarray([_to_view(point) for point in points], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_view], isClosed=len(points) >= 3 and points[0] == points[-1], color=(0, 180, 0), thickness=2)

        cv2.putText(
            canvas,
            "Mode: R=box, F=freehand | Middle-drag=pan | +/- or wheel=zoom | Enter=finish | C=clear | Esc=cancel",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 100, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Selection mode: {mode['value']} | Zoom: {zoom['value']:.2f}x",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 100, 0),
            2,
            cv2.LINE_AA,
        )

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            base_point = _to_base((x, y))
            if mode["value"] == "rect":
                selection["start"] = base_point
                selection["end"] = base_point
                selection["dragging"] = True
                selection["rect"] = None
                freehand["points"] = []
            else:
                freehand["drawing"] = True
                freehand["points"] = [base_point]
                selection["start"] = None
                selection["end"] = None
                selection["dragging"] = False
                selection["rect"] = None
            redraw()
        elif event == cv2.EVENT_MBUTTONDOWN:
            pan["active"] = True
            pan["last"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and selection["dragging"] and mode["value"] == "rect":
            selection["end"] = _to_base((x, y))
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and freehand["drawing"] and mode["value"] == "freehand":
            point = _to_base((x, y))
            if not freehand["points"] or abs(point[0] - freehand["points"][-1][0]) + abs(point[1] - freehand["points"][-1][1]) >= 3:
                freehand["points"].append(point)
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and pan["active"] and pan["last"] is not None:
            last_x, last_y = pan["last"]
            _pan_by_canvas_delta(x - last_x, y - last_y)
            pan["last"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and selection["dragging"] and mode["value"] == "rect":
            selection["end"] = _to_base((x, y))
            selection["dragging"] = False
            start = selection["start"]
            end = selection["end"]
            if start is None or end is None:
                selection["rect"] = None
            else:
                x0 = min(start[0], end[0])
                y0 = min(start[1], end[1])
                x1 = max(start[0], end[0])
                y1 = max(start[1], end[1])
                selection["rect"] = None if (x1 - x0 < 4 or y1 - y0 < 4) else (x0, y0, x1, y1)
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and freehand["drawing"] and mode["value"] == "freehand":
            freehand["drawing"] = False
            pts = _simplify_points(freehand["points"], min_gap=3)
            if len(pts) >= 3 and pts[-1] != pts[0]:
                pts.append(pts[0])
            freehand["points"] = pts
            redraw()
        elif event == cv2.EVENT_MBUTTONUP:
            pan["active"] = False
            pan["last"] = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            if hasattr(cv2, "getMouseWheelDelta"):
                delta = cv2.getMouseWheelDelta(flags)
            else:
                delta = 1 if flags > 0 else -1
            if delta > 0:
                _change_zoom(1.20)
            elif delta < 0:
                _change_zoom(1.0 / 1.20)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    result = None
    while True:
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if len(freehand["points"]) >= 4:
                result = {"kind": "poly", "points": tuple(freehand["points"])}
            elif selection["rect"] is not None:
                result = {"kind": "rect", "rect": selection["rect"]}
            else:
                result = None
            break
        if key in (27,):
            result = None
            break
        if key in (ord("+"), ord("=")):
            _change_zoom(1.20)
        if key in (ord("-"), ord("_")):
            _change_zoom(1.0 / 1.20)
        if key in (ord("0"),):
            zoom["value"] = 1.0
            center["x"] = base.shape[1] / 2.0
            center["y"] = base.shape[0] / 2.0
            redraw()
        if key in (81,):  # left arrow
            _pan_by_canvas_delta(24, 0)
        if key in (83,):  # right arrow
            _pan_by_canvas_delta(-24, 0)
        if key in (82,):  # up arrow
            _pan_by_canvas_delta(0, 24)
        if key in (84,):  # down arrow
            _pan_by_canvas_delta(0, -24)
        if key in (ord("c"), ord("C")):
            selection = {"start": None, "end": None, "dragging": False, "rect": None}
            freehand = {"drawing": False, "points": []}
            redraw()
        if key in (ord("r"), ord("R")):
            mode["value"] = "rect"
            redraw()
        if key in (ord("f"), ord("F")):
            mode["value"] = "freehand"
            redraw()

    cv2.destroyWindow(window)
    return result


def _extract_markup_crop_from_selection(
    image_bgr: np.ndarray,
    selection: dict[str, object],
    preview_scale: float,
) -> np.ndarray:
    kind = str(selection.get("kind", ""))
    if kind == "rect":
        rect = selection.get("rect")
        if not isinstance(rect, tuple) or len(rect) != 4:
            raise ValueError("Invalid rectangle selection.")
        x0, y0, x1, y1 = rect
        ox0 = max(0, int(round(float(x0) / preview_scale)))
        oy0 = max(0, int(round(float(y0) / preview_scale)))
        ox1 = min(image_bgr.shape[1], int(round(float(x1) / preview_scale)))
        oy1 = min(image_bgr.shape[0], int(round(float(y1) / preview_scale)))
        if ox1 - ox0 < 4 or oy1 - oy0 < 4:
            raise ValueError("Markup crop is too small.")
        return image_bgr[oy0:oy1, ox0:ox1].copy()

    if kind == "poly":
        points = selection.get("points")
        if not isinstance(points, tuple) or len(points) < 4:
            raise ValueError("Invalid freehand selection.")
        orig_points: list[tuple[int, int]] = []
        for x, y in points:
            px = int(np.clip(round(float(x) / preview_scale), 0, image_bgr.shape[1] - 1))
            py = int(np.clip(round(float(y) / preview_scale), 0, image_bgr.shape[0] - 1))
            orig_points.append((px, py))
        pts = np.asarray(orig_points, dtype=np.int32)
        x0 = int(np.clip(np.min(pts[:, 0]), 0, image_bgr.shape[1] - 1))
        y0 = int(np.clip(np.min(pts[:, 1]), 0, image_bgr.shape[0] - 1))
        x1 = int(np.clip(np.max(pts[:, 0]) + 1, 1, image_bgr.shape[1]))
        y1 = int(np.clip(np.max(pts[:, 1]) + 1, 1, image_bgr.shape[0]))
        if x1 - x0 < 4 or y1 - y0 < 4:
            raise ValueError("Markup crop is too small.")

        crop = image_bgr[y0:y1, x0:x1].copy()
        local_pts = pts.copy()
        local_pts[:, 0] -= x0
        local_pts[:, 1] -= y0
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [local_pts.reshape((-1, 1, 2))], 255)
        if int(np.count_nonzero(mask)) < 16:
            raise ValueError("Freehand selection is too small.")

        border = np.concatenate((crop[0, :, :], crop[-1, :, :], crop[:, 0, :], crop[:, -1, :]), axis=0)
        bg = np.median(border.astype(np.float32), axis=0).astype(np.uint8)
        crop[mask == 0] = bg
        return crop

    raise ValueError("Unknown selection type.")


def _simplify_points(points, min_gap: int = 8):
    simplified = [points[0]]
    for point in points[1:]:
        if abs(point[0] - simplified[-1][0]) + abs(point[1] - simplified[-1][1]) >= min_gap:
            simplified.append(point)
    if len(simplified) >= 3 and simplified[-1] != simplified[0]:
        simplified.append(simplified[0])
    return simplified


# (OpenCV review functions removed)



def _infer_markup_hint(crop_bgr: np.ndarray) -> tuple[str, str]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32)
    chromatic = float(np.percentile(sat, 70)) >= 20.0
    color_hint = "chromatic" if chromatic else "low-chroma"

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    comps, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, comps)]
    if not areas:
        return color_hint, "unknown-shape"
    max_area = max(areas)
    total = max(1, crop_bgr.shape[0] * crop_bgr.shape[1])
    fill = max_area / float(total)
    shape_hint = "blob-like" if fill >= 0.28 else "complex"
    return color_hint, shape_hint


if __name__ == "__main__":
    raise SystemExit(main())
