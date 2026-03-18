from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from tkinter import END, BOTH, LEFT, RIGHT, Button, Entry, Frame, Label, StringVar, Tk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from app.detector import DetectionConfig, detect_document


class PatternDetectionApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Pattern Markup Counter")
        self.root.geometry("860x580")

        self.input_var = StringVar()
        self.template_var = StringVar()
        self.output_var = StringVar(value=str((Path.cwd() / "outputs").resolve()))

        self.threshold_var = StringVar(value="0.50")
        self.dpi_var = StringVar(value="220")
        self.min_scale_var = StringVar(value="0.35")
        self.max_scale_var = StringVar(value="8.0")
        self.dark_threshold_var = StringVar(value="0")

        self._build_ui()

    def _build_ui(self) -> None:
        main = Frame(self.root, padx=12, pady=12)
        main.pack(fill=BOTH, expand=True)

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
            button_text="Browse",
        )

        Label(main, text="Detection settings").pack(anchor="w", pady=(10, 4))
        settings = Frame(main)
        settings.pack(fill="x")

        self._add_setting(settings, "Match threshold (0.45-0.9):", self.threshold_var, row=0, col=0)
        self._add_setting(settings, "PDF render DPI:", self.dpi_var, row=0, col=1)
        self._add_setting(settings, "Min scale:", self.min_scale_var, row=1, col=0)
        self._add_setting(settings, "Max scale:", self.max_scale_var, row=1, col=1)
        self._add_setting(
            settings,
            "Dark threshold (0=auto):",
            self.dark_threshold_var,
            row=2,
            col=0,
        )

        action = Frame(main, pady=12)
        action.pack(fill="x")
        Button(action, text="Run Detection", command=self._run_detection, height=2).pack(side=LEFT)
        Button(action, text="Clear Log", command=self._clear_log).pack(side=LEFT, padx=(8, 0))

        Label(main, text="Run log").pack(anchor="w", pady=(2, 4))
        self.log = ScrolledText(main, height=15)
        self.log.pack(fill=BOTH, expand=True)
        self._append_log("Ready. Pick input file + template image, then click Run Detection.")

    def _add_picker_row(
        self,
        parent: Frame,
        label: str,
        variable: StringVar,
        browse_handler,
        button_text: str = "Select",
    ) -> None:
        row = Frame(parent, pady=4)
        row.pack(fill="x")
        Label(row, text=label, width=34, anchor="w").pack(side=LEFT)
        Entry(row, textvariable=variable).pack(side=LEFT, fill="x", expand=True, padx=(0, 8))
        Button(row, text=button_text, command=browse_handler).pack(side=RIGHT)

    def _add_setting(self, parent: Frame, text: str, variable: StringVar, row: int, col: int) -> None:
        holder = Frame(parent)
        holder.grid(row=row, column=col, padx=6, pady=4, sticky="ew")
        Label(holder, text=text, width=22, anchor="w").pack(side=LEFT)
        Entry(holder, textvariable=variable, width=12).pack(side=LEFT)
        parent.grid_columnconfigure(col, weight=1)

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

            config = DetectionConfig(
                dpi=int(self.dpi_var.get()),
                match_threshold=float(self.threshold_var.get()),
                min_scale=float(self.min_scale_var.get()),
                max_scale=float(self.max_scale_var.get()),
                dark_threshold=int(self.dark_threshold_var.get()),
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

    def _clear_log(self) -> None:
        self.log.delete("1.0", END)

    def _append_log(self, text: str) -> None:
        self.log.insert(END, text + "\n")
        self.log.see(END)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pattern markup counter")
    parser.add_argument("--input", type=Path, help="Input file (.png/.jpg/.jpeg/.pdf)")
    parser.add_argument("--template", type=Path, help="Template image of the markup symbol")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output folder")
    parser.add_argument("--threshold", type=float, default=0.50, help="Match threshold")
    parser.add_argument("--dpi", type=int, default=220, help="PDF rendering DPI")
    parser.add_argument("--min-scale", type=float, default=0.35, help="Minimum template scale")
    parser.add_argument("--max-scale", type=float, default=8.0, help="Maximum template scale")
    parser.add_argument("--dark-threshold", type=int, default=0, help="Dark pixel threshold (0=auto)")
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

    root = Tk()
    PatternDetectionApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
