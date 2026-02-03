import logging
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import time

PlugIn_Name = "Console"
PlugIn_Id = "LogsPlugin"

app_logger = logging.getLogger("logging_app")
app_logger.setLevel(logging.INFO)
if not app_logger.handlers:
    console_handler = logging.StreamHandler()
    app_logger.addHandler(console_handler)


class ConfigTabPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()

    def setup_ui(self):
        self.request_global("get_current_model_settings")
        self.request_component("refresh_form_trigger")
        self.request_component("state")
        self.request_component("resolution")
        self.request_component("main_tabs")

        self.add_tab(
            tab_id=PlugIn_Id,
            label=PlugIn_Name,
            component_constructor=self.create_config_ui,
        )

    def on_tab_select(self, state: dict) -> None:
        pass

    def on_tab_deselect(self, state: dict) -> None:
        pass

    def create_config_ui(self):
        gr.Markdown("### Auto-streaming logs from file")

        log_file = "/workspace/wan2gp_log.txt"

        gr.HTML("""
        <style>
            .terminal-output {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 13px;
                padding: 15px;
                border-radius: 5px;
                overflow-y: auto;
                max-height: 500px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .log-error { color: #f48771; }
            .log-warn { color: #cca700; }
            .log-info { color: #4fc1ff; }
            .log-success { color: #4ec9b0; }
            .log-debug { color: #808080; }
            .log-timestamp { color: #6a9955; }
        </style>
        """)

        with gr.Column():
            self.log_output = gr.HTML(
                value='<div class="terminal-output">Waiting for logs...</div>',
                label="Terminal Output",
            )

            file_info = gr.Markdown(
                f"**Log file:** `{log_file}` - Auto-refreshing every 2s"
            )

        self.line_tracker = gr.Number(value=0, visible=False)
        self.tick_counter = gr.Number(value=0, visible=False)

        def escape_html(text):
            result = (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#x27;")
            )
            return result

        def highlight_log_line(line):
            import re

            escaped_line = escape_html(line)
            highlighted = escaped_line

            # Highlight log levels
            if re.search(r"\b(ERROR|FATAL|CRITICAL)\b", highlighted, re.IGNORECASE):
                highlighted = f'<span class="log-error">{highlighted}</span>'
            elif re.search(r"\b(WARN|WARNING)\b", highlighted, re.IGNORECASE):
                highlighted = f'<span class="log-warn">{highlighted}</span>'
            elif re.search(r"\b(INFO|INFORMATION)\b", highlighted, re.IGNORECASE):
                highlighted = f'<span class="log-info">{highlighted}</span>'
            elif re.search(r"\b(DEBUG)\b", highlighted, re.IGNORECASE):
                highlighted = f'<span class="log-debug">{highlighted}</span>'
            elif re.search(r"\b(SUCCESS|COMPLETED|DONE)\b", highlighted, re.IGNORECASE):
                highlighted = f'<span class="log-success">{highlighted}</span>'

            # Highlight timestamps
            timestamp_match = re.search(
                r"(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})", highlighted
            )
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                highlighted = highlighted.replace(
                    timestamp, f'<span class="log-timestamp">{timestamp}</span>'
                )

            return highlighted

        def read_new_lines(current_line: int, existing_html: str):
            new_line = current_line
            new_lines = []
            import os

            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if current_line < len(lines):
                        for i in range(current_line, len(lines)):
                            line = lines[i].rstrip("\n")
                            if line:
                                new_lines.append(line)
                                new_line = i + 1

            # Extract existing log content from HTML
            existing_content = ""
            if existing_html:
                import re

                match = re.search(
                    r'<div class="terminal-output">(.*?)</div>',
                    existing_html,
                    re.DOTALL,
                )
                if match:
                    existing_content = match.group(1)
                elif (
                    existing_html
                    != '<div class="terminal-output">Waiting for logs...</div>'
                ):
                    existing_content = existing_html

            # Highlight and append new lines
            for line in new_lines:
                highlighted = highlight_log_line(line)
                if existing_content:
                    existing_content += "\n" + highlighted
                else:
                    existing_content = highlighted

            # Wrap in terminal div
            combined_html = f'<div class="terminal-output">{existing_content if existing_content else "Waiting for logs..."}</div>'

            return new_line, combined_html

        # Auto-refresh every 0.5 seconds using Timer (requires Gradio 4.0+)
        self.timer = gr.Timer(2.0)
        self.timer.tick(
            fn=read_new_lines,
            inputs=[self.line_tracker, self.log_output],
            outputs=[self.line_tracker, self.log_output],
        )
