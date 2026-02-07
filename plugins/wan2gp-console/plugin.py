import logging
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import time
import os

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

        log_file = "/app/Wan2GP/wan2gp_log.txt"

        with gr.Column():
            # TextArea for logs - no flicker, native scroll
            self.log_output = gr.TextArea(
                value="Waiting for logs...",
                label="Terminal Output",
                lines=25,
                max_lines=25,
                interactive=False,
                autoscroll=True,
            )
            
            file_info = gr.Markdown(
                f"**Log file:** `{log_file}` - Auto-refreshing every 2s"
            )

        # Track which lines we've read
        self.line_tracker = gr.State(value=0)

        def read_logs(current_line: int):
            """Read new lines from log file."""
            import html as html_module
            import re
            
            if not os.path.exists(log_file):
                return current_line, "Waiting for logs..."
            
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    all_lines = f.readlines()
            except Exception:
                return current_line, "Waiting for logs..."
            
            # Get new lines
            new_lines = []
            new_line_count = current_line
            
            if current_line < len(all_lines):
                for i in range(current_line, len(all_lines)):
                    line = all_lines[i].rstrip("\n\r")
                    if line:
                        new_lines.append(line)
                        new_line_count = i + 1
            
            # Build display text (last 500 lines for performance)
            display_lines = all_lines[-500:] if len(all_lines) > 500 else all_lines
            
            # Format with simple highlighting markers
            output_lines = []
            for line in display_lines:
                line = line.rstrip("\n\r")
                if not line:
                    continue
                    
                # Simple visual markers for log levels
                upper_line = line.upper()
                if any(x in upper_line for x in ["ERROR", "FATAL", "CRITICAL"]):
                    line = "[ERR] " + line
                elif any(x in upper_line for x in ["WARN"]):
                    line = "[WRN] " + line
                elif any(x in upper_line for x in ["INFO"]):
                    line = "[INF] " + line
                elif any(x in upper_line for x in ["DEBUG"]):
                    line = "[DBG] " + line
                elif any(x in upper_line for x in ["SUCCESS", "COMPLETED", "DONE"]):
                    line = "[OK]  " + line
                    
                output_lines.append(line)
            
            text = "\n".join(output_lines) if output_lines else "Waiting for logs..."
            
            return new_line_count, text

        # Auto-refresh every 2 seconds
        self.timer = gr.Timer(2.0)
        self.timer.tick(
            fn=read_logs,
            inputs=[self.line_tracker],
            outputs=[self.line_tracker, self.log_output],
        )
