import logging
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import time

PlugIn_Name = "Logs Plugin"
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
            position=10000,
        )

    def on_tab_select(self, state: dict) -> None:
        pass

    def on_tab_deselect(self, state: dict) -> None:
        pass

    def create_config_ui(self):
        gr.Markdown("### Auto-streaming logs from file (sample.log)")

        log_file = "/workspace/wan2gp_log.txt"

        with gr.Column():
            self.log_output = gr.Textbox(
                label="Process Output",
                lines=30,
                max_lines=30,
                show_copy_button=True,
                autoscroll=True,
                interactive=False,
                value="",
            )

            file_info = gr.Markdown(
                f"**Log file:** `{log_file}` - Auto-refreshing every 0.5s"
            )

        self.line_tracker = gr.Number(value=0, visible=False)

        def read_new_lines(current_line: int, existing_text: str):
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

            log_text = "\n".join(new_lines) if new_lines else ""

            # Append new lines to existing text
            if existing_text and log_text:
                combined_text = existing_text + "\n" + log_text
            elif log_text:
                combined_text = log_text
            else:
                combined_text = existing_text

            return new_line, combined_text

        try:
            self.timer = gr.Timer(interval=0.5)
            self.timer.tick(
                fn=read_new_lines,
                inputs=[self.line_tracker, self.log_output],
                outputs=[self.line_tracker, self.log_output],
            )
        except:
            timer_btn = gr.Button("Refresh Logs", size="sm", visible=True)
            timer_btn.click(
                fn=read_new_lines,
                inputs=[self.line_tracker, self.log_output],
                outputs=[self.line_tracker, self.log_output],
            )
