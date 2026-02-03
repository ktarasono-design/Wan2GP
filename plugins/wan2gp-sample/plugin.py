import logging
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
from gradio_livelog import LiveLog
from gradio_livelog.utils import livelog
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
        tracker_file = ".line_tracker.txt"

        with gr.Column():
            self.feature_logger = LiveLog(
                label="Process Output",
                line_numbers=True,
                height=450,
                background_color="#000000",
                display_mode="log",
            )

            file_info = gr.Markdown(
                f"**Log file:** `{log_file}` - Auto-refreshing every 0.5s"
            )

        @livelog(
            log_names=["logging_app"],
            outputs_for_yield=[self.feature_logger],
            log_output_index=0,
            result_output_index=0,
            use_tracker=False,
        )
        def read_new_lines(**kwargs):
            log_callback = kwargs["log_callback"]
            import os

            # Read current line from file
            current_line = 0
            if os.path.exists(tracker_file):
                try:
                    with open(tracker_file, "r") as tf:
                        current_line = int(tf.read())
                except:
                    pass

            # Read new lines from log file
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                    if current_line < len(lines):
                        for i in range(current_line, len(lines)):
                            line = lines[i].strip()
                            if line:
                                app_logger.info(line)
                                log_callback(log_content=line)
                                current_line = i + 1
                                time.sleep(0.01)

                        # Save new line position
                        with open(tracker_file, "w") as tf:
                            tf.write(str(current_line))

        try:
            self.timer = gr.Timer(interval=0.5)
            self.timer.tick(
                fn=read_new_lines,
                outputs=[self.feature_logger],
            )
        except:
            timer_btn = gr.Button("Refresh Logs", size="sm", visible=True)
            timer_btn.click(
                fn=read_new_lines,
                outputs=[self.feature_logger],
            )
