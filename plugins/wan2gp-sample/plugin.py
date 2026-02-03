import logging
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
from gradio_livelog import LiveLog
from gradio_livelog.utils import ProgressTracker, livelog
import time

PlugIn_Name = "Logs Plugin"
PlugIn_Id = "LogsPlugin"

app_logger = logging.getLogger("logging_app")
app_logger.setLevel(logging.INFO)
if not app_logger.handlers:
    console_handler = logging.StreamHandler()
    app_logger.addHandler(console_handler)


def read_file_lines(log_file: str, **kwargs):
    tracker: ProgressTracker = kwargs["tracker"]
    log_callback = kwargs["log_callback"]
    logger = logging.getLogger(kwargs.get("log_name", "logging_app"))

    import os

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

            tracker.total = len(lines)
            logger.info(f"Found {len(lines)} lines in log file")
            log_callback(
                advance=0, log_content=f"Reading {len(lines)} lines from {log_file}..."
            )
            time.sleep(0.1)

            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    logger.info(f"Line {i + 1}: {line[:50]}...")
                    log_callback(advance=1, log_content=line)
                    time.sleep(0.05)
    else:
        logger.error(f"Log file '{log_file}' not found")
        log_callback(status="error", log_content=f"Log file '{log_file}' not found")
        return

    logger.info("Finished reading log file")
    log_callback(status="success", log_content="Finished reading log file")


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
        gr.Markdown("### Read logs from file")

        log_file = "/workspace/wan2gp_log.txt"

        with gr.Row():
            self.log_file_input = gr.Textbox(
                label="Log file path", value=log_file, scale=2
            )
            self.start_btn = gr.Button("▶ Load Log File", variant="primary", scale=1)

        with gr.Column():
            self.feature_logger = LiveLog(
                label="Process Output",
                line_numbers=True,
                height=450,
                background_color="#000000",
                display_mode="log",
            )

        @livelog(
            log_names=["logging_app"],
            outputs_for_yield=[self.feature_logger, self.start_btn],
            log_output_index=0,
            result_output_index=0,
            use_tracker=True,
            tracker_mode="auto",
            tracker_description="Reading log file...",
            tracker_rate_unit="lines/s",
            disable_console_logs="disable_console",
        )
        def load_log_file(disable_console: bool, log_file_path: str, **kwargs):
            kwargs["disable_console"] = disable_console
            kwargs["log_name"] = "logging_app"
            read_file_lines(log_file_path, **kwargs)

        self.start_btn.click(
            fn=load_log_file,
            inputs=[gr.State(False), self.log_file_input],
            outputs=[self.feature_logger, self.start_btn],
        )
