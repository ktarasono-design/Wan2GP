import gradio as gr
import threading
import time
import logging
import random
import numpy as np
from typing import Callable

from gradio_livelog import LiveLog
from gradio_livelog.utils import ProgressTracker, livelog

# --- 1. SETUP ---

def configure_logging():
    """
    Configure logging for the application with two separate loggers:
    - 'logging_app' for the LiveLog Feature Demo tab.
    - 'diffusion_app' for the Diffusion Pipeline Integration tab.
    Each logger outputs to the console with DEBUG level.
    """
      
    #logging.basicConfig(level=logging.DEBUG)

    # Logger for LiveLog Feature Demo
    app_logger = logging.getLogger("logging_app")
    app_logger.setLevel(logging.INFO)
    if not app_logger.handlers:
        console_handler = logging.StreamHandler()
        #console_handler.flush = sys.stderr.flush
        app_logger.addHandler(console_handler)

# --- 2. BUSINESS LOGIC FUNCTIONS ---
def _run_process_logic(run_error_case: bool, **kwargs):
    """
    Simulate a process with multiple steps, logging progress and status updates to the LiveLog component.
    Used in the LiveLog Feature Demo tab to demonstrate logging and progress tracking.

    Args:
        run_error_case (bool): If True, simulates an error at step 25 to test error handling.
        **kwargs: Additional arguments including:
            - tracker (ProgressTracker): Tracker for progress updates.
            - log_callback (Callable): Callback to send logs and progress to LiveLog.
            - total_steps (int): Total number of steps for the process.
            - log_name (str, optional): Logger name, defaults to 'logging_app'.

    Raises:
        RuntimeError: If run_error_case is True, raises an error at step 25.
    """
    tracker: ProgressTracker = kwargs['tracker']
    log_callback: Callable = kwargs['log_callback']
    total_steps = kwargs.get('total_steps', tracker.total)
    logger = logging.getLogger(kwargs.get('log_name', 'logging_app'))

    logger.info(f"Starting simulated process with {total_steps} steps...")
    log_callback(advance=0, log_content=f"Starting simulated process with {total_steps} steps...")
    time.sleep(0.01)
    
    logger.info("Initializing system parameters...")
    logger.info("Verifying asset integrity (check 1/3)...")
    logger.info("Verifying asset integrity (check 2/3)...")
    logger.info("Verifying asset integrity (check 3/3)...")
    logger.info("Checking for required dependencies...")
    logger.info("  - Dependency 'numpy' found.")
    logger.info("  - Dependency 'torch' found.")
    logger.info("Pre-allocating memory buffer (1024 MB)...")
    logger.info("Initialization complete. Starting main loop.")
    log_callback(log_content="Simulating a process...")
    time.sleep(0.01)

    sub_tasks = ["Reading data block...", "Applying filter algorithm...", "Normalizing values...", "Checking for anomalies..."]

    update_interval = 2  # Update every 2 steps to reduce overhead
    for i in range(total_steps):
        time.sleep(0.03)
        current_step = i + 1
        logger.info(f"--- Begin Step {current_step}/{total_steps} ---")
        for task in sub_tasks:
            logger.info(f"  - {task} (completed)")

        if current_step == 10:
            logger.warning(f"Low disk space warning at step {current_step}.")
        elif current_step == 30:
            logger.log(logging.INFO + 5, f"Asset pack loaded at step {current_step}.")
        elif current_step == 40:
            logger.critical(f"Checksum mismatch at step {current_step}.")

        logger.info(f"--- End Step {current_step}/{total_steps} ---")

        if run_error_case and current_step == 25:
            logger.error("A fatal simulation error occurred! Aborting.")
            log_callback(status="error", log_content="A fatal simulation error occurred! Aborting.")
            time.sleep(0.01)
            raise RuntimeError("A fatal simulation error occurred! Aborting.")

        if current_step % update_interval == 0 or current_step == total_steps:
            log_callback(advance=min(update_interval, total_steps - (current_step - update_interval)), log_content=f"Processing step {current_step}/{total_steps}")
            time.sleep(0.01)

    logger.log(logging.INFO + 5, "Process completed successfully!")
    log_callback(status="success", log_content="Process completed successfully!")
    time.sleep(0.01)
    logger.info("Performing final integrity check.")
    logger.info("Saving results to 'output.log'...")
    logger.info("Cleaning up temporary files...")
    logger.info("Releasing memory buffer.")
    logger.info("Disconnecting from all services.")
    logger.info("Process finished.")

# --- 4. GRADIO UI ---
with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.HTML("<h1><center>LiveLog Component Showcase</center></h1>")

    with gr.Tabs():
        with gr.TabItem("LiveLog Feature Demo"):
            """Interactive tab to test LiveLog features with customizable properties and simulated processes."""
            gr.Markdown("### Test all features of the LiveLog component interactively.")
            with gr.Row():
                with gr.Column(scale=3):
                    feature_logger = LiveLog(
                        label="Process Output",
                        line_numbers=True,
                        height=450,
                        background_color="#000000",
                        display_mode="log"
                    )
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Component Properties")
                        display_mode_radio = gr.Radio(["full", "log", "progress"], label="Display Mode", value="full")
                        rate_unit = gr.Radio(["it/s", "s/it"], label="Progress rate unit", value="it/s")
                        bg_color_picker = gr.ColorPicker(label="Background Color", value="#000000")
                        line_numbers_checkbox = gr.Checkbox(label="Show Line Numbers", value=True)
                        autoscroll_checkbox = gr.Checkbox(label="Autoscroll", value=True)
                        disable_console_checkbox = gr.Checkbox(label="Disable Python Console Output", value=False)
                    with gr.Group():
                        gr.Markdown("### Simulation Controls")
                        start_btn = gr.Button("Run Success Case", variant="primary")
                        error_btn = gr.Button("Run Error Case")

            @livelog(
                log_names=["logging_app"],
                outputs_for_yield=[feature_logger, start_btn, error_btn],
                log_output_index=0,
                # interactive_outputs_indices=[1, 2],
                result_output_index=0,
                use_tracker=True,
                tracker_mode="manual",
                tracker_total_arg_name="total_steps",
                tracker_description="Simulating a process...",
                tracker_rate_unit="it/s",
                disable_console_logs="disable_console",
                tracker_total_steps=100
            )
            def run_success_case(disable_console: bool, rate_unit: str, total_steps: int = 100, **kwargs):
                """
                Run a simulated process that completes successfully, logging progress and status to feature_logger.

                Args:
                    disable_console (bool): If True, suppress console logs.
                    rate_unit (str): Unit for progress rate ('it/s' or 's/it').
                    total_steps (int, optional): Total steps for the process. Defaults to 100.
                    **kwargs: Additional arguments passed to _run_process_logic.
                """
                kwargs["total_steps"] = total_steps
                kwargs["rate_unit"] = rate_unit
                kwargs["disable_console"] = disable_console
                kwargs["log_name"] = "logging_app"
                _run_process_logic(run_error_case=False, **kwargs)

            @livelog(
                log_names=["logging_app"],
                outputs_for_yield=[feature_logger, start_btn, error_btn],
                log_output_index=0,
                # interactive_outputs_indices=[1, 2],
                result_output_index=0,
                use_tracker=True,
                tracker_mode="manual",
                tracker_total_arg_name="total_steps",
                tracker_description="Simulating an error...",
                tracker_rate_unit="it/s",
                disable_console_logs="disable_console",
                tracker_total_steps=100
            )
            def run_error_case(disable_console: bool, rate_unit: str, total_steps: int = 100, **kwargs):
                """
                Run a simulated process that triggers an error, logging progress and error to feature_logger.

                Args:
                    disable_console (bool): If True, suppress console logs.
                    rate_unit (str): Unit for progress rate ('it/s' or 's/it').
                    total_steps (int, optional): Total steps for the process. Defaults to 100.
                    **kwargs: Additional arguments passed to _run_process_logic.
                """
                kwargs["total_steps"] = total_steps
                kwargs["rate_unit"] = rate_unit
                kwargs["disable_console"] = disable_console
                kwargs["log_name"] = "logging_app"
                _run_process_logic(run_error_case=True, **kwargs)

            start_btn.click(
                fn=run_success_case,
                inputs=[disable_console_checkbox, rate_unit],
                outputs=[feature_logger, start_btn, error_btn]
            )
            error_btn.click(
                fn=run_error_case,
                inputs=[disable_console_checkbox, rate_unit],
                outputs=[feature_logger, start_btn, error_btn]
            )
            feature_logger.clear(fn=lambda: None, outputs=[feature_logger])
            
            controls = [display_mode_radio, bg_color_picker, line_numbers_checkbox, autoscroll_checkbox]
            def update_livelog_properties(mode, color, lines, scroll):
                """Update LiveLog properties dynamically based on user input."""
                return gr.update(display_mode=mode, background_color=color, line_numbers=lines, autoscroll=scroll)
            for control in controls:
                control.change(fn=update_livelog_properties, inputs=controls, outputs=feature_logger)
    
if __name__ == "__main__":
    configure_logging()
    demo.queue(max_size=50).launch(debug=True)