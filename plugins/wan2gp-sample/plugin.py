import logging
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
from typing import Callable
from gradio_livelog import LiveLog
from gradio_livelog.utils import ProgressTracker, livelog
import time

PlugIn_Name = "Logs Plugin"
PlugIn_Id ="LogsPlugin"

# def acquire_GPU(state):
#     GPU_process_running = any_GPU_process_running(state, PlugIn_Id)
#     if GPU_process_running:
#         gr.Error("Another PlugIn is using the GPU")
#     acquire_GPU_ressources(state, PlugIn_Id, PlugIn_Name, gr= gr)      

# def release_GPU(state):
#     release_GPU_ressources(state, PlugIn_Id)

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
            position=-10000
        )


    def configure_logging():
        #logging.basicConfig(level=logging.DEBUG)

        # Logger for LiveLog Feature Demo
        app_logger = logging.getLogger("logging_app")
        app_logger.setLevel(logging.INFO)
        if not app_logger.handlers:
            console_handler = logging.StreamHandler()
            #console_handler.flush = sys.stderr.flush
            app_logger.addHandler(console_handler)


    def on_tab_select(self, state: dict) -> None:
        self._run_process_logic(True)
        return

    @livelog(
        log_names=["logging_app"],
        outputs_for_yield=[],
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

    def on_tab_deselect(self, state: dict) -> None:
        pass

    def create_config_ui(self):
        # def update_prompt(state, text):
        #     settings = self.get_current_model_settings(state)
        #     settings["prompt"] = text
        #     return time.time()

        # def big_process(state):
        #     acquire_GPU(state)
        #     gr.Info("Doing something important")
        #     time.sleep(30)
        #     release_GPU(state)
        #     return "42"
        gr.Markdown("### Test all features of the LiveLog component interactively.")

        with gr.Column():
            feature_logger = LiveLog(
                label="Process Output",
                line_numbers=True,
                height=450,
                background_color="#000000",
                display_mode="full"                            
            )
            


   
