import logging
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import time
import json
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
            # Initial HTML with terminal container
            gr.HTML("""
            <style>
                .terminal-container {
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
                    scroll-behavior: smooth;
                }
                .log-line {
                    display: block;
                    margin: 2px 0;
                    line-height: 1.4;
                }
                .log-error { color: #f48771; }
                .log-warn { color: #cca700; }
                .log-info { color: #4fc1ff; }
                .log-success { color: #4ec9b0; }
                .log-debug { color: #808080; }
                .log-timestamp { color: #6a9955; }
            </style>
            <div id="console-terminal" class="terminal-container">Waiting for logs...</div>
            <script>
                (function() {
                    window.consoleLines = [];
                    let lastUpdateId = 0;
                    
                    function escapeHtml(text) {
                        return text
                            .replace(/&/g, "&amp;")
                            .replace(/</g, "&lt;")
                            .replace(/>/g, "&gt;")
                            .replace(/"/g, "&quot;")
                            .replace(/'/g, "&#x27;");
                    }
                    
                    function highlightLogLine(line) {
                        let highlighted = escapeHtml(line);
                        
                        if (/\\b(ERROR|FATAL|CRITICAL)\\b/i.test(highlighted)) {
                            highlighted = '<span class="log-error">' + highlighted + '</span>';
                        } else if (/\\b(WARN|WARNING)\\b/i.test(highlighted)) {
                            highlighted = '<span class="log-warn">' + highlighted + '</span>';
                        } else if (/\\b(INFO|INFORMATION)\\b/i.test(highlighted)) {
                            highlighted = '<span class="log-info">' + highlighted + '</span>';
                        } else if (/\\b(DEBUG)\\b/i.test(highlighted)) {
                            highlighted = '<span class="log-debug">' + highlighted + '</span>';
                        } else if (/\\b(SUCCESS|COMPLETED|DONE)\\b/i.test(highlighted)) {
                            highlighted = '<span class="log-success">' + highlighted + '</span>';
                        }
                        
                        const timestampMatch = highlighted.match(/(\\d{4}-\\d{2}-\\d{2}[\\sT]\\d{2}:\\d{2}:\\d{2})/);
                        if (timestampMatch) {
                            highlighted = highlighted.replace(
                                timestampMatch[1],
                                '<span class="log-timestamp">' + timestampMatch[1] + '</span>'
                            );
                        }
                        
                        return highlighted;
                    }
                    
                    // Function called by Gradio when new data arrives
                    window.updateConsole = function(jsonData) {
                        try {
                            const data = JSON.parse(jsonData);
                            if (!data || data.update_id <= lastUpdateId) return;
                            
                            lastUpdateId = data.update_id;
                            const terminal = document.getElementById('console-terminal');
                            if (!terminal) return;
                            
                            if (terminal.textContent === 'Waiting for logs...') {
                                terminal.innerHTML = '';
                            }
                            
                            const wasNearBottom = terminal.scrollHeight - terminal.scrollTop - terminal.clientHeight < 50;
                            
                            // Add new lines
                            data.lines.forEach(function(line) {
                                window.consoleLines.push(line);
                                const div = document.createElement('div');
                                div.className = 'log-line';
                                div.innerHTML = highlightLogLine(line);
                                terminal.appendChild(div);
                            });
                            
                            // Keep only last 500 lines
                            while (window.consoleLines.length > 500) {
                                window.consoleLines.shift();
                                if (terminal.firstChild) {
                                    terminal.removeChild(terminal.firstChild);
                                }
                            }
                            
                            if (wasNearBottom && data.lines.length > 0) {
                                terminal.scrollTop = terminal.scrollHeight;
                            }
                        } catch (e) {
                            console.error('Console update error:', e);
                        }
                    };
                    
                    // Also expose function to load initial content
                    window.loadInitialLogs = function(lines) {
                        const terminal = document.getElementById('console-terminal');
                        if (!terminal || lines.length === 0) return;
                        
                        terminal.innerHTML = '';
                        window.consoleLines = lines;
                        
                        lines.forEach(function(line) {
                            const div = document.createElement('div');
                            div.className = 'log-line';
                            div.innerHTML = highlightLogLine(line);
                            terminal.appendChild(div);
                        });
                        
                        terminal.scrollTop = terminal.scrollHeight;
                    };
                })();
            </script>
            """)
            
            # Hidden component to trigger updates
            self.update_trigger = gr.HTML(value="", visible=False)
            
            file_info = gr.Markdown(
                f"**Log file:** `{log_file}` - Auto-refreshing every 2s"
            )

        # Hidden state to track line position
        self.line_tracker = gr.State(value=0)

        def read_new_lines(current_line: int):
            new_line = current_line
            new_lines = []
            
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if current_line < len(lines):
                            for i in range(current_line, len(lines)):
                                line = lines[i].rstrip("\n")
                                if line:
                                    new_lines.append(line)
                                    new_line = i + 1
                except Exception:
                    pass

            # Create update payload
            update_id = int(time.time() * 1000)
            payload = json.dumps({"update_id": update_id, "lines": new_lines})
            
            # Create HTML that calls the JavaScript function
            if new_lines:
                trigger_html = f'<script>window.updateConsole({json.dumps(payload)});</script>'
            else:
                trigger_html = ""
            
            return new_line, trigger_html

        # Auto-refresh every 2 seconds
        self.timer = gr.Timer(2.0)
        self.timer.tick(
            fn=read_new_lines,
            inputs=[self.line_tracker],
            outputs=[self.line_tracker, self.update_trigger],
        )
