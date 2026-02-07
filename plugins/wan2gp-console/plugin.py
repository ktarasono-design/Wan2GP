import logging
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import time
import json

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
                let lastProcessedId = 0;
                
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
                
                function processNewLines() {
                    const jsonInput = document.getElementById('console-data');
                    if (!jsonInput) return;
                    
                    // Get text content, skip if it's a queue status message
                    const jsonText = jsonInput.value || jsonInput.textContent || jsonInput.innerText || '';
                    if (!jsonText || jsonText.startsWith('queue:') || jsonText.startsWith('progress:')) return;
                    
                    try {
                        const data = JSON.parse(jsonText);
                        if (!data || !data.lines || data.id <= lastProcessedId) return;
                        
                        lastProcessedId = data.id;
                        const terminal = document.getElementById('console-terminal');
                        if (!terminal) return;
                        
                        if (terminal.textContent === 'Waiting for logs...') {
                            terminal.innerHTML = '';
                        }
                        
                        const wasNearBottom = terminal.scrollHeight - terminal.scrollTop - terminal.clientHeight < 50;
                        
                        data.lines.forEach(function(line) {
                            const div = document.createElement('div');
                            div.className = 'log-line';
                            div.innerHTML = highlightLogLine(line);
                            terminal.appendChild(div);
                        });
                        
                        // Keep only last 500 lines to prevent memory issues
                        while (terminal.children.length > 500) {
                            terminal.removeChild(terminal.firstChild);
                        }
                        
                        if (wasNearBottom) {
                            terminal.scrollTop = terminal.scrollHeight;
                        }
                    } catch (e) {
                        // Silently ignore parse errors from queue status messages
                    }
                }
                
                // Check for new lines every 500ms
                setInterval(processNewLines, 500);
                
                // Also scroll to bottom on load
                setTimeout(function() {
                    const terminal = document.getElementById('console-terminal');
                    if (terminal) terminal.scrollTop = terminal.scrollHeight;
                }, 500);
            })();
        </script>
        """)

        with gr.Column():
            # Hidden Text component to pass new lines to JavaScript as JSON string
            self.new_lines_json = gr.Text(
                value='{"id": 0, "lines": []}',
                visible=False,
                elem_id="console-data"
            )

            file_info = gr.Markdown(
                f"**Log file:** `{log_file}` - Auto-refreshing every 2s"
            )

        self.line_tracker = gr.Number(value=0, visible=False)

        def read_new_lines(current_line: int):
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

            # Return update ID and new lines as JSON string
            update_id = int(time.time() * 1000) if new_lines else current_line
            return new_line, json.dumps({"id": update_id, "lines": new_lines})

        # Auto-refresh every 2 seconds using Timer
        self.timer = gr.Timer(2.0)
        self.timer.tick(
            fn=read_new_lines,
            inputs=[self.line_tracker],
            outputs=[self.line_tracker, self.new_lines_json],
        )
