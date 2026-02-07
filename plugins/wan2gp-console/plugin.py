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

        log_file = "/app/Wan2GP/wan2gp_log.txt"

        # Store for new lines data - used by JavaScript
        self.line_tracker = gr.State(value=0)
        self.log_buffer = gr.State(value=[])

        gr.HTML(f"""
        <style>
            .terminal-container {{
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
            }}
            .log-line {{
                display: block;
                margin: 2px 0;
                line-height: 1.4;
            }}
            .log-error {{ color: #f48771; }}
            .log-warn {{ color: #cca700; }}
            .log-info {{ color: #4fc1ff; }}
            .log-success {{ color: #4ec9b0; }}
            .log-debug {{ color: #808080; }}
            .log-timestamp {{ color: #6a9955; }}
        </style>
        <div id="console-terminal" class="terminal-container">Waiting for logs...</div>
        <script>
            (function() {{
                let lastLineCount = 0;
                const logFile = "{log_file}";
                
                function escapeHtml(text) {{
                    return text
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;")
                        .replace(/"/g, "&quot;")
                        .replace(/'/g, "&#x27;");
                }}
                
                function highlightLogLine(line) {{
                    let highlighted = escapeHtml(line);
                    
                    if (/\\b(ERROR|FATAL|CRITICAL)\\b/i.test(highlighted)) {{
                        highlighted = '<span class="log-error">' + highlighted + '</span>';
                    }} else if (/\\b(WARN|WARNING)\\b/i.test(highlighted)) {{
                        highlighted = '<span class="log-warn">' + highlighted + '</span>';
                    }} else if (/\\b(INFO|INFORMATION)\\b/i.test(highlighted)) {{
                        highlighted = '<span class="log-info">' + highlighted + '</span>';
                    }} else if (/\\b(DEBUG)\\b/i.test(highlighted)) {{
                        highlighted = '<span class="log-debug">' + highlighted + '</span>';
                    }} else if (/\\b(SUCCESS|COMPLETED|DONE)\\b/i.test(highlighted)) {{
                        highlighted = '<span class="log-success">' + highlighted + '</span>';
                    }}
                    
                    const timestampMatch = highlighted.match(/(\\d{{4}}-\\d{{2}}-\\d{{2}}[\\sT]\\d{{2}}:\\d{{2}}:\\d{{2}})/);
                    if (timestampMatch) {{
                        highlighted = highlighted.replace(
                            timestampMatch[1],
                            '<span class="log-timestamp">' + timestampMatch[1] + '</span>'
                        );
                    }}
                    
                    return highlighted;
                }}
                
                async function fetchLogs() {{
                    try {{
                        const response = await fetch('/file=' + logFile);
                        if (!response.ok) return;
                        
                        const text = await response.text();
                        const lines = text.split('\\n').filter(line => line.trim());
                        
                        if (lines.length === 0 || lines.length <= lastLineCount) return;
                        
                        const terminal = document.getElementById('console-terminal');
                        if (!terminal) return;
                        
                        if (terminal.textContent === 'Waiting for logs...') {{
                            terminal.innerHTML = '';
                        }}
                        
                        const wasNearBottom = terminal.scrollHeight - terminal.scrollTop - terminal.clientHeight < 50;
                        
                        for (let i = lastLineCount; i < lines.length; i++) {{
                            const div = document.createElement('div');
                            div.className = 'log-line';
                            div.innerHTML = highlightLogLine(lines[i]);
                            terminal.appendChild(div);
                        }}
                        
                        lastLineCount = lines.length;
                        
                        // Keep only last 500 lines to prevent memory issues
                        while (terminal.children.length > 500) {{
                            terminal.removeChild(terminal.firstChild);
                            lastLineCount--;
                        }}
                        
                        if (wasNearBottom) {{
                            terminal.scrollTop = terminal.scrollHeight;
                        }}
                    }} catch (e) {{
                        // Silently fail - file might not exist yet
                    }}
                }}
                
                // Fetch immediately and then every 2 seconds
                setTimeout(fetchLogs, 500);
                setInterval(fetchLogs, 2000);
            }})();
        </script>
        """)

        file_info = gr.Markdown(
            f"**Log file:** `{log_file}` - Auto-refreshing every 2s"
        )
