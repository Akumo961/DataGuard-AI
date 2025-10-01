#!/usr/bin/env python3
"""
DataGuard AI - Gradio Web Interface
"""

import gradio as gr
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import sys
from pathlib import Path

# Fix Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class DataGuardInterface:
    """Gradio interface for DataGuard AI"""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.setup_interface()

    def setup_interface(self):
        """Setup the Gradio interface"""

        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
        }
        .risk-critical { color: #dc3545 !important; font-weight: bold; font-size: 1.2em; }
        .risk-high { color: #fd7e14 !important; font-weight: bold; font-size: 1.2em; }
        .risk-medium { color: #ffc107 !important; font-weight: bold; font-size: 1.2em; }
        .risk-low { color: #28a745 !important; font-weight: bold; font-size: 1.2em; }
        .pii-item { 
            background-color: #f8f9fa; 
            border-left: 4px solid #007bff; 
            padding: 10px; 
            margin: 5px 0; 
            border-radius: 5px;
        }
        .success-box { 
            background-color: #d4edda; 
            border: 1px solid #c3e6cb; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
        }
        .error-box { 
            background-color: #f8d7da; 
            border: 1px solid #f5c6cb; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
        }
        """

        with gr.Blocks(css=css, title="DataGuard AI - Privacy Compliance Checker") as self.interface:
            # Header
            gr.HTML("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2.5em;">🛡️ DataGuard AI</h1>
                <p style="font-size: 1.2em; margin: 10px 0;">Intelligent Privacy Compliance Checker</p>
                <p style="opacity: 0.9;">Automatically detect PII and assess compliance risks in your documents</p>
            </div>
            """)

            with gr.Tabs():
                # Text Analysis Tab
                with gr.Tab("📝 Text Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            text_input = gr.Textbox(
                                label="Enter Text to Analyze",
                                placeholder="Paste your text here...\nExample: Contact John at john.doe@example.com or 555-123-4567. SSN: 123-45-6789",
                                lines=8,
                                max_lines=15
                            )
                            context_dropdown = gr.Dropdown(
                                choices=["general", "public_document", "internal_memo",
                                         "customer_data", "employee_data", "financial_data", "medical_data"],
                                value="general",
                                label="Document Context"
                            )
                            analyze_btn = gr.Button("🔍 Analyze Text", variant="primary", size="lg")

                        with gr.Column(scale=2):
                            with gr.Row():
                                risk_score = gr.HTML(label="Risk Assessment")
                            with gr.Row():
                                pii_detection = gr.HTML(label="PII Detection Results")
                            with gr.Row():
                                recommendations = gr.HTML(label="Recommendations")

                # File Analysis Tab
                with gr.Tab("📄 File Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            file_input = gr.File(
                                label="Upload Document",
                                file_types=[".txt", ".pdf", ".docx"],
                                type="filepath"
                            )
                            file_analyze_btn = gr.Button("🔍 Analyze File", variant="primary", size="lg")

                        with gr.Column(scale=2):
                            file_results = gr.HTML(label="File Analysis Results")

                # API Test Tab
                with gr.Tab("🔧 API Test"):
                    with gr.Row():
                        with gr.Column():
                            gr.HTML("<h3>API Endpoint Tester</h3>")
                            test_endpoint = gr.Dropdown(
                                choices=["/", "/health", "/test", "/analyze/text"],
                                value="/health",
                                label="Select Endpoint"
                            )
                            test_input = gr.Textbox(
                                label="Test Input (for analyze/text)",
                                placeholder='{"text": "test@example.com", "context": "general"}',
                                lines=3
                            )
                            test_btn = gr.Button("🧪 Test API", variant="secondary")
                            test_output = gr.JSON(label="API Response")

            # Event handlers
            analyze_btn.click(
                fn=self.analyze_text,
                inputs=[text_input, context_dropdown],
                outputs=[risk_score, pii_detection, recommendations]
            )

            file_analyze_btn.click(
                fn=self.analyze_file,
                inputs=[file_input],
                outputs=[file_results]
            )

            test_btn.click(
                fn=self.test_api,
                inputs=[test_endpoint, test_input],
                outputs=[test_output]
            )

    def analyze_text(self, text: str, context: str) -> tuple:
        """Analyze text for PII"""
        if not text.strip():
            return self._create_error_html("Please enter some text to analyze"), "", ""

        try:
            # Call the API
            payload = {"text": text, "context": context}
            response = requests.post(
                f"{self.api_base_url}/analyze/text",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                results = response.json()
                return (
                    self._create_risk_html(results),
                    self._create_pii_html(results),
                    self._create_recommendations_html(results)
                )
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                return self._create_error_html(error_msg), "", ""

        except requests.exceptions.ConnectionError:
            error_msg = f"❌ Cannot connect to API at {self.api_base_url}. Make sure the API server is running."
            return self._create_error_html(error_msg), "", ""
        except Exception as e:
            error_msg = f"Error analyzing text: {str(e)}"
            return self._create_error_html(error_msg), "", ""

    def analyze_file(self, file_path: str) -> str:
        """Analyze uploaded file"""
        if not file_path:
            return self._create_error_html("Please upload a file first")

        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.api_base_url}/analyze/file",
                    files=files,
                    timeout=60
                )

            if response.status_code == 200:
                results = response.json()
                return self._create_file_results_html(results)
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                return self._create_error_html(error_msg)

        except Exception as e:
            error_msg = f"Error analyzing file: {str(e)}"
            return self._create_error_html(error_msg)

    def test_api(self, endpoint: str, input_text: str) -> dict:
        """Test API endpoints"""
        try:
            if endpoint == "/analyze/text" and input_text.strip():
                # Parse JSON input for analyze/text endpoint
                try:
                    payload = json.loads(input_text)
                    response = requests.post(
                        f"{self.api_base_url}{endpoint}",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON input"}
            else:
                response = requests.get(f"{self.api_base_url}{endpoint}")

            return {
                "status_code": response.status_code,
                "endpoint": endpoint,
                "response": response.json() if response.status_code == 200 else response.text
            }

        except Exception as e:
            return {"error": str(e)}

    def _create_risk_html(self, results: Dict[str, Any]) -> str:
        """Create risk assessment HTML"""
        analysis = results.get('analysis', {})
        risk_assessment = analysis.get('risk_assessment', {})
        risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
        risk_score = risk_assessment.get('final_risk_score', 0)

        risk_class = f"risk-{risk_level.lower()}"

        return f"""
        <div class="success-box">
            <h3>📊 Risk Assessment</h3>
            <div style="display: flex; justify-content: space-around; align-items: center; text-align: center;">
                <div>
                    <div style="font-size: 2em; font-weight: bold;" class="{risk_class}">{risk_level}</div>
                    <div>Risk Level</div>
                </div>
                <div>
                    <div style="font-size: 2em; font-weight: bold;">{risk_score:.2f}</div>
                    <div>Risk Score (0-1)</div>
                </div>
            </div>
        </div>
        """

    def _create_pii_html(self, results: Dict[str, Any]) -> str:
        """Create PII detection results HTML"""
        analysis = results.get('analysis', {})
        pii_detected = analysis.get('pii_detected', {})

        if not pii_detected:
            return """
            <div class="success-box">
                <h3>🔍 PII Detection</h3>
                <p>✅ No PII detected in the text.</p>
            </div>
            """

        pii_html = "<div class='success-box'><h3>🔍 PII Detection</h3>"

        for pii_type, items in pii_detected.items():
            pii_html += f"""
            <div class="pii-item">
                <strong>{pii_type.upper()}:</strong> {len(items)} detected
                <div style="font-size: 0.9em; color: #666;">
                    {', '.join([str(item)[:50] + '...' if len(str(item)) > 50 else str(item) for item in items[:3]])}
                    {f'<br>... and {len(items) - 3} more' if len(items) > 3 else ''}
                </div>
            </div>
            """

        pii_html += f"<p><strong>Total PII items found:</strong> {sum(len(items) for items in pii_detected.values())}</p>"
        pii_html += "</div>"

        return pii_html

    def _create_recommendations_html(self, results: Dict[str, Any]) -> str:
        """Create recommendations HTML"""
        analysis = results.get('analysis', {})
        compliance_report = analysis.get('compliance_report', {})
        recommendations = compliance_report.get('recommendations', [])

        if not recommendations:
            return """
            <div class="success-box">
                <h3>💡 Recommendations</h3>
                <p>✅ No specific recommendations. The document appears to be low risk.</p>
            </div>
            """

        rec_html = "<div class='success-box'><h3>💡 Recommendations</h3><ul>"
        for rec in recommendations:
            rec_html += f"<li style='margin: 8px 0;'>{rec}</li>"
        rec_html += "</ul></div>"

        return rec_html

    def _create_file_results_html(self, results: Dict[str, Any]) -> str:
        """Create file analysis results HTML"""
        return f"""
        <div class="success-box">
            <h3>📄 File Analysis Complete</h3>
            <p><strong>Status:</strong> {results.get('status', 'Unknown')}</p>
            {self._create_risk_html(results)}
            {self._create_pii_html(results)}
            {self._create_recommendations_html(results)}
        </div>
        """

    def _create_error_html(self, message: str) -> str:
        """Create error message HTML"""
        return f"""
        <div class="error-box">
            <h3>❌ Error</h3>
            <p>{message}</p>
        </div>
        """

    def launch(self, share=False):
        """Launch the Gradio interface"""
        print("🚀 Launching DataGuard AI Web Interface...")
        print("📍 URL: http://localhost:7860")
        print("📡 API: http://localhost:8000")
        print("⏹️  Press Ctrl+C to stop the server")

        return self.interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=share,
            show_error=True
        )


def main():
    """Main function to launch the interface"""
    try:
        # Test API connection first
        print("🔌 Testing API connection...")
        try:
            response = requests.get("http://localhost:8000/", timeout=5)
            if response.status_code == 200:
                print("✅ API connection successful!")
            else:
                print("⚠️  API responded with unexpected status:", response.status_code)
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API. Please make sure the API server is running on http://localhost:8000")
            print("💡 Start the API with: python src/api/main.py")
            return

        # Launch the interface
        interface = DataGuardInterface()
        interface.launch(share=False)

    except Exception as e:
        print(f"❌ Error launching interface: {e}")


if __name__ == "__main__":
    main()