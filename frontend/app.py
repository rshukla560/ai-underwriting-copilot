import gradio as gr
import requests

RAILWAY_URL = "https://ai-underwriting-copilot-production.up.railway.app"


def analyze(pdf_file, applicant_id):
    if pdf_file is None:
        return "Please upload a PDF."

    applicant_id = applicant_id.strip() or "DEMO_001"

    with open(pdf_file, "rb") as f:
        response = requests.post(
            f"{RAILWAY_URL}/api/v1/analyze/{applicant_id}",
            files={"file": f},
            timeout=60
        )

    if response.status_code != 200:
        return f"Error {response.status_code}: {response.text}"

    return response.json()


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.File(label="Upload Application PDF", file_types=[".pdf"]),
        gr.Textbox(label="Applicant ID", value="DEMO_001"],
    outputs=gr.JSON(label="Result"),
    title="AI Underwriting Copilot",
    description="Upload an insurance application PDF for AI-powered risk assessment. No PDF? Download sample: https://github.com/rshukla560/ai-underwriting-copilot/raw/master/tests/fixtures/sample_application.pdf"
)

if __name__ == "__main__":
    demo.launch()
