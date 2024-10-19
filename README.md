# ObjectColorSolidRenderer

An object color solid renderer backend.

## How to Run

### Prerequisites
- `conda`

### Setup Instructions

- Create and activate a virtual environment (recommended):
```bash
conda create --name ocsr_backend python=3.11 
conda activate ocsr_backend
```
- Install Python dependencies:
```bash
pip install -r requirements.txt
```
- Run the Flask app:
```bash
python app.py
```

### Accessing the Application
- The Flask backend will be running on `http://localhost:5000`.

### Additional Information
- Ensure both servers are running simultaneously for full functionality.
- The frontend is built with React and TypeScript, using Vite as the build tool.
- The project uses Yarn for package management, but npm can be used as an alternative.
- CORS should be enabled in the Flask app to allow requests from the frontend.
