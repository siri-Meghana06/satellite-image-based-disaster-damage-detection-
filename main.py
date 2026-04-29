from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import DamageAssessor
import base64
from io import BytesIO
import traceback

app = FastAPI(title="Disaster Damage Assessment System")

# FIXED: Enable CORS so Live Server (5500) can talk to FastAPI (8001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_engine = DamageAssessor()

@app.get("/")
def home():
    return {"message": "Server is running!", "status": "Ready"}
async def analyze_damage(
    pre_disaster: UploadFile = File(...), 
    post_disaster: UploadFile = File(...)
):
    try:
        pre_bytes = await pre_disaster.read()
        post_bytes = await post_disaster.read()

        # Model Inference
        result_image, damage_stats = ai_engine.predict(pre_bytes, post_bytes)

        # Convert result to Base64
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "severity": damage_stats.get("Overall_Severity", "Unknown"),
            "details": damage_stats,
            "processed_image": f"data:image/png;base64,{img_str}"
        }
        
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Change 8001 to 8002
    uvicorn.run(app, host="127.0.0.1", port=8002)
