import os
import json
import pandas as pd
import joblib

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "crop_yield_model.joblib"
)

WEB_DIR = os.path.join(
    BASE_DIR,
    "web"
)


print("=" * 60)
print("             AGRIPREDICT AI")
print("          MAIN WEBSITE SERVER")
print("=" * 60)

print("\nLoading crop yield model...")

model = joblib.load(MODEL_PATH)

print("Crop yield model loaded successfully.")


class WebsiteHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):

        response = json.dumps(data).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, OPTIONS"
        )

        self.end_headers()

        self.wfile.write(response)


    def do_OPTIONS(self):

        self.send_json({
            "success": True
        })


    def do_GET(self):

        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/":
            path = "/index.html"

        file_path = os.path.join(
            WEB_DIR,
            path.lstrip("/")
        )

        file_path = os.path.abspath(file_path)

        if not file_path.startswith(
            os.path.abspath(WEB_DIR)
        ):
            self.send_error(403)
            return

        if not os.path.isfile(file_path):
            self.send_error(404)
            return

        try:

            with open(file_path, "rb") as file:
                content = file.read()

            if file_path.endswith(".html"):
                content_type = "text/html"

            elif file_path.endswith(".css"):
                content_type = "text/css"

            elif file_path.endswith(".js"):
                content_type = "application/javascript"

            elif file_path.endswith(".png"):
                content_type = "image/png"

            elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                content_type = "image/jpeg"

            elif file_path.endswith(".svg"):
                content_type = "image/svg+xml"

            else:
                content_type = "application/octet-stream"

            self.send_response(200)

            self.send_header(
                "Content-Type",
                content_type
            )

            self.send_header(
                "Content-Length",
                str(len(content))
            )

            self.end_headers()

            self.wfile.write(content)

        except Exception as error:

            print("ERROR:", error)

            self.send_error(500)


    def do_POST(self):

        if self.path != "/predict":

            self.send_json({
                "success": False,
                "error": "Invalid endpoint"
            }, 404)

            return

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            body = self.rfile.read(
                content_length
            )

            data = json.loads(
                body.decode("utf-8")
            )

            # SHOW EXACT DATA RECEIVED FROM BROWSER
            print("BROWSER DATA:", data)


            crop = data.get("crop")

            rainfall = float(
                data.get(
                    "rainfall_mm",
                    data.get("rainfall")
                )
            )

            temperature = float(
                data.get(
                    "temperature_c",
                    data.get("temperature")
                )
            )

            soil_ph = float(
                data.get(
                    "soil_ph",
                    data.get("soilPh")
                )
            )

            fertilizer = float(
                data.get(
                    "fertilizer_kg",
                    data.get("fertilizer")
                )
            )

            area = float(
                data.get(
                    "area_hectare",
                    data.get("area")
                )
            )


            input_data = pd.DataFrame([
                {
                    "crop": crop,
                    "rainfall_mm": rainfall,
                    "temperature_c": temperature,
                    "soil_ph": soil_ph,
                    "fertilizer_kg": fertilizer,
                    "area_hectare": area
                }
            ])


            prediction = model.predict(
                input_data
            )[0]

            prediction = round(
                float(prediction),
                2
            )


            print(
                f"Prediction: {prediction} tons"
            )


            self.send_json({
                "success": True,
                "prediction": prediction,
                "predicted_yield": prediction,
                "unit": "tons"
            })


        except Exception as error:

            print(
                "Prediction ERROR:",
                error
            )

            self.send_json({
                "success": False,
                "error": str(error)
            }, 500)


HOST = "127.0.0.1"
PORT = 8000


server = HTTPServer(
    (HOST, PORT),
    WebsiteHandler
)


print("\n" + "=" * 60)
print("Website server is running")
print("=" * 60)

print("\nWebsite:")
print(
    f"http://{HOST}:{PORT}"
)

print("\nWaiting for requests...")

server.serve_forever()