import os
from typing import Any

import cv2
from deepface import DeepFace
from ollama import chat


MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
DB_PATH = os.getenv("FACE_DB_PATH", "known_faces")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
RECOGNITION_INTERVAL = int(os.getenv("RECOGNITION_INTERVAL", "30"))


def ask_qwen(name: str) -> str:
    """Ask the local Ollama model to generate a short greeting."""
    try:
        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"Say hi to {name} in 5 words",
                }
            ],
        )

        return response["message"]["content"].strip()

    except Exception as error:
        print(f"Ollama failed: {error}")
        return f"Hi {name}"


def get_name_from_identity(identity: str) -> str:
    """Extract a clean name from a matched image path."""
    filename = os.path.basename(identity)
    name, _ = os.path.splitext(filename)
    return name


def recognize_face(frame: Any) -> str:
    """Recognize a face from the current webcam frame."""
    result = DeepFace.find(
        img_path=frame,
        db_path=DB_PATH,
        enforce_detection=False,
        silent=True,
    )

    if result and len(result[0]) > 0:
        identity = result[0].iloc[0]["identity"]
        return get_name_from_identity(identity)

    return "Unknown"


def draw_text(frame: Any, name: str, greeting: str) -> None:
    """Draw the detected name and greeting on the webcam frame."""
    cv2.putText(
        frame,
        name,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    if greeting:
        cv2.putText(
            frame,
            greeting,
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )


def main() -> None:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Missing folder: {DB_PATH}. Create it and add face images."
        )

    video = cv2.VideoCapture(CAMERA_INDEX)

    if not video.isOpened():
        raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}")

    frame_count = 0
    current_name = "Unknown"
    current_greeting = ""
    greetings_cache: dict[str, str] = {}

    print("face-id started. Press q to quit.")

    try:
        while True:
            ret, frame = video.read()

            if not ret:
                print("Failed to read from webcam.")
                break

            frame_count += 1

            if frame_count % RECOGNITION_INTERVAL == 0:
                try:
                    current_name = recognize_face(frame)

                    if current_name != "Unknown":
                        if current_name not in greetings_cache:
                            greetings_cache[current_name] = ask_qwen(current_name)

                        current_greeting = greetings_cache[current_name]
                    else:
                        current_greeting = ""

                except Exception as error:
                    print(f"Face lookup failed: {error}")
                    current_name = "Unknown"
                    current_greeting = ""

            draw_text(frame, current_name, current_greeting)

            cv2.imshow("face-id", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()