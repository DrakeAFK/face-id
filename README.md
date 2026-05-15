# face-id

simple webcam face identification

uses deepface to match faces from `known_faces/` and ollama/qwen to generate a short greeting 

## setup
**run**
```bash
pip install -r requirements.txt
ollama pull qwen3.5:9b
```

**add face images:**
```
known_faces/
  donald_knuth.png
```

**filename becomes the displayed name:**
`donald_knuth.png -> donald_knuth`

**run**
`python app.py`

press q to quit 

## notes
recognition runs every 30 frames
qwen greetings are cached per person
known_faces/ is ignored except for .gitkeep
structure:
```
face-id/
  app.py
  requirements.txt
  README.md
  LICENSE
  .gitignore
  known_faces/
    .gitkeep
```