```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    Start([Start]):::first
    Analysis([Analysis]):::analysis
    emotion_face_image_classification_v3([emotion_face_image_classification_v3]):::recommended
    Start --> Analysis;
    Analysis -->|recommends| emotion_face_image_classification_v3;
    classDef first fill:#ffdfba,stroke:#ff9a00,color:black
    classDef analysis fill:#bae1ff,stroke:#0077ff,color:black
    classDef recommended fill:#baffc9,stroke:#00b050,color:black,stroke-width:2px
    classDef model fill:#f2f0ff,stroke:#9c88ff,color:black

```