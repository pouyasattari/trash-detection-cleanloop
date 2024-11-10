



# CleanLoop Project
### Trash Detection Feature with Cleaning Time Estimation

CleanLoop is a project focused on detecting trash in images and estimating the cleaning time required for the identified areas. This feature leverages **YOLOv8**, along with **Cloudinary** and **Imagga** APIs, to process, recognize, and tag trash in sample images, offering a visual representation of detected areas alongside an estimation of cleaning time.

This project was developed at the **SFSCON Hackathon (Nov 2024),  Bolzano, Italy 🇮🇹**. Learn more about the event [here](https://hackathon.bz.it/).

---

## Features

- **Trash Detection**: Identifies and tags trash in images using the YOLOv8 model, optimized for real-time object detection.
- **Cleaning Time Estimation**: Calculates an estimated cleaning time based on detected trash areas.
- **Cloud Storage and Management**: Utilizes Cloudinary for image hosting and management.

---

## Sample Images

**Original Image**  
<img width="700" alt="Original" src="https://github.com/user-attachments/assets/21443585-a4b9-4455-a7fe-befac64b4325">

**Tagged Trashes**  
<img width="700" alt="Detected" src="https://github.com/user-attachments/assets/dda625aa-d95b-4e04-86f6-2e44fd96ed50">

---

## Estimation Algorithm

The project uses an algorithm designed to estimate cleaning time by analyzing the tagged trash areas within the image.

<img width="328" alt="Screenshot 2024-11-09 at 05 51 01" src="https://github.com/user-attachments/assets/34f31bfa-b059-4aad-8fe4-7e4b93773c48">

---

## Model and APIs Used

- **Model**: [YOLOv8](https://github.com/ultralytics/ultralytics) - An object detection model, optimized for identifying trash in real-time.
- **[Cloudinary](https://cloudinary.com/)**: For image storage, transformation, and optimization.
- **[Imagga](https://imagga.com/)**: For image tagging and analysis to enhance trash detection.

---

## Getting Started

To run this feature locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/pouyasattari/trash-detection-cleanloop-project.git
   cd trash-detection-cleanloop-project/trash-detection

