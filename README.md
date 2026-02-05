# VisDrone Detection Segmentation

## Project Structure
The VisDrone Detection Segmentation project is structured to support clear organization and easy navigation. Below is an overview of the folder structure:

```
visdrone-detection-segmentation/
├── data/                 # Contains dataset files
├── models/               # Contains trained model files
├── notebooks/            # Jupyter notebooks for experimentation
├── src/                 # Source code for the project
│   ├── preprocessing/    # Scripts for data preprocessing
│   ├── training/        # Scripts for model training
│   ├── evaluation/      # Scripts for model evaluation
│   ├── utils/           # Utility functions
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
└── config.yaml           # Configuration file
```

## Dataset Requirements
The VisDrone Detection Segmentation project requires specific datasets to function effectively. For this project, the following dataset is necessary:
- **VisDrone Dataset**: Download the dataset from [VisDrone](https://visdrone.net/dataset.html)

Make sure to place the extracted dataset in the `data/` directory.

## Models Used
This project utilizes various models for detection and segmentation tasks. Key models include:
- **YOLOv5**: For object detection tasks
- **U-Net**: For image segmentation tasks
- **Faster R-CNN**: Alternative object detection model that can be used based on requirement.

## How to Run the Pipeline
Follow the instructions below to run the full pipeline:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Runa-Rameena/visdrone-detection-segmentation.git
   cd visdrone-detection-segmentation
   ```

2. **Install required packages**:
   Make sure to create a virtual environment for the project and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Preprocessing**:
   Execute the preprocessing scripts to prepare the data:
   ```bash
   python src/preprocessing/preprocess_data.py
   ```

4. **Train the model**:
   Train the desired model using the provided scripts:
   ```bash
   python src/training/train_yolov5.py
   python src/training/train_unet.py
   ```

5. **Evaluate the model**:
   After training, evaluate the model's performance:
   ```bash
   python src/evaluation/evaluate_model.py
   ```

6. **Run Inference**:
   Use the trained model to run inference on new images:
   ```bash
   python src/inference/run_inference.py --image_path path/to/test/image.jpg
   ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the creators of the VisDrone dataset.
- Special thanks to the ML community for inspiration and support.