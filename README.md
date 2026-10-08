# Potato Leaf Doctor (CNN learning project)

This project has a training script, a small CNN, and a phone-friendly web app. The current dataset supports **potato leaves only**, with three labels: Early blight, Late blight, and healthy. It cannot reliably recognize other crops or conditions missing from the dataset.

## 1. Prepare the environment

From this project folder, activate the existing Windows environment. TensorFlow and Pillow are the only external packages used:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Train and save the CNN

```powershell
python train.py
```

The script creates `model/potato_leaf_cnn.keras` and `model/labels.json`. It prints the class counts, CNN layer summary, and held-out test accuracy. First run may take a while on CPU.

## 3. Run the app on your computer

```powershell
python app.py
```

On the same computer, open `http://localhost:5000`.

## 4. Open it on a phone on the same Wi-Fi

Find the computer's local IPv4 address with `ipconfig`, then on the phone open `http://COMPUTER_IP:5000` (replace `COMPUTER_IP`). Windows Firewall may ask to allow Python on your private network. The phone and computer must be on the same Wi-Fi. Browser camera access on a phone generally requires HTTPS; if the camera button does not open, use **Choose photo** or run the site through a secure HTTPS tunnel/deployment. This first version runs inference on the computer, so the computer must stay on.

## What happens at each stage

1. **Dataset:** `image_dataset_from_directory` gets the label from each folder name.
2. **Split:** images are stratified into 80% training, 10% validation, and 10% test sets. This avoids evaluating the model on images it trained on.
3. **Preprocessing:** each image is resized to 160×160 RGB pixels. The model rescales pixel values from 0–255 to 0–1.
4. **Augmentation:** random flip, rotation, and zoom create small training variations. Class weights compensate for fewer healthy examples.
5. **CNN:** convolution layers learn local visual patterns, pooling reduces their spatial size, and a final softmax gives a score for each of the three classes.
6. **Evaluation:** validation data guides early stopping; the separate test split gives a more honest final accuracy estimate.
7. **Prediction:** the phone sends a photo to Flask, which applies the same image size and model. Below 60% top score, the app says the result is uncertain.

The displayed percentage is a model confidence score, not a guarantee of real-world accuracy. Treat results as a learning aid and confirm suspected disease with an agricultural expert.
