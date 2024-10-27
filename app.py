import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model structure (needs to match the one used to create `best_model.pth`)
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(64 * (256 // 8) * (256 // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Load the model and class names
num_classes = 5  # Replace with actual number of classes in your dataset
class_names = ['cat', 'dog', 'elephant', 'horse', 'lion']  # Replace with actual class names
model = CNNModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Animal descriptions
descriptions = {
    "cat": (
        "Kucing adalah hewan peliharaan populer yang dikenal dengan ukuran tubuh kecil hingga sedang, wajah bulat, dan bulu lembutnya yang bervariasi dalam warna dan pola. "
        "Kucing adalah hewan yang mandiri dan seringkali memiliki naluri berburu yang tinggi. Mereka cenderung tenang, namun juga bisa sangat aktif, terutama saat masih muda. "
        "Kucing dikenal bersih karena sering membersihkan tubuhnya sendiri, dan mereka bisa menjadi hewan yang penuh kasih atau pendiam tergantung pada kepribadian individu masing-masing. "
        "Kucing berkomunikasi melalui suara seperti meongan dan purring, serta bahasa tubuh seperti melengkungkan punggung dan menggerakkan ekor."
    ),
    "dog": (
        "Anjing adalah hewan sosial dan loyal yang sering kali dikenal sebagai sahabat manusia. Mereka hadir dalam berbagai ukuran, bentuk, dan jenis, yang setiapnya memiliki karakteristik unik. "
        "Anjing cenderung mudah dilatih dan dapat membantu manusia dalam berbagai tugas, termasuk sebagai hewan penjaga, penyelamat, atau pemburu. "
        "Dengan kemampuan berinteraksi sosial yang tinggi, anjing membangun ikatan yang kuat dengan pemilik dan keluarganya. "
        "Mereka berkomunikasi melalui berbagai suara seperti gonggongan, lolongan, dan geraman, serta menggunakan bahasa tubuh untuk mengekspresikan suasana hati mereka."
    ),
    "elephant": (
        "Gajah adalah mamalia darat terbesar di dunia, dikenal dengan tubuh besar, telinga lebar yang membantu mengatur suhu tubuh, serta belalai yang panjang dan kuat. "
        "Belalai ini digunakan untuk berbagai fungsi, termasuk mengambil makanan, minum, dan bahkan menunjukkan kasih sayang. Gajah dikenal sangat cerdas dan memiliki memori yang tajam, "
        "yang membantu mereka dalam kehidupan sosial yang kompleks. Mereka hidup dalam kelompok keluarga yang dipimpin oleh betina tertua, dan mereka menunjukkan ikatan yang kuat antaranggota kelompok."
    ),
    "horse": (
        "Kuda adalah hewan herbivora dengan tubuh yang berotot dan kaki panjang yang membuat mereka terkenal dengan kekuatan, stamina, dan kecepatan. "
        "Kuda sering digunakan sebagai hewan transportasi, dalam olahraga, atau sebagai hewan terapi. "
        "Mereka adalah hewan yang memiliki struktur sosial kuat di alam liar, dan mereka dapat menjalin hubungan erat dengan manusia. "
        "Kuda berkomunikasi melalui gerakan tubuh dan suara seperti meringkik, dan mereka sangat responsif terhadap sinyal-sinyal kecil dari pengendara mereka."
    ),
    "lion": (
        "Singa adalah hewan karnivora besar yang terkenal sebagai 'raja hutan' karena keberanian dan kekuatan mereka. "
        "Mereka memiliki tubuh yang kuat, kaki yang besar, dan cakar yang tajam, serta bulu lebat di sekitar kepala jantan yang disebut surai. "
        "Singa hidup dalam kelompok yang disebut kawanan, dengan struktur sosial yang terorganisir di mana betina biasanya berburu bersama untuk makanan. "
        "Mereka adalah pemangsa yang tangguh dan biasanya memangsa hewan besar seperti zebra dan kerbau di habitatnya. "
        "Singa memiliki auman yang kuat yang dapat terdengar dari jarak yang jauh, yang sering digunakan untuk menandai wilayah dan berkomunikasi dengan anggota kelompok."
    )
}

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit app
st.title("Animal Classifier")
st.write("Upload an image to see which animal it is.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
    
    st.write(f"Predicted Animal: **{predicted_class}**")
    st.write("Description:")
    st.write(descriptions[predicted_class])
