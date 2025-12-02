VTuber Python Tracker (MediaPipe to VMC)
Aplikasi Python ini mengimplementasikan sistem Motion Capture menggunakan webcam standar. Program ini memanfaatkan MediaPipe Holistic untuk mendeteksi gerakan wajah, tangan, dan tubuh, lalu mengirimkan data tersebut melalui VMC Protocol (OSC).

Aplikasi ini dirancang untuk bekerja dengan aplikasi VTuber populer yang mendukung penerimaan data VMC/OSC, seperti VSeeFace, Warudo, atau project Unity kustom.

Fitur
Face Tracking: Melacak rotasi kepala (Pitch, Yaw, Roll), posisi mata (Iris Tracking), kedipan mata, dan pergerakan mulut.

Upper Body Tracking: Melacak rotasi tulang belakang (Spine), bahu, lengan atas, dan lengan bawah.

Finger Tracking: Melacak pergerakan dan lekukan 5 jari pada kedua tangan.

Smoothing System: Mengimplementasikan algoritma One Euro Filter untuk mengurangi jitter (getaran) dan memperhalus gerakan tracking.

VMC Protocol Support: Mengirim data tulang (Bone) dan Blendshape via OSC ke port default (39539).

Visualisasi: Menampilkan wireframe deteksi pada jendela preview.

Persyaratan Sistem
Python 3.7 atau versi yang lebih baru.

Webcam yang berfungsi.

Pustaka Python (Dependencies)
opencv-python

mediapipe

numpy

python-osc

Instalasi
Clone repository ini:

Bash

git clone https://github.com/username/repository-anda.git
cd repository-anda
Instal pustaka yang dibutuhkan: Gunakan pip untuk menginstal dependensi:

Bash

pip install opencv-python mediapipe numpy python-osc
Penggunaan
1. Menjalankan Tracker
Jalankan script utama melalui terminal:

Bash

python main.py
Sebuah jendela akan muncul menampilkan umpan kamera dengan visualisasi tracking. Tekan tombol q pada keyboard saat jendela aktif untuk menghentikan program.

2. Integrasi dengan VSeeFace (Contoh)
Secara default, script ini mengirim data ke 127.0.0.1 pada port 39539.

Buka VSeeFace.

Masuk ke Settings > General Settings.

Pada bagian OSC/VMC receiver, centang opsi Enable.

Pastikan port diatur ke 39539.

Avatar akan mulai bergerak mengikuti gerakan Anda.

Catatan: Disarankan untuk tidak menggunakan kamera yang sama pada input kamera VSeeFace saat script ini berjalan, karena dapat menyebabkan konflik akses perangkat keras.

Konfigurasi
Anda dapat mengubah parameter konfigurasi langsung di bagian atas file main.py:

Koneksi OSC
Python

OSC_IP = "127.0.0.1"    # Alamat IP tujuan (Localhost)
OSC_PORT = 39539        # Port VMC
Kamera
Python

WEBCAM_ID = 0           # Index kamera (0 biasanya kamera default)
TARGET_FPS = 30         # Target FPS
Kalibrasi & Sensitivitas
FINGER_SENSITIVITY: Mengatur responsivitas lekukan jari.

EAR_THRESH: Mengatur ambang batas deteksi kedipan mata (Close/Open).

ARM_INVERT: Mengatur inversi sumbu rotasi lengan jika gerakan terbalik di aplikasi target.

Smoothing (One Euro Filter)
Parameter ini mengatur keseimbangan antara smoothness (kehalusan) dan latency (responsivitas):

MIN_CUTOFF: Nilai lebih rendah meningkatkan kehalusan namun menambah latensi.

BETA: Nilai lebih tinggi mengurangi latensi pada gerakan cepat.

Python

HEAD_MIN_CUTOFF = 0.05
HEAD_BETA       = 0.5
Pemecahan Masalah (Troubleshooting)
Error: AttributeError: 'MessageFactory' object has no attribute 'GetPrototype' Ini adalah masalah kompatibilitas umum antara versi protobuf dan mediapipe. Solusinya adalah menurunkan versi protobuf:

Bash

pip install protobuf==3.20.*
Kamera tidak terbuka Pastikan WEBCAM_ID sesuai dengan perangkat Anda dan tidak ada aplikasi lain yang sedang menggunakan kamera tersebut.
