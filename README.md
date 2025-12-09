# VTuber Python Tracker (MediaPipe to VMC)

Aplikasi Python ini mengimplementasikan sistem *Motion Capture* tubuh penuh (*Full Body*) menggunakan webcam standar. Program ini memanfaatkan **MediaPipe Holistic** untuk mendeteksi gerakan wajah, mata, tangan, tubuh, hingga kaki, lalu mengirimkan data tersebut melalui **VMC Protocol (OSC)**.

Aplikasi ini dirancang untuk bekerja dengan aplikasi VTuber populer yang mendukung penerimaan data VMC/OSC, seperti **VSeeFace**, **Warudo**, atau project Unity kustom.

## Fitur Utama

  * **Face Tracking:** Melacak rotasi kepala (Pitch, Yaw, Roll), posisi iris mata (Iris Tracking), kedipan mata, dan pergerakan mulut.
  * **Upper Body Tracking:** Melacak rotasi tulang belakang (Spine), bahu, lengan atas, dan lengan bawah.
  * **Leg Tracking (Baru):** Melacak pergerakan kaki (Paha dan Betis/Upper & Lower Leg) menggunakan deteksi pose MediaPipe.
  * **Finger Tracking:** Melacak pergerakan dan lekukan 5 jari pada kedua tangan secara detail.
  * **Smoothing System (One Euro Filter):** Mengimplementasikan algoritma *One Euro Filter* yang telah dikalibrasi ulang untuk mengurangi *jitter* dan membuat gerakan lebih natural.
  * **Visualisasi Tesselation:** Menampilkan jaring-jaring wajah (*Face Mesh Tesselation*) dan *skeleton* tubuh pada jendela preview.
  * **Auto-Fix Protobuf:** Menyertakan *shim* kompatibilitas bawaan untuk mencegah error versi `google.protobuf`.
  * **VMC Protocol Support:** Mengirim data tulang (Bone) dan Blendshape via OSC ke port default (39539).

## Persyaratan Sistem

  * Python 3.7 atau versi yang lebih baru.
  * Webcam yang berfungsi.

### Pustaka Python (Dependencies)

Gunakan `pip` untuk menginstal pustaka yang diperlukan:

```bash
pip install opencv-python mediapipe numpy python-osc
```

## Instalasi & Penggunaan

1.  **Clone atau Download repository ini.**

2.  **Jalankan Script:**

    Buka terminal di folder project dan jalankan:

    ```bash
    python main.py
    ```

    *Jika Anda menggunakan Virtual Environment (venv), pastikan sudah diaktifkan.*

3.  **Integrasi dengan Aplikasi VTuber (Contoh: VSeeFace)**

      * Buka VSeeFace.
      * Masuk ke **Settings** \> **General Settings**.
      * Pada bagian **OSC/VMC receiver**, centang **Enable**.
      * Pastikan port diatur ke **39539**.
      * Avatar akan mulai bergerak mengikuti gerakan tubuh Anda.

## Konfigurasi (Tweak)

Anda dapat mengubah parameter konfigurasi langsung di bagian atas file `main.py` untuk menyesuaikan sensitivitas:

### Koneksi OSC & Kamera

```python
OSC_IP = "127.0.0.1"    # IP Tujuan
OSC_PORT = 39539        # Port VMC
WEBCAM_ID = 0           # ID Kamera
TARGET_FPS = 30         # Target FPS
```

### Smoothing & Sensitivitas

Kode ini menggunakan parameter yang telah disesuaikan untuk gerakan yang lebih stabil:

```python
HEAD_MIN_CUTOFF = 0.03  # Semakin kecil = semakin smooth (kurang jitter) tapi sedikit delay
HEAD_BETA       = 1.5   # Responsivitas gerakan cepat
FINGER_MIN_CUTOFF = 0.8 # Smoothing tinggi untuk jari agar tidak "twitchy"
```

### Kalibrasi Gerakan

```python
ARM_GAIN_XY, ARM_GAIN_Z = 0.8, 0.35  # Mengatur seberapa lebar rentang gerak lengan
NECK_RATIO = 0.4                     # Rasio pergerakan leher terhadap kepala
PITCH_CORRECTION_FACTOR = 0.01       # Koreksi posisi mata saat menunduk/mendongak
```

## Pemecahan Masalah (Troubleshooting)

**1. Error: "The OpenCV build in use does not include GUI support"**
Jika muncul pesan ini, artinya Python yang Anda gunakan menginstal versi OpenCV "headless" (tanpa fitur tampilan jendela).

  * **Solusi:** Pastikan Anda menjalankan script menggunakan python di dalam virtual environment (`.venv`) yang memiliki paket `opencv-python` lengkap, bukan `opencv-python-headless`.

**2. Error Google Protobuf / GetMessageClass**
Script ini sudah memiliki kode *shim* otomatis di baris awal untuk mengatasi masalah ini. Namun, jika masih bermasalah, coba instal versi protobuf spesifik:

```bash
pip install protobuf==3.20.*
```

**3. Gerakan Lambat / FPS Rendah**

  * Pastikan pencahayaan ruangan cukup terang.
  * Coba turunkan resolusi kamera di kode (`cap.set`) jika spesifikasi PC terbatas.

**4. Kamera tidak terbuka**
Pastikan tidak ada aplikasi lain (seperti Zoom/Discord atau VSeeFace bagian "Camera") yang sedang menggunakan webcam yang sama saat script ini dijalankan.

## Tombol Kontrol

  * Tekan **`q`** pada keyboard saat jendela preview aktif untuk menutup aplikasi.
