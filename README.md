# Python VTuber Full Body Tracker (MediaPipe to VMC Protocol)

**Python VTuber Tracker** adalah sistem *Motion Capture* (MoCap) berbasis webcam yang mengubah input video standar menjadi data gerakan avatar 3D secara *real-time*.

Menggunakan kekuatan **Google MediaPipe Holistic**, aplikasi ini mendeteksi titik-titik (landmarks) pada wajah, tangan, dan seluruh tubuh (termasuk kaki), lalu mengonversinya menjadi rotasi tulang (Quaternion) dan Blendshapes yang dikirimkan melalui protokol **VMC (Virtual Motion Capture)** via OSC.

Sistem ini dirancang sebagai alternatif gratis untuk perangkat keras MoCap mahal, kompatibel dengan aplikasi VTuber populer seperti **VSeeFace**, **Warudo**, **VNyan**, atau proyek Unity/Unreal Engine yang mendukung VMC.

-----

## Fitur Utama

### 1\. Pelacakan Wajah (Face Tracking) High-Fidelity

  * **Rotasi Kepala:** Menghitung Pitch, Yaw, dan Roll kepala dengan *dampening* agar gerakan tidak patah-patah.
  * **Iris Tracking:** Melacak pergerakan bola mata secara independen dari kepala.
  * **Blink Detection:** Mendeteksi kedipan mata kiri dan kanan menggunakan *Eye Aspect Ratio* (EAR).
  * **Mouth Tracking:** Mendeteksi pembukaan mulut berdasarkan jarak bibir atas dan bawah.
  * **Tesselation Visualization:** Menampilkan jaring-jaring wajah (mesh) secara visual di jendela preview.

### 2\. Pelacakan Tubuh & Kaki (Full Body & Leg Tracking)

  * **Upper Body:** Melacak rotasi Tulang Belakang (Spine), Bahu, Lengan Atas, dan Lengan Bawah.
  * **Lower Body (Baru):** Fitur eksperimental untuk melacak Paha (Upper Leg) dan Betis (Lower Leg) saat pengguna berdiri jauh dari kamera.
  * **Spine Logic:** Menghitung rotasi tubuh berdasarkan sudut bahu untuk gerakan leaning kiri/kanan yang natural.

### 3\. Pelacakan Jari (Finger Tracking)

  * **5-Finger Detection:** Melacak kelima jari pada kedua tangan.
  * **Curling Logic:** Menghitung tingkat lekukan (curl) jari berdasarkan rasio jarak ujung jari ke pergelangan tangan vs pangkal jari.
  * **Per-Bone Rotation:** Mengirimkan data rotasi untuk setiap ruas jari (Proximal, Intermediate, Distal).

### 4\. Sistem Stabilisasi (One Euro Filter)

Kode ini mengimplementasikan algoritma **One Euro Filter** yang dikustomisasi. Filter ini bekerja secara adaptif:

  * **Gerakan Lambat:** Filter sangat agresif untuk menghilangkan *jitter* (getaran mikro) kamera.
  * **Gerakan Cepat:** Filter mengurangi latensi agar gerakan tetap responsif.

### 5\. Kompatibilitas & Keamanan

  * **Auto-Fix Protobuf:** Dilengkapi *shim* kode otomatis untuk menangani inkompatibilitas versi `google.protobuf` yang sering terjadi pada MediaPipe.
  * **GUI Check:** Mendeteksi jika OpenCV yang terinstal tidak mendukung tampilan jendela (headless) dan memberi peringatan yang jelas.

-----

## Persyaratan Sistem

  * **OS:** Windows 10/11 (Disarankan), Linux, atau macOS.
  * **Python:** Versi 3.8 hingga 3.11.
  * **Hardware:** Webcam standar (720p 30fps sudah cukup).

### Instalasi Dependensi

Pastikan Anda berada di dalam *Virtual Environment* (disarankan), lalu instal pustaka berikut:

```bash
pip install opencv-python mediapipe numpy python-osc
```

*Catatan: Jika mengalami error protobuf, script ini sudah memiliki penanganan otomatis, namun jika tetap gagal, coba `pip install protobuf==3.20.0`*

-----

## Cara Penggunaan

### 1\. Menjalankan Tracker

Jalankan script utama melalui terminal atau Command Prompt:

```bash
python main.py
```

Sebuah jendela berjudul **"VTuber TESSELATION ONLY"** akan muncul. Pastikan kamera mendeteksi tubuh Anda (garis-garis skeleton akan muncul di layar).

### 2\. Menghubungkan ke Aplikasi VTuber (Contoh: VSeeFace)

Secara default, tracker mengirim data ke `127.0.0.1` port `39539`.

1.  Buka **VSeeFace**.
2.  Pergi ke **Settings** \> **General Settings**.
3.  Cari bagian **OSC/VMC Receiver**.
4.  Centang **Enable**.
5.  Pastikan port terisi **39539**.
6.  Pilih **"VMC source adds to VSeeFace tracking"** (atau matikan tracking bawaan VSeeFace jika ingin full control dari script ini).

-----

## Panduan Konfigurasi Mendalam

Anda dapat menyetel parameter di bagian atas file `main.py` untuk menyesuaikan "rasa" gerakan avatar Anda.

### A. Konfigurasi Jaringan

```python
OSC_IP = "127.0.0.1"   # Ganti ke IP komputer lain jika menjalankan dual-PC setup
OSC_PORT = 39539       # Port standar VMC. Ubah jika bentrok.
WEBCAM_ID = 0          # Ganti ke 1, 2, dst jika Anda memiliki banyak kamera.
```

### B. Konfigurasi Smoothing (PENTING)

Parameter ini mengatur keseimbangan antara *Smoothness* (Kehalusan) dan *Latency* (Kecepatan).

  * **`MIN_CUTOFF`**: Nilai lebih kecil = Lebih halus tapi sedikit delay. Nilai besar = Lebih responsif tapi bergetar.
  * **`BETA`**: Nilai responsivitas. Naikkan jika gerakan cepat terasa tertinggal.

<!-- end list -->

```python
HEAD_MIN_CUTOFF = 0.03   # Sangat stabil untuk wajah
HEAD_BETA       = 1.5    # Cukup responsif
FINGER_MIN_CUTOFF = 0.8  # Nilai tinggi agar jari tidak bergetar sendiri
```

### C. Kalibrasi Gerakan (Motion Tuning)

Sesuaikan nilai ini agar gerakan avatar sesuai dengan proporsi tubuh asli Anda.

```python
# Mengontrol seberapa luas lengan avatar bergerak dibanding aslinya
ARM_GAIN_XY = 0.8        # 1.0 = 1:1. 0.8 = Gerakan avatar sedikit lebih kecil (aman dari clipping)
ARM_GAIN_Z  = 0.35       # Kedalaman lengan (maju/mundur)

# Sensitivitas Jari
FINGER_SENSITIVITY = 0.9 # Turunkan jika jari avatar terlalu cepat menutup

# Rasio Leher
NECK_RATIO = 0.4         # 40% rotasi kepala dibebankan ke leher, 60% ke kepala
```

-----

## Cara Kerja Teknis (Under the Hood)

1.  **Image Capture:** OpenCV mengambil frame dari webcam dan membalikkannya (mirroring).
2.  **Inference:** Frame dikirim ke `mediapipe.solutions.holistic` yang menghasilkan 3 set landmarks: Face (468 titik), Hands (21 titik per tangan), dan Pose (33 titik tubuh).
3.  **Face Geometry (PnP):** Script menggunakan algoritma *Perspective-n-Point* (PnP) untuk menghitung rotasi 3D kepala berdasarkan posisi titik wajah 2D di layar.
4.  **Vector Calculation:**
      * Untuk lengan dan kaki, script menghitung vektor antara bahu-\>siku, siku-\>pergelangan, pinggul-\>lutut, dst.
      * Vektor ini dibandingkan dengan "Rest Pose" (posisi T-pose/I-pose) menggunakan *Dot Product* dan *Cross Product* untuk menghasilkan **Quaternion** rotasi.
5.  **Filtering:** Semua nilai mentah (Raw) masuk ke **OneEuroFilter** untuk dibersihkan dari noise.
6.  **OSC Dispatch:** Data bersih dikemas dalam format VMC (`/VMC/Ext/Bone/Pos`) dan dikirim via UDP ke aplikasi target.

-----

## Analisis Teknis & Evaluasi Sistem

Bagian ini menyajikan evaluasi teknis terhadap arsitektur kode, performa, dan validitas implementasi matematika yang digunakan dalam proyek ini.

### 1\. Kualitas & Struktur Kode

  * **Manajemen Dependensi:** Kode menerapkan *compatibility shim* (penyesuaian otomatis) untuk `google.protobuf.message_factory`. Hal ini mencegah *runtime error* yang umum terjadi akibat ketidakcocokan versi antara MediaPipe dan Protobuf.
  * **Modularitas:** Fungsi perhitungan kompleks seperti konversi Euler-ke-Quaternion dan rotasi vektor (`get_limb_rotation`) dipisahkan dari loop utama, meningkatkan keterbacaan kode.

### 2\. Logika Matematika (Vector Math)

  * **Quaternion vs Euler:** Sistem ini memprioritaskan penggunaan Quaternion untuk rotasi tulang. Pendekatan ini menghindari masalah *Gimbal Lock* yang sering terjadi jika hanya mengandalkan sudut Euler (Pitch/Yaw/Roll).
  * **Kinematika Jari:** Menggunakan metode aproksimasi berbasis rasio jarak (*distance ratio*) daripada sudut sendi murni. Meskipun merupakan penyederhanaan, metode ini terbukti jauh lebih stabil untuk data resolusi rendah dari webcam dibandingkan deteksi sudut sendi tradisional.

### 3\. Implementasi Smoothing (One Euro Filter)

Proyek ini tidak menggunakan *Moving Average* sederhana, melainkan **One Euro Filter**. Algoritma ini dievaluasi sangat efektif karena sifatnya yang adaptif:

  * **Pada kondisi diam:** *Cutoff frequency* menurun, menghilangkan *micro-jitter* secara agresif.
  * **Pada kondisi bergerak:** *Cutoff frequency* meningkat instan berdasarkan parameter `beta`, meminimalkan latensi (lag) yang dirasakan pengguna.

### 4\. Performa & Batasan

  * **CPU Bound:** Karena Python menggunakan *Global Interpreter Lock (GIL)*, proses pengambilan gambar (IO) dan inferensi AI (CPU) berjalan pada thread yang sama. Pada perangkat spesifikasi rendah, hal ini dapat menyebabkan penurunan FPS jika resolusi kamera terlalu tinggi.
  * **Single-Camera Occlusion:** Seperti semua sistem berbasis satu kamera, sistem ini memiliki batasan oklusi. Jika tangan menyilang menutupi tubuh, atau kaki tertutup meja, MediaPipe akan melakukan prediksi (tebakan) yang mungkin tidak akurat.

### 5\. Roadmap Pengembangan

Rencana pengembangan ke depan untuk meningkatkan stabilitas sistem:

1.  **Multithreading:** Memisahkan pembacaan kamera (`cv2.VideoCapture`) ke thread terpisah untuk meningkatkan throughput FPS.
2.  **Kalibrasi T-Pose:** Menambahkan fitur reset posisi nol secara dinamis via keyboard.
3.  **Mode Duduk (Sitting Mode):** Opsi untuk menonaktifkan pelacakan kaki saat pengguna duduk untuk mengurangi *noise*.

-----

## Pemecahan Masalah (Troubleshooting)

**Q: Muncul error `cv2.error: ... The function is not implemented.`**

  * **Solusi:** Uninstall versi headless dan instal versi standar: `pip install opencv-python`.

**Q: Tangan saya bergerak terbalik.**

  * **Solusi:** Ubah parameter `ARM_INVERT` di konfigurasi menjadi nilai negatif.

**Q: Gerakan kaki berantakan/melayang.**

  * **Solusi:** Fitur *Leg Tracking* membutuhkan visibilitas penuh. Mundurlah dari kamera, atau abaikan gerakan kaki jika Anda hanya duduk (avatar akan tetap mengikuti tubuh bagian atas).

**Q: FPS sangat rendah.**

  * **Solusi:** Pastikan cahaya ruangan terang dan coba turunkan resolusi kamera di kode: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)`.
