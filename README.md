# Archer Trajectory Alpha Version 0.0.1

## Installation
```
pip install flask cv2 uuid numpy
```

## Run Development
``` 
python app/app.py
```
or
```
python app/app2.py
```

Terdapat dua versi yaitu app dan app2. Untuk versi yang paling terakhir gunakan app2.

## API Endpoint

### POST /api/predict 
Request
``` JSON
{
    "video": file video,
    "y_threshold": int 0-200
}
```
y_threshold berfungsi sebagai toleransi nilai y terdapat median kumpulan centroid. misal median centroid terletak pada y = 570 maka centroid yang diluar y+-y_threshold akan di abaikan pada pendeteksian lintasan.


Response
``` JSON
{
    "status": "processing",
    "code": code
}
```
code berisi kode unik sebagai identifier sebuah file yang dimana bisa dipakai untuk mencari history atau hasil anak panah.


### POST /api/result
Request
``` JSON
{
    "code": code
}
```
code berisikan dengan kode unik yang didapatkan pada api prediksi.

Response
``` JSON
{
    "status": "processing"
}
```

``` JSON
{
    "status": "error",
    "message": "Failed to process video"
}
```

``` JSON
image
```

Terdapat tiga kondisi respon yaitu processing jika sedang memproses. error jika terjadi error dan langsung image jika telah berhasil terprediksi.


### POST /api/velocity
Request
``` JSON
{
    "video": file video,
    "distance": jarak dalam meter
}
```
Response 
``` JSON
{
    "elapsed_time": waktu sampai target,
    "velocity": kecepatan (m/s)
}
```