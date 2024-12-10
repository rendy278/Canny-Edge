import cv2
import matplotlib.pyplot as plt

# Read the original image 
img = cv2.imread('images/malsook.jpeg') 

# Konversi ke grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur untuk memperbaiki deteksi tepi
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=200, threshold2=300)

# Konversi Sobel dan Canny ke uint8 untuk kompatibilitas dengan Matplotlib
sobelx_uint8 = cv2.convertScaleAbs(sobelx)
sobely_uint8 = cv2.convertScaleAbs(sobely)
sobelxy_uint8 = cv2.convertScaleAbs(sobelxy)

# Konversi gambar asli ke RGB untuk Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Tampilkan hasil dalam grid menggunakan Matplotlib
titles = ['Original', 'Grayscale', 'Sobel X', 'Sobel Y', 'Sobel XY', 'Canny Edge']
images = [img_rgb, img_gray, sobelx_uint8, sobely_uint8, sobelxy_uint8, edges]

# Atur ukuran grid
plt.figure(figsize=(12, 8))

# Gambar Original
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original', color='blue')
plt.axis('off')

# Gambar Grayscale
plt.subplot(2, 3, 2)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale', color='blue')
plt.axis('off')

# Gambar Sobel X
plt.subplot(2, 3, 3)
plt.imshow(sobelx_uint8, cmap='gray')
plt.title('Sobel X', color='blue')
plt.axis('off')

# Gambar Sobel Y
plt.subplot(2, 3, 4)
plt.imshow(sobely_uint8, cmap='gray')
plt.title('Sobel Y', color='blue')
plt.axis('off')

# Gambar Sobel XY
plt.subplot(2, 3, 5)
plt.imshow(sobelxy_uint8, cmap='gray')
plt.title('Sobel XY', color='blue')
plt.axis('off')

# Gambar Canny Edge
plt.subplot(2, 3, 6)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge', color='blue')
plt.axis('off')

# Tata letak
plt.tight_layout()
plt.show()
