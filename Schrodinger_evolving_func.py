import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation

# Parametreler
hbar = 1  # Planck sabiti (doğal birimlerde)
m = 1.0  # Parçacık kütlesi
L = 10  # Alan boyutu
num_points = 100  # Pozisyon gridindeki noktalar
t_max = 5  # Zaman aralığı (5 saniye)
num_frames = 200  # Zaman adımları

# Alanın başlangıç pozisyonu ve momentum grid'i
x = np.linspace(-L/2, L/2, num_points)

# Başlangıç dalga fonksiyonu: Gauss tipi dalga
sigma = 1.0  # Dalga fonksiyonunun genişliği
k0 = 2.0  # Dalga sayısı
psi_0 = np.exp(-(x**2)/(2*sigma**2)) * np.exp(1j * k0 * x)
psi_0 = psi_0 / np.linalg.norm(psi_0)  # Normalizasyon

# Hamiltonyen Operatörü (Serbest parçacık için)
H_matrix = -hbar**2 / (2*m) * (-np.diag(np.ones(num_points-1), -1) + 2 * np.diag(np.ones(num_points), 0) - np.diag(np.ones(num_points-1), 1)) / (x[1] - x[0])**2
H = Qobj(H_matrix)

# Zaman evrimi operatörü
t_list = np.linspace(0, t_max, num_frames)  # Zaman adımları

# Başlangıç durumunun zamanla evrimini hesaplayalım
psi_t_list = []

for t in t_list:
    # Zaman evrimi operatörünü uygulayalım
    U_t = (-1j * H * t / hbar).expm()  # Zaman evrimi operatörü
    psi_t = U_t * Qobj(psi_0)  # Zamanla evrilen dalga fonksiyonu (Qobj'e dönüştürülmüş)
    psi_t_list.append(psi_t)

# Momentum uzayında dalga fonksiyonunun evrimi (Fourier dönüşümü)
def fourier_transform(psi_x, x, p):
    dx = x[1] - x[0]
    return np.fft.fftshift(np.fft.fft(psi_x)) * np.sqrt(dx)

# Animasyon için figür ve eksenler
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Başlangıçta pozisyon uzayında dalga fonksiyonu ve momentum uzayında dalga fonksiyonu
line_real_x, = ax1.plot(x, np.real(psi_t_list[0].full()), label='Re(ψ(x,t))')
line_imag_x, = ax1.plot(x, np.imag(psi_t_list[0].full()), label='Im(ψ(x,t))')
line_real_p, = ax2.plot(np.fft.fftshift(np.fft.fftfreq(num_points, x[1]-x[0])), np.real(fourier_transform(psi_t_list[0].full(), x, np.fft.fftshift(np.fft.fftfreq(num_points, x[1]-x[0])))), label='Re(φ(p,t))')
line_imag_p, = ax2.plot(np.fft.fftshift(np.fft.fftfreq(num_points, x[1]-x[0])), np.imag(fourier_transform(psi_t_list[0].full(), x, np.fft.fftshift(np.fft.fftfreq(num_points, x[1]-x[0])))), label='Im(φ(p,t))')

# Grafik başlıkları ve etiketler
ax1.set_title("Pozisyon Uzayındaki Dalga Fonksiyonunun Zamanla Evrimi")
ax1.set_xlabel("Pozisyon (x)")
ax1.set_ylabel("Dalga Fonksiyonu (ψ(x,t))")
ax1.legend()

ax2.set_title("Momentum Uzayındaki Dalga Fonksiyonunun Zamanla Evrimi")
ax2.set_xlabel("Momentum (p)")
ax2.set_ylabel("Dalga Fonksiyonu (φ(p,t))")
ax2.legend()

# Animasyon fonksiyonu
def update(frame):
    # Pozisyon uzayındaki dalga fonksiyonu
    line_real_x.set_ydata(np.real(psi_t_list[frame].full()))
    line_imag_x.set_ydata(np.imag(psi_t_list[frame].full()))
    
    # Fourier dönüşümü ile momentum uzayındaki dalga fonksiyonu
    psi_p = fourier_transform(psi_t_list[frame].full(), x, np.fft.fftshift(np.fft.fftfreq(num_points, x[1]-x[0])))
    line_real_p.set_ydata(np.real(psi_p))
    line_imag_p.set_ydata(np.imag(psi_p))
    
    return line_real_x, line_imag_x, line_real_p, line_imag_p

# Animasyonu oluştur
ani = FuncAnimation(fig, update, frames=len(t_list), interval=50, blit=True)

plt.show()
