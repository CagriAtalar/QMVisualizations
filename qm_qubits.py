import numpy as np
from qutip import *

# Qubit'lerin temel durumları (kendi başlarına |0> ve |1> durumları)
q0 = basis(2, 0)  # |0> durumu
q1 = basis(2, 1)  # |1> durumu

# Bireysel qubit durumları
alpha = 1 / np.sqrt(2)  # Örneğin |0> + |1> durumunun katsayısı
beta = 1 / np.sqrt(2)   # Örneğin |0> + |1> durumunun katsayısı

# İlk ve ikinci qubit'in süperpozisyon durumlarını tanımlayalım
psi_1 = alpha * q0 + beta * q1  # |psi_1> = α|0> + β|1>
psi_2 = alpha * q0 + beta * q1  # |psi_2> = α|0> + β|1>

# Bu iki durumu tensör çarpımı ile birleştiriyoruz
psi_composite = tensor(psi_1, psi_2)  # |ψ_composite> = |ψ_1> ⊗ |ψ_2>

# Bu durumu yazdırıyoruz
print("Birleşik Kuantum Durumu (|ψ_composite>):")
print(psi_composite)

# Qubit'leri ayrı ayrı Bloch Sphere üzerinde görselleştiriyoruz

# İlk qubit'in Bloch Sphere'ini görselleştiriyoruz
bloch1 = Bloch()
bloch1.add_states(psi_1)  # İlk qubit'in durumu
bloch1.show()

# İkinci qubit'in Bloch Sphere'ini görselleştiriyoruz
bloch2 = Bloch()
bloch2.add_states(psi_2)  # İkinci qubit'in durumu
bloch2.show()
