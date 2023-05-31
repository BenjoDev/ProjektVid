import os
import cv2
import joblib
import numpy as np
from sklearn.svm import SVC

def ustvari_polje_slik_in_znacilnic(pot_mape, znacilnica):
    slike = []
    znacilnice = []

    for ime_datoteke in os.listdir(pot_mape):
        pot_slike = os.path.join(pot_mape, ime_datoteke)
        slika = cv2.imread(pot_slike)
        if slika is not None:
            slika2 = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
            slika2 = cv2.resize(slika2, (100, 100), interpolation = cv2.INTER_AREA)
            slike.append(slika2)
            znacilnice.append(znacilnica)
    return slike, znacilnice

def lbp(slika):
    visina, sirina = slika.shape[:2]
    lbp_slika = np.zeros_like(slika, dtype=np.uint8)
    sosedi = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    for i in range(1, visina - 1):
        for j in range(1, sirina - 1):
            srednji_pixel = slika[i, j]
            novi_srednji_pixel = 0
            for k in range(8):
                sosed_i = i + sosedi[k][0]
                sosed_j = j + sosedi[k][1]
                sosedni_pixel = slika[sosed_i, sosed_j]
                if sosedni_pixel >= srednji_pixel:
                    novi_srednji_pixel |= 1 << (7 - k)
            lbp_slika[i, j] = novi_srednji_pixel

    hist, _ = np.histogram(lbp_slika.ravel(), bins=np.arange(0, 256), range=(0, 255))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def hog(slika, vel_celice=8, vel_blok=2, razdelki=8):
    visina, sirina = slika.shape[:2]
    gx = cv2.Sobel(slika.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3).astype(np.float64)
    gy = cv2.Sobel(slika.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3).astype(np.float64)
    moci = np.sqrt(np.square(gx) + np.square(gy))
    koti = np.arctan2(gy, gx)
    
    st_blokov_x = visina // vel_celice // vel_blok
    st_blokov_y = sirina // vel_celice // vel_blok
    
    hog_znacilke = []
    for i in range(st_blokov_y):
        for j in range(st_blokov_x):            
            i_zacetek = i * vel_celice * vel_blok
            i_konec = i_zacetek + vel_blok * vel_celice
            j_zacetek = j * vel_celice * vel_blok
            j_konec = j_zacetek + vel_blok * vel_celice
            
            blok_moci = moci[i_zacetek:i_konec, j_zacetek:j_konec]
            blok_koti = koti[i_zacetek:i_konec, j_zacetek:j_konec]
            
            hist, _ = np.histogram(blok_koti, bins=razdelki, range=(-np.pi, np.pi), weights=blok_moci)
            hist_norm = np.linalg.norm(hist)
            if hist_norm > 0.0:
                hist /= hist_norm
            else:
                hist = np.zeros_like(hist)
            hog_znacilke.extend(hist)
    return np.array(hog_znacilke)

def izlocanje_znacilk(slike):
    vektor_znacilk = []
    for slika in slike:
        lbp_znacilke = lbp(slika)
        hog_znacilke = hog(slika)
        vektor_znacilk.append(np.concatenate((lbp_znacilke, hog_znacilke)))
    return vektor_znacilk

slike_mack, znacilnice_mack = ustvari_polje_slik_in_znacilnic('Zan', 0)
slike_psov, znacilnice_psov = ustvari_polje_slik_in_znacilnic('other', 1)

slike = np.array(slike_mack + slike_psov, dtype=object)
znacilnice = znacilnice_mack + znacilnice_psov

ucni_vektor_znacilk = izlocanje_znacilk(slike)

razvrscanje = SVC()
razvrscanje.fit(ucni_vektor_znacilk, znacilnice)

joblib.dump(razvrscanje, 'treniran_model.pkl')
