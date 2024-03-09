import cv2  
import mediapipe as mp
import numpy as np
import math  

mp_desen = mp.solutions.drawing_utils  
mp_pozitie = mp.solutions.pose   
mp_holistic = mp.solutions.holistic
 
def gaseste_distanta(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist
 
def calculeaza_unghiul_3_puncte(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
 
    ba = a - b
    bc = c - b
 
    unghi_cosinus = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)) 
    unghi = np.arccos(unghi_cosinus)
 
    return np.degrees(unghi)
 
def calculeaza_unghiul_2_puncte(a, b):
    a = np.array(a)
    b = np.array(b)
 
    delta = b - a 
    unghi = np.arctan2(delta[1], delta[0]) 
 
    return np.degrees(unghi)  

def deseneaza_scheletul_lateral(imagine, repere):
    puncte_laterale = [mp_holistic.PoseLandmark.LEFT_SHOULDER,  
                       mp_holistic.PoseLandmark.LEFT_ELBOW, 
                       mp_holistic.PoseLandmark.LEFT_WRIST,  
                       mp_holistic.PoseLandmark.LEFT_HIP, 
                       mp_holistic.PoseLandmark.LEFT_KNEE, 
                       mp_holistic.PoseLandmark.LEFT_ANKLE]
 
    for punct1, punct2 in zip(puncte_laterale, puncte_laterale[1:]):
        x1, y1 = int(repere[punct1].x * imagine.shape[1]), int(repere[punct1].y * imagine.shape[0])
        x2, y2 = int(repere[punct2].x * imagine.shape[1]), int(repere[punct2].y * imagine.shape[0])
 
        cv2.line(imagine, (x1, y1), (x2, y2), (0, 255, 0), 2) 
 
    umar_stanga_x, umar_stanga_y = int(repere[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * imagine.shape[1]), int(repere[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * imagine.shape[0])
    nas_x, nas_y = int(repere[mp_holistic.PoseLandmark.NOSE].x * imagine.shape[1]), int(repere[mp_holistic.PoseLandmark.NOSE].y * imagine.shape[0])
    ureche_stanga_x, ureche_stanga_y = int(repere[mp_holistic.PoseLandmark.LEFT_EAR].x * imagine.shape[1]), int(repere[mp_holistic.PoseLandmark.LEFT_EAR].y * imagine.shape[0])
 
    cv2.line(imagine, (umar_stanga_x, umar_stanga_y), (nas_x, nas_y), (0, 255, 0), 2)
    cv2.line(imagine, (umar_stanga_x, umar_stanga_y), (ureche_stanga_x, ureche_stanga_y), (0, 255, 0), 2)
 
def deseneaza_inclinarea_gatului(imagine, repere):
    umar_stanga = np.array([repere[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * imagine.shape[1], repere[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * imagine.shape[0]])
    umar_drept = np.array([repere[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * imagine.shape[1], repere[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * imagine.shape[0]])
    nas = np.array([repere[mp_holistic.PoseLandmark.NOSE].x * imagine.shape[1], repere[mp_holistic.PoseLandmark.NOSE].y * imagine.shape[0]])
 
 
    centru_umar = (umar_stanga + umar_drept) / 2
    vector_gat_nas = nas - centru_umar
    vector_gat_nas = vector_gat_nas / np.linalg.norm(vector_gat_nas)
 
    vector_umar = umar_stanga - umar_drept
    vector_umar = vector_umar / np.linalg.norm(vector_umar)
 
    unghi = math.degrees(np.arccos(np.dot(vector_gat_nas, vector_umar)))
 
    inclinare = unghi - 90
 
    interval_acceptabil = (-10, 10)
    culoare = (0, 255, 0) if interval_acceptabil[0] <= inclinare <= interval_acceptabil[1] else (0, 0, 255)
 
    cv2.putText(imagine, f'Inclinarea gatului: {inclinare:.2f}°', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, culoare, 2)
 
def main():
    cap = cv2.VideoCapture(0)  
    cap2 = cv2.VideoCapture(1)    
 
    with mp_pozitie.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pozitie:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened() and cap2.isOpened():
                ret, frame = cap.read()     
                ret2, frame2 = cap2.read()   
 
                imagine = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imagine.flags.writeable = False
                rezultate = pozitie.process(imagine)
                mp_desen.draw_landmarks(imagine, rezultate.pose_landmarks, mp_pozitie.POSE_CONNECTIONS)
 
                umar_stang, umar_drept, sold_stang, sold_drept, nas = None, None, None, None, None
                
                if rezultate.pose_landmarks:
                    for id, lmk in enumerate(rezultate.pose_landmarks.landmark):
                        if lmk.visibility > 0.5:
                            if id == mp_pozitie.PoseLandmark.LEFT_SHOULDER.value:
                                umar_stang = [lmk.x * imagine.shape[1], lmk.y * imagine.shape[0]]
                            if id == mp_pozitie.PoseLandmark.RIGHT_SHOULDER.value:
                                umar_drept = [lmk.x * imagine.shape[1], lmk.y * imagine.shape[0]]
                            if id == mp_pozitie.PoseLandmark.LEFT_HIP.value:
                                sold_stang = [lmk.x * imagine.shape[1], lmk.y * imagine.shape[0]]
                            if id == mp_pozitie.PoseLandmark.RIGHT_HIP.value:
                                sold_drept = [lmk.x * imagine.shape[1], lmk.y * imagine.shape[0]]  
                            if id == mp_pozitie.PoseLandmark.NOSE.value:
                                nas = [lmk.x * imagine.shape[1], lmk.y * imagine.shape[0]]  
                    
                    
 
                    if umar_stang and umar_drept and nas:   
                        punct_mediu = np.mean([umar_stang, umar_drept], axis=0)
                        unghi_gat = calculeaza_unghiul_3_puncte(umar_stang, punct_mediu, nas)
 
                        text_gat = "Drept" if 86 <= unghi_gat <= 100 else "Nu este drept"  
                        text_unghi = f"Unghi: {unghi_gat:.2f}°"    
                        cv2.putText(imagine, f"Gat: {text_gat}", tuple(np.array(nas, dtype=int)),   
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,cv2.LINE_AA)
                        pozitie_unghi = tuple(np.array(nas, dtype=int) + np.array([0, 15]))
                        cv2.putText(imagine, text_unghi, pozitie_unghi,   
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
 
                    if umar_stang and umar_drept:    
                        unghi_umeri = calculeaza_unghiul_2_puncte(umar_stang, umar_drept)
                        punct_mediu_umeri = [(umar_stang[0] + umar_drept[0]) / 2, (umar_stang[1] + umar_drept[1]) / 2]
                        cv2.putText(imagine, f"Umeri: {round(unghi_umeri,2)}", tuple(np.array(punct_mediu_umeri, dtype=int)),   
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
 
                    if sold_stang and sold_drept:  
                        unghi_solduri = calculeaza_unghiul_2_puncte(sold_stang, sold_drept)
                        punct_mediu_solduri = [(sold_stang[0] + sold_drept[0]) / 2, (sold_stang[1] + sold_drept[1]) / 2]
                        cv2.putText(imagine, f"Șolduri: {round(unghi_solduri, 2)}",   
                                    tuple(np.array(punct_mediu_solduri, dtype=int)),   
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
 
                imagine2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                imagine2.flags.writeable = False
                rezultate2 = holistic.process(imagine2)
                imagine = cv2.cvtColor(imagine, cv2.COLOR_RGB2BGR)
                imagine2.flags.writeable = True    
                imagine2 = cv2.cvtColor(imagine2, cv2.COLOR_RGB2BGR)
                if rezultate2.pose_landmarks:
                    deseneaza_scheletul_lateral(imagine2, rezultate2.pose_landmarks.landmark)
                    deseneaza_inclinarea_gatului(imagine2, rezultate2.pose_landmarks.landmark)
 
                width = 500
                height = 500
                imagine = cv2.resize(imagine, (width, height))  
                imagine2 = cv2.resize(imagine2, (width, height))

                rezultat = np.zeros((height, width*2, 3), np.uint8)
                rezultat[:, :width] = imagine  
                rezultat[:, width:] = imagine2
                


                cv2.imshow('Postura', rezultat)  
 
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
 
    cap.release()
    cap2.release()  
    cv2.destroyAllWindows() 
 
if __name__ == "__main__":
    main()