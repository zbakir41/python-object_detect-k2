import cv2
import os
import threading

font = cv2.FONT_HERSHEY_SIMPLEX

new_width = 1200
new_height = 800

def object_detection(image, template_folder):
    # Giriş görüntüsü üzerinde Canny kenar dedektörü uygulanır
    edges_image = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)

    # Template klasöründeki tüm şablonları işleyin
    for template_filename in os.listdir(template_folder):
        template_path = os.path.join(template_folder, template_filename)
        if not os.path.isfile(template_path):
            continue

        # Şablonu yükleyin
        template = cv2.imread(template_path, 0)  # Gri tonlamalı şablon

        # Şablon üzerinde Canny kenar dedektörü uygulanır
        edges_template = cv2.Canny(template, 50, 150)

        # Giriş görüntüsü üzerinde eşleşmeyi bulmak için şablon eşleme yapılır
        result = cv2.matchTemplate(edges_image, edges_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Eşleşme alanının köşe noktaları
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        if int(bottom_right[1]) < 270 :
            #print(int(bottom_right[1]))
            continue
        if int(bottom_right[1]) > 345 :
            #print(int(bottom_right[1]))
            continue

        # Eşleşmeyi gösteren bir dikdörtgen çizilir
        cv2.putText(image, template_filename, top_left, font, 0.6, (0, 0, 255), 2)
        cv2.putText(image, str(bottom_right), bottom_right, font, 1, (0, 255, 255), 2)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    return image

def capture_and_display(cap, template_folder):
    while True:
        # Kameradan bir kare al
        ret, frame1 = cap.read()
        frame = cv2.resize(frame1, (new_width, new_height))

        # Nesne tanıma fonksiyonunu çağır ve sonucu al
        try:
            result_frame = object_detection(frame, template_folder)
            pass
        except :
            result_frame =frame.copy()
        
        #result_frame = object_detection(frame, template_folder)

        # Sonucu göster
        cv2.imshow('Object Detection', result_frame)

        # Çıkış için 'q' tuşuna basın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kamerayı serbest bırakın ve pencereyi kapatın
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Kamerayı başlatın (0 yerine kamera indeksini belirtin veya IP adresi kullanın)
    cap = cv2.VideoCapture(0)

    # Template klasörünün dosya yolu
    template_folder = './template_folder'

    # İş parçacığı oluştur ve başlat
    thread = threading.Thread(target=capture_and_display, args=(cap, template_folder))
    thread.start()

    # Ana iş parçacığında bekle
    thread.join()
