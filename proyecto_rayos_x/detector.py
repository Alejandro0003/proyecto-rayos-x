import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw

class DetectorTransformer:
    def __init__(self, model_name: str = "hustvl/yolos-tiny"):
        self.processor = YolosImageProcessor.from_pretrained(model_name)
        self.model = YolosForObjectDetection.from_pretrained(model_name)

    def procesar_imagen(self, imagen_pil: Image.Image, umbral: float):
        imagen_dibujo = imagen_pil.copy()
        inputs = self.processor(images=imagen_dibujo, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([imagen_dibujo.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=umbral
        )[0]
        
        dibujo = ImageDraw.Draw(imagen_dibujo)
        objeto_detectado = "Inspección Requerida"
        nombre_objeto_crudo = "Masa Opaca Desconocida"
        datos_inventario = []

        clases_sospechosas = ["knife", "scissors", "tool", "fork", "weapon", "umbrella", "bottle", "handbag", "suitcase"]

        if len(results["scores"]) > 0:
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                confianza = round(score.item() * 100, 1)
                label_name = self.model.config.id2label[label.item()].upper()
                
                es_sospechoso = any(x in label_name.lower() for x in clases_sospechosas)
                if es_sospechoso:
                    nombre_objeto_crudo = f"SILUETA CRÍTICA ({label_name})"
                    objeto_detectado = "AMENAZA CONFIRMADA / OBJETO PROHIBIDO"
                    color_caja = "#FF1744"
                else:
                    color_caja = "#FF9100"
                
                dibujo.rectangle(box, outline=color_caja, width=6)
                dibujo.text((box[0] + 5, box[1] + 5), f"{label_name} ({confianza}%)", fill=color_caja)
                
                datos_inventario.append({"Objeto": label_name, "Clasificación": "Revisión Obligatoria" if es_sospechoso else "Ordinario", "Confianza": f"{confianza}%"})
        
        return imagen_dibujo, objeto_detectado, nombre_objeto_crudo, datos_inventario