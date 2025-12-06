%%writefile app.py
import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import re

import sys, types
from torchvision.transforms import functional as F

fake_ft = types.ModuleType("torchvision.transforms.functional_tensor")
fake_ft.rgb_to_grayscale = F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = fake_ft

from ultralytics import YOLO
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# YOLO
# -----------------------------
vehicle_model = YOLO("best_Vehicules.pt")
plate_model   = YOLO("best_plaques_final.pt")

# -----------------------------
# OCR
# -----------------------------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)

# -----------------------------
# ESRGAN
# -----------------------------
rrdbnet = RRDBNet(
    num_in_ch=3, num_out_ch=3,
    num_feat=64, num_block=23,
    num_grow_ch=32, scale=4
)
upsampler = RealESRGANer(
    scale=4,
    model_path="RealESRGAN_x4plus.pth",
    model=rrdbnet,
    tile=0, tile_pad=10, pre_pad=0,
    half=(device == "cuda")
)

# ===========================
# 2) Pipeline
# ===========================

def enhance_image(img_pil):
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    output, _ = upsampler.enhance(img_bgr, outscale=1)
    return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


def read_plate(img_pil):
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    pix = processor(images=img_pil, return_tensors="pt").pixel_values.to(device)
    ids = ocr_model.generate(pix, max_length=16)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].replace(" ", "")


def detect_and_recognize_all(img_pil):
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    results = []
    veh_pred = vehicle_model(img_cv2)[0]
    veh_boxes = veh_pred.boxes.xyxy.cpu().numpy()
    veh_classes = veh_pred.boxes.cls.cpu().numpy().astype(int)

    if len(veh_boxes) == 0:
        return []

    for idx, (box, cls_id) in enumerate(zip(veh_boxes, veh_classes), 1):
        x1, y1, x2, y2 = box

        # Nom de la classe (car, truck, bus, ...)
        veh_label = vehicle_model.names.get(cls_id, f"class_{cls_id}")

        crop = img_pil.crop((x1, y1, x2, y2))
        cv2crop = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)

        plate_pred = plate_model(cv2crop)[0]
        plates = plate_pred.boxes.xyxy.cpu().numpy()

        if len(plates) == 0:
            results.append({
                "vehicle_id": idx,
                "vehicle_label": veh_label,
                "vehicle_crop": crop,
                "status": "Plaque non détectée",
                "plate_raw": None
            })
            continue

        px1, py1, px2, py2 = plates[0]
        plate_raw = crop.crop((px1, py1, px2, py2))
        plate_gan = enhance_image(plate_raw)

        w, h = plate_gan.size
        plate_ocr = plate_gan.resize((int(w * (64 / h)), 64), Image.BICUBIC)

        raw = read_plate(plate_ocr)
        norm = normalize_tunisian_plate(raw)

        results.append({
            "vehicle_id": idx,
            "vehicle_label": veh_label,
            "vehicle_crop": crop,
            "plate_raw": plate_raw,
            "plate_gan": plate_gan,
            "plate_ocr": plate_ocr,
            "text_raw": raw,
            "text_norm": norm,
            "status": "Plaque détectée"
        })

    return results


import re

def normalize_tunisian_plate(text: str) -> str:
    """
    Formate une plaque tunisienne sous la forme :
        XXX تونس YYYY

    - XXX : au plus 3 chiffres (à gauche)
    - YYYY : au plus 4 chiffres (à droite)

    """
    text = text.strip()
    nums = re.findall(r"\d+", text)

    if not nums:
        return text   
    if len(nums) >= 2:
        left = nums[0][-3:]      # maximum 3 à gauche
        right = nums[-1][-4:]    # maximum 4 à droite
        return f"{left} تونس {right}"  
    n = nums[0]
    if len(n) >= 6:
        
        raw_left = n[:-4]
        raw_right = n[-4:]        
        left = raw_left[-3:]
        right = raw_right
        return f"{left} تونس {right}"

    return text


# Pour forcer l'affichage correct LTR/RTL
LRE = "\u202A"   # Left-to-Right Embedding
RLE = "\u202B"   # Right-to-Left Embedding
PDF = "\u202C"   # Pop directional formatting


def display_plate_pretty(text: str) -> str:
    """
    Affiche joliment une plaque déjà normalisée du type '245 تونس 5911'
    en forçant le sens LTR/RTL.
    """
    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        left = nums[0]
        right = nums[1]
        return f"{LRE}{left}{PDF} {RLE}تونس{PDF} {LRE}{right}{PDF}"
    return text

st.title("🚍🚘 Automatic Number Plate Recognition System 🚐🚛")
st.markdown(
    "<h4 style='color:#555;'>🌶️ With the Tunisian Touch 🌶️ </h4>",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("⚙️ Options")

    source = st.radio(
        "Source de l'image :",
        ["📤 Uploader une image", "🧪 Image test 1", "🧪 Image test 2"],
    )

    st.markdown("---")
    st.subheader("ℹ️ Modèles utilisés")
    st.markdown(
        """
        - **Vehicule Detection** : YOLOv8 fine tuné
        - **Plate Detection** : YOLOv8 fine tuné
        - **Image enhancement** : Real-ESRGAN x4
        - **OCR** : Microsoft TrOCR
        """
    )


image = None

if source == "📤 Uploader une image":
    uploaded = st.file_uploader("📤 Importer une image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
else:
    
    test_file_1 = "test7.jpg"   
    test_file_2 = "test11.jpg"  

    try:
        if source == "🧪 Image test 1":
            image = Image.open(test_file_1).convert("RGB")
        elif source == "🧪 Image test 2":
            image = Image.open(test_file_2).convert("RGB")
    except FileNotFoundError:
        st.error("⚠️ Les images de test ne sont pas trouvées. Vérifie les noms de fichiers dans app.py.")


if image is not None:
    st.subheader("🖼️ Image originale")
    st.image(image, use_column_width=True)

    res = detect_and_recognize_all(image)

    if not res:
        st.error("❌ Aucun véhicule détecté.")
    else:
        st.subheader("📊 Résultats par véhicule détecté")

        for r in res:
            veh_label = r.get("vehicle_label", "vehicle")

            # Titre du véhicule avec label 2× plus grand + gras
            st.markdown(
                f"""
                <div style="margin-top: 10px; margin-bottom: 5px;">
                    <span style="font-size:1.1rem;">Véhicule #{r['vehicle_id']} — </span>
                    <span style="font-size:1.8rem; font-weight:800;">
                        {veh_label.upper()}
                    </span>
                    <span style="font-size:1.1rem;"> — {r['status']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                st.caption("Crop véhicule")
                st.image(r["vehicle_crop"], use_column_width=True)

            if r.get("plate_raw") is not None:
                with col2:
                    st.caption("🔳 Plaque brute")
                    st.image(r["plate_raw"], use_column_width=True)

                col3, col4 = st.columns([1, 1])
                with col3:
                    st.caption("🟦 Plaque après GAN")
                    st.image(r["plate_gan"], use_column_width=True)
                with col4:
                    st.caption("⚪ Plaque utilisée pour OCR")
                    st.image(r["plate_ocr"], use_column_width=True)

                st.markdown(
                    f"**Texte brut OCR :** `{r['text_raw']}`  \n"
                    f"**Texte normalisé :** {display_plate_pretty(r['text_norm'])}"
                )
            else:
                st.warning("Aucune plaque détectée pour ce véhicule.")
else:
    st.info("➡️ Choisis une source dans la barre latérale (upload ou image test) pour commencer.")


st.markdown(
    """
    <hr>
    <div style="text-align:center; color:#888; font-size:0.9rem; margin-top:10px;">
        Réalisé par: Salsabil Ben Halima — Étudiante en terminale Ingénierie des données  
        Faculté des Sciences — Université de Sfax —  
        Département d'Informatique et des Communications
    </div>
    """,
    unsafe_allow_html=True,
)
