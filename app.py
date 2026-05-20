import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Configuration des classes et du modèle
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
model_id = "facebook/convnext-tiny-224"
num_labels = len(class_names)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Wrapper pour le modèle HuggingFace pour GradCAM
class WrappedModel(torch.nn.Module):
    def __init__(self, model_to_wrap):
        super().__init__()
        self.model_to_wrap = model_to_wrap

    def forward(self, x):
        return self.model_to_wrap(x).logits

@st.cache_resource
def load_model_and_resources():
    # Charger le modèle pré-entraîné
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    # Geler le backbone (si non fait dans le modèle sauvegardé)
    for param in model.convnext.parameters():
        param.requires_grad = False
    model.eval() # Important pour l'inférence
    model.to(device)

    # Mapper les labels
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in enumerate(class_names)}
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Transforms pour la validation/prédiction (sans augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), # Convertir en 3 canaux si niveaux de gris
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Définir la target layer pour GradCAM (pour ConvNeXt-Tiny)
    target_layer = model.convnext.encoder.stages[-1].layers[-1]

    return model, val_transform, id2label, target_layer

# Charger le modèle et les ressources (sera fait une seule fois)
model, val_transform, id2label, target_layer = load_model_and_resources()

# ── Configuration de la page ──
st.set_page_config(
    page_title="Detecteur d'emotions faciales",
    page_icon="🔬",
    layout="wide"
)

# ── Sidebar ──
st.sidebar.title("A propos")
st.sidebar.info(
    "Prototype de recherche — la reconnaissance d'emotions par IA est un sujet controverse. Les resultats sont approximatifs et culturellement biaises."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modele :** ConvNeXt-Tiny")
st.sidebar.markdown("**Classes :** " + ", ".join(class_names))

# ── Page principale ──
st.title("Detecteur d'emotions faciales")
st.markdown(
    "Uploadez une image pour obtenir une prediction "
    "avec score de confiance et visualisation GradCAM."
)

# ── Upload d'image ──
uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptes : JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Layout en 2 colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analyse")

        with st.spinner("Analyse en cours..."):
            # 1. Appliquer val_transform a l'image
            input_tensor = val_transform(image).unsqueeze(0).to(device)

            # 2. Predire avec le modele
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits

            # 3. Calculer les probabilites (softmax)
            probabilities = torch.softmax(logits, dim=-1)[0]
            predicted_class_id = torch.argmax(probabilities).item()
            predicted_confidence = probabilities[predicted_class_id].item()
            predicted_label = id2label[predicted_class_id]

            # 4. Generer le GradCAM
            # Obtenir l'image originale (denormalisée) pour l'affichage du GradCAM
            # On doit inverser la normalisation pour afficher l'image correctement
            mean_val = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std_val = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

            # Detacher de cuda et convertir en numpy pour matplotlib
            image_for_cam_display = (input_tensor[0].cpu() * std_val.cpu() + mean_val.cpu()).permute(1, 2, 0).numpy()
            image_for_cam_display = np.clip(image_for_cam_display, 0, 1)

            # Créer l'objet GradCAM avec le modèle wrappé
            wrapped_model = WrappedModel(model)
            cam_generator = GradCAM(model=wrapped_model, target_layers=[target_layer])
            targets = [ClassifierOutputTarget(predicted_class_id)]

            # Temporarily enable gradients for the target layer parameters (required by GradCAM)
            original_requires_grad_states = {}
            for param in target_layer.parameters():
                original_requires_grad_states[param] = param.requires_grad
                param.requires_grad = True

            # Générer la heatmap
            grayscale_cam = cam_generator(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

            # Revert gradients for the target layer parameters
            for param, original_state in original_requires_grad_states.items():
                param.requires_grad = original_state

            # Superposer la heatmap sur l'image originale
            gradcam_overlay = show_cam_on_image(image_for_cam_display, grayscale_cam, use_rgb=True)

            # 5. Afficher les resultats
            st.metric(label="Prediction", value=f"{predicted_label}", delta=f"{predicted_confidence:.2%}")
            st.image(gradcam_overlay, caption="Heatmap GradCAM", use_container_width=True)

    # ── Disclaimer en bas ──
    st.markdown("---")
    st.warning("Prototype de recherche — la reconnaissance d'emotions par IA est un sujet controverse. Les resultats sont approximatifs et culturellement biaises.")

else:
    st.info("Uploadez une image pour commencer l'analyse.")
