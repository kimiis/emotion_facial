import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
import matplotlib.cm as cm

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
model_id = "facebook/convnext-tiny-224"
num_labels = len(class_names)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WrappedModel(torch.nn.Module):
    def __init__(self, model_to_wrap):
        super().__init__()
        self.model_to_wrap = model_to_wrap

    def forward(self, x):
        return self.model_to_wrap(x).logits


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.hooks = []
        self.hooks.append(target_layer.register_forward_hook(self._save_activation))
        self.hooks.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam[0, 0].cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def overlay_cam(image_np, cam):
    h, w = image_np.shape[:2]
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cam_np = np.array(cam_pil) / 255.0
    heatmap = cm.jet(cam_np)[:, :, :3]
    overlay = np.clip(0.5 * image_np + 0.5 * heatmap, 0, 1)
    return (overlay * 255).astype(np.uint8)


@st.cache_resource
def load_model_and_resources():
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    for param in model.convnext.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in enumerate(class_names)}
    model.config.id2label = id2label
    model.config.label2id = label2id
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_layer = model.convnext.encoder.stages[-1].layers[-1]
    return model, val_transform, id2label, target_layer


model, val_transform, id2label, target_layer = load_model_and_resources()

st.set_page_config(
    page_title="Detecteur d'emotions faciales",
    page_icon="🔬",
    layout="wide"
)

st.sidebar.title("A propos")
st.sidebar.info(
    "Prototype de recherche — la reconnaissance d'emotions par IA est un sujet controverse. "
    "Les resultats sont approximatifs et culturellement biaises."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modele :** ConvNeXt-Tiny")
st.sidebar.markdown("**Classes :** " + ", ".join(class_names))

st.title("Detecteur d'emotions faciales")
st.markdown(
    "Uploadez une image pour obtenir une prediction "
    "avec score de confiance et visualisation GradCAM."
)

uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptes : JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analyse")
        with st.spinner("Analyse en cours..."):
            input_tensor = val_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits

            probabilities = torch.softmax(logits, dim=-1)[0]
            predicted_class_id = torch.argmax(probabilities).item()
            predicted_confidence = probabilities[predicted_class_id].item()
            predicted_label = id2label[predicted_class_id]

            mean_val = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std_val = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_for_cam = (input_tensor[0].cpu() * std_val + mean_val).permute(1, 2, 0).numpy()
            image_for_cam = np.clip(image_for_cam, 0, 1)

            wrapped_model = WrappedModel(model)
            input_tensor_grad = input_tensor.clone().requires_grad_(True)

            for param in target_layer.parameters():
                param.requires_grad = True

            cam_gen = GradCAM(wrapped_model, target_layer)
            grayscale_cam = cam_gen.generate(input_tensor_grad, predicted_class_id)
            cam_gen.remove_hooks()

            for param in target_layer.parameters():
                param.requires_grad = False

            gradcam_overlay = overlay_cam(image_for_cam, grayscale_cam)

            st.metric(label="Prediction", value=predicted_label, delta=f"{predicted_confidence:.2%}")
            st.image(gradcam_overlay, caption="Heatmap GradCAM", use_container_width=True)

    st.markdown("---")
    st.warning(
        "Prototype de recherche — la reconnaissance d'emotions par IA est un sujet controverse. "
        "Les resultats sont approximatifs et culturellement biaises."
    )
else:
    st.info("Uploadez une image pour commencer l'analyse.")