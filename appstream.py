import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.builders import model_builder
# from object_detection.utils import config_util
import easyocr
import pandas as pd
import os
import uuid
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# from utils import label_map_util
# from utils import visualization_utils as viz_utils
# from builders import model_builder
# from utils import config_util

global df
df = pd.DataFrame(columns=["Detected Text", "Date", "Time"])
# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load the configuration and model
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
paths = {
    'CHECKPOINT_PATH': os.path.join('model', 'checkpoints'),
}
files = {
    'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-61')).expect_partial()

# Load the label map
category_index = label_map_util.create_category_index_from_labelmap(
    'Tensorflow/workspace/annotations/label_map.pbtxt'
)

@tf.function
def detect_fn(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Number Plate Detection")
option = st.sidebar.selectbox("Choose an option", ("Select", "Import Image", "Live Video", "Download Results"))

if option == "Import Image":
    img_option = st.selectbox("Choose Image Option", ["Select", "Take Picture", "Import from Gallery"])

    if img_option == "Take Picture":
        st.write("Take Picture functionality is currently not supported directly in Streamlit.")
    
    elif img_option == "Import from Gallery":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Convert image to tensor and make detections
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)

            # Extract the number of detections and reshape the outputs
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # Visualize detection boxes on the image
            label_id_offset = 1
            for i in range(num_detections):
                if detections['detection_scores'][i] > 0.8:
                    ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
                    (left, right, top, bottom) = (xmin * image.width, xmax * image.width, ymin * image.height, ymax * image.height)
                    cutout = image_np[int(top):int(bottom), int(left):int(right)]
                    st.image(cutout, caption="Detected Number Plate", use_column_width=True)

                    # Detect text from number plate using EasyOCR
                    result = reader.readtext(cutout)
                    detected_text = ' '.join([text[1] for text in result])
                    st.write(f"Detected Number Plate Text: {detected_text}")
                    break

elif option == "Live Video":
    # Set up the GPU memory limit
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]
            )
        except RuntimeError as e:
            print(e)
   
    # Start video capture
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_placeholder = st.empty()
    table_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_np = np.array(frame)

        # Convert image to tensor and make detections
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        # Extract the number of detections and reshape the outputs
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # Convert detection classes to integer
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        # Visualize detection boxes
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=0.8,
            agnostic_mode=False,
        )

        # Extract and display the number plate cutout
        for i in range(num_detections):
            if detections['detection_scores'][i] > 0.8:
                ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
                (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
                cutout = image_np[int(top):int(bottom), int(left):int(right)]
                
                # Detect text from number plate using EasyOCR
                result = reader.readtext(cutout)
                detected_text = ' '.join([text[1] for text in result])
                
                # Add the new detection to the global DataFrame
                new_row = pd.DataFrame({
                    'Detected Text': [detected_text],
                    'Date': [datetime.now().strftime("%Y-%m-%d")],
                    'Time': [datetime.now().strftime("%H:%M:%S")]
                })
                df = pd.concat([df, new_row], ignore_index=True)

        # Update the video feed
        frame_placeholder.image(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))

        # Update the table
        if not df.empty:
            table_placeholder.table(df)
            
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

elif option == "Download Results":
    if df.empty:
        st.warning("No results available for download yet.")
    else:
        # Add a download button for the table
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Detected Results",
            data=csv,
            file_name='detected_results.csv',
            mime='text/csv',
            key="download_button_unique_key"
        )
