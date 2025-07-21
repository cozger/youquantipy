# test_trt_conversion.py
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

def test_model_outputs(onnx_path, trt_path, test_image_path):
    """Compare ONNX and TensorRT outputs"""
    
    # Load test image
    img = cv2.imread(test_image_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 608))  # Note: 608 height based on your model
    
    # Preprocess
    img_float = img.astype(np.float32)
    img_float -= np.array([104, 117, 123], dtype=np.float32)
    img_input = np.expand_dims(img_float, 0)  # Add batch dimension
    
    # Test ONNX
    print("\n=== Testing ONNX Model ===")
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get input name and shape
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    print(f"Input: {input_name}, shape: {input_shape}")
    
    # Get output info
    for i, output in enumerate(ort_session.get_outputs()):
        print(f"Output {i}: {output.name}, shape: {output.shape}")
    
    # Run inference
    ort_outputs = ort_session.run(None, {input_name: img_input})
    
    print("\nONNX Outputs:")
    for i, output in enumerate(ort_outputs):
        print(f"  Output {i}: shape={output.shape}, dtype={output.dtype}")
        print(f"    min={output.min():.3f}, max={output.max():.3f}, mean={output.mean():.3f}")
        if output.shape[-1] == 2:  # Likely scores
            # Apply softmax
            exp_scores = np.exp(output - output.max(axis=-1, keepdims=True))
            softmax_scores = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
            face_scores = softmax_scores[0, :, 1]  # Get face class
            print(f"    Face scores after softmax: min={face_scores.min():.3f}, "
                  f"max={face_scores.max():.3f}, count > 0.5: {np.sum(face_scores > 0.5)}")
    
    return ort_outputs

# Run the test
if __name__ == "__main__":
    onnx_path = "D:/Projects/youquantipy/retinaface.onnx"
    trt_path = "D:/Projects/youquantipy/retinaface.trt"
    test_image ="C:/Users/canoz/OneDrive/Masaüstü/pp.jpg"  # Use any test image
    
    test_model_outputs(onnx_path, trt_path, test_image)