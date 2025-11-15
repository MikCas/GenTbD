#!/usr/bin/env python3
"""
Step-by-step MPS profiler to identify performance bottlenecks.

This script profiles each stage of the detection pipeline separately:
1. Model loading
2. Warmup (shader compilation)
3. Preprocessing (BGR→RGB, normalize, tensor conversion)
4. CPU→MPS transfer
5. Model inference
6. MPS→CPU transfer
7. Postprocessing

Usage:
    python profile_mps.py --video data/TownCent.mp4
    python profile_mps.py --model resnet50  # Test different models
"""

import sys
sys.path.insert(0, '.')

import cv2
import time
import torch
import numpy as np
import argparse
from src.detecting.detectors import ObjectDetector

def profile_step(name, func, *args, **kwargs):
    """Profile a single step and return result + timing."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")

    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start

    print(f"✓ Completed in: {elapsed*1000:.1f}ms ({elapsed:.3f}s)")

    return result, elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='data/TownCent.mp4')
    parser.add_argument('--model', default='mobilenet', choices=['mobilenet', 'resnet50', 'retinanet'])
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--max-dimension', type=int, default=640)
    args = parser.parse_args()

    print("="*60)
    print("MPS STEP-BY-STEP PROFILER")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Max dimension: {args.max_dimension}")
    print("="*60)

    timings = {}

    # ========================================================================
    # STEP 1: Load video frame
    # ========================================================================
    def load_frame():
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Failed to read video")
        print(f"  Frame shape: {frame.shape}")
        print(f"  Frame dtype: {frame.dtype}")
        return frame

    frame, timings['load_frame'] = profile_step("Load video frame", load_frame)

    # ========================================================================
    # STEP 2: Resize frame
    # ========================================================================
    def resize_frame(frame):
        h, w = frame.shape[:2]
        if max(h, w) > args.max_dimension:
            scale = args.max_dimension / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))
            print(f"  Original: {w}x{h}")
            print(f"  Resized: {new_w}x{new_h}")
            print(f"  Scale: {scale:.3f}")
            return resized
        return frame

    frame_resized, timings['resize'] = profile_step("Resize frame", resize_frame, frame)

    # ========================================================================
    # STEP 3: Create detector (model loading)
    # ========================================================================
    def create_detector():
        print(f"  Loading {args.model} model...")
        detector = ObjectDetector(
            model=args.model,
            device=args.device,
            conf_threshold=0.5,
            min_size=args.max_dimension,
            max_size=args.max_dimension
        )
        print(f"  Device: {detector.device}")
        print(f"  Model loaded: {type(detector.model).__name__}")
        return detector

    detector, timings['model_load'] = profile_step("Create detector", create_detector)

    # ========================================================================
    # STEP 4: First inference (includes shader compilation for MPS)
    # ========================================================================
    def first_inference(detector, frame):
        print(f"  Running first inference (includes warmup)...")
        detections = detector.detect(frame)
        print(f"  Detections: {len(detections)}")
        return detections

    detections, timings['first_inference'] = profile_step(
        "First inference (warmup + shader compilation)",
        first_inference,
        detector,
        frame_resized
    )

    # ========================================================================
    # STEP 5: Subsequent inferences (no shader compilation)
    # ========================================================================
    print("\n" + "="*60)
    print("STEP: Running 5 more inferences to measure steady-state")
    print("="*60)

    subsequent_times = []
    for i in range(5):
        start = time.time()
        detections = detector.detect(frame_resized)
        elapsed = time.time() - start
        subsequent_times.append(elapsed * 1000)
        print(f"  Inference {i+1}: {elapsed*1000:.1f}ms ({len(detections)} detections)")

    avg_subsequent = np.mean(subsequent_times)
    std_subsequent = np.std(subsequent_times)
    timings['steady_state_avg'] = avg_subsequent / 1000
    timings['steady_state_std'] = std_subsequent / 1000

    print(f"\n✓ Average: {avg_subsequent:.1f}ms ± {std_subsequent:.1f}ms")
    print(f"✓ Expected FPS: {1000/avg_subsequent:.1f}")

    # ========================================================================
    # STEP 6: Profile individual pipeline stages
    # ========================================================================
    print("\n" + "="*60)
    print("DETAILED PIPELINE PROFILING")
    print("="*60)

    # Preprocessing
    def preprocess_only(image):
        tensor = detector.preprocess(image)
        print(f"  Input shape: {image.shape}")
        print(f"  Output shape: {tensor.shape}")
        print(f"  Output dtype: {tensor.dtype}")
        return tensor

    tensor, timings['preprocess'] = profile_step(
        "Preprocessing (BGR→RGB, normalize, to tensor)",
        preprocess_only,
        frame_resized
    )

    # Transfer to device
    def transfer_to_device(tensor):
        # Add batch dimension
        tensor_batch = tensor.unsqueeze(0)
        print(f"  Tensor shape: {tensor_batch.shape}")
        print(f"  Moving to {detector.device}...")
        tensor_device = tensor_batch.to(detector.device)
        # Force synchronization
        if args.device == 'mps':
            torch.mps.synchronize()
        elif args.device == 'cuda':
            torch.cuda.synchronize()
        print(f"  Tensor device: {tensor_device.device}")
        return tensor_device

    tensor_device, timings['transfer_to_device'] = profile_step(
        "Transfer tensor to device",
        transfer_to_device,
        tensor
    )

    # Model inference (on device)
    def inference_only(tensor_device):
        print(f"  Running model forward pass...")
        with torch.no_grad():
            output = detector.model(tensor_device)
        # Force synchronization
        if args.device == 'mps':
            torch.mps.synchronize()
        elif args.device == 'cuda':
            torch.cuda.synchronize()
        print(f"  Output boxes: {len(output[0]['boxes'])}")
        print(f"  Output scores: {len(output[0]['scores'])}")
        return output[0]

    output, timings['inference'] = profile_step(
        "Model inference (forward pass only)",
        inference_only,
        tensor_device
    )

    # Transfer back to CPU
    def transfer_to_cpu(output):
        print(f"  Boxes device: {output['boxes'].device}")
        print(f"  Transferring to CPU...")
        boxes_cpu = output['boxes'].cpu()
        scores_cpu = output['scores'].cpu()
        labels_cpu = output['labels'].cpu()
        print(f"  Boxes CPU: {boxes_cpu.device}")
        return {
            'boxes': boxes_cpu,
            'scores': scores_cpu,
            'labels': labels_cpu
        }

    output_cpu, timings['transfer_to_cpu'] = profile_step(
        "Transfer results back to CPU",
        transfer_to_cpu,
        output
    )

    # Postprocessing
    def postprocess_only(output, image_shape):
        print(f"  Filtering detections...")
        print(f"  Confidence threshold: {detector.conf_threshold}")
        detections = detector.postprocess(output, image_shape)
        print(f"  Final detections: {len(detections)}")
        return detections

    final_detections, timings['postprocess'] = profile_step(
        "Postprocessing (filter + create Detection objects)",
        postprocess_only,
        output_cpu,
        frame_resized.shape[:2]
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)

    # One-time costs
    print("\n--- One-time Costs ---")
    print(f"Model loading:          {timings['model_load']*1000:>8.1f}ms")
    print(f"First inference:        {timings['first_inference']*1000:>8.1f}ms  (includes shader compilation)")

    # Per-frame costs (steady state)
    print("\n--- Per-Frame Costs (Steady State) ---")
    print(f"Preprocessing:          {timings['preprocess']*1000:>8.1f}ms")
    print(f"CPU→Device transfer:    {timings['transfer_to_device']*1000:>8.1f}ms")
    print(f"Model inference:        {timings['inference']*1000:>8.1f}ms")
    print(f"Device→CPU transfer:    {timings['transfer_to_cpu']*1000:>8.1f}ms")
    print(f"Postprocessing:         {timings['postprocess']*1000:>8.1f}ms")

    pipeline_total = (timings['preprocess'] +
                     timings['transfer_to_device'] +
                     timings['inference'] +
                     timings['transfer_to_cpu'] +
                     timings['postprocess'])

    print(f"\nPipeline total:         {pipeline_total*1000:>8.1f}ms")
    print(f"Measured avg:           {timings['steady_state_avg']:>8.1f}ms")
    print(f"Difference:             {abs(pipeline_total*1000 - timings['steady_state_avg']):>8.1f}ms")

    # ========================================================================
    # BOTTLENECK ANALYSIS
    # ========================================================================
    print("\n" + "="*60)
    print("BOTTLENECK ANALYSIS")
    print("="*60)

    # Calculate percentages
    stages = {
        'Preprocessing': timings['preprocess'],
        'CPU→Device': timings['transfer_to_device'],
        'Inference': timings['inference'],
        'Device→CPU': timings['transfer_to_cpu'],
        'Postprocessing': timings['postprocess']
    }

    sorted_stages = sorted(stages.items(), key=lambda x: x[1], reverse=True)

    print("\nTime breakdown (sorted by duration):")
    for stage, duration in sorted_stages:
        percentage = (duration / pipeline_total) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"{stage:20s} {duration*1000:>7.1f}ms ({percentage:>5.1f}%) {bar}")

    # Identify bottleneck
    bottleneck, bottleneck_time = sorted_stages[0]
    print(f"\n⚠️  BOTTLENECK: {bottleneck} ({bottleneck_time*1000:.1f}ms, {(bottleneck_time/pipeline_total)*100:.1f}%)")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if bottleneck == 'Inference':
        print("\n✓ Expected: Inference should be the bottleneck")
        print("  This is normal and indicates the pipeline is well-optimized.")

        if args.device == 'mps' and timings['inference'] > 0.1:  # >100ms
            print("\n⚠️  However, MPS inference is slower than expected:")
            print(f"    Current: {timings['inference']*1000:.1f}ms")
            print(f"    Expected for {args.model}: 20-50ms")
            print("\n  Possible issues:")
            print("    1. MPS is falling back to CPU (check model compatibility)")
            print("    2. Model has unsupported operations on MPS")
            print("    3. System is thermal throttling")
            print("    4. Background apps using GPU")

    elif bottleneck == 'CPU→Device' or bottleneck == 'Device→CPU':
        print("\n⚠️  Transfer overhead is too high!")
        print("  Recommendations:")
        print("    1. Keep more operations on device")
        print("    2. Use tensor pinning for faster transfers")
        print("    3. Batch process multiple frames")

    elif bottleneck == 'Preprocessing':
        print("\n⚠️  Preprocessing is the bottleneck!")
        print("  Recommendations:")
        print("    1. Move preprocessing to GPU if possible")
        print("    2. Use torch.jit.script for preprocessing")
        print("    3. Optimize cv2.resize (use INTER_NEAREST)")

    elif bottleneck == 'Postprocessing':
        print("\n⚠️  Postprocessing is the bottleneck!")
        print("  Recommendations:")
        print("    1. Keep filtering on device (don't call .cpu() early)")
        print("    2. Reduce number of detections (higher conf threshold)")
        print("    3. Optimize BoundingBox creation")

    # Check if MPS is actually being used
    print("\n" + "="*60)
    print("DEVICE CHECK")
    print("="*60)

    print(f"\nRequested device: {args.device}")
    print(f"Detector device: {detector.device}")
    print(f"Model device: {next(detector.model.parameters()).device}")

    if args.device == 'mps':
        print(f"\nMPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")

        if timings['inference'] > 0.1:
            print("\n⚠️  WARNING: Inference is suspiciously slow for MPS")
            print("   This might indicate MPS fallback to CPU.")
            print("   Check PyTorch version and MPS support.")

    print("\n" + "="*60)

if __name__ == '__main__':
    main()
