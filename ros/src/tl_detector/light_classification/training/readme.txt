# Start tensorboard (http://localhost:6006 to view)
tensorboard --logdir ./summaries &


# Retrain mobilenet (1.0x size, 128x128) for 250 epochs
python retrain.py \
  --bottleneck_dir=./bottlenecks \
  --how_many_training_steps=250 \
  --model_dir=./model \
  --summaries_dir=./summaries \
  --output_graph=../graph.pb \
  --output_labels=../labels.txt \
  --architecture="mobilenet_1.0_128" \
  --image_dir=./samples \
  --print_misclassified_test_images

