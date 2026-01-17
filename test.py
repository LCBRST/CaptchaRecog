import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import random
import pickle
import time
import warnings
import tensorflow as tf
import matplotlib.patches as patches
from matplotlib import cm

# Suppress warnings
warnings.filterwarnings('ignore')

class CaptchaModelTester:
    def __init__(self):
        # Character set
        self.characters = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ'
        self.char2label = {char: i for i, char in enumerate(self.characters)}
        self.label2char = {i: char for i, char in enumerate(self.characters)}
        self.num_classes = len(self.characters)
        
        # Store loaded models
        self.knn_model = None
        self.svm_model = None
        self.rf_model = None
        self.cnn_model = None
        
        # Test results storage
        self.test_results = {}
        
        # Set matplotlib
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("="*60)
        print("CAPTCHA MODEL TESTING SYSTEM")
        print("="*60)
        print(f"Character Set: {self.characters}")
        print(f"Number of Classes: {self.num_classes}")
    
    def _load_font(self, size=28):
        """Helper method to safely load font with fallback"""
        font = None
        font_specs = [
            ("arial.ttf", size),
            ("DejaVuSans.ttf", size),
            ("LiberationSans-Regular.ttf", size),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size),
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)
        ]
        
        for font_name, font_size in font_specs:
            try:
                font = ImageFont.truetype(font_name, font_size)
                print(f"  Using font: {font_name}")
                break
            except (OSError, IOError):
                continue
        
        if font is None:
            print("  Using default font")
            font = ImageFont.load_default()
        
        return font
    
    def load_models(self):
        """Load all trained models"""
        print("\n" + "="*60)
        print("LOADING MODELS...")
        print("="*60)
        
        models_loaded = 0
        model_paths = {
            'KNN': 'models/knn.pkl',
            'SVM': 'models/svm.pkl', 
            'Random Forest': 'models/random_forest.pkl',
            'CNN': 'models/cnn_best.h5'
        }
        
        for model_name, model_path in model_paths.items():
            try:
                if model_name == 'CNN':
                    if os.path.exists(model_path):
                        self.cnn_model = tf.keras.models.load_model(model_path)
                        print(f"✓ CNN model loaded successfully")
                        models_loaded += 1
                    else:
                        print(f"✗ CNN model file not found: {model_path}")
                else:
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            if model_name == 'KNN':
                                self.knn_model = pickle.load(f)
                            elif model_name == 'SVM':
                                self.svm_model = pickle.load(f)
                            elif model_name == 'Random Forest':
                                self.rf_model = pickle.load(f)
                        print(f"✓ {model_name} model loaded successfully")
                        models_loaded += 1
                    else:
                        print(f"✗ {model_name} model file not found: {model_path}")
            except Exception as e:
                print(f"✗ Failed to load {model_name} model: {e}")
        
        print(f"\nTotal models loaded: {models_loaded}/4")
        return models_loaded
    
    def generate_test_captcha(self, num_samples=20, complexity='mixed'):
        """Generate test CAPTCHA samples"""
        print(f"\nGenerating {num_samples} {complexity} complexity test samples...")
        
        test_samples = []
        complexities = ['simple', 'medium', 'hard']
        
        for i in range(num_samples):
            if complexity == 'mixed':
                comp = random.choice(complexities)
            else:
                comp = complexity
            
            # Generate random text
            text = ''.join(random.choice(self.characters) for _ in range(4))
            
            # Generate CAPTCHA image
            width, height = 120, 50
            image = Image.new('RGB', (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            
            # Load font using helper method
            font = self._load_font(28)
            
            char_width = 25
            total_width = 4 * char_width
            start_x = (width - total_width) // 2
            start_y = (height - 28) // 2
            
            for j, char in enumerate(text):
                x = start_x + j * char_width
                y = start_y
                
                if comp == 'medium':
                    angle = random.randint(-10, 10)
                    char_image = Image.new('RGBA', (30, 40), (255, 255, 255, 0))
                    char_draw = ImageDraw.Draw(char_image)
                    char_draw.text((5, 5), char, font=font, fill=(0, 0, 0))
                    rotated_char = char_image.rotate(angle, expand=1)
                    image.paste(rotated_char, (x, y), rotated_char)
                elif comp == 'hard':
                    angle = random.randint(-15, 15)
                    char_image = Image.new('RGBA', (30, 40), (255, 255, 255, 0))
                    char_draw = ImageDraw.Draw(char_image)
                    char_draw.text((5, 5), char, font=font, fill=(0, 0, 0))
                    rotated_char = char_image.rotate(angle, expand=1)
                    image.paste(rotated_char, (x, y), rotated_char)
                else:
                    draw.text((x, y), char, font=font, fill=(0, 0, 0))
            
            if comp != 'simple':
                for _ in range(random.randint(2, 4)):
                    x1 = random.randint(0, width)
                    y1 = random.randint(0, height)
                    x2 = random.randint(0, width)
                    y2 = random.randint(0, height)
                    draw.line([(x1, y1), (x2, y2)], 
                             fill=(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)), 
                             width=1)
                
                for _ in range(random.randint(20, 40)):
                    x = random.randint(0, width - 1)
                    y = random.randint(0, height - 1)
                    draw.point((x, y), 
                              fill=(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)))
            
            # Convert to numpy array
            image_np = np.array(image)
            test_samples.append({
                'image': image_np,
                'text': text,
                'complexity': comp,
                'id': i
            })
            
            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        print(f"Test sample generation completed!")
        return test_samples
    
    def preprocess_for_ml(self, image, text):
        """Preprocess single CAPTCHA for traditional ML"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        height, width = binary.shape
        char_width = width // 4
        
        features_list = []
        labels_list = []
        
        for i in range(4):
            x_start = i * char_width
            x_end = (i + 1) * char_width
            
            char_img = binary[:, x_start:x_end]
            resized = cv2.resize(char_img, (20, 20))
            features = resized.flatten() / 255.0
            
            features_list.append(features)
            labels_list.append(self.char2label[text[i]])
        
        return features_list, labels_list
    
    def preprocess_for_cnn(self, image, text):
        """Preprocess single CAPTCHA for CNN"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        height, width = binary.shape
        char_width = width // 4
        
        char_images = []
        char_labels = []
        
        for i in range(4):
            x_start = i * char_width
            x_end = (i + 1) * char_width
            
            char_img = binary[:, x_start:x_end]
            resized = cv2.resize(char_img, (28, 28))
            char_img_processed = resized.reshape(28, 28, 1) / 255.0
            
            char_images.append(char_img_processed)
            char_labels.append(self.char2label[text[i]])
        
        return char_images, char_labels
    
    def predict_with_model(self, image, text, model_name):
        """Predict using specified model"""
        try:
            if model_name == 'CNN':
                if self.cnn_model is None:
                    return None, 0, False
                
                char_images, _ = self.preprocess_for_cnn(image, text)
                char_images = np.array(char_images)
                
                predicted_text = ''
                probabilities = []
                for i, char_img in enumerate(char_images):
                    char_img = char_img.reshape(1, 28, 28, 1)
                    pred_prob = self.cnn_model.predict(char_img, verbose=0)
                    pred_label = np.argmax(pred_prob)
                    predicted_text += self.label2char[pred_label]
                    probabilities.append(pred_prob[0][pred_label])
                
                avg_prob = np.mean(probabilities)
                
            else:  # Traditional models
                if model_name == 'KNN':
                    model = self.knn_model
                elif model_name == 'SVM':
                    model = self.svm_model
                elif model_name == 'Random Forest':
                    model = self.rf_model
                else:
                    return None, 0, False
                
                if model is None:
                    return None, 0, False
                
                features_ml, _ = self.preprocess_for_ml(image, text)
                features_ml = np.array(features_ml)
                
                predicted_text = ''
                probabilities = []
                for i, features in enumerate(features_ml):
                    features = features.reshape(1, -1)
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(features)[0]
                        pred_label = np.argmax(prob)
                        predicted_text += self.label2char[pred_label]
                        probabilities.append(prob[pred_label])
                    else:
                        pred_label = model.predict(features)[0]
                        predicted_text += self.label2char[pred_label]
                        probabilities.append(1.0)  # If no probability output, set to 1.0
                
                avg_prob = np.mean(probabilities)
            
            correct = predicted_text == text
            return predicted_text, avg_prob, correct
            
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
            return None, 0, False
    
    def test_all_models_on_samples(self, test_samples):
        """Test all models on test samples"""
        print("\n" + "="*60)
        print("STARTING MODEL TESTING...")
        print("="*60)
        
        models_to_test = []
        if self.knn_model: models_to_test.append('KNN')
        if self.svm_model: models_to_test.append('SVM')
        if self.rf_model: models_to_test.append('Random Forest')
        if self.cnn_model: models_to_test.append('CNN')
        
        if not models_to_test:
            print("No models available for testing!")
            return
        
        # Initialize result storage
        results = {model: {'total': 0, 'correct': 0, 'time': 0, 'details': []} for model in models_to_test}
        complexity_results = {}
        
        print(f"\nTesting {len(test_samples)} samples with models: {', '.join(models_to_test)}")
        
        for idx, sample in enumerate(test_samples):
            print(f"\nSample {idx + 1}/{len(test_samples)}: True text='{sample['text']}', Complexity={sample['complexity']}")
            
            for model_name in models_to_test:
                start_time = time.time()
                predicted_text, confidence, correct = self.predict_with_model(
                    sample['image'], sample['text'], model_name
                )
                end_time = time.time()
                
                if predicted_text is not None:
                    results[model_name]['total'] += 1
                    results[model_name]['correct'] += 1 if correct else 0
                    results[model_name]['time'] += (end_time - start_time)
                    
                    # Record detailed results
                    results[model_name]['details'].append({
                        'sample_id': idx,
                        'true_text': sample['text'],
                        'predicted': predicted_text,
                        'correct': correct,
                        'confidence': confidence,
                        'time': end_time - start_time,
                        'complexity': sample['complexity']
                    })
                    
                    status = "✓" if correct else "✗"
                    print(f"  {model_name:15s}: {predicted_text} (Confidence: {confidence:.3f}, Time: {(end_time-start_time)*1000:.1f}ms) {status}")
                else:
                    print(f"  {model_name:15s}: Prediction failed")
        
        # Calculate statistics
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        table_data = []
        for model_name in models_to_test:
            if results[model_name]['total'] > 0:
                accuracy = results[model_name]['correct'] / results[model_name]['total']
                avg_time = results[model_name]['time'] / results[model_name]['total'] * 1000
                table_data.append([
                    model_name,
                    f"{accuracy:.4f}",
                    f"{results[model_name]['correct']}/{results[model_name]['total']}",
                    f"{avg_time:.1f}ms"
                ])
        
        # Print table without tabulate
        self._print_table(table_data, ['Model', 'Accuracy', 'Correct/Total', 'Avg Time'])
        
        # Complexity analysis
        print("\n" + "="*60)
        print("COMPLEXITY ANALYSIS")
        print("="*60)
        
        complexities = ['simple', 'medium', 'hard']
        complexity_table = []
        
        for complexity in complexities:
            complexity_samples = [s for s in test_samples if s['complexity'] == complexity]
            if not complexity_samples:
                continue
            
            complexity_results[complexity] = {}
            for model_name in models_to_test:
                model_details = results[model_name]['details']
                complexity_details = [d for d in model_details if d['complexity'] == complexity]
                if complexity_details:
                    correct_count = sum(1 for d in complexity_details if d['correct'])
                    total_count = len(complexity_details)
                    accuracy = correct_count / total_count if total_count > 0 else 0
                    complexity_results[complexity][model_name] = accuracy
                    complexity_table.append([
                        complexity,
                        model_name,
                        f"{accuracy:.4f}",
                        f"{correct_count}/{total_count}"
                    ])
        
        self._print_table(complexity_table, ['Complexity', 'Model', 'Accuracy', 'Correct/Total'])
        
        self.test_results = results
        self.complexity_results = complexity_results
        
        return results
    
    def _print_table(self, data, headers):
        """Print table without external dependencies"""
        if not data:
            return
        
        # Calculate column widths
        col_widths = []
        for i in range(len(headers)):
            max_width = len(headers[i])
            for row in data:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)  # Add padding
        
        # Print header
        header_row = ""
        for i, header in enumerate(headers):
            header_row += f"{header:<{col_widths[i]}}"
        print(header_row)
        print("-" * len(header_row))
        
        # Print data rows
        for row in data:
            row_str = ""
            for i, cell in enumerate(row):
                row_str += f"{str(cell):<{col_widths[i]}}"
            print(row_str)
    
    def visualize_results(self, test_samples):
        """Visualize test results"""
        if not hasattr(self, 'test_results'):
            print("No test results available for visualization!")
            return
        
        models = [model for model in self.test_results.keys() if self.test_results[model]['total'] > 0]
        
        if not models:
            return
        
        # Create visualization charts
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Model Performance Comparison
        ax1 = plt.subplot(2, 3, 1)
        accuracies = [self.test_results[model]['correct'] / self.test_results[model]['total'] for model in models]
        times = [self.test_results[model]['time'] / self.test_results[model]['total'] * 1000 for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Inference Time Comparison
        ax2 = plt.subplot(2, 3, 2)
        bars2 = ax2.bar(models, times, color='lightcoral')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Average Inference Time (ms)')
        ax2.set_title('Model Inference Time Comparison')
        ax2.set_xticklabels(models, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, t in zip(bars2, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f'{t:.1f}ms', ha='center', va='bottom')
        
        # 3. Accuracy by Complexity
        ax3 = plt.subplot(2, 3, 3)
        complexities = sorted(set([d['complexity'] for model in models for d in self.test_results[model]['details']]))
        
        if complexities:
            x = np.arange(len(complexities))
            width = 0.2
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, model in enumerate(models[:4]):  # Show max 4 models
                model_accuracies = []
                for comp in complexities:
                    comp_details = [d for d in self.test_results[model]['details'] if d['complexity'] == comp]
                    if comp_details:
                        correct = sum(1 for d in comp_details if d['correct'])
                        model_accuracies.append(correct / len(comp_details))
                    else:
                        model_accuracies.append(0)
                
                ax3.bar(x + (i-1.5)*width, model_accuracies, width, label=model, color=colors[i % len(colors)])
            
            ax3.set_xlabel('CAPTCHA Complexity')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Model Performance by Complexity')
            ax3.set_xticks(x)
            ax3.set_xticklabels(complexities)
            ax3.set_ylim(0, 1.1)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Example CAPTCHA Recognition Results
        ax4 = plt.subplot(2, 3, 4)
        if test_samples:
            sample_idx = 0
            sample = test_samples[sample_idx]
            ax4.imshow(sample['image'])
            ax4.set_title(f"Example CAPTCHA (ID: {sample_idx})")
            ax4.axis('off')
            
            # Show true text
            ax4.text(0.5, -0.1, f"True Text: {sample['text']}", 
                    transform=ax4.transAxes, ha='center', fontsize=12, fontweight='bold')
            
            # Show prediction results
            y_pos = 0.85
            ax4.text(-0.1, y_pos, "Model Predictions:", transform=ax4.transAxes, 
                    ha='right', fontsize=10, fontweight='bold')
            
            y_offset = 0.12
            for i, model in enumerate(models):
                if sample_idx < len(self.test_results[model]['details']):
                    detail = self.test_results[model]['details'][sample_idx]
                    color = 'green' if detail['correct'] else 'red'
                    ax4.text(-0.1, y_pos - (i+1)*y_offset, f"{model}:", 
                            transform=ax4.transAxes, ha='right', fontsize=9)
                    ax4.text(0.1, y_pos - (i+1)*y_offset, detail['predicted'], 
                            transform=ax4.transAxes, ha='left', fontsize=9, 
                            color=color, fontweight='bold')
        
        # 5. Model Confidence Distribution
        ax5 = plt.subplot(2, 3, 5)
        all_confidences = []
        model_labels = []
        
        for model in models:
            confidences = [d['confidence'] for d in self.test_results[model]['details']]
            all_confidences.append(confidences)
            model_labels.append(model)
        
        if all_confidences:
            bp = ax5.boxplot(all_confidences, labels=model_labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(models)]):
                patch.set_facecolor(color)
            ax5.set_xlabel('Model')
            ax5.set_ylabel('Confidence')
            ax5.set_title('Model Prediction Confidence Distribution')
            ax5.grid(True, alpha=0.3)
        
        # 6. Character Recognition Error Analysis
        ax6 = plt.subplot(2, 3, 6)
        if hasattr(self, 'test_results') and test_samples:
            all_errors = {model: {'total_chars': 0, 'errors': 0} for model in models}
            
            for model in models:
                for detail in self.test_results[model]['details']:
                    all_errors[model]['total_chars'] += 4
                    for true_char, pred_char in zip(detail['true_text'], detail['predicted']):
                        if true_char != pred_char:
                            all_errors[model]['errors'] += 1
            
            error_rates = []
            for model in models:
                if all_errors[model]['total_chars'] > 0:
                    error_rate = all_errors[model]['errors'] / all_errors[model]['total_chars']
                else:
                    error_rate = 0
                error_rates.append(error_rate)
            
            bars = ax6.bar(models, error_rates, color='salmon')
            ax6.set_xlabel('Model')
            ax6.set_ylabel('Character Error Rate')
            ax6.set_title('Character-Level Error Rate')
            ax6.set_xticklabels(models, rotation=45)
            ax6.grid(True, alpha=0.3, axis='y')
            
            for bar, rate in zip(bars, error_rates):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/model_testing_results.png', dpi=120, bbox_inches='tight')
        plt.show()
        
        # Output detailed error analysis
        print("\n" + "="*60)
        print("DETAILED ERROR ANALYSIS")
        print("="*60)
        
        for model in models:
            print(f"\n{model} Model:")
            incorrect_samples = [d for d in self.test_results[model]['details'] if not d['correct']]
            if incorrect_samples:
                print(f"  Incorrect samples ({len(incorrect_samples)}):")
                for i, detail in enumerate(incorrect_samples[:5]):  # Show first 5 errors
                    print(f"    Sample {detail['sample_id']}: True='{detail['true_text']}', "
                          f"Pred='{detail['predicted']}', Confidence={detail['confidence']:.3f}")
            else:
                print("  All samples recognized correctly!")
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n" + "="*60)
        print("INTERACTIVE TESTING MODE")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("1. Test random CAPTCHA")
            print("2. Enter custom CAPTCHA text")
            print("3. Return to main menu")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '1':
                complexity = input("Select complexity (simple/medium/hard, default mixed): ").strip()
                if complexity not in ['simple', 'medium', 'hard']:
                    complexity = 'mixed'
                
                text = None
                
            elif choice == '2':
                text = input("Enter 4-character CAPTCHA text (use 23456789ABCDEFGHJKLMNPQRSTUVWXYZ): ").strip().upper()
                if len(text) != 4 or not all(c in self.characters for c in text):
                    print("Invalid input! Must be 4 characters from the specified set")
                    continue
                
                complexity = input("Select complexity (simple/medium/hard, default mixed): ").strip()
                if complexity not in ['simple', 'medium', 'hard']:
                    complexity = 'mixed'
            
            elif choice == '3':
                break
            else:
                print("Invalid choice!")
                continue
            
            # Generate CAPTCHA
            if complexity == 'mixed':
                comp = random.choice(['simple', 'medium', 'hard'])
            else:
                comp = complexity
            
            if text is None:
                text = ''.join(random.choice(self.characters) for _ in range(4))
            
            width, height = 120, 50
            image = Image.new('RGB', (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            
            # Use helper method to load font
            font = self._load_font(28)
            
            char_width = 25
            total_width = 4 * char_width
            start_x = (width - total_width) // 2
            start_y = (height - 28) // 2
            
            for i, char in enumerate(text):
                x = start_x + i * char_width
                y = start_y
                
                if comp == 'medium':
                    angle = random.randint(-10, 10)
                    char_image = Image.new('RGBA', (30, 40), (255, 255, 255, 0))
                    char_draw = ImageDraw.Draw(char_image)
                    char_draw.text((5, 5), char, font=font, fill=(0, 0, 0))
                    rotated_char = char_image.rotate(angle, expand=1)
                    image.paste(rotated_char, (x, y), rotated_char)
                elif comp == 'hard':
                    angle = random.randint(-15, 15)
                    char_image = Image.new('RGBA', (30, 40), (255, 255, 255, 0))
                    char_draw = ImageDraw.Draw(char_image)
                    char_draw.text((5, 5), char, font=font, fill=(0, 0, 0))
                    rotated_char = char_image.rotate(angle, expand=1)
                    image.paste(rotated_char, (x, y), rotated_char)
                else:
                    draw.text((x, y), char, font=font, fill=(0, 0, 0))
            
            if comp != 'simple':
                for _ in range(random.randint(2, 4)):
                    x1 = random.randint(0, width)
                    y1 = random.randint(0, height)
                    x2 = random.randint(0, width)
                    y2 = random.randint(0, height)
                    draw.line([(x1, y1), (x2, y2)], 
                             fill=(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)), 
                             width=1)
                
                for _ in range(random.randint(20, 40)):
                    x = random.randint(0, width - 1)
                    y = random.randint(0, height - 1)
                    draw.point((x, y), 
                              fill=(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)))
            
            image_np = np.array(image)
            
            # Display CAPTCHA
            plt.figure(figsize=(4, 2))
            plt.imshow(image_np)
            plt.title(f"CAPTCHA: {text} (Complexity: {comp})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Predict with all models
            print(f"\nModel predictions (True text: {text}):")
            models_to_test = []
            if self.knn_model: models_to_test.append('KNN')
            if self.svm_model: models_to_test.append('SVM')
            if self.rf_model: models_to_test.append('Random Forest')
            if self.cnn_model: models_to_test.append('CNN')
            
            for model_name in models_to_test:
                start_time = time.time()
                predicted_text, confidence, correct = self.predict_with_model(image_np, text, model_name)
                end_time = time.time()
                
                if predicted_text is not None:
                    status = "✓" if correct else "✗"
                    print(f"  {model_name:15s}: {predicted_text} (Confidence: {confidence:.3f}, "
                          f"Time: {(end_time-start_time)*1000:.1f}ms) {status}")
    
    def run_complete_test(self):
        """Run complete testing pipeline"""
        print("="*60)
        print("CAPTCHA MODEL TESTING SYSTEM")
        print("="*60)
        
        # Create directories
        os.makedirs('results', exist_ok=True)
        
        # 1. Load models
        models_loaded = self.load_models()
        if models_loaded == 0:
            print("No models to load, please train models first!")
            return
        
        while True:
            print("\n" + "="*60)
            print("MAIN MENU")
            print("="*60)
            print("1. Batch Testing (auto-generated samples)")
            print("2. Interactive Testing (single test)")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '1':
                # Batch testing
                num_samples = input("Enter number of test samples (default 20): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 20
                
                complexity = input("Select complexity (simple/medium/hard/mixed, default mixed): ").strip()
                if complexity not in ['simple', 'medium', 'hard', 'mixed']:
                    complexity = 'mixed'
                
                # Generate test samples
                test_samples = self.generate_test_captcha(num_samples, complexity)
                
                # Test all models
                results = self.test_all_models_on_samples(test_samples)
                
                # Visualize results
                if results:
                    self.visualize_results(test_samples)
                    
                    # Save results to file
                    with open('results/test_results_summary.txt', 'w') as f:
                        f.write("CAPTCHA MODEL TESTING RESULTS SUMMARY\n")
                        f.write("="*50 + "\n\n")
                        for model, stats in results.items():
                            if stats['total'] > 0:
                                accuracy = stats['correct'] / stats['total']
                                f.write(f"{model}:\n")
                                f.write(f"  Accuracy: {accuracy:.4f}\n")
                                f.write(f"  Correct: {stats['correct']}/{stats['total']}\n")
                                f.write(f"  Total time: {stats['time']:.2f}s\n")
                                f.write(f"  Average time: {stats['time']/stats['total']*1000:.1f}ms\n")
                                f.write("\n")
                    
                    print("\nResults saved to results/test_results_summary.txt")
                    print("Charts saved to results/model_testing_results.png")
                
            elif choice == '2':
                # Interactive testing
                self.interactive_test()
                
            elif choice == '3':
                print("Exiting program...")
                break
            else:
                print("Invalid choice!")

def main():
    """Main function"""
    # Create tester
    tester = CaptchaModelTester()
    
    # Run complete test
    tester.run_complete_test()

if __name__ == '__main__':
    main()