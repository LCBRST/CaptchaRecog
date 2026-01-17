import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import string
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 忽略警告
warnings.filterwarnings('ignore')

class CompleteCaptchaSystem:
    def __init__(self):
        # 字符集
        self.characters = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ'
        self.char2label = {char: i for i, char in enumerate(self.characters)}
        self.label2char = {i: char for i, char in enumerate(self.characters)}
        self.num_classes = len(self.characters)
        
        # 设置matplotlib
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
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
                break
            except (OSError, IOError):
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        return font
    
    def generate_captcha(self, text=None, complexity='simple'):
        """生成验证码（使用测试代码的改进版本）"""
        if text is None:
            text = ''.join(random.choice(self.characters) for _ in range(4))
        
        width, height = 120, 50
        image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # 使用改进的字体加载方法
        font = self._load_font(28)
        
        char_width = 25
        total_width = 4 * char_width
        start_x = (width - total_width) // 2
        start_y = (height - 28) // 2
        
        for i, char in enumerate(text):
            x = start_x + i * char_width
            y = start_y
            
            if complexity == 'medium':
                # 中等复杂度：添加轻微旋转
                angle = random.randint(-10, 10)
                char_image = Image.new('RGBA', (30, 40), (255, 255, 255, 0))
                char_draw = ImageDraw.Draw(char_image)
                char_draw.text((5, 5), char, font=font, fill=(0, 0, 0))
                rotated_char = char_image.rotate(angle, expand=1)
                image.paste(rotated_char, (x, y), rotated_char)
            elif complexity == 'hard':
                # 高复杂度：添加更多干扰
                angle = random.randint(-15, 15)
                char_image = Image.new('RGBA', (30, 40), (255, 255, 255, 0))
                char_draw = ImageDraw.Draw(char_image)
                char_draw.text((5, 5), char, font=font, fill=(0, 0, 0))
                rotated_char = char_image.rotate(angle, expand=1)
                image.paste(rotated_char, (x, y), rotated_char)
            else:
                # 简单复杂度：直接绘制
                draw.text((x, y), char, font=font, fill=(0, 0, 0))
        
        # 添加干扰（仅对中高复杂度）
        if complexity != 'simple':
            # 添加干扰线
            for _ in range(random.randint(2, 4)):
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                x2 = random.randint(0, width)
                y2 = random.randint(0, height)
                draw.line([(x1, y1), (x2, y2)], 
                         fill=(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)), 
                         width=1)
            
            # 添加噪点
            for _ in range(random.randint(20, 40)):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                draw.point((x, y), 
                          fill=(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)))
        
        return image, text
    
    def generate_dataset(self, num_samples=5000):
        """生成包含不同复杂度的数据集"""
        os.makedirs('data/captchas', exist_ok=True)
        
        complexities = ['simple', 'medium', 'hard']
        labels = []
        
        print(f"Generating {num_samples} CAPTCHA samples...")
        
        for i in range(num_samples):
            # 轮流使用不同复杂度
            complexity = complexities[i % len(complexities)]
            text = ''.join(random.choice(self.characters) for _ in range(4))
            labels.append((text, complexity))
            
            image, _ = self.generate_captcha(text, complexity)
            filename = f'data/captchas/captcha_{i:05d}.png'
            image.save(filename)
            
            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{num_samples}")
        
        # 保存标签
        with open('data/captchas/labels.txt', 'w') as f:
            for i, (text, complexity) in enumerate(labels):
                filename = f'data/captchas/captcha_{i:05d}.png'
                f.write(f'{filename} {text} {complexity}\n')
        
        print("Dataset generation completed!")
    
    def preprocess_for_cnn(self, image_path, text):
        """为CNN预处理数据"""
        # 加载图片
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        
        # 转换为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 二值化
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # 分割字符
        height, width = binary.shape
        char_width = width // 4
        
        char_images = []
        char_labels = []
        
        for i in range(4):
            x_start = i * char_width
            x_end = (i + 1) * char_width
            
            char_img = binary[:, x_start:x_end]
            resized = cv2.resize(char_img, (28, 28))
            
            # 为CNN准备数据（添加通道维度）
            char_img_processed = resized.reshape(28, 28, 1) / 255.0
            
            char_images.append(char_img_processed)
            char_labels.append(self.char2label[text[i]])
        
        return char_images, char_labels
    
    def preprocess_for_ml(self, image_path, text):
        """为传统ML预处理数据"""
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        
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
    
    def load_datasets(self):
        """加载所有数据集"""
        print("\nLoading and processing datasets...")
        
        labels_file = 'data/captchas/labels.txt'
        if not os.path.exists(labels_file):
            print("Error: Labels file not found")
            return None, None, None, None
        
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        
        # 为不同模型准备数据
        X_ml = []  # 传统机器学习特征
        y_ml = []  # 传统机器学习标签
        X_cnn = [] # CNN特征
        y_cnn = [] # CNN标签
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
                
            img_path, text, complexity = parts
            
            if not os.path.exists(img_path):
                continue
            
            try:
                # 为传统ML处理
                features_ml, labels_ml = self.preprocess_for_ml(img_path, text)
                X_ml.extend(features_ml)
                y_ml.extend(labels_ml)
                
                # 为CNN处理
                features_cnn, labels_cnn = self.preprocess_for_cnn(img_path, text)
                X_cnn.extend(features_cnn)
                y_cnn.extend(labels_cnn)
                
            except Exception as e:
                continue
        
        X_ml = np.array(X_ml)
        y_ml = np.array(y_ml)
        X_cnn = np.array(X_cnn)
        y_cnn = np.array(y_cnn)
        
        print(f"Traditional ML dataset: {X_ml.shape[0]} samples")
        print(f"CNN dataset: {X_cnn.shape[0]} samples")
        
        return X_ml, y_ml, X_cnn, y_cnn
    
    def build_cnn_model(self, input_shape, num_classes):
        """构建CNN模型"""
        model = models.Sequential([
            # 第一卷积层
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第二卷积层
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第三卷积层
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Dropout(0.25),
            
            # 全连接层
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train_cnn_model(self, X_train, y_train, X_val, y_val, input_shape, num_classes):
        """训练CNN模型"""
        print("\nTraining CNN Model...")
        
        # 构建模型
        model = self.build_cnn_model(input_shape, num_classes)
        
        # 编译模型
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # 回调函数
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint('models/cnn_best.h5', save_best_only=True)
        ]
        
        # 训练模型
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def train_all_models(self, X_ml, y_ml, X_cnn, y_cnn):
        """训练所有模型"""
        print("\n" + "="*60)
        print("Training All Models")
        print("="*60)
        
        results = {}
        
        # 训练传统机器学习模型
        if len(X_ml) > 0:
            X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
                X_ml, y_ml, test_size=0.2, random_state=42
            )
            
            print(f"Traditional ML - Training: {X_train_ml.shape[0]}, Test: {X_test_ml.shape[0]}")
            
            # KNN
            print("\nTraining KNN...")
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train_ml, y_train_ml)
            knn_acc = accuracy_score(y_test_ml, knn.predict(X_test_ml))
            results['KNN'] = knn_acc
            print(f"KNN Accuracy: {knn_acc:.4f}")
            
            # SVM
            print("Training SVM...")
            svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            svm.fit(X_train_ml, y_train_ml)
            svm_acc = accuracy_score(y_test_ml, svm.predict(X_test_ml))
            results['SVM'] = svm_acc
            print(f"SVM Accuracy: {svm_acc:.4f}")
            
            # Random Forest
            print("Training Random Forest...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_ml, y_train_ml)
            rf_acc = accuracy_score(y_test_ml, rf.predict(X_test_ml))
            results['Random Forest'] = rf_acc
            print(f"Random Forest Accuracy: {rf_acc:.4f}")
            
            # 保存传统模型
            import pickle
            pickle.dump(knn, open('models/knn.pkl', 'wb'))
            pickle.dump(svm, open('models/svm.pkl', 'wb'))
            pickle.dump(rf, open('models/random_forest.pkl', 'wb'))
        
        # 训练CNN模型
        if len(X_cnn) > 0:
            X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
                X_cnn, y_cnn, test_size=0.2, random_state=42
            )
            
            # 进一步分割出验证集
            X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
                X_train_cnn, y_train_cnn, test_size=0.2, random_state=42
            )
            
            print(f"\nCNN - Training: {X_train_cnn.shape[0]}, Validation: {X_val_cnn.shape[0]}, Test: {X_test_cnn.shape[0]}")
            
            input_shape = X_cnn.shape[1:]
            cnn_model, history = self.train_cnn_model(
                X_train_cnn, y_train_cnn, X_val_cnn, y_val_cnn,
                input_shape, self.num_classes
            )
            
            # 评估CNN
            cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
            results['CNN'] = cnn_test_acc
            print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")
            
            # 保存训练历史用于绘图
            self.cnn_history = history
            self.cnn_model = cnn_model
        
        return results
    
    def plot_comparison(self, results):
        """绘制模型比较图"""
        plt.figure(figsize=(12, 6))
        
        # 模型性能比较
        plt.subplot(1, 2, 1)
        models = list(results.keys())
        accuracies = list(results.values())
        
        bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom')
        
        # CNN训练历史
        if hasattr(self, 'cnn_history'):
            plt.subplot(1, 2, 2)
            history = self.cnn_history.history
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('CNN Training History')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison_complete.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def test_models(self):
        """测试所有模型"""
        print("\n" + "="*60)
        print("Testing Models")
        print("="*60)
        
        # 生成测试验证码
        complexities = ['simple', 'medium', 'hard']
        test_results = {}
        
        for complexity in complexities:
            print(f"\nTesting {complexity} CAPTCHAs:")
            test_results[complexity] = {}
            
            # 生成5个测试样本
            for i in range(5):
                text = ''.join(random.choice(self.characters) for _ in range(4))
                image, _ = self.generate_captcha(text, complexity)
                test_path = f'data/test_{complexity}_{i}.png'
                image.save(test_path)
                
                # 测试每个模型
                for model_name in ['KNN', 'SVM', 'Random Forest', 'CNN']:
                    if model_name not in test_results[complexity]:
                        test_results[complexity][model_name] = []
                    
                    correct = self.test_single_captcha(test_path, text, model_name)
                    test_results[complexity][model_name].append(correct)
            
            # 计算准确率
            for model_name in test_results[complexity]:
                accuracy = np.mean(test_results[complexity][model_name])
                print(f"  {model_name}: {accuracy:.2f}")
        
        return test_results
    
    def test_single_captcha(self, image_path, true_text, model_name):
        """测试单张验证码"""
        try:
            if model_name == 'CNN':
                # CNN测试
                char_images, char_labels = self.preprocess_for_cnn(image_path, true_text)
                char_images = np.array(char_images)
                
                predicted_text = ''
                for i, char_img in enumerate(char_images):
                    char_img = char_img.reshape(1, 28, 28, 1)
                    pred_prob = self.cnn_model.predict(char_img, verbose=0)
                    pred_label = np.argmax(pred_prob)
                    predicted_text += self.label2char[pred_label]
            else:
                # 传统模型测试
                import pickle
                if model_name == 'KNN':
                    model = pickle.load(open('models/knn.pkl', 'rb'))
                elif model_name == 'SVM':
                    model = pickle.load(open('models/svm.pkl', 'rb'))
                elif model_name == 'Random Forest':
                    model = pickle.load(open('models/random_forest.pkl', 'rb'))
                
                features_ml, labels_ml = self.preprocess_for_ml(image_path, true_text)
                features_ml = np.array(features_ml)
                
                predicted_text = ''
                for i, features in enumerate(features_ml):
                    features = features.reshape(1, -1)
                    pred_label = model.predict(features)[0]
                    predicted_text += self.label2char[pred_label]
            
            return predicted_text == true_text
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            return False
    
    def run_complete_pipeline(self):
        """运行完整流程"""
        print("="*70)
        print("Complete CAPTCHA Recognition System with CNN")
        print("="*70)
        
        # 创建目录
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # 1. 生成数据集
        print("\n1. Generating Dataset with Different Complexities")
        self.generate_dataset(num_samples=5000)
        
        # 2. 加载数据
        print("\n2. Loading and Preprocessing Data")
        X_ml, y_ml, X_cnn, y_cnn = self.load_datasets()
        
        if X_ml is None or X_cnn is None:
            print("Data loading failed")
            return
        
        # 3. 训练所有模型
        print("\n3. Training All Models")
        results = self.train_all_models(X_ml, y_ml, X_cnn, y_cnn)
        
        # 4. 可视化结果
        print("\n4. Visualizing Results")
        self.plot_comparison(results)
        
        # 5. 测试模型
        print("\n5. Comprehensive Testing")
        test_results = self.test_models()
        
        # 6. 总结
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        
        best_model = max(results, key=results.get)
        best_accuracy = results[best_model]
        
        print(f"Best Model: {best_model}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"\nAll Model Accuracies:")
        for model, acc in results.items():
            print(f"  {model}: {acc:.4f}")
        
        print(f"\nModels saved to: models/")
        print(f"Results saved to: results/")
        print(f"Dataset location: data/captchas/")

def main():
    """主函数"""
    # 创建系统
    system = CompleteCaptchaSystem()
    
    # 运行完整流程
    system.run_complete_pipeline()

if __name__ == '__main__':
    main()