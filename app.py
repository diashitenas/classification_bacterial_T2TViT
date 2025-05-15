from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from vit_pytorch.t2t import T2TViT
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
from torch.nn import functional as F
import time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model Database
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    processing_time = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.predicted_class}>'

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definition
v = T2TViT(
    dim=512,
    image_size=256,
    depth=5,
    heads=8,
    mlp_dim=512,
    num_classes=33,
    t2t_layers=((7, 4), (3, 2), (3, 2))
)

# Load model
model = v
model.load_state_dict(torch.load('model_T2TViT.pth', map_location=device))
model.to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Classes
classes = ['Acinetobacter baumannii', 'Actinomyces israelii', 'Bacteroides fragilis', 'Bifidobacterium spp', 
           'Candida albicans', 'Clostridium perfringens', 'Enterococcus faecalis', 'Enterococcus faecium', 
           'Escherichia coli', 'Fusobacterium', 'Lactobacillus casei', 'Lactobacillus crispatus', 
           'Lactobacillus delbrueckii', 'Lactobacillus gasseri', 'Lactobacillus jensenii', 'Lactobacillus johnsonii', 
           'Lactobacillus paracasei', 'Lactobacillus plantarum', 'Lactobacillus reuteri', 'Lactobacillus rhamnosus', 
           'Lactobacillus salivarius', 'Listeria monocytogenes', 'Micrococcus spp', 'Neisseria gonorrhoeae', 
           'Porphyromonas gingivalis', 'Propionibacterium acnes', 'Proteus', 'Pseudomonas aeruginosa', 
           'Staphylococcus aureus', 'Staphylococcus epidermidis', 'Staphylococcus saprophyticus', 
           'Streptococcus agalactiae', 'Veillonella']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        start_time = time.time()

        if not os.path.exists('static'):
            os.makedirs('static')

        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        image_path = os.path.join('static', filename)
        file.save(image_path)
        
        # Save only filename to database
        db_image_path = filename

        # Process image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            _, predicted_class = torch.max(probabilities, 1)

        processing_time = round(time.time() - start_time, 2)
        predicted_label = classes[predicted_class.item()]
        confidence = probabilities[0][predicted_class.item()].item() * 100

        # Save to database
        prediction = Prediction(
            image_path=db_image_path,
            predicted_class=predicted_label,
            confidence=confidence,
            processing_time=processing_time
        )
        db.session.add(prediction)
        db.session.commit()

        return render_template('result.html',
                             image_path=db_image_path,
                             predicted_class=predicted_label,
                             confidence=f"{confidence:.2f}%",
                             processing_time=f"{processing_time}s",
                             timestamp=datetime.now())

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/history')
def history():
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/delete_prediction/<int:prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    try:
        prediction = Prediction.query.get_or_404(prediction_id)
        
        # Delete image file
        image_path = os.path.join('static', prediction.image_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Delete from database
        db.session.delete(prediction)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)