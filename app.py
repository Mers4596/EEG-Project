import os
import json
import uuid
import numpy as np
import random
from datetime import datetime
from functools import wraps
from scipy.signal import welch

from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model

# --- UYGULAMA KURULUMU ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

USER_DATA_DIR = 'userData'
USER_DATA_FILE = os.path.join(USER_DATA_DIR, 'users.json')
ANALYSIS_HISTORY_DIR = os.path.join(USER_DATA_DIR, 'history')
REPORTED_ERRORS_FILE = os.path.join(USER_DATA_DIR, 'reported_errors.json')
CHATBOT_KNOWLEDGE_FILE = os.path.join(USER_DATA_DIR, 'chatbot_knowledge.json')
os.makedirs(USER_DATA_DIR, exist_ok=True)
os.makedirs(ANALYSIS_HISTORY_DIR, exist_ok=True)

# --- MODEL VE SABİTLER ---
try:
    model = load_model("models/final_model.keras")
except Exception as e:
    print(f"!!! DİKKAT: Sınıflandırma modeli yüklenemedi: {e}. Analiz özelliği çalışmayacak.")
    model = None

CLASSES = ['F', 'N', 'O', 'S', 'Z']
TARGET_LEN = 4096
FS = 256

# --- YARDIMCI FONKSİYONLAR ---

def load_json_file(file_path, default_value=None):
    """Bir JSON dosyasını güvenli bir şekilde okur."""
    if default_value is None:
        default_value = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_value
    return default_value

def save_json_file(file_path, data):
    """Bir Python nesnesini JSON dosyasına yazar."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_users():
    """Kullanıcı verilerini users.json dosyasından yükler."""
    return load_json_file(USER_DATA_FILE, default_value=[])

def save_users(users):
    """Kullanıcı verilerini users.json dosyasına kaydeder."""
    save_json_file(USER_DATA_FILE, users)

def get_user_history_path(user_id):
    """Belirli bir kullanıcının geçmiş dosyasının yolunu döndürür."""
    return os.path.join(ANALYSIS_HISTORY_DIR, f'history_{user_id}.json')

def load_user_history(user_id):
    """Kullanıcının analiz geçmişini yükler."""
    history_file = get_user_history_path(user_id)
    return load_json_file(history_file)

def save_analysis_to_history(user_id, analysis_result):
    """Yeni bir analizi kullanıcının geçmişine kaydeder."""
    history = load_user_history(user_id)
    history.insert(0, analysis_result)
    history_file = get_user_history_path(user_id)
    save_json_file(history_file, history)

def overwrite_user_history(user_id, history_data):
    """Bir kullanıcının tüm geçmişinin üzerine yazar."""
    history_file = get_user_history_path(user_id)
    save_json_file(history_file, history_data)

def load_reported_errors():
    """Kullanıcıların bildirdiği hataları yükler."""
    return load_json_file(REPORTED_ERRORS_FILE)

def save_reported_error(error_data):
    """Yeni bir hata bildirimini JSON dosyasına kaydeder."""
    errors = load_reported_errors()
    errors.insert(0, error_data)
    save_json_file(REPORTED_ERRORS_FILE, errors)

def preprocess_file(path):
    """Bir dosyayı modelin anlayacağı formata getirir."""
    x = np.loadtxt(path, dtype=float)
    if x.ndim > 1: x = x.squeeze()
    if len(x) > TARGET_LEN:
        start = (len(x) - TARGET_LEN) // 2
        x = x[start:start + TARGET_LEN]
    elif len(x) < TARGET_LEN:
        x = np.pad(x, (0, TARGET_LEN - len(x)), mode="constant")
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return x.astype(np.float32)

def time_ago(timestamp_str):
    """ISO formatındaki bir zaman damgasını '... önce' formatına çevirir."""
    if not timestamp_str: return ""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        diff = datetime.now() - timestamp
        seconds = diff.total_seconds()
        if seconds < 60: return "az önce"
        if seconds < 3600: return f"{int(seconds / 60)} dakika önce"
        if seconds < 86400: return f"{int(seconds / 3600)} saat önce"
        return f"{int(seconds / 86400)} gün önce"
    except (ValueError, TypeError):
        return ""

@app.context_processor
def inject_utility_functions():
    """time_ago fonksiyonunu tüm Jinja2 şablonlarında kullanılabilir hale getirir."""
    return dict(time_ago=time_ago)

def get_analysis_data(class_label):
    """Sınıf etiketine göre açıklama ve öneri metinlerini döndürür."""
    analysis_dict = {
        'Z': {'title': 'Anormal (Gözler Kapalı)', 'explanation': 'Sinyal, gözler kapalıyken kaydedilen anormal bir EEG aktivitesini temsil eder.', 'suggestion': 'Epileptik aktiviteye işaret edebilir. Bir nörolog ile iletişime geçin.'},
        'O': {'title': 'Normal (Gözler Kapalı)', 'explanation': 'Sinyal, sağlıklı bir bireyin gözler kapalıyken kaydedilen normal EEG aktivitesini temsil eder.', 'suggestion': 'Özel bir önlem gerekmemektedir.'},
        'N': {'title': 'Anormal (Gözler Açık)', 'explanation': 'Sinyal, gözler açıkken kaydedilen anormal bir EEG aktivitesini temsil eder.', 'suggestion': 'Altta yatan nörolojik bir sorunu gösterebilir. Bir uzmana başvurun.'},
        'F': {'title': 'Anormal (Fokal Odaklı)', 'explanation': 'Sinyal, beynin belirli bir bölgesinden kaynaklanan fokal (yerel) epileptik aktiviteyi temsil eder.', 'suggestion': 'Fokal epilepsiye işaret edebilir. Bir nörolog ile konsültasyon gereklidir.'},
        'S': {'title': 'Anormal (Nöbet Anı)', 'explanation': 'Sinyal, bir epileptik nöbet (seizure) sırasında kaydedilmiştir.', 'suggestion': 'Aktif bir nöbet durumu gösterir. Acil tıbbi yardım çağırın.'}
    }
    return analysis_dict.get(class_label, {'title': 'Bilinmeyen Sınıf', 'explanation': 'Analiz sonucu bilinmeyen bir sınıf döndürdü.', 'suggestion': 'Bir uzmana danışın.'})

def get_band_power(freqs, psd, band):
    """Belirli bir frekans bandındaki gücü hesaplar."""
    low, high = band
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return np.sum(psd[idx_band])

def generate_statistical_comparison(data1, data2, label1, label2):
    """İki sinyalin istatistiksel özelliklerini karşılaştırıp okunabilir bir HTML metni oluşturur."""
    bands = {"Delta (0.5-4 Hz)": (0.5, 4), "Theta (4-8 Hz)": (4, 8),
             "Alfa (8-13 Hz)": (8, 13), "Beta (13-30 Hz)": (13, 30)}
    powers1 = {name: get_band_power(np.array(data1['freqs']), np.array(data1['psd']), band) for name, band in bands.items()}
    powers2 = {name: get_band_power(np.array(data2['freqs']), np.array(data2['psd']), band) for name, band in bands.items()}
    std1, std2 = np.std(data1['signal']), np.std(data2['signal'])
    yorum = f"<p><strong>'{label1}'</strong> ve <strong>'{label2}'</strong> analizleri sayısal olarak karşılaştırıldığında şu farklar gözlemlenmiştir:</p>"
    yorum += "<ul class='list-disc list-inside mt-4 space-y-2'>"
    if std1 > 1e-9 and abs(std1 - std2) / std1 > 0.1:
        yorum += f"<li>Genel sinyal gücü (amplitüd), '{label2}' analizinde '{label1}'e göre belirgin şekilde <strong>{'daha yüksek' if std2 > std1 else 'daha düşük'}</strong> bulunmuştur.</li>"
    else:
        yorum += "<li>İki sinyalin genel gücü arasında anlamlı bir fark saptanmamıştır.</li>"
    degisiklik_var = False
    band_yorumlari = ""
    for name in bands:
        p1, p2 = powers1[name], powers2[name]
        if p1 > 1e-9 and abs(p1 - p2) / p1 > 0.15:
            degisim = "artış" if p2 > p1 else "azalış"
            oran = abs(p2 - p1) / p1 * 100
            band_yorumlari += f"<li><strong>{name}</strong> bandında <strong>%{oran:.0f}</strong> oranında bir <strong>{degisim}</strong> gözlemlenmiştir.</li>"
            degisiklik_var = True
    if degisiklik_var:
        yorum += "<li>Frekans bandı güçlerindeki değişimler:</li><ul class='list-disc list-inside ml-6 mt-2 space-y-1'>" + band_yorumlari + "</ul>"
    else:
        yorum += "<li>Frekans bantları arasında belirgin bir güç değişimi saptanmamıştır.</li>"
    yorum += "</ul>"
    return yorum

# --- DÜZELTİLMİŞ FONKSİYON ---
def is_intent_match(user_message, patterns):
    """Kullanıcı mesajının, verilen kalıplarla esnek bir şekilde eşleşip eşleşmediğini kontrol eder."""
    user_words = set(user_message.lower().split())
    for pattern in patterns:
        pattern_words = set(pattern.lower().split())
        # HATA BURADAYDI: issubset yerine, kesişim kontrolü yaparak daha esnek hale getiriyoruz.
        # Eğer ortak en az bir kelime varsa, bunu bir potansiyel eşleşme olarak kabul edebiliriz.
        # Daha da iyisi: Kalıptaki kelimelerden herhangi biri mesajda geçiyor mu?
        if not pattern_words.isdisjoint(user_words): # Kesişimleri boş değilse, yani ortak kelime varsa
            return True
    return False

# --- ROUTE (SAYFA) TANIMLARI ---

def login_required(f):
    """Kullanıcı girişi gerektiren sayfalar için decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def home():
    """Ana sayfaya yönlendirme yapar."""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    """Ana panel sayfasını yönetir ve dosya yükleme işlemini yapar."""
    user_id = session['user_id']
    user_name = session.get('user_name', '')
    history = load_user_history(user_id)
    result = None
    if request.method == "POST":
        if model is None:
            flash("Model yüklenemediği için analiz yapılamıyor.", "error")
            return render_template("index.html", user_name=user_name, history=history)
        file = request.files.get("file")
        if not file or file.filename == '':
            flash('Analiz için bir dosya seçmediniz.', 'error')
            return render_template("index.html", user_name=user_name, history=history)
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            processed_signal = preprocess_file(filepath)
            processed_signal = np.expand_dims(processed_signal, (0, -1))
            pred_probs = model.predict(processed_signal)
            pred_class = CLASSES[np.argmax(pred_probs)]
            analysis_label = request.form.get('analysis_label', '').strip()
            display_name = analysis_label if analysis_label else file.filename
            result = {'sonuc': pred_class, 'display_name': display_name, 'unique_filename': unique_filename,
                      'original_filename': file.filename, 'timestamp': datetime.now().isoformat()}
            save_analysis_to_history(user_id, result)
            history = load_user_history(user_id)
            flash(f"'{display_name}' analizi başarıyla tamamlandı!", "success")
        except Exception as e:
            flash(f'Dosya işlenirken bir hata oluştu: {str(e)}', 'error')
    return render_template("index.html", user_name=user_name, result=result, history=history)

@app.route("/karsilastir", methods=["GET", "POST"])
@login_required
def karsilastir():
    """İki analizin seçilip karşılaştırıldığı sayfayı yönetir."""
    user_id = session['user_id']
    history = load_user_history(user_id)
    if request.method == 'POST':
        file1_name = request.form.get('file1')
        file2_name = request.form.get('file2')
        if not file1_name or not file2_name or file1_name == file2_name:
            flash("Lütfen karşılaştırmak için iki FARKLI analiz seçin.", "error")
            return render_template("karsilastirma.html", history=history)
        analysis_1 = next((i for i in history if i.get('unique_filename') == file1_name), None)
        analysis_2 = next((i for i in history if i.get('unique_filename') == file2_name), None)
        if not analysis_1 or not analysis_2:
            flash("Seçilen analiz(ler) bulunamadı.", "error")
            return render_template("karsilastirma.html", history=history)
        file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], analysis_1['unique_filename'])
        file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], analysis_2['unique_filename'])
        signal1, signal2 = np.loadtxt(file_path1), np.loadtxt(file_path2)
        freqs1, psd1 = welch(signal1, fs=FS, nperseg=1024)
        freqs2, psd2 = welch(signal2, fs=FS, nperseg=1024)
        mask1, mask2 = freqs1 <= 50, freqs2 <= 50
        data1 = {'signal': signal1.tolist(), 'psd': psd1[mask1].tolist(), 'freqs': freqs1[mask1].tolist()}
        data2 = {'signal': signal2.tolist(), 'psd': psd2[mask2].tolist(), 'freqs': freqs2[mask2].tolist()}
        comparison_suggestion = generate_statistical_comparison(data1, data2, analysis_1['display_name'], analysis_2['display_name'])
        return render_template("karsilastirma.html", history=history, analysis_1=analysis_1,
                               analysis_2=analysis_2, comparison_suggestion=comparison_suggestion,
                               selected_file1=file1_name, selected_file2=file2_name)
    selected_file1 = request.args.get('file1')
    selected_file2 = request.args.get('file2')
    return render_template("karsilastirma.html", history=history,
                           selected_file1=selected_file1, selected_file2=selected_file2)

@app.route("/analiz_detay/<dosya_adi>")
@login_required
def analiz_detay(dosya_adi):
    """Tek bir analizin detaylarını ve grafiklerini gösterir."""
    user_id = session['user_id']
    history = load_user_history(user_id)
    analysis_item = next((i for i in history if i.get('unique_filename') == dosya_adi), None)
    if not analysis_item:
        flash("Analiz detayı bulunamadı.", "error")
        return redirect(url_for('dashboard'))
    interpretation = get_analysis_data(analysis_item['sonuc'])
    return render_template("analiz.html", new_analysis=analysis_item, analysis=interpretation)

@app.route("/delete_analysis/<timestamp>", methods=["POST"])
@login_required
def delete_analysis(timestamp):
    """AJAX isteği ile bir analiz kaydını siler."""
    user_id = session['user_id']
    history = load_user_history(user_id)
    updated_history = [item for item in history if item.get('timestamp') != timestamp]
    if len(updated_history) < len(history):
        overwrite_user_history(user_id, updated_history)
        return jsonify({'success': True, 'message': 'Analiz başarıyla silindi.'})
    return jsonify({'success': False, 'message': 'Silinecek analiz bulunamadı.'}), 404

@app.route("/api/analyze_file_data/<filename>")
@login_required
def analyze_file_data(filename):
    """Grafikler için işlenmiş sinyal verisini JSON olarak döndürür."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'Dosya bulunamadı.'}), 404
    try:
        signal = np.loadtxt(file_path)
        if signal.ndim > 1: signal = signal.squeeze()
        time_domain_data = signal[:2048].tolist()
        freqs, psd = welch(signal, fs=FS, nperseg=1024)
        mask = freqs <= 50
        psd_data = psd[mask].tolist()
        freqs_data = freqs[mask].tolist()
        return jsonify({'signal': time_domain_data, 'psd': psd_data, 'freqs': freqs_data})
    except Exception as e:
        return jsonify({'error': f'Dosya işlenirken hata oluştu: {str(e)}'}), 500

@app.route("/login", methods=["GET", "POST"])
def login():
    """Kullanıcı giriş sayfasını yönetir."""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        users = load_users()
        user = next((u for u in users if u.get('email') == email), None)
        if user and check_password_hash(user.get('password_hash', ''), password):
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            session['user_name'] = user['name']
            return redirect(url_for('dashboard'))
        flash('E-posta veya şifre hatalı.', 'error')
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Yeni kullanıcı kayıt sayfasını yönetir."""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == "POST":
        name = request.form.get('fullname')
        email = request.form.get('email')
        password = request.form.get('password')
        if not all([name, email, password]):
            flash("Lütfen tüm alanları doldurun.", "error")
            return render_template("register.html")
        users = load_users()
        if any(u.get('email') == email for u in users):
            flash('Bu e-posta adresi zaten kayıtlı.', 'error')
            return render_template("register.html")
        password_hash = generate_password_hash(password)
        new_user = {'id': str(uuid.uuid4()), 'name': name, 'email': email,
                    'password_hash': password_hash, 'created_at': datetime.now().isoformat()}
        users.append(new_user)
        save_users(users)
        flash('Hesabınız başarıyla oluşturuldu! Lütfen giriş yapın.', 'success')
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    """Kullanıcı oturumunu sonlandırır."""
    session.clear()
    flash("Başarıyla çıkış yaptınız.", "info")
    return redirect(url_for('login'))

@app.route("/chatbot", methods=["POST"])
@login_required
def handle_chatbot():
    """Gelen sohbet mesajlarını işler ve cevap döndürür."""
    data = request.get_json()
    user_message = data.get('message', '')
    user_id = session.get('user_id')
    user_name = session.get('user_name', 'Bilinmeyen Kullanıcı')
    error_keywords = ['hata', 'sorun', 'çalışmıyor', 'bozuk', 'problem', 'yanlış']
    if any(keyword in user_message.lower() for keyword in error_keywords):
        error_report = {'user_id': user_id, 'user_name': user_name, 'message': data.get('message'),
                        'timestamp': datetime.now().isoformat(), 'status': 'kırmızı'}
        save_reported_error(error_report)
        reply = "Hata bildiriminiz için teşekkürler! Mesajınızı kaydettim ve en kısa sürede ilgileneceğiz."
        return jsonify({'reply': reply})
    knowledge_base = load_json_file(CHATBOT_KNOWLEDGE_FILE)
    for intent in knowledge_base:
        if is_intent_match(user_message, intent['patterns']):
            reply = random.choice(intent['responses'])
            return jsonify({'reply': reply})
    fallback_intent = next((i for i in knowledge_base if i['tag'] == 'anlamadim'), None)
    if fallback_intent:
        reply = random.choice(fallback_intent['responses'])
    else:
        reply = "Üzgünüm, size nasıl cevap vereceğimi bilemiyorum."
    return jsonify({'reply': reply})

@app.route("/report_js_error", methods=['POST'])
def report_js_error():
    """Ön yüzden (JavaScript) gelen otomatik hata raporlarını kaydeder."""
    try:
        data = request.get_json()
        user_id = session.get('user_id', 'Giriş Yapılmamış')
        user_name = session.get('user_name', 'Bilinmeyen Kullanıcı')
        error_report = { 'type': 'otomatik_js_hatasi', 'user_id': user_id, 'user_name': user_name, 'error_message': data.get('message'), 'details': { 'source_file': data.get('source'), 'line_number': data.get('lineno'), 'column_number': data.get('colno'), 'stack_trace': data.get('error'), 'page_url': data.get('url') }, 'timestamp': datetime.now().isoformat(), 'status': 'kırmızı' }
        save_reported_error(error_report)
        return jsonify({'success': True, 'message': 'Hata raporu başarıyla alındı.'}), 200
    except Exception as e:
        print(f"Hata raporu kaydedilirken bir sunucu hatası oluştu: {e}")
        return jsonify({'success': False, 'message': 'Sunucu hatası.'}), 500

if __name__ == "__main__":
    app.run(debug=True)