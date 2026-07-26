import io
import os
import re
import sys
import tempfile
import threading
import shutil
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for, send_file, jsonify
from markdown import markdown as md_to_html

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from utils.translate import translate_en_to_ar
from utils.stt import stt as stt_tool
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Ensure `src` package is importable when running this script directly
sys.path.append(str(Path(__file__).parent.parent))

from agent import MedicineAssistantAgent

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# OCR-RAG — إضافة مسار scripts/ حتى يجد Python ملف ingest_pdf_OCR.py
# ═══════════════════════════════════════════════════════════════════════════════
_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from ingest_pdf_OCR import ingest_path, query_pipeline, validate_config, OCR_LANGUAGES
    _OCR_IMPORT_OK = True
except ImportError as _e:
    print(f"[OCR] Warning: could not import ingest_pdf_OCR: {_e}")
    _OCR_IMPORT_OK = False

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-key-medicine-assistant")

# ═══════════════════════════════════════════════════════════════════════════════
# OCR-RAG — تحميل الـ index في الخلفية عند بدء السيرفر مع معالجة خطأ التوافق (_type)
# ═══════════════════════════════════════════════════════════════════════════════
_ocr_ready = False
_ocr_error = None


def _load_ocr_index() -> None:
    """
    تُشغَّل مرة واحدة في background thread عند بدء التشغيل.
    - يقوم بفحص الملف وفهرسته، وإذا حدث خطأ بسبب صيغة قديمة (_type)، يتم مسح الكاش وإعادة البناء تلقائياً.
    """
    global _ocr_ready, _ocr_error

    if not _OCR_IMPORT_OK:
        _ocr_error = "ingest_pdf_OCR module not found — check scripts/ folder."
        return

    try:
        validate_config()
        pdf_path = Path(
            os.getenv(
                "OCR_PDF_PATH",
                "scripts/hcea guidelines_BW 1-25-2021-4-7_page-0001.pdf",
            )
        )
        print(f"[OCR] Starting ingestion/check: {pdf_path.name}")
        
        try:
            ingest_path(str(pdf_path), ocr_lang=OCR_LANGUAGES)
        except Exception as inner_exc:
            if "_type" in str(inner_exc) or "chroma" in str(inner_exc).lower():
                print(f"[OCR] Detected incompatible vector store format ({inner_exc}). Clearing chroma_db cache and re-indexing...")
                chroma_dir = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"))
                if chroma_dir.exists():
                    shutil.rmtree(chroma_dir)
                ingest_path(str(pdf_path), ocr_lang=OCR_LANGUAGES)
            else:
                raise inner_exc

        _ocr_ready = True
        print("[OCR] ✓ Ready — vector store loaded.")
    except Exception as exc:
        _ocr_error = str(exc)
        print(f"[OCR] ✗ Ingestion failed: {exc}")


# يبدأ تلقائياً عند import الـ module (أي عند تشغيل Flask)
threading.Thread(target=_load_ocr_index, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# Database connection
# ═══════════════════════════════════════════════════════════════════════════════
def get_db_connection():
    database_url = os.getenv("DATABASE_URL")

    fallback_url = (
        "postgresql://medicine_db_o4sx_user:"
        "PVfrBQ4jOYsvmy78uUQXdqaK9ow0NU7O@"
        "dpg-d9hs93svct5s73abbgu0-a.oregon-postgres.render.com/medicine_db_o4sx"
    )

    url_to_use = database_url if database_url else fallback_url

    if url_to_use.startswith("postgres://"):
        url_to_use = url_to_use.replace("postgres://", "postgresql://", 1)

    return psycopg2.connect(url_to_use)


# ═══════════════════════════════════════════════════════════════════════════════
# Initialize Agent
# ═══════════════════════════════════════════════════════════════════════════════
try:
    agent = MedicineAssistantAgent()
except Exception as e:
    print(f"Warning: Could not initialize Agent: {e}")
    agent = None


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — General
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/patients')
def patients():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute('SELECT * FROM Patients ORDER BY created_at DESC')
        patients = cur.fetchall()
    except Exception as e:
        flash(f"Error fetching patients: {e}", "danger")
        patients = []
    finally:
        cur.close()
        conn.close()
    return render_template('patients.html', patients=patients)


@app.route('/patient/add', methods=('GET', 'POST'))
def add_patient():
    if request.method == 'POST':
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO Patients (
                    Patient_ID, Name, Age, Gender, Height_cm, Weight_kg,
                    Diabetes_Type, Duration_Years, Comorbidities, Latest_HbA1c,
                    Current_Meds, eGFR_ml_min, Recent_Symptoms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    request.form['patient_id'],
                    request.form['name'],
                    request.form['age'],
                    request.form['gender'],
                    request.form['height_cm'],
                    request.form['weight_kg'],
                    request.form['diabetes_type'],
                    request.form['duration_years'],
                    request.form['comorbidities'],
                    request.form['latest_hba1c'],
                    request.form['current_meds'],
                    request.form['egfr_ml_min'],
                    request.form['recent_symptoms'],
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
            flash('Patient added successfully!', 'success')
            return redirect(url_for('patients'))
        except Exception as e:
            flash(f"Error adding patient: {e}", 'danger')

    return render_template('patient_form.html', action='Add', patient={})


@app.route('/patient/edit/<patient_id>', methods=('GET', 'POST'))
def edit_patient(patient_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    if request.method == 'POST':
        try:
            cur.execute(
                """
                UPDATE Patients SET
                    Name = %s, Age = %s, Gender = %s, Height_cm = %s, Weight_kg = %s,
                    Diabetes_Type = %s, Duration_Years = %s, Comorbidities = %s,
                    Latest_HbA1c = %s, Current_Meds = %s, eGFR_ml_min = %s,
                    Recent_Symptoms = %s
                WHERE Patient_ID = %s
                """,
                (
                    request.form['name'],
                    request.form['age'],
                    request.form['gender'],
                    request.form['height_cm'],
                    request.form['weight_kg'],
                    request.form['diabetes_type'],
                    request.form['duration_years'],
                    request.form['comorbidities'],
                    request.form['latest_hba1c'],
                    request.form['current_meds'],
                    request.form['egfr_ml_min'],
                    request.form['recent_symptoms'],
                    patient_id,
                ),
            )
            conn.commit()
            flash('Patient updated successfully!', 'success')
            return redirect(url_for('patients'))
        except Exception as e:
            flash(f"Error updating patient: {e}", 'danger')
        finally:
            cur.close()
            conn.close()
    else:
        cur.execute('SELECT * FROM Patients WHERE Patient_ID = %s', (patient_id,))
        patient = cur.fetchone()
        cur.close()
        conn.close()
        if not patient:
            flash('Patient not found', 'danger')
            return redirect(url_for('patients'))
        return render_template('patient_form.html', action='Edit', patient=patient)


@app.route('/patient/delete/<patient_id>', methods=('POST',))
def delete_patient(patient_id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('DELETE FROM Patients WHERE Patient_ID = %s', (patient_id,))
        conn.commit()
        flash('Patient deleted successfully!', 'success')
    except Exception as e:
        flash(f"Error deleting patient: {e}", 'danger')
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('index'))


@app.route('/consult', methods=('GET', 'POST'))
def consult():
    patient_id = request.args.get('patient_id') or request.form.get('patient_id')
    patient = None

    if patient_id:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('SELECT * FROM Patients WHERE Patient_ID = %s', (patient_id,))
        patient = cur.fetchone()
        cur.close()
        conn.close()

    result = None
    if request.method == 'POST':
        if not agent:
            flash("Agent not initialized. Check configuration.", "danger")
        else:
            try:
                consult_data = {
                    "patient_id": patient_id,
                    "name": patient.get('name') if patient else request.form.get('name', 'Unknown'),
                    "age": request.form.get('age') or (patient.get('age') if patient else None),
                    "gender": request.form.get('gender') or (patient.get('gender') if patient else None),
                    "weight": request.form.get('weight') or (patient.get('weight_kg') if patient else None),
                    "diabetes_type": request.form.get('diabetes_type') or (patient.get('diabetes_type') if patient else None),
                    "duration_years": request.form.get('duration_years') or (patient.get('duration_years') if patient else None),
                    "latest_hba1c": request.form.get('latest_hba1c') or (patient.get('latest_hba1c') if patient else None),
                    "blood_glucose": request.form.get('blood_glucose'),
                    "blood_pressure": request.form.get('blood_pressure'),
                    "egfr": request.form.get('egfr') or (patient.get('egfr_ml_min') if patient else None),
                    "lipid_panel": request.form.get('lipid_panel'),
                    "symptoms_notes": request.form.get('symptoms_notes') or (patient.get('recent_symptoms') if patient else ''),
                    "treatment_adjustments": request.form.get('treatment_adjustments'),
                    "current_meds": patient.get('current_meds') if patient else request.form.get('current_meds', ''),
                    "comorbidities": request.form.get('comorbidities') or (patient.get('comorbidities') if patient else None),
                    "allergies": request.form.get('allergies') or (patient.get('allergies') if patient else None),
                }

                consult_data = {k: (v if v is not None else '') for k, v in consult_data.items()}

                query = f"""
Please analyze the patient data and provide comprehensive physician and patient reports.

Patient: {consult_data['name']} (ID: {consult_data['patient_id']})

Latest vitals and labs provided in structured data.
"""
                result = agent.invoke(query, patient_info=consult_data)

                if result.get("needs_clarification"):
                    flash("Additional patient information is required for a complete analysis.", "warning")

                if result.get("safety_alerts"):
                    for alert in result["safety_alerts"]:
                        flash(alert, "danger")

            except Exception as e:
                flash(f"Error generating report: {str(e)}", "danger")
                import traceback
                print(f"Error in consult route: {traceback.format_exc()}")

    phys_md = result.get('physician_report', '') if result else ''
    pat_md  = result.get('patient_report', '')   if result else ''

    pat_md_ar = ''
    try:
        if pat_md:
            pat_md_ar = translate_en_to_ar(pat_md)
    except Exception as e:
        print(f"Translation error: {e}")

    phys_html    = md_to_html(phys_md,    extensions=['extra', 'nl2br']) if phys_md    else ''
    pat_html     = md_to_html(pat_md,     extensions=['extra', 'nl2br']) if pat_md     else ''
    pat_html_ar  = md_to_html(pat_md_ar,  extensions=['extra', 'nl2br']) if pat_md_ar  else ''

    return render_template(
        'consult.html',
        patient=patient,
        result=result,
        physician_html=phys_html,
        patient_html=pat_html,
        patient_html_ar=pat_html_ar,
    )


@app.route('/consult/pdf/<report_type>', methods=('POST',))
def consult_pdf(report_type: str):
    """Generate a PDF for a given report type using ReportLab (Pure Python)."""
    try:
        data     = request.get_json(force=True, silent=True) or request.form or {}
        filename = data.get('filename') or f"{report_type}_report.pdf"

        html_input = data.get('html', '')
        md_input   = data.get('md') or data.get('report', '')
        text_content = md_input or html_input

        clean_text = re.sub('<[^<]+?>', '', text_content)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=40,   bottomMargin=40,
        )

        styles       = getSampleStyleSheet()
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10, leading=14, spaceAfter=6,
        )
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16, leading=20, spaceAfter=12,
        )

        story = [Paragraph(f"{report_type.title()} Report", title_style), Spacer(1, 10)]
        for line in clean_text.split('\n'):
            line_str = line.strip()
            if line_str:
                story.append(Paragraph(line_str.replace('**', ''), normal_style))

        doc.build(story)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf',
        )

    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/consult/stt', methods=['POST'])
def consult_stt():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return jsonify({"error": "OpenRouter API key not configured"}), 500

    try:
        original_ext = os.path.splitext(audio_file.filename)[1] or '.webm'
        with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as tmp_audio:
            audio_file.save(tmp_audio.name)
            tmp_audio_path = tmp_audio.name

        try:
            result = stt_tool(audio_path=tmp_audio_path, api_key=api_key)
            return jsonify(result)
        except Exception as inner_e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"STT processing failed: {str(inner_e)}"}), 500
        finally:
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)

    except Exception as e:
        print(f"STT API error: {e}")
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Route: POST /api/ocr-ask  or  POST /api/ask-qna  — OCR-RAG endpoint
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/api/ocr-ask', methods=['POST'])
@app.route('/api/ask-qna', methods=['POST'])
def ocr_ask():
    global _ocr_ready, _ocr_error

    # ── التحقق من الـ input أولاً ───────────────────────────────────────────
    body   = request.get_json(silent=True) or {}
    question = (body.get("question") or body.get("query") or "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400

    # ── معالجة تلقائية ومسح الكاش إذا كان هناك خطأ '_type' سابق ───────────
    if _ocr_error and ("_type" in str(_ocr_error) or "chroma" in str(_ocr_error).lower()):
        print(f"[OCR] Recovering from stored error: {_ocr_error}. Clearing cache and re-indexing...")
        try:
            chroma_dir = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"))
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir)
            
            pdf_path = Path(
                os.getenv(
                    "OCR_PDF_PATH",
                    "scripts/hcea guidelines_BW 1-25-2021-4-7_page-0001.pdf",
                )
            )
            ingest_path(str(pdf_path), ocr_lang=OCR_LANGUAGES)
            _ocr_ready = True
            _ocr_error = None
            print("[OCR] ✓ Recovery successful — index rebuilt.")
        except Exception as recovery_exc:
            import traceback
            tb_str = traceback.format_exc()
            print(f"[OCR Recovery Error]: {tb_str}")
            return jsonify({"error": f"Recovery failed: {str(recovery_exc)}", "traceback": tb_str}), 500

    # إذا كان لا يزال هناك خطأ آخر غير متعلق بـ chroma/_type
    if _ocr_error:
        return jsonify({"error": f"OCR pipeline failed: {_ocr_error}"}), 500

    if not _ocr_ready:
        return jsonify({
            "error": "OCR index is still loading — please wait 30–60 s and retry."
        }), 503

    # ── تشغيل RAG مع حماية إضافية أثناء الاستعلام وطباعة الخطأ بالتفصيل ───────────
    try:
        results = query_pipeline(question)
    except Exception as exc:
        import traceback
        tb_str = traceback.format_exc()
        print(f"[OCR Error Traceback]:\n{tb_str}")
        
        if "_type" in str(exc) or "chroma" in str(exc).lower():
            print(f"[OCR] Detected incompatible format during query ({exc}). Re-indexing...")
            try:
                chroma_dir = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"))
                if chroma_dir.exists():
                    shutil.rmtree(chroma_dir)
                
                pdf_path = Path(
                    os.getenv(
                        "OCR_PDF_PATH",
                        "scripts/hcea guidelines_BW 1-25-2021-4-7_page-0001.pdf",
                    )
                )
                ingest_path(str(pdf_path), ocr_lang=OCR_LANGUAGES)
                results = query_pipeline(question)
            except Exception as inner_exc:
                inner_tb = traceback.format_exc()
                print(f"[OCR Error during query recovery]: {inner_tb}")
                return jsonify({"error": str(inner_exc), "traceback": inner_tb}), 500
        else:
            return jsonify({"error": str(exc), "traceback": tb_str}), 500
        
    rag_ans = results.get("rag_answer", "") if isinstance(results, dict) else str(results)
    no_rag_ans = results.get("no_rag_answer", "") if isinstance(results, dict) else ""
    
    return jsonify({
        "answer": rag_ans,
        "rag_answer": rag_ans,
        "no_rag_answer": no_rag_ans,
        "chunks": results.get("retrieved_chunks", []) if isinstance(results, dict) else [],
        "avg_similarity": results.get("overall_similarity_score", 0) if isinstance(results, dict) else 0
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Route: GET /api/ocr-status  — تحقق هل الـ index جاهز
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/api/ocr-status', methods=('GET',))
def ocr_status():
    """
    واجهة بسيطة يستطيع الـ frontend استخدامها للتحقق من جاهزية الـ OCR index.
    Returns: { "ready": true/false, "error": null / "message" }
    """
    return jsonify({
        "ready": _ocr_ready,
        "error": _ocr_error,
    })


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
