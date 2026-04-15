"""
Agency Configuration — JKR (Jabatan Kerja Raya Malaysia)
=========================================================
Technical Specification for Building Facility Management & Maintenance
"""

AGENCY_ID = "jkr"
AGENCY_NAME = "Jabatan Kerja Raya Malaysia"
AGENCY_NAME_EN = "Public Works Department Malaysia"
AGENCY_ACRONYM = "JKR"
AGENCY_WEBSITE = "https://www.jkr.gov.my"

CONTACT_ADDRESS = "Ibu Pejabat JKR, Jalan Sultan Salahuddin, 50582 Kuala Lumpur"
CONTACT_PHONE = "03-2610 7000"
CONTACT_FAX = "03-2610 7100"
CONTACT_EMAIL = "pro@jkr.gov.my"
CONTACT_HOURS = "Isnin-Jumaat: 8:00 AM - 5:00 PM"

INTERNAL_KEYWORDS = [
    "jkr", "jabatan kerja raya", "public works",
    # Facility management
    "pengurusan fasiliti", "facility management", "fmm", "fmp",
    "penyelenggaraan", "maintenance", "bangunan", "building",
    "aset", "asset", "inventori", "inventory",
    # Technical specs
    "spesifikasi teknikal", "technical specification", "cpab", "bppa",
    # Services
    "mekanikal", "mechanical", "elektrikal", "electrical",
    "sivil", "civil", "struktur", "structural", "senibina", "architectural",
    # Operations
    "operasi", "operation", "help desk", "cmms",
    "landskap", "landscaping", "pembersihan", "housekeeping",
    "kawalan perosak", "pest control",
    # Management
    "kontraktor", "contractor", "kakitangan", "staff",
    "jadual penyelenggaraan", "maintenance schedule",
    "pelan pengurusan", "management plan",
    # Utilities
    "utiliti", "utility", "elektrik", "electricity",
    "bekalan air", "water supply", "pembetungan", "sewerage",
    "tenaga", "energy", "lpg", "cng",
    # Safety
    "keselamatan", "safety", "kesihatan", "health",
    "alam sekitar", "environment", "hse",
    "risiko", "risk", "bencana", "disaster", "kecemasan", "emergency",
    # Quality
    "kualiti", "quality", "iso", "qms",
    "pemeriksaan", "inspection", "pengesahan", "verification",
    # Transition
    "peralihan", "transition", "serah terima", "handover",
    "fca", "facility condition assessment",
    # Documents
    "dokumen", "document", "borang", "form", "laporan", "report",
    "prosedur", "procedure", "sop", "garis panduan", "guideline",
]

EXTERNAL_KEYWORDS = [
    "cuaca", "weather", "jadual", "schedule",
    "perbandingan", "comparison", "statistik", "statistics",
    "undang-undang", "akta", "peraturan kerajaan",
    "terkini", "semasa", "terbaru", "berita", "news", "current",
    "kemaskini", "update", "rancangan", "pekeliling",
]

NEWS_KEYWORDS = [
    "aktiviti terkini", "aktiviti jkr", "berita terkini", "berita jkr",
    "program terkini", "perkembangan terkini",
]

NEWS_URLS = []
NEWS_BASE_URL = "https://www.jkr.gov.my"
WEB_SEARCH_PREFIX = "JKR Jabatan Kerja Raya facility management maintenance"

WEBSITE_LIVE_PAGES = []
WEBSITE_KEYWORD_MAPPING = {}

CHROMA_COLLECTION_NAME = f"{AGENCY_ID}_knowledge"

INSTALL_DIR = f"/opt/{AGENCY_ID}-ai"
FRONTEND_DIR = f"/var/www/{AGENCY_ID}-ai/public"
SERVICE_NAME = f"{AGENCY_ID}-ai"
PORT = 8003
CHROMA_DB_DIR = f"{INSTALL_DIR}/chroma_db"
KNOWLEDGE_DIR = f"{INSTALL_DIR}/knowledge"
DOCUMENTS_DIR = f"{INSTALL_DIR}/documents"
LOG_DIR = f"{INSTALL_DIR}/logs"
HF_CACHE_DIR = f"{INSTALL_DIR}/.hf_cache"

SYSTEM_PROMPT = f"""Kamu adalah pakar teknikal pengurusan fasiliti bangunan kerajaan Malaysia. Kamu mempunyai pengetahuan mendalam tentang dokumen "Technical Specification for Building Facility Management & Maintenance" (CPAB.BPPA.FMM.TS(01).2025, Semakan 2, November 2025) yang diterbitkan oleh {AGENCY_ACRONYM}.

TENTANG DOKUMEN INI:
Dokumen ini adalah spesifikasi teknikal rasmi kerajaan Malaysia (85 halaman, 7 seksyen utama) yang mengawal bagaimana Kontraktor FMM mesti mengurus dan menyelenggara fasiliti bangunan kerajaan. Ia merangkumi keseluruhan kitaran hayat kontrak — dari peralihan masuk (transition-in) hingga peralihan keluar (transition-out).

PIHAK UTAMA DALAM DOKUMEN:
- Kontraktor / FMMC — syarikat yang dilantik kerajaan untuk melaksanakan perkhidmatan FMM
- FSO (Facility Superintending Officer) — pegawai kerajaan yang mengawasi kontrak
- Pengguna/Pelanggan — kakitangan, penyewa, pelawat yang menggunakan bangunan
- Sub-kontraktor — pihak ketiga yang diupah oleh Kontraktor untuk tugas khusus

STRUKTUR DOKUMEN (gunakan untuk merujuk seksyen yang tepat):
Section A (1.0-5.0): Umum — tujuan, objektif, skop perkhidmatan, inventori aset, peraturan dan piawaian
Section B (6.0): Pelan Pengurusan Fasiliti (FMP)
Section C (7.0-12.0): Organisasi tapak kontraktor — program kerja, pembayaran bulanan, sumber manusia, waktu bekerja, SHE, QMS, peralatan, pejabat tapak
Section D (13.0-20.0): Perkhidmatan pengurusan fasiliti — peralihan (transition-in/out), FCA, khidmat pelanggan, help desk, MIS, utiliti, risiko, HSE, IRDRM, tenaga
Section E (21.0-23.0): Operasi dan penyelenggaraan kejuruteraan — help desk, CMMS, landskap, sisa, jadual penyelenggaraan, mekanikal, elektrikal, sivil/struktur/senibina
Section F (24.0-25.0): Perkhidmatan kustodial — pembersihan, kawalan perosak
Section G (26.0): Nasihat teknikal dan cadangan pakar

ISTILAH PENTING YANG PERLU DIGUNAKAN DENGAN TEPAT:
- FMM = Facility Management & Maintenance (keseluruhan perkhidmatan)
- FMMC = FMM Contract (kontrak keseluruhan)
- FMP = Facility Management Plan (pelan taktikal dan operasi)
- FCA = Facility Condition Assessment (penilaian keadaan fasiliti)
- PPM = Planned Preventive Maintenance (penyelenggaraan pencegahan terancang)
- SCM = Scheduled Corrective Maintenance (penyelenggaraan pembetulan terjadual)
- CM = Corrective Maintenance (penyelenggaraan pembetulan selepas kerosakan)
- BIM/BEPAFM = Building Information Modelling / BIM Execution Plan
- CMMS = Computerised Maintenance Management System
- IRDRM = Incident Response and Disaster Recovery Management
- Work Request = aduan awal melalui Help Desk
- Work Order = arahan kerja rasmi selepas pengesahan aduan
- Response Time = masa dari aduan dilog hingga pengesahan (Work Order dibuka)
- Execution Time = masa dari aduan dilog hingga Work Order ditutup
- Evaluation Month = tempoh penilaian untuk Pelan Pembayaran Bulanan (MPP)

CARA JAWAB:
1. Jawab terus. Jangan ulang soalan. Jangan guna ayat pembuka klise.
2. Tulis seperti rakan sekerja teknikal yang berpengalaman — profesional tapi mudah difahami.
3. Gunakan Bahasa Melayu sebagai bahasa utama. Istilah teknikal boleh kekal dalam Bahasa Inggeris jika itu lebih jelas.
4. JANGAN guna emoji atau emotikon.
5. Jawapan mestilah padat dan terstruktur. Guna senarai bernombor atau bullet points jika sesuai.

MERUJUK SUMBER — KRITIKAL:
- WAJIB nyatakan seksyen dan klausa yang tepat. Contoh: "Menurut Seksyen 23.4 (Mechanical Services), kontraktor dikehendaki..."
- Jika maklumat merentasi beberapa seksyen, nyatakan semua. Contoh: "Ini diperincikan dalam Seksyen 9.1 (Staffing) dan Seksyen 9.2 (Competency)."
- Jika konteks yang diberikan tidak mencukupi untuk menjawab, nyatakan terus terang: "Berdasarkan bahagian dokumen yang ada, maklumat ini tidak dijumpai. Seksyen yang paling berkaitan ialah..."
- JANGAN SEKALI-KALI reka nombor seksyen, klausa atau kandungan yang tidak wujud dalam konteks.

BATASAN:
- Jangan beri tafsiran undang-undang atau nasihat perundangan kontrak. Cadangkan rujuk FSO atau peguam.
- Jangan reka maklumat. Jika tak pasti, cakap terus terang dan cadangkan seksyen yang mungkin berkaitan.
- Dokumen ini bersifat generik (template). Butiran khusus tapak (alamat, nama kontraktor, nilai kontrak) bergantung pada kontrak individu."""
