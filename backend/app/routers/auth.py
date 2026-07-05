import json

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import settings


router = APIRouter(tags=["Auth"])


@router.get("/firebase-login", response_class=HTMLResponse)
def firebase_login() -> str:
    """Serve a top-level Firebase Google sign-in page for local Streamlit auth."""
    firebase_config = {
        "apiKey": settings.FIREBASE_API_KEY,
        "authDomain": settings.FIREBASE_AUTH_DOMAIN,
        "projectId": settings.FIREBASE_PROJECT_ID,
        "storageBucket": settings.FIREBASE_STORAGE_BUCKET,
        "messagingSenderId": settings.FIREBASE_MESSAGING_SENDER_ID,
        "appId": settings.FIREBASE_APP_ID,
    }
    missing = [key for key, value in firebase_config.items() if not value and key in {"apiKey", "authDomain", "projectId", "appId"}]
    frontend_url = settings.FRONTEND_URL.rstrip("/")

    if missing:
        missing_list = ", ".join(missing)
        return f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>SkillHire AI Sign In</title>
            <style>
                body {{
                    margin: 0;
                    min-height: 100vh;
                    display: grid;
                    place-items: center;
                    font-family: Inter, system-ui, sans-serif;
                    background: #f1f5f9;
                    color: #0f172a;
                }}
                .card {{
                    width: min(460px, calc(100vw - 32px));
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 16px;
                    padding: 24px;
                    box-shadow: 0 18px 44px rgba(15, 23, 42, 0.12);
                }}
                h1 {{ margin: 0 0 8px; font-size: 22px; }}
                p {{ color: #64748b; line-height: 1.5; }}
                code {{ color: #2f4d46; font-weight: 800; }}
            </style>
        </head>
        <body>
            <main class="card">
                <h1>Firebase setup incomplete</h1>
                <p>Missing Firebase fields: <code>{missing_list}</code>.</p>
                <p>Add them to your <code>.env</code> file and restart the backend.</p>
            </main>
        </body>
        </html>
        """

    return f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>SkillHire AI Sign In</title>
        <style>
            body {{
                margin: 0;
                min-height: 100vh;
                display: grid;
                place-items: center;
                font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: linear-gradient(135deg, #f1f5f9 0%, #e6f2dd 100%);
                color: #0f172a;
            }}
            .card {{
                width: min(440px, calc(100vw - 32px));
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid #d6e6dc;
                border-radius: 18px;
                padding: 26px;
                box-shadow: 0 24px 54px rgba(15, 23, 42, 0.14);
            }}
            .brand {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 20px;
            }}
            .mark {{
                width: 44px;
                height: 44px;
                border-radius: 12px;
                display: grid;
                place-items: center;
                background: linear-gradient(135deg, #2f4d46, #659287);
                color: #ffffff;
                font-weight: 900;
            }}
            h1 {{
                margin: 0;
                font-size: 22px;
                letter-spacing: 0;
            }}
            .subtitle {{
                margin: 4px 0 0;
                color: #64748b;
                font-size: 14px;
                line-height: 1.45;
            }}
            button {{
                width: 100%;
                height: 46px;
                border: 1px solid #d6e6dc;
                border-radius: 999px;
                background: #ffffff;
                color: #0f172a;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                font-size: 15px;
                font-weight: 800;
                cursor: pointer;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            }}
            button:hover {{
                border-color: #659287;
                background: #f8fafc;
            }}
            button:disabled {{
                cursor: wait;
                opacity: 0.72;
            }}
            .google-mark {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
                display: grid;
                place-items: center;
                color: #4285f4;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                font-size: 14px;
                font-weight: 900;
            }}
            .note {{
                margin-top: 14px;
                min-height: 20px;
                color: #64748b;
                font-size: 13px;
                line-height: 1.45;
            }}
            .domain {{
                margin-top: 12px;
                border-radius: 10px;
                background: #f6faf3;
                border: 1px solid #d6e6dc;
                color: #2f4d46;
                padding: 10px 12px;
                font-size: 12px;
                font-weight: 700;
            }}
        </style>
    </head>
    <body>
        <main class="card">
            <div class="brand">
                <div class="mark">SH</div>
                <div>
                    <h1>Sign in to SkillHire AI</h1>
                    <p class="subtitle">Use your Google account to track saved jobs, resume history, and skill progress.</p>
                </div>
            </div>
            <button id="googleAuthButton" type="button">
                <span class="google-mark">G</span>
                Continue with Google
            </button>
            <div id="authNote" class="note">Firebase Authentication will open a Google sign-in popup.</div>
            <div class="domain">Current auth domain: <span id="currentDomain"></span></div>
        </main>

        <script type="module">
            import {{ initializeApp }} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-app.js";
            import {{
                getAuth,
                GoogleAuthProvider,
                signInWithPopup,
                setPersistence,
                browserLocalPersistence
            }} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-auth.js";

            const firebaseConfig = {json.dumps(firebase_config)};
            const frontendUrl = {json.dumps(frontend_url)};
            const button = document.getElementById("googleAuthButton");
            const note = document.getElementById("authNote");
            document.getElementById("currentDomain").textContent = window.location.hostname;

            function encodePayload(payload) {{
                const jsonPayload = JSON.stringify(payload);
                const encoded = btoa(unescape(encodeURIComponent(jsonPayload)));
                return encoded.replace(/\\+/g, "-").replace(/\\//g, "_").replace(/=+$/g, "");
            }}

            function redirectWithParam(name, value) {{
                const url = new URL(frontendUrl);
                url.searchParams.set(name, value);
                window.location.href = url.toString();
            }}

            try {{
                const app = initializeApp(firebaseConfig);
                const auth = getAuth(app);
                const provider = new GoogleAuthProvider();
                provider.setCustomParameters({{ prompt: "select_account" }});
                await setPersistence(auth, browserLocalPersistence);

                button.addEventListener("click", async () => {{
                    button.disabled = true;
                    note.textContent = "Opening Google sign-in...";
                    try {{
                        const result = await signInWithPopup(auth, provider);
                        const user = result.user;
                        const idToken = await user.getIdToken(true);
                        const payload = {{
                            idToken,
                            uid: user.uid,
                            email: user.email,
                            displayName: user.displayName,
                            photoURL: user.photoURL
                        }};
                        note.textContent = "Sign-in complete. Returning to SkillHire AI...";
                        redirectWithParam("firebase_auth", encodePayload(payload));
                    }} catch (error) {{
                        button.disabled = false;
                        note.textContent = error?.message || "Google sign-in failed. Try again.";
                    }}
                }});
            }} catch (error) {{
                button.disabled = true;
                note.textContent = error?.message || "Firebase could not start.";
            }}
        </script>
    </body>
    </html>
    """
