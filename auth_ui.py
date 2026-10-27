"""Login presentation and the server-side gate for the Streamlit application."""

from pathlib import Path
import sqlite3

import streamlit as st

from auth import AuthError, AuthStore

ASSETS = Path(__file__).resolve().parent / "assets"

BRAND = """
<div class="brand-row">
  <div class="brand"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>
  <span>Student<span class="brand-light">Insight</span></span></div>
  <span class="brand-tag">Insights for educators and school leaders</span>
</div>
"""

def _clear_session():
    # Clear form inputs and results as well as the authentication token.
    for key in list(st.session_state):
        del st.session_state[key]


def _switch_view(view):
    for key in ("auth_password", "auth_confirmation", "auth_error", "account_created"):
        st.session_state.pop(key, None)
    st.session_state["auth_view"] = view


def _submit_account(store, setup):
    """Callbacks can safely clear password widgets before the next render."""
    username = st.session_state.get("auth_username", "")
    password = st.session_state.get("auth_password", "")
    try:
        if setup:
            if password != st.session_state.get("auth_confirmation", ""):
                raise AuthError("The passwords do not match. Please enter them again.")
            store.create_account(st.session_state.get("auth_name", ""), username, password)
            _clear_session()
            st.session_state.update(auth_view="sign_in", auth_username=username.strip().lower(), account_created=True)
        else:
            if not username.strip() or not password:
                raise AuthError("Enter your username and password.")
            if store.needs_setup():
                raise AuthError("No account is registered on this app yet. Choose Sign up below to create one.")
            token = store.login(username, password)
            _clear_session()
            st.session_state["auth_token"] = token
    except AuthError as exc:
        st.session_state["auth_error"] = str(exc)
    except (sqlite3.Error, OSError):
        st.session_state["auth_error"] = "The account store is unavailable. Please try again or contact the administrator."
    finally:
        st.session_state.pop("auth_password", None)
        st.session_state.pop("auth_confirmation", None)


def require_login():
    st.html(ASSETS / "app.css")
    st.html(BRAND)
    try:
        store = AuthStore()
        token = st.session_state.get("auth_token")
        user = store.current_user(token)
        if user:
            with st.sidebar:
                st.caption("YOUR WORKSPACE")
                st.subheader(user["display_name"])
                st.caption(f"@{user['username']}")
                if st.button("Sign out", key="sign_out", width="stretch"):
                    store.logout(token)
                    _clear_session()
                    st.rerun()
                st.divider()
            return user
        expired = bool(token)
        if expired:
            _clear_session()
        setup = st.session_state.get("auth_view") == "create_account"
        _login_page(store, setup, expired)
    except (sqlite3.Error, OSError):
        st.error("The account store is unavailable. Check that the app can write to its .local folder, then refresh.")
    st.stop()


def _login_page(store, setup, expired):
    with st.container(key="auth_shell"):
        with st.container(key="auth_card", border=False):
            st.html('<div class="form-eyebrow">YOUR STAFF WORKSPACE</div>')
            st.title("Create your staff account" if setup else "Welcome back.")
            st.write("Create your account to review class and student performance." if setup else
                     "Sign in to review classes, student progress, and reports.")
            if expired:
                st.info("Your session has ended. Sign in again to continue.")
            if st.session_state.pop("account_created", False):
                st.success("Your account is ready. Sign in to open your workspace.")
            if error := st.session_state.pop("auth_error", None):
                st.error(error)
            with st.form("account_setup" if setup else "sign_in", border=False):
                if setup:
                    st.text_input("Your name", key="auth_name", max_chars=80, placeholder="e.g. Alex Morgan", autocomplete="name")
                st.text_input("Username", key="auth_username", max_chars=40, placeholder="Enter your username", autocomplete="username")
                st.text_input("Password", key="auth_password", type="password", max_chars=128,
                                         placeholder="Choose a memorable passphrase" if setup else "Enter your password",
                                         autocomplete="new-password" if setup else "current-password")
                if setup:
                    st.caption("Use at least 15 characters. A few memorable words work well.")
                    st.text_input("Confirm password", key="auth_confirmation", type="password", max_chars=128,
                                                 placeholder="Enter your password again", autocomplete="new-password")
                st.form_submit_button("Create account →" if setup else "Sign in →",
                                      type="primary", width="stretch", on_click=_submit_account, args=(store, setup))
            st.html('<div class="form-divider"></div>')
            if setup:
                st.write("Already have an account?")
                st.button("Sign in", key="show_sign_in", width="stretch", on_click=_switch_view, args=("sign_in",))
            else:
                st.write("Don't have an account?")
                st.button("Sign up", key="show_create_account", width="stretch", on_click=_switch_view, args=("create_account",))
    st.html('<footer class="login-footer"><span>STUDENT PERFORMANCE PREDICTION</span><span>Machine learning to support learning.</span></footer>')
