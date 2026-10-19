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
  <span class="brand-tag">A clearer picture of learning</span>
</div>
"""

def _clear_session():
    # Clear form inputs and results as well as the authentication token.
    for key in list(st.session_state):
        del st.session_state[key]


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
        setup = store.needs_setup()
        _login_page(store, setup, expired)
    except (sqlite3.Error, OSError):
        st.error("The account store is unavailable. Check that the app can write to its .local folder, then refresh.")
    st.stop()


def _login_page(store, setup, expired):
    with st.container(key="auth_shell"):
        with st.container(key="auth_card", border=False):
            st.html('<div class="form-eyebrow">YOUR LEARNING WORKSPACE</div>')
            st.title("Make it yours." if setup else "Welcome back.")
            st.write("Create your account to start exploring student performance." if setup else
                     "Sign in to turn learning data into useful insights.")
            if expired:
                st.info("Your session has ended. Sign in again to continue.")
            if st.session_state.pop("account_created", False):
                st.success("Your account is ready. Sign in to open your workspace.")
            with st.form("account_setup" if setup else "sign_in", border=False, clear_on_submit=True):
                if setup:
                    name = st.text_input("Your name", max_chars=80, placeholder="e.g. Alex Morgan", autocomplete="name")
                username = st.text_input("Username", max_chars=40, placeholder="Enter your username", autocomplete="username")
                password = st.text_input("Password", type="password", max_chars=128,
                                         placeholder="Choose a memorable passphrase" if setup else "Enter your password",
                                         autocomplete="new-password" if setup else "current-password")
                if setup:
                    st.caption("Use at least 15 characters. A few memorable words work well.")
                    confirmation = st.text_input("Confirm password", type="password", max_chars=128,
                                                 placeholder="Enter your password again", autocomplete="new-password")
                submitted = st.form_submit_button("Create account →" if setup else "Sign in →",
                                                  type="primary", width="stretch")
            if submitted:
                try:
                    if setup:
                        if password != confirmation:
                            raise AuthError("The passwords do not match. Please enter them again.")
                        store.create_owner(name, username, password)
                        st.session_state["account_created"] = True
                    else:
                        st.session_state["auth_token"] = store.login(username, password)
                    st.rerun()
                except AuthError as exc:
                    st.error(str(exc))
            st.html('<div class="form-divider"></div>')
            st.caption("FIRST TIME HERE?" if setup else "YOUR ACCOUNT, YOUR WORKSPACE")
            st.write("This one-time setup creates the owner account for this workspace." if setup else
                     "Use the account you created when you first opened this workspace.")
    st.html('<footer class="login-footer"><span>STUDENT PERFORMANCE PREDICTION</span><span>Machine learning to support learning.</span></footer>')
