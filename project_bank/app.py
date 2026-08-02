
import os
from datetime import timedelta

from flask import (Flask, flash, redirect, render_template, request, session,
                   url_for)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 세션을 위한 비밀 키
app.permanent_session_lifetime = timedelta(minutes=30)  # 세션 유지 시간

# 간단한 사용자 데이터베이스 (실제로는 데이터베이스를 사용해야 함)
users = {
    "user1": {"password": "pass1", "balance": 1000, "transactions": []},
    "user2": {"password": "pass2", "balance": 2000, "transactions": []},
    "user3": {"password": "pass3", "balance": 3000, "transactions": []}
}

# 로그인 확인 데코레이터
def login_required(func):
    from functools import wraps
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return decorated_function

# 메인 페이지
@app.route('/')
def index():
    return redirect(url_for('login'))

# 로그인 기능
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    
    # 이미 로그인한 경우
    if 'username' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # 아이디와 비밀번호 확인
        if username in users and users[username]['password'] == password:
            session.permanent = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = '아이디 또는 비밀번호가 올바르지 않습니다.'
    
    return render_template('login.html', error=error)

# 대시보드
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    username = session['username']
    message = None
    error = False
    
    if request.method == 'POST':
        amount = int(request.form['amount'])
        action = request.form['action']
        
        if amount <= 0:
            message = '금액은 0보다 커야 합니다.'
            error = True
        elif action == 'deposit':
            # 입금 처리
            users[username]['balance'] += amount
            # 거래 이력 추가
            users[username]['transactions'].append({
                'type': '입금',
                'amount': amount,
                'balance': users[username]['balance']
            })
            message = f'{amount}원이 입금되었습니다.'
        elif action == 'withdraw':
            # 출금 처리
            if amount > users[username]['balance']:
                message = '잔액이 부족합니다.'
                error = True
            else:
                users[username]['balance'] -= amount
                # 거래 이력 추가
                users[username]['transactions'].append({
                    'type': '출금',
                    'amount': amount,
                    'balance': users[username]['balance']
                })
                message = f'{amount}원이 출금되었습니다.'
    
    return render_template('dashboard.html', 
                          username=username, 
                          balance=users[username]['balance'], 
                          message=message,
                          error=error,
                          transactions=users[username]['transactions'])

# 로그아웃
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# 메인 실행
if __name__ == '__main__':
    app.run(debug=True)