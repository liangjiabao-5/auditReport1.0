# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import render_template, redirect, request, url_for, jsonify, Blueprint, Response, stream_with_context
from flask_login import (
    current_user,
    login_user,
    logout_user
)
import pandas as pd
import subprocess
import time
import os

from apps import db, login_manager
from apps.authentication import blueprint
from apps.authentication.forms import LoginForm, CreateAccountForm
from apps.authentication.models import Users

from apps.authentication.util import verify_pass


chatglm4Path='/root/miniconda3/envs/chatglm4/bin/python'


@blueprint.route('/')
def route_default():
    return redirect(url_for('authentication_blueprint.login'))


# Login & Registration

@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)
    if 'login' in request.form:

        # read form data
        username = request.form['username']
        password = request.form['password']

        # Locate user
        user = Users.query.filter_by(username=username).first()

        # Check the password
        if user and verify_pass(password, user.password):

            login_user(user)
            return redirect(url_for('authentication_blueprint.route_default'))

        # Something (user or pass) is not ok
        return render_template('accounts/login.html',
                               msg='Wrong user or password',
                               form=login_form)

    if not current_user.is_authenticated:
        return render_template('accounts/login.html',
                               form=login_form)
    return redirect(url_for('home_blueprint.report_review'))


@blueprint.route('/register', methods=['GET', 'POST'])
def register():
    create_account_form = CreateAccountForm(request.form)
    if 'register' in request.form:

        username = request.form['username']
        email = request.form['email']

        # Check usename exists
        user = Users.query.filter_by(username=username).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Username already registered',
                                   success=False,
                                   form=create_account_form)

        # Check email exists
        user = Users.query.filter_by(email=email).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Email already registered',
                                   success=False,
                                   form=create_account_form)

        # else we can create the user
        user = Users(**request.form)
        db.session.add(user)
        db.session.commit()

        return render_template('accounts/register.html',
                               msg='User created please <a href="/login">login</a>',
                               success=True,
                               form=create_account_form)

    else:
        return render_template('accounts/register.html', form=create_account_form)


@blueprint.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('authentication_blueprint.login'))



# --------------------------------------------------------第一页-----------------------------------------------------------------

@blueprint.route('/uploadReview', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify('No file part'), 400

    file = request.files['file']
    if file:
        # 获取时间戳
        timestamp = request.form.get('timestamp')
        # 安全地获取文件名
        filename = file.filename
        print(filename)
        # 构建带有时间戳的新文件名
        timestamped_filename = f"{timestamp}_{filename}"
        # 设置保存文件的路径
        save_path = os.path.join(os.path.dirname(__file__), '../../reviewFile', timestamped_filename)
        # 保存文件
        file.save(save_path)

        # 读取Excel文件
        df = pd.read_excel(file, engine='openpyxl')
        # 只选择特定的字段
        selected_columns = ['report_id', 'obj_type', 'style_name', 'security_class', 'detail', 'result_record', 'result_score']
        df = df[selected_columns]
        # 将DataFrame转换为JSON
        data = df.to_dict(orient='records')
        return jsonify({"data": data, "filename": timestamped_filename})  # 返回JSON数据



@blueprint.route('/streamLoadModel', methods=['GET'])
def streamLoadModel():
    modelName = request.args.get('modelName', default=None, type=str)
    if modelName == "glm4":
        MODEL_PATH = '/mnt/workspace/glm4/model'
        SCRIPT_PATH = '/mnt/workspace/glm4/GLM-4/basic_demo/glm_server.py'
        print("glm4模型启动")
        def generate():
            try:
                with subprocess.Popen(
                    [chatglm4Path, '-u', SCRIPT_PATH, MODEL_PATH],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # 将标准错误重定向到标准输出
                    text=True,
                    bufsize=1  # 行缓冲
                ) as process:
                    for line in process.stdout:
                        # 实时发送每一行到前端
                        yield f"data: {line}\n\n"
                        time.sleep(0.1)  # 防止过快发送导致浏览器压力

                    # 等待进程结束
                    process.wait()

                    # 检查返回码
                    if process.returncode == 0:
                        yield "data: Script executed successfully.\n\n"
                    else:
                        yield f"data: Script failed with return code {process.returncode}.\n\n"

                    # 发送完成事件
                    yield "event: complete\ndata: All data has been sent.\n\n"

            except Exception as e:
                yield f"data: An error occurred: {str(e)}\n\n"
                import traceback
                yield f"data: Traceback: {traceback.format_exc()}\n\n"
                yield "event: complete\ndata: Error during script execution.\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')
    else :
        MODEL_PATH = '/mnt/workspace/qwen/model-7B'
        print("qwen模型启动")
        def generate():
            try:
                # vllm serve 命令及其参数
                command = [
                    'vllm', 'serve',
                    MODEL_PATH,  # 模型路径
                    '--port', '8000',                # 端口
                    '--dtype', 'float16',            # 数据类型
                    '--max-model-len', '2048'        # 最大模型长度
                ]

                with subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # 将标准错误重定向到标准输出
                    text=True,
                    bufsize=1  # 行缓冲
                ) as process:
                    for line in process.stdout:
                        # 实时发送每一行到前端
                        yield f"data: {line}\n\n"
                        time.sleep(0.1)  # 防止过快发送导致浏览器压力

                    # 等待进程结束
                    process.wait()

                    # 检查返回码
                    if process.returncode == 0:
                        yield "data: VLLM server started successfully.\n\n"
                    else:
                        yield f"data: VLLM server failed to start with return code {process.returncode}.\n\n"

                    # 发送完成事件
                    yield "event: complete\ndata: All data has been sent.\n\n"

            except Exception as e:
                yield f"data: An error occurred: {str(e)}\n\n"
                import traceback
                yield f"data: Traceback: {traceback.format_exc()}\n\n"
                yield "event: complete\ndata: Error during server startup.\n\n"
        return Response(stream_with_context(generate()), content_type='text/event-stream')

    return jsonify("Invalid model name"), 400


@blueprint.route('/unloadModel', methods=['GET'])
def unloadModel():
    # 获取查询参数
    modelName = request.args.get('modelName', default=None, type=str)
    
    if modelName == 'glm4':
        # 特定于 glm4 的卸载逻辑
        return _unload_glm4_model()
    else:
        # 对于其他模型名，使用 pgrep 查找并杀死进程
        return _kill_vllm_process()

def _unload_glm4_model():
    port = 8000
    try:
        # 查找占用端口的进程ID
        result = subprocess.run(
            ['netstat', '-tulpn'], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        pids = []
        for line in result.stdout.splitlines():
            if f":{port} " in line:
                parts = line.split()
                # PID is the second to last element, and program name is the last
                pid_with_program = parts[-1]
                pid = pid_with_program.split('/')[0]
                pids.append(pid)

        if not pids:
            print(f"没有找到占用端口 {port} 的进程")
            return jsonify("没有找到占用端口的进程"), 200

        # 终止找到的进程
        for pid in set(pids):  # 使用set去重
            try:
                subprocess.run(
                    ['kill', '-9', pid],
                    check=True
                )
                print(f"已成功终止PID为 {pid} 的进程，该进程占用了端口 {port}")
            except subprocess.CalledProcessError as e:
                print(f"终止PID为 {pid} 的进程失败: {e.stderr.strip()}")
                return jsonify(f"终止PID为 {pid} 的进程失败: {e.stderr.strip()}"), 500
        return jsonify("卸载模型成功"), 200

    except subprocess.CalledProcessError as e:
        print(f"执行命令时出错: {e.stderr.strip()}")
        return jsonify(f"执行命令时出错: {e.stderr.strip()}"), 500
    except Exception as e:
        print(f"尝试终止占用端口 {port} 的进程时出错: {e}")
        return jsonify(f"尝试终止占用端口 {port} 的进程时出错: {e}"), 500
    
    return jsonify("卸载模型失败！"), 500

def _kill_vllm_process():
    try:
        # 使用 pgrep -f vllm 命令查找进程
        result = subprocess.run(
            ['pgrep', '-f', 'vllm'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(result)

        pids = result.stdout.split()
        if not pids:
            print("没有找到与 vllm 相关的进程")
            return jsonify("没有找到与 vllm 相关的进程"), 200
        # 终止找到的进程
        for pid in pids:
            try:
                subprocess.run(
                    ['kill', '-9', pid],
                    check=True
                )
                print(f"已成功终止PID为 {pid} 的进程")
                
            except subprocess.CalledProcessError as e:
                print(f"终止PID为 {pid} 的进程失败: {e.stderr.strip()}")
                return jsonify(f"终止PID为 {pid} 的进程失败: {e.stderr.strip()}"), 500
        return jsonify("卸载模型成功"), 200

    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            # pgrep 返回 1 表示未找到匹配的进程，这并不是一个错误
            print("没有找到与 vllm 相关的进程")
            return jsonify("没有找到与 vllm 相关的进程"), 200
        else:
            print(f"执行 pgrep 命令时出错: {e.stderr.strip()}")
            return jsonify(f"执行 pgrep 命令时出错: {e.stderr.strip()}"), 500
    except Exception as e:
        print(f"尝试终止与 vllm 相关的进程时出错: {e}")
        return jsonify(f"尝试终止与 vllm 相关的进程时出错: {e}"), 500
    
    return jsonify("卸载模型失败！"), 500


@blueprint.route('/startReview', methods=['GET'])
def startReview():

    modelName = request.args.get('modelName', default=None, type=str)
    filename = request.args.get('fileName', default=None, type=str)
    securityClass = request.args.get('securityClass', default=None, type=str)
    # modelName = 'qwen'
    # filename = '20241219130022_测试优化数据结果6.0.xlsx'
    # securityClass = "安全物理环境"
    
    if modelName is None:
        return jsonify('没有选择模型'), 400
    if filename is None or filename == 'null':
        return jsonify('没有选择审核文件'), 400
    
    SCRIPT_PATH = "./script/sh2.0.py"
    FILE_PATH = "./reviewFile/" + filename
    print(modelName)
    print(filename)
    print(securityClass)
    # model_script(SCRIPT_PATH, modelName, FILE_PATH, securityClass)
    
    # 使用 Popen 启动脚本并在后台运行
    try:
        process = subprocess.Popen(
            [chatglm4Path, SCRIPT_PATH, modelName, FILE_PATH, securityClass],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # 可选：记录进程 ID 以便后续跟踪或管理
        print(f"Started review script with PID: {process.pid}")
    except Exception as e:
        print(str(e))
        return jsonify(f'启动脚本时出错: {str(e)}'), 500

    return jsonify("审核已启动，请稍后查看结果"), 202

# # 使用这个时，会多次重复执行
# def model_script(SCRIPT_PATH, modelName, FILE_PATH, securityClass):
#     # 定义要执行的命令和参数
#     command = [
#         chatglm4Path,  # Python解释器路径
#         SCRIPT_PATH,  # 脚本路径
#         modelName,  # 第一个参数
#         FILE_PATH,  # 第二个参数（文件路径）
#         securityClass  # 第三个参数
#     ]


#     try:
#         print("-------------开始调用审核脚本--------------")
#         # 使用subprocess.run来执行命令
#         result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
#         # 打印标准输出
#         print("标准输出：", result.stdout)
        
#         # 如果有标准错误输出，也打印出来
#         if result.stderr:
#             print("标准错误输出：", result.stderr)

#         print("命令成功执行")
#     except subprocess.CalledProcessError as e:
#         # 如果命令执行失败，则会抛出CalledProcessError异常
#         print("执行命令时发生错误:")
#         print("返回码:", e.returncode)
#         print("输出:", e.output)
#         print("标准错误:", e.stderr)

# --------------------------------------------------------第二页-----------------------------------------------------------------

@blueprint.route('/getReviewFiles', methods=['GET'])
def get_xlsx_files():
    REVIEW_FILE_PATH = './reviewFile'
    try:
        # 获取所有 .xlsx 文件的列表
        files = [f for f in os.listdir(REVIEW_FILE_PATH) if f.endswith('.xlsx')]
        return jsonify(files), 200
    except Exception as e:
        print(str(e))
        return jsonify(str(e)), 500

@blueprint.route('/selectResult', methods=['GET'])
def selectResult():
    REVIEW_FILE_PATH = './reviewFile'
    fileName = request.args.get('fileName', default=None, type=str)
    
    try:
        # 构建文件路径
        file_path = os.path.join(REVIEW_FILE_PATH, fileName)
        
        # 检查文件是否存在
        if not os.path.isfile(file_path):
            return jsonify('文件不存在'), 404
        
        # 使用pandas读取Excel文件
        df = pd.read_excel(file_path)
        
        # 筛选前端需要的字段
        required_fields = ["report_id", "obj_type", "style_name", "security_class", "detail", "result_record", "result_score", "qwen_result", "qwen_reason", "glm4_result", "glm4_reason"]
        filtered_df = df[required_fields]
        
        # 将DataFrame转换为字典列表
        data = filtered_df.to_dict(orient='records')
        
        # 检查是否有NaN或Infinity值，并将其替换为null或适当的值
        for item in data:
            for key, value in item.items():
                if pd.isna(value) or isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf')):
                    item[key] = None
        
        return jsonify(data), 200
    except Exception as e:
        print(str(e))
        return jsonify(str(e)), 500
    
@blueprint.route('/filterProblemItems', methods=['GET'])
def filterProblemItems():
    REVIEW_FILE_PATH = './reviewFile'
    fileName = request.args.get('fileName', default=None, type=str)
    
    try:
        # 构建文件路径
        file_path = os.path.join(REVIEW_FILE_PATH, fileName)
        
        # 检查文件是否存在
        if not os.path.isfile(file_path):
            return jsonify({'error': '文件不存在'}), 404
        
        # 使用pandas读取Excel文件
        df = pd.read_excel(file_path)
        
        # 筛选前端需要的字段
        required_fields = ["report_id", "obj_type", "style_name", "security_class", "detail", "result_record", "result_score", "qwen_result", "qwen_reason", "glm4_result", "glm4_reason"]
        filtered_df = df[required_fields]
        
        # 过滤出result_score和qwen_result不一致的行
        inconsistent_df = filtered_df[filtered_df['result_score'] != filtered_df['qwen_result']]
        
        # 将DataFrame转换为字典列表
        data = inconsistent_df.to_dict(orient='records')
        
        # 检查是否有NaN或Infinity值，并将其替换为null或适当的值
        for item in data:
            for key, value in item.items():
                if pd.isna(value) or (isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf'))):
                    item[key] = None
        
        return jsonify(data), 200
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

# --------------------------------------------------------第三页-----------------------------------------------------------------
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

    
@blueprint.route('/historyReference', methods=['GET'])
def historyReference():
    
    inputText = request.args.get('inputText', default=None, type=str)
    
     # 使用 Popen 启动脚本并在后台运行
    try:
        result = subprocess.run(['python', 'script/text.py', inputText],
                stdout=subprocess.PIPE,       # 捕获标准输出
                stderr=subprocess.PIPE,       # 捕获标准错误
                text=True)
    except Exception as e:
        print(str(e))
        return jsonify(f'启动脚本时出错: {str(e)}'), 500
    print(result.stdout.split())
    page_contents = ""
    for result in result.stdout.split(','):
        page_contents += result + "\n"
        print(result + "\n")
    # 打印或返回response
    return jsonify(page_contents),200


# --------------------------------------------------------第四页-----------------------------------------------------------------
@blueprint.route('/resultRecordOptimization', methods=['GET'])
def resultRecordOptimization():
    inputText = request.args.get('inputText', default=None, type=str)
    SCRIPT_PATH = "./script/TemperaturePrompt.py"
    # model_script(SCRIPT_PATH, modelName, FILE_PATH, securityClass)
    
    # 使用 Popen 启动脚本并在后台运行
    try:
        result = subprocess.run([chatglm4Path, SCRIPT_PATH, inputText],
                stdout=subprocess.PIPE,       # 捕获标准输出
                stderr=subprocess.PIPE,       # 捕获标准错误
                text=True)
        # 可选：记录进程 ID 以便后续跟踪或管理
        print(str(result))
    except Exception as e:
        print(str(e))
        return jsonify(f'启动脚本时出错: {str(e)}'), 500

    return jsonify(result.stdout), 200

# Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('home/page-404.html'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('home/page-500.html'), 500
