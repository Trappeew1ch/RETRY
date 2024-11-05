import gradio as gr
import requests
import cv2
import numpy as np
from PIL import Image
import io
import re
import os
import subprocess
import zipfile
import tempfile
import shutil
import uuid
from flask import Flask, request, send_file, jsonify
import threading
import asyncio
import aiohttp
import signal
import logging
import atexit
import sys
from queue import Queue
from threading import Thread
import functools
import traceback
from googlesearch import googlesearch
from bs4 import BeautifulSoup
from groq import Groq
from mistralai.client import MistralClient
from mistralai import Mistral
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import zipfile
import tempfile
import shutil
from flask import send_file
import sys
from typing import List, Dict, Any, Optional, Tuple, NamedTuple  # Добавьте NamedTuple здесь
import ast
import traceback
import os
from pathlib import Path
import google.generativeai as genai
import openai
import httpx  # Добавлено для исправления ошибки
from contextlib import contextmanager  # Убедитесь, что эта строка также присутствует
# Добавляем путь к TaskWeaver в sys.path
sys.path.append(r'C:\Users\Богдан\Documents\TaskWeaver')

# Настройка логгера
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")

# Инициализация Flask приложения
app = Flask(__name__)
file_storage = {}

GOOGLE_NEWS_API_KEY = "AIzaSyA4_loK5yODgDG5IHl1SjZC-Ru8xa8wkmM"
SEARCH_ENGINE_ID = "812fc1deaea1242ff"
# Инициализация API ключей и клиентов
GOOGLE_API_KEY = "AIzaSyB9eYeIwSrH0m0vxG0eszqly67tEL0U8NA"
GROQ_API_KEY = "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"
HYPERBOLIC_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0cmFwcGVldzFjaEBnbWFpbC5jb20iLCJpYXQiOjE3MjY5MzkwMzZ9.Mph5RQUydak_BI9jp2nQBbttO2AF5kwb6yiGnJJL_GI"

GROQ_API_BASE = "https://api.groq.com/v1/chat/completions"

genai.configure(api_key=GOOGLE_API_KEY)
try:
    groq_client = Groq(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)
except Exception as e:
    logger.error(f"Ошибка при инициализации Groq клиента: {str(e)}")
    groq_client = None
hyperbolic_client = openai.OpenAI(
    api_key=HYPERBOLIC_API_KEY,
    base_url="https://api.hyperbolic.xyz/v1"
)

MODELS = [
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-exp-0827",
    "gemini-1.5-flash-exp-0827",
    "gemini-1.5-flash-8b-exp-0827",
    "Qwen/Qwen2.5-72B-Instruct"
]
SYSTEM_PROMPT = """
You are a code development assistant. Use the following commands if necessary:

1. /web_search <query> - to search for information on the Internet
2. /create_file <file_name> <extension> - to create a new file in the project
3. /read_file <file_id> - to read the contents of the file
4. /write_file <file_id> <content> - to write to a file
5. /list_files - to view the list of files in the project
6. /delete_file <file_id> - to delete a file
7. /run_code <file_id> - to execute code from a file
8. /test <file_id> - to run tests for a file
9. /install_package <package name> - to install the Python package
Use these commands before the appropriate actions. For example, before testing, use /create_file to create a test file, then /write_file to write tests, and finally /test to run tests.
Every time I ask you to generate a code, send me only the code, without any explanations or comments. Never send me a message with the code unless I asked you for the code. Never repeat yourself, if 1 solution does not fit, generate another one.
"""

# Словарь для хранения виртуальных файлов
virtual_files = {}


async def generate_content_google(prompt, model):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Ошибка при вызове Google AI Studio API: {str(e)}")
        return None


async def internet_search(query):
    try:
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_NEWS_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": query,
            "num": 5
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'items' in data:
                        search_results = []
                        for item in data['items']:
                            search_results.append(
                                f"Заголовок: {item['title']}\nОписание: {item['snippet']}\nURL: {item['link']}\n")
                        return "\n".join(search_results)
                    else:
                        return "Результаты поиска не найдены"
                elif response.status == 403:
                    logger.error("Ошибка доступа к API Google Search. Проверьте ваш API ключ и квоты.")
                    return "Ошибка доступа к API поиска. Пожалуйста, попробуйте позже."
                else:
                    return f"Ошибка при выполнении поиска: HTTP статус {response.status}"
    except Exception as e:
        logger.error(f"Ошибка при выполнении поиска в интернете: {str(e)}")
        return f"Ошибка при выполнении поиска: {str(e)}"


# Функция для выполнения HTTP-запросов
async def safe_request(method, url, **kwargs):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка при выполнении запроса: {e}")
            return None


async def create_virtual_file(file_name, file_type):
    file_id = str(uuid.uuid4())
    virtual_files[file_id] = {
        'name': f"{file_name}.{file_type}",
        'content': ''
    }
    return file_id


async def write_virtual_file(file_id, content):
    if file_id in virtual_files:
        virtual_files[file_id]['content'] = content
        return True
    return False


async def read_virtual_file(file_id):
    if file_id in virtual_files:
        return virtual_files[file_id]['content']
    return None


async def run_virtual_code(file_id):
    if file_id not in virtual_files:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"Файл с ID {file_id} не найден"
        }

    code = virtual_files[file_id]['content']
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    redirected_error = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_error
    try:
        exec(code)
        return {
            "returncode": 0,
            "stdout": redirected_output.getvalue(),
            "stderr": redirected_error.getvalue()
        }
    except Exception:
        return {
            "returncode": 1,
            "stdout": redirected_output.getvalue(),
            "stderr": traceback.format_exc()
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


async def run_real_code(code):
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        result = subprocess.run(['python', temp_file_path], capture_output=True, text=True, timeout=30)
        os.unlink(temp_file_path)

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {"returncode": 1, "stdout": "", "stderr": "Выполнение кода превысило лимит времени"}
    except Exception as e:
        return {"returncode": 1, "stdout": "", "stderr": str(e)}


# Обработчики команд
@app.route('/create-file', methods=['POST'])
async def create_file():
    file_name = request.json['file_name']
    file_type = request.json['file_type']
    full_file_name = f"{file_name}.{file_type}"
    file_id = str(uuid.uuid4())
    virtual_files[file_id] = {'name': full_file_name, 'content': ''}
    return jsonify({"file_id": file_id, "message": f"Файл {full_file_name} создан"})


@app.route('/generate-code', methods=['POST'])
async def generate_code_api():
    user_request = request.json['user_request']
    model = request.json['model']
    use_pro_mode = request.json.get('use_pro_mode', False)
    use_model_to_model = request.json.get('use_model_to_model', False)

    progress_messages = []
    process_steps = []
    code_output = ""
    file_path = None
    result_queue = asyncio.Queue()
    code_executor = CodeExecutor()

    async def progress_callback(message):
        logger.debug(f"Progress: {message}")
        progress_messages.append(message)
        await result_queue.put((progress_messages, process_steps, code_output, file_path))

    async def process_callback(step_name, content):
        logger.debug(f"Process step: {step_name}")
        process_steps.append(f"### {step_name}\n{content}")
        nonlocal code_output
        if step_name.startswith("Код для") or step_name == "Итоговый код":
            code_output = content
        await result_queue.put((progress_messages, process_steps, code_output, file_path))

    try:
        await progress_callback("Начало обработки запроса")
        # Вызов функции генерации кода
        await generate_code(user_request, model, progress_callback)
        await progress_callback("Обработка запроса завершена")
    except Exception as e:
        logger.error(f"Ошибка в generate_code_api: {str(e)}", exc_info=True)
        error_message = f"Произошла ошибка: {str(e)}"
        await result_queue.put((progress_messages + [error_message], process_steps, code_output, file_path))
    finally:
        await result_queue.put(None)  # Сигнал о завершении

    return jsonify({
        "progress": progress_messages,
        "process_steps": process_steps,
        "code_output": code_output,
        "file_path": file_path
    })


@app.route('/get-results', methods=['POST'])
async def get_results_api():
    code = request.json['code']
    file_id = request.json['file_id']
    model = request.json['model']

    stdout, stderr = await run_final_tests(code, file_id, model)
    return jsonify({
        "stdout": stdout,
        "stderr": stderr
    })


@app.route('/read-file', methods=['GET'])
async def read_file():
    file_id = request.args.get('file_id')
    if file_id in virtual_files:
        return jsonify({"content": virtual_files[file_id]['content']})
    return jsonify({"error": "Файл не найден"}), 404


@app.route('/write-file', methods=['POST'])
async def write_file():
    file_id = request.json['file_id']
    content = request.json['content']
    if file_id in virtual_files:
        virtual_files[file_id]['content'] = content
        return jsonify({"message": "Содержимое файла обновлено"})
    return jsonify({"error": "Файл не найден"}), 404


@app.route('/list-files', methods=['GET'])
async def list_files():
    files = [{"id": k, "name": v['name']} for k, v in virtual_files.items()]
    return jsonify({"files": files})


@app.route('/delete-file', methods=['DELETE'])
async def delete_file():
    file_id = request.json['file_id']
    if file_id in virtual_files:
        del virtual_files[file_id]
        return jsonify({"message": "Файл удален"})
    return jsonify({"error": "Файл не найден"}), 404


@app.route('/run-code', methods=['POST'])
async def run_code():
    file_id = request.json['file_id']
    if file_id in virtual_files:
        code = virtual_files[file_id]['content']
        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            result = subprocess.run(['python', temp_file_path], capture_output=True, text=True, timeout=30)
            os.unlink(temp_file_path)

            return jsonify({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            })
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Выполнение кода превысило лимит времени"}), 408
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Файл не найден"}), 404


@app.route('/test', methods=['POST'])
async def run_tests():
    file_id = request.json['file_id']
    if file_id in virtual_files:
        code = virtual_files[file_id]['content']
        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            result = subprocess.run(['python', '-m', 'unittest', temp_file_path], capture_output=True, text=True,
                                    timeout=30)
            os.unlink(temp_file_path)

            return jsonify({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            })
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Выполнение тестов превысило лимит времени"}), 408
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Файл не найден"}), 404


@app.route('/install-package', methods=['POST'])
async def install_package():
    package_name = request.json['package_name']
    try:
        result = subprocess.run(['pip', 'install', package_name], capture_output=True, text=True, timeout=300)
        return jsonify({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Установка пакета превысила лимит времени"}), 408
    except Exception as e:
        return jsonify({"error": str(e)}), 500


async def execute_command(command):
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    try:
        if cmd == "/web_search":
            return await internet_search(args)
        elif cmd == "/create_file":
            file_name, file_type = args.split()
            file_id = await create_virtual_file(file_name, file_type)
            return f"Файл создан: {file_id}"
        elif cmd == "/read_file":
            content = await read_virtual_file(args)
            return f"Содержимое файла:\n{content}"
        elif cmd == "/write_file":
            file_id, content = args.split(maxsplit=1)
            success = await write_virtual_file(file_id, content)
            return "Файл обновлен" if success else "Ошибка при обновлении файла"
        elif cmd == "/list_files":
            return f"Список файлов:\n{virtual_files}"
        elif cmd == "/delete_file":
            if args in virtual_files:
                del virtual_files[args]
                return "Файл удален"
            return "Файл не найден"
        elif cmd == "/run_code":
            result = await run_virtual_code(args)
            return f"Результат выполнения:\nstdout: {result['stdout']}\nstderr: {result['stderr']}"
        elif cmd == "/test":
            # Здесь должна быть реализация запуска тестов
            return "Функция тестирования не реализована"
        elif cmd == "/install_package":
            # Здесь должна быть реализация установки пакета
            return f"Установка пакета {args} не реализована"
        else:
            return f"Неизвестная команда: {cmd}"
    except Exception as e:
        return f"Ошибка при выполнении команды {cmd}: {str(e)}"


# Вспомогательные функции
async def safe_request(method, url, **kwargs):
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug(f"Выполнение запроса: {method} {url} с параметрами {kwargs}")
            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка при выполнении запроса: {e}")
            return None


import ast
import re
from typing import List, Dict, Any, Optional
import io
import sys
from aiogram import types
import subprocess


class FileHandler:
    def __init__(self, base_dir: str = 'files'):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def write_file(self, filename: str, content: str) -> None:
        path = os.path.join(self.base_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def read_file(self, filename: str) -> Optional[str]:
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def delete_file(self, filename: str) -> None:
        path = os.path.join(self.base_dir, filename)
        if os.path.exists(path):
            os.remove(path)

    def send_file(self, filename: str, bot, message: types.Message) -> None:
        path = os.path.join(self.base_dir, filename)
        if os.path.exists(path):
            bot.send_document(chat_id=message.chat.id, document=types.FSInputFile(path, filename=filename))
            self.delete_file(filename)
        else:
            bot.send_message(chat_id=message.chat.id, text="Файл не найден.")


file_handler = FileHandler()


def send_file(self, filename: str, bot, message: types.Message) -> None:
    path = os.path.join(self.base_dir, filename)
    if os.path.exists(path):
        bot.send_document(chat_id=message.chat.id, document=types.FSInputFile(path, filename=filename))
        self.delete_file(filename)
    else:
        bot.send_message(chat_id=message.chat.id, text="Файл не найден.")


class CodeExecutionResult:
    def __init__(self, is_success: bool, output: str, error: Optional[str] = None,
                 logs: list = None, execution_time: float = 0):
        self.is_success = is_success
        self.output = output
        self.error = error
        self.logs = logs or []
        self.execution_time = execution_time


class CodeExecutor:
    def __init__(self):
        self.global_vars: Dict[str, Any] = {}
        self.timeout = 30
        self.max_memory = 512 * 1024 * 1024
        self.blocked_modules = {'os', 'subprocess', 'sys', 'importlib'}
        self.version_history = []
        self.validator = CodeValidator()
        self.resource_limiter = ResourceLimiter(self.max_memory, self.timeout)
        self._setup_logging()

    def _setup_logging(self):
        """Настройка логирования для экземпляра"""
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """Расширенная валидация кода"""
        try:
            # Проверка синтаксиса
            ast.parse(code)

            # Проверка безопасности
            is_safe, message = self.validator.check_code(code)
            if not is_safe:
                return False, message

            # Проверка импортов и вызовов
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = node.names[0].name.split('.')[0]
                    if module in self.blocked_modules:
                        return False, f"Запрещенный модуль: {module}"

                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in {'eval', 'exec', 'compile'}:
                            return False, f"Запрещенная функция: {node.func.id}"

            return True, "Код прошел валидацию"
        except SyntaxError as e:
            return False, f"Синтаксическая ошибка: {str(e)}"
        except Exception as e:
            return False, f"Ошибка валидации: {str(e)}"

    def create_sandbox(self) -> dict:
        """Создание изолированной среды выполнения"""
        safe_builtins = {
            name: getattr(__builtins__, name)
            for name in dir(__builtins__)
            if name not in {'eval', 'exec', 'compile', '__import__', 'open', 'input'}
        }
        return {'__builtins__': safe_builtins}

    def execute_code(self, code: str, language: str) -> CodeExecutionResult:
        """Безопасное выполнение кода с контролем версий"""
        start_time = time.time()
        logs = []

        # Валидация кода
        is_valid, message = self.validate_code(code)
        if not is_valid:
            return CodeExecutionResult(False, "", message, logs)

        try:
            # Сохранение версии
            version = {
                'timestamp': start_time,
                'code': code,
                'language': language
            }
            self.version_history.append(version)

            # Выполнение кода в зависимости от языка
            if language.lower() == 'python':
                result = self.execute_python(code)
            else:
                return CodeExecutionResult(
                    False,
                    "",
                    f"Неподдерживаемый язык: {language}",
                    logs
                )

            execution_time = time.time() - start_time
            result.execution_time = execution_time

            # Логирование результата
            self.logger.info(
                f"Код выполнен за {execution_time:.2f} сек. "
                f"Успешно: {result.is_success}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Ошибка выполнения кода: {str(e)}", exc_info=True)
            return CodeExecutionResult(False, "", str(e), logs)

    def execute_python(self, code: str) -> CodeExecutionResult:
        """Выполнение Python кода в песочнице с контролем ресурсов"""
        output_queue = Queue()
        error_queue = Queue()

        def execute_in_thread():
            try:
                with self.resource_limiter.limit_resources():
                    # Перенаправление stdout
                    old_stdout = sys.stdout
                    redirected_output = io.StringIO()
                    sys.stdout = redirected_output

                    try:
                        # Выполнение в песочнице
                        sandbox = self.create_sandbox()
                        exec(code, sandbox)
                        output_queue.put(redirected_output.getvalue())
                    except Exception as e:
                        error_queue.put(str(e))
                    finally:
                        sys.stdout = old_stdout
            except Exception as e:
                error_queue.put(str(e))

        # Запуск выполнения в отдельном потоке
        thread = threading.Thread(target=execute_in_thread)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            thread.join(0)  # Принудительное завершение потока
            return CodeExecutionResult(
                False,
                "",
                "Превышен таймаут выполнения",
                ["Выполнение прервано по таймауту"]
            )

        # Проверка результатов
        if not error_queue.empty():
            error = error_queue.get()
            return CodeExecutionResult(
                False,
                "",
                error,
                [f"Ошибка выполнения: {error}"]
            )

        output = output_queue.get() if not output_queue.empty() else ""
        return CodeExecutionResult(
            True,
            output,
            logs=["Код успешно выполнен"]
        )

    def rollback_to_version(self, timestamp: float) -> Optional[dict]:
        """Откат к предыдущей версии кода"""
        for version in reversed(self.version_history):
            if version['timestamp'] <= timestamp:
                self.logger.info(f"Выполнен откат к версии от {version['timestamp']}")
                return version
        return None


class CodeValidator:
    def __init__(self):
        self.dangerous_patterns = [
            r'os\.',
            r'subprocess\.',
            r'sys\.',
            r'importlib\.',
            r'eval\(',
            r'exec\(',
            r'__import__',
            r'open\(',
            r'input\(',
        ]

    def check_code(self, code: str) -> Tuple[bool, str]:
        """Проверка кода на наличие опасных паттернов"""
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code):
                return False, f"Обнаружен опасный паттерн: {pattern}"
        return True, ""


class ResourceLimiter:
    def __init__(self, max_memory: int, timeout: int):
        self.max_memory = max_memory
        self.timeout = timeout

    @contextmanager
    def limit_resources(self):
        """Установка ограничений на ресурсы"""
        try:
            if sys.platform != 'win32':  # Только для UNIX-подобных систем
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (self.max_memory, self.max_memory)
                )
            signal.alarm(self.timeout)
            yield
        finally:
            signal.alarm(0)


import logging.handlers


def setup_logging():
    """Расширенная настройка логирования"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Ротация логов по размеру
    file_handler = logging.handlers.RotatingFileHandler(
        'app.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    # Форматирование логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Добавляем обработчик к логгеру
    logger.addHandler(file_handler)

    return logger


class VerificationResult(NamedTuple):
    is_valid: bool
    error_message: Optional[str]
    logs: List[str]


class Verifier:
    def __init__(self, blocked_functions):
        self.blocked_functions = blocked_functions
        self.allowed_modules = set(sys.builtin_module_names)
        self.allowed_modules.update(sys.stdlib_module_names)

    def verify_code(self, code: str, language: str) -> VerificationResult:
        # Реализация метода verify_code остается без изменений
        pass

    def add_custom_module(self, module_name):
        self.allowed_modules.add(module_name)

    def verify_code(self, code: str, language: str) -> VerificationResult:
        if language.lower() == 'python':
            return self.verify_python(code)
        else:
            return VerificationResult(False, f"Верификация для языка {language} не реализована", [])

    def verify_python(self, code: str) -> VerificationResult:
        logs = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.allowed_modules:
                            # Попытка импортировать модуль
                            try:
                                importlib.import_module(alias.name)
                                self.allowed_modules.add(alias.name)
                                logs.append(
                                    f"Модуль '{alias.name}' успешно импортирован и добавлен в список разрешенных.")
                            except ImportError:
                                return VerificationResult(False, f"Не удалось импортировать модуль '{alias.name}'.",
                                                          logs)
                elif isinstance(node, ast.ImportFrom):
                    if node.module not in self.allowed_modules:
                        try:
                            importlib.import_module(node.module)
                            self.allowed_modules.add(node.module)
                            logs.append(f"Модуль '{node.module}' успешно импортирован и добавлен в список разрешенных.")
                        except ImportError:
                            return VerificationResult(False, f"Не удалось импортировать модуль '{node.module}'.", logs)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in self.blocked_functions:
                        return VerificationResult(False, f"Использование функции '{node.func.id}' запрещено.", logs)

            return VerificationResult(True, None, logs)
        except SyntaxError as e:
            return VerificationResult(False, f"Синтаксическая ошибка: {str(e)}", logs)
        except Exception as e:
            return VerificationResult(False, f"Неожиданная ошибка при верификации: {str(e)}", logs)


class CodeInterpreter:
    def __init__(self, blocked_functions):
        self.verifier = Verifier(blocked_functions)
        self.executor = CodeExecutor()

    @staticmethod
    def extract_code(text: str) -> List[Dict[str, str]]:
        # Ищем код в блоках, обрамленных тройными обратными кавычками
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        code_blocks = [{'language': match[0] or 'python', 'code': match[1].strip()} for match in matches]

        # Если код не найден в блоках, считаем весь текст Python-кодом
        if not code_blocks:
            code_blocks = [{'language': 'python', 'code': text.strip()}]

        return code_blocks

    def process_code(self, text: str) -> Dict[str, Any]:
        logger.info("Обработка кода")
        code_blocks = self.extract_code(text)
        if not code_blocks:
            logger.warning("Не найден исполняемый код")
            return {
                "verification_status": "INCORRECT",
                "verification_error": "Не найден исполняемый код.",
                "execution_status": "NONE",
                "execution_result": "Код не был выполнен, так как не найден исполняемый код.",
                "logs": ["Не найден исполняемый код"]
            }

        combined_result = {
            "verification_status": "CORRECT",
            "verification_error": None,
            "execution_status": "SUCCESS",
            "execution_result": "",
            "logs": []
        }

        for block in code_blocks:
            language = block['language']
            code = block['code']

            # Проверка на пустой код
            if not code.strip():
                logger.warning("Найден пустой блок кода")
                continue

            # Верификация кода
            verification_result = self.verifier.verify_code(code, language)
            combined_result["logs"].extend(verification_result.logs)

            if not verification_result.is_valid:
                combined_result["verification_status"] = "INCORRECT"
                combined_result["verification_error"] = verification_result.error_message
                combined_result["execution_status"] = "NONE"
                combined_result[
                    "execution_result"] += f"[{language}] Код не был выполнен из-за ошибки верификации: {verification_result.error_message}\n"
                continue

            # Выполнение кода
            result = self.executor.execute_code(code, language)
            combined_result["logs"].extend(result.logs)

            if result.is_success:
                combined_result["execution_result"] += f"[{language}] {result.output}\n"
            else:
                combined_result["execution_status"] = "FAILURE"
                combined_result["execution_result"] += f"[{language}] Ошибка: {result.error}\n"

        combined_result["execution_result"] = combined_result["execution_result"].strip()
        return combined_result

    def update_session_variables(self, session_variables: Dict[str, Any]):
        self.executor.update_session_var(session_variables)


async def create_project_structure(code):
    logger.debug("Создание структуры проекта...")
    try:
        file_data = await safe_request('POST', 'http://127.0.0.1:5000/create-file', json={
            "file_name": "main",
            "file_type": "py"
        })
        if file_data and 'file_id' in file_data:
            file_id = file_data['file_id']
            logger.debug(f"Создан файл с ID: {file_id}")
            await safe_request('POST', 'http://127.0.0.1:5000/fill-file', json={
                "file_id": file_id,
                "content": code
            })
            return file_id, f"{file_id}/main.py"
        else:
            logger.error("Не удалось получить file_id при создании файла")
            return None, None
    except Exception as e:
        logger.error(f"Ошибка при создании структуры проекта: {str(e)}")
        return None, None


async def install_dependencies(code):
    logger.info("Установка зависимостей...")
    requirements = await generate_safe_content(
        f"Перечисли все необходимые библиотеки Python для этого кода в формате requirements.txt (только названия библиотек, по одной на строку, без версий):\n{code}")
    if not requirements or "не требуется" in requirements.lower():
        logger.info("Зависимости не требуются.")
        return
    req_file_id = (await safe_request('POST', 'http://127.0.0.1:5000/create-file', json={
        "file_name": "requirements",
        "file_type": "txt"
    }))['file_id']
    logger.debug(f"Создан файл requirements с ID: {req_file_id}")
    await safe_request('POST', 'http://127.0.0.1:5000/fill-file', json={
        "file_id": req_file_id,
        "content": requirements
    })
    clean_requirements = "\n".join([line.strip() for line in requirements.split("\n") if
                                    line.strip() and not line.startswith("#") and not line.startswith("**")])
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(clean_requirements)
        temp_file_path = temp_file.name
    try:
        logger.debug(f"Установка зависимостей из файла: {temp_file_path}")
        process = await asyncio.create_subprocess_exec(
            "pip", "install", "-r", temp_file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            logger.info("Зависимости успешно установлены.")
        else:
            logger.error(f"Ошибка при установке зависимостей: {stderr.decode()}")
    except Exception as e:
        logger.error(f"Ошибка при установке зависимостей: {str(e)}")
        logger.info("Продолжаем выполнение без установки зависимостей")
    finally:
        os.unlink(temp_file_path)


async def run_code(file_path):
    logger.info(f"Запуск кода из файла {file_path}...")
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as temp_file:
        with open(f"temp/{file_path}", 'r') as f:
            temp_file.write(f.read())
        temp_file_path = temp_file.name
    try:
        process = await asyncio.create_subprocess_exec(
            "python", temp_file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        logger.info("Код выполнен.")
        logger.debug(f"Результат выполнения: stdout: {stdout.decode()}, stderr: {stderr.decode()}")
        return stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        logger.error("Время выполнения кода превысило лимит.")
        return "", "Время выполнения кода превысило лимит."
    finally:
        os.unlink(temp_file_path)


def search_internet(query):
    logger.info(f"Поиск в интернете: {query}")
    response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
    return response.json()


def create_zip_file(file_id):
    logger.info("Создание zip-архива проекта...")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(f"temp/{file_id}"):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    zip_buffer.seek(0)
    logger.info("Архив проекта создан.")
    return zip_buffer


def generate_content_groq(prompt, model):
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка при вызове Groq API: {str(e)}")
        return None


logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((asyncio.TimeoutError, ConnectionError, httpx.HTTPStatusError))
)
async def generate_safe_content(prompt: str, model: str) -> Optional[str]:
    try:
        if model in ["gemini-1.5-pro-002", "gemini-1.5-flash-002", "gemini-1.5-flash-001",
                     "gemini-1.5-pro-exp-0827", "gemini-1.5-flash-exp-0827", "gemini-1.5-flash-8b-exp-0827"]:
            genai.configure(api_key=GOOGLE_API_KEY)
            model_instance = genai.GenerativeModel(model)
            response = await asyncio.wait_for(asyncio.to_thread(model_instance.generate_content, prompt), timeout=30)
            return response.text
        elif model == "Qwen/Qwen2.5-72B-Instruct":
            response = await asyncio.wait_for(asyncio.to_thread(
                hyperbolic_client.completions.create,
                model=model,
                prompt=prompt,
                max_tokens=1000
            ), timeout=30)
            return response.choices[0].text
        else:
            logger.error(f"Неподдерживаемая модель: {model}")
            return None
    except asyncio.TimeoutError:
        logger.error(f"Таймаут при запросе к модели {model}")
        logger.error(traceback.format_exc())
        raise
    except ConnectionError as e:
        logger.error(f"Ошибка подключения к API для модели {model}: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    except openai.APIError as e:
        logger.error(f"Ошибка API при запросе к модели {model}: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка при генерации контента для модели {model}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


async def run_code_and_get_output(code, file_id):
    logger.info("Запуск кода и получение вывода...")
    try:
        file_data = await safe_request('POST', 'http://127.0.0.1:5000/create-file', json={
            "file_name": "temp_execution",
            "file_type": "py"
        })
        if file_data and 'file_id' in file_data:
            temp_file_id = file_data['file_id']
            await safe_request('POST', 'http://127.0.0.1:5000/fill-file', json={
                "file_id": temp_file_id,
                "content": code
            })
            stdout, stderr = await run_code(f"{temp_file_id}/temp_execution.py")
            return stdout, stderr
        else:
            return "", "Ошибка при создании временного файла для выполнения"
    except Exception as e:
        logger.error(f"Ошибка при выполнении кода: {str(e)}")
        return "", str(e)


async def internet_search(query):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_NEWS_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": 5
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'items' in data:
                        search_results = [
                            f"Заголовок: {item['title']}\nОписание: {item['snippet']}\nURL: {item['link']}\n"
                            for item in data['items']
                        ]
                        result = "\n".join(search_results)
                        logger.info(f"Результаты поиска:\n{result}")
                        return result
                    else:
                        logger.warning("Результаты поиска не найдены")
                        return "Результаты поиска не найдены"
                else:
                    logger.error(f"Ошибка при выполнении поиска: HTTP статус {response.status}")
                    return f"Ошибка при выполнении поиска: HTTP статус {response.status}"
    except Exception as e:
        logger.error(f"Ошибка при выполнении поиска в интернете: {str(e)}")
        return f"Ошибка при выполнении поиска: {str(e)}"


async def generate_code(user_request, model, progress_callback):
    completed_stages = []
    failed_stages = []

    # Шаг 1: Планирование
    await progress_callback("Начало: Планирование")
    try:
        plan = await generate_safe_content(f"Создай пошаговый план для решения следующей задачи: {user_request}", model)
        await progress_callback(f"Планирование завершено:\n{plan}")
        completed_stages.append("Планирование")
    except Exception as e:
        await progress_callback(f"Планирование не удалось: {str(e)}")
        failed_stages.append("Планирование")

    # Шаг 2: Генерация кода
    await progress_callback("Начало: Генерация кода")
    code = ""
    try:
        for step in plan.strip().split('\n'):
            if step.strip():
                step_code = await generate_safe_content(f"Напиши код для этого шага: {step}", model)
                code += step_code + "\n\n"
                await progress_callback(f"Код для шага '{step}' сгенерирован.")
        await progress_callback("Генерация кода завершена.")
        completed_stages.append("Генерация кода")
    except Exception as e:
        await progress_callback(f"Генерация кода не удалась: {str(e)}")
        failed_stages.append("Генерация кода")

    # Шаг 3-5: Проверка кода на ошибки и исправление
    await progress_callback("Начало: Проверка и исправление кода")
    error_count = 0
    max_attempts = 3
    while error_count < max_attempts:
        try:
            file_id, main_file = await create_project_structure(code)
            if file_id is None:
                await progress_callback("Проверка кода не удалась: Не удалось создать структуру проекта.")
                failed_stages.append("Проверка кода")
                break
            await install_dependencies(code)
            stdout, stderr = await run_code(main_file)
            if not stderr:
                await progress_callback("Код успешно прошел проверку.")
                completed_stages.append("Проверка кода")
                break
            await progress_callback(f"Обнаружены ошибки:\n{stderr}")
            error_fix_plan = await generate_safe_content(f"Создай план для исправления этих ошибок: {stderr}", model)
            code = await generate_safe_content(f"Исправь код согласно этому плану:\n{error_fix_plan}\n\nКод:\n{code}",
                                               model)
            error_count += 1
            await progress_callback(f"Попытка исправления #{error_count} выполнена.")
        except Exception as e:
            await progress_callback(f"Проверка кода не удалась: {str(e)}")
            failed_stages.append("Проверка кода")
            break

    # Запасной метод, если код все еще содержит ошибки
    if error_count == max_attempts:
        try:
            backup_code = await generate_safe_content(
                f"Создай простой рабочий код, который частично решает задачу: {user_request}", model)
            code = backup_code
            await progress_callback("Создание запасного кода завершено.")
            completed_stages.append("Создание запасного кода")
        except Exception as e:
            await progress_callback(f"Создание запасного кода не удалось: {str(e)}")
            failed_stages.append("Создание запасного кода")

    # Шаг 6: Автоматическое добавление библиотек
    await progress_callback("Начало: Установка зависимостей")
    try:
        await install_dependencies(code)
        await progress_callback("Установка зависимостей завершена.")
        completed_stages.append("Установка зависимостей")
    except Exception as e:
        await progress_callback(f"Установка зависимостей не удалась: {str(e)}")
        failed_stages.append("Установка зависимостей")

    # Шаг 7: Проверка соответствия первоначальному запросу
    await progress_callback("Начало: Проверка соответствия запросу")
    try:
        compliance_check = await generate_safe_content(
            f"Проверь, соответствует ли этот код запросу пользователя:\n{user_request}\n\nКод:\n{code}", model)
        if "не соответствует" in compliance_check.lower():
            improvement_plan = await generate_safe_content(
                f"Создай план для улучшения кода, чтобы он соответствовал запросу: {user_request}", model)
            code = await generate_safe_content(
                f"Улучшите код согласно этому плану:\n{improvement_plan}\n\nКод:\n{code}", model)
            await progress_callback("Код был улучшен для соответствия запросу.")
        else:
            await progress_callback("Код соответствует запросу.")
        completed_stages.append("Проверка соответствия запросу")
    except Exception as e:
        await progress_callback(f"Проверка соответствия запросу не удалась: {str(e)}")
        failed_stages.append("Проверка соответствия запросу")

    # Шаг 8: Оптимизация кода
    await progress_callback("Начало: Оптимизация кода")
    try:
        optimization_plan = await generate_safe_content(f"Создай план для оптимизации этого кода:\n{code}", model)
        code = await generate_safe_content(
            f"Оптимизируй код согласно этому плану:\n{optimization_plan}\n\nКод:\n{code}", model)
        await progress_callback("Оптимизация кода завершена.")
        completed_stages.append("Оптимизация кода")
    except Exception as e:
        await progress_callback(f"Оптимизация кода не удалась: {str(e)}")
        failed_stages.append("Оптимизация кода")

    return code, file_id, completed_stages, failed_stages


def parse_file_structure(code: str) -> List[Tuple[str, str]]:
    files = []
    current_file = ""
    current_content = []

    for line in code.split('\n'):
        if line.startswith("# File: "):
            if current_file and current_content:
                files.append((current_file, '\n'.join(current_content)))
            current_file = line[7:].strip()
            current_content = []
        else:
            current_content.append(line)

    if current_file and current_content:
        files.append((current_file, '\n'.join(current_content)))

    return files


def parse_code_blocks(text: str) -> List[Dict[str, str]]:
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    code_blocks = []
    for match in matches:
        language = match[0] or 'python'
        code = match[1].strip()
        code_blocks.append({'language': language, 'code': code})
    return code_blocks


async def pro_mode(full_code: str, user_request: str, model, progress_callback, process_callback) -> str:
    try:
        await progress_callback("Начало Pro Mode")

        # Анализ соответствия требованиям
        await progress_callback("Анализ соответствия требованиям...")
        analysis = await generate_safe_content(f"""Проанализируй следующий код на соответствие первоначальному запросу:

        Первоначальный запрос:
        {user_request}

        Код:
        {full_code}

        Оцени наличие всех требуемых функций, внешних показателей и соответствие общей структуры запросу.
        Предложи конкретные улучшения, если что-то не соответствует или может быть улучшено.""", model)

        if analysis is None:
            raise Exception("Не удалось получить анализ от модели")

        yield await process_callback("Анализ соответствия", analysis)

        # Шаг 2: Планирование улучшений
        await progress_callback("Планирование улучшений...")
        improvement_plan = await generate_safe_content(f"""На основе предыдущего анализа, составь пошаговый план улучшения кода:

        Анализ:
        {analysis}

        Код:
        {code}

        Составь список конкретных шагов для улучшения кода, нумеруя каждый шаг.""", model)

        await process_callback("План улучшений", improvement_plan)

        # Шаг 3: Реализация улучшений
        await progress_callback("Реализация улучшений...")
        improvement_steps = improvement_plan.split('\n')
        for step in improvement_steps:
            if step.strip() and step[0].isdigit():
                await progress_callback(f"Выполнение шага: {step}")
                improved_code = await generate_safe_content(f"""Реализуй следующее улучшение в коде:

                Улучшение: {step}

                Текущий код:
                {improved_code}

                Внеси необходимые изменения, сохраняя общую структуру и функциональность кода.""", model)

                await process_callback(f"Результат шага: {step}", improved_code)

        # Шаг 4: Инновационная система улучшения - "Контекстное обогащение"
        await progress_callback("Применение контекстного обогащения...")
        enrichment_prompts = [
            "Как можно применить последние тенденции в разработке ПО к этому коду?",
            "Какие паттерны проектирования могли бы улучшить структуру этого кода?",
            "Как можно оптимизировать производительность этого кода?",
            "Какие методы обеспечения безопасности следует применить к этому коду?",
            "Как можно улучшить читаемость и поддерживаемость этого кода?"
        ]

        for prompt in enrichment_prompts:
            enhancement = await generate_safe_content(f"{prompt}\n\nТекущий код:\n{improved_code}", model)
            improved_code = await generate_safe_content(
                f"Примени следующее улучшение к коду, сохраняя его основную функциональность:\n{enhancement}\n\nТекущий код:\n{improved_code}",
                model)
            await process_callback(f"Контекстное обогащение: {prompt}", improved_code)

        # Новый шаг: Анализ структуры кода
        await progress_callback("Анализ структуры кода...")
        try:
            code_structure = list_code_definition_names('.')
            code_analysis = await generate_safe_content(
                f"Проанализируй структуру кода:\n{code_structure}\n\nПредложи улучшения на основе этого анализа.",
                model)
            await process_callback("Анализ структуры кода", code_analysis)
        except UnicodeDecodeError:
            logger.warning("Не удалось проанализировать структуру кода из-за ошибки кодировки")
            code_analysis = "Не удалось проанализировать структуру кода из-за ошибки кодировки"

        # Применение улучшений на основе анализа структуры
        improved_code = await generate_safe_content(
            f"Примени следующие улучшения к коду:\n{code_analysis}\n\nТекущий код:\n{improved_code}", model)
        await process_callback("Улучшения на основе анализа структуры", improved_code)

        # Поиск потенциальных проблем
        await progress_callback("Поиск потенциальных проблем...")
        potential_issues = search_files('.', r'\b(TODO|FIXME|XXX|HACK)\b')
        if potential_issues:
            issues_fix = await generate_safe_content(
                f"Исправь следующие потенциальные проблемы в коде:\n{potential_issues}\n\nТекущий код:\n{improved_code}",
                model)
            improved_code = issues_fix

        # Шаг 5: Финальная оптимизация
        await progress_callback("Финальная оптимизация...")
        final_code = await generate_safe_content(f"""Проведи финальную оптимизацию кода, учитывая все предыдущие улучшения и анализ. 
        Убедись, что код соответствует изначальным требованиям, оптимизирован по производительности, 
        безопасен, читабелен и легко поддерживаем. Добавь подробные комментарии к ключевым частям кода.

        Код для оптимизации:
        {improved_code}""", model)

        await process_callback("Финальная оптимизация", final_code)

        await progress_callback("Pro Mode завершен")
        yield final_code
    except Exception as e:
        logger.error(f"Ошибка в pro_mode: {str(e)}", exc_info=True)
        await progress_callback(f"Произошла ошибка в Pro Mode: {str(e)}")
        yield full_code


# Глобальная переменная для управления состоянием программы
running = True


def signal_handler(signum, frame):
    global running
    print("Получен сигнал завершения. Завершение работы...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def process_wrapper(user_request, use_pro_mode, model):
    progress_messages = []
    process_steps = []
    code_output = ""
    file_path = None

    async def progress_callback(message):
        progress_messages.append(message)
        return "\n".join(progress_messages)

    async def process_callback(step_name, model_response):
        process_steps.append(f"### {step_name}\n{model_response}")
        nonlocal code_output
        if step_name.startswith("Код для") or step_name == "Итоговый код":
            code_output = model_response
        return "\n".join(progress_messages), "\n\n".join(process_steps), code_output, file_path

    try:
        async for result in process_request(user_request, use_pro_mode, model, progress_callback, process_callback):
            yield result
    except Exception as e:
        logger.error(f"Ошибка в process_wrapper: {str(e)}", exc_info=True)
        error_message = f"### Ошибка\n{str(e)}"
        yield error_message, "\n\n".join(process_steps), code_output, None


async def process_request(user_request, use_pro_mode, model, progress_callback, process_callback,
                          use_virtual_files=True):
    try:
        start_time = time.time()
        await progress_callback("Начало обработки запроса")

        # Список всех доступных моделей
        all_models = MODELS.copy()
        if model in all_models:
            all_models.remove(model)
            all_models.insert(0, model)

        # Шаг 1: Улучшенное планирование
        await progress_callback("Генерация плана...")
        plan = None
        for current_model in all_models:
            try:
                plan = await generate_safe_content(
                    f"""Создай детальный план разработки для решения следующей задачи: {user_request}
                План должен включать:
                1. Анализ требований
                2. Структуру проекта
                3. Основные функции и классы
                4. Порядок реализации
                5. Потенциальные сложности
                Не пиши код, только план разработки.""", current_model)
                if plan:
                    await progress_callback(f"План успешно сгенерирован с использованием модели {current_model}")
                    logger.info(f"План сгенерирован: {plan[:100]}...")  # Логируем первые 100 символов плана
                    yield await process_callback("Планирование", plan)
                    break
            except Exception as e:
                logger.error(f"Ошибка при использовании модели {current_model}: {str(e)}")
                await progress_callback(f"Не удалось использовать модель {current_model}: {str(e)}")

        if plan is None:
            error_msg = "Не удалось сгенерировать план ни с одной из доступных моделей"
            logger.error(error_msg)
            await progress_callback(error_msg)
            yield await process_callback("Ошибка планирования", error_msg)
            return

        # Шаг 2: Генерация структуры проекта
        await progress_callback("Создание структуры проекта...")
        project_structure = None
        for current_model in all_models:
            try:
                project_structure = await generate_safe_content(
                    f"На основе этого плана:\n{plan}\nСоздай структуру проекта с именами файлов и их кратким описанием.",
                    current_model)
                if project_structure:
                    await progress_callback(
                        f"Структура проекта успешно сгенерирована с использованием модели {current_model}")
                    logger.info(f"Структура проекта сгенерирована: {project_structure[:100]}...")
                    yield await process_callback("Структура проекта", project_structure)
                    break
            except Exception as e:
                logger.warning(f"Не удалось использовать модель {current_model} для структуры проекта: {str(e)}")
                await progress_callback(
                    f"Не удалось использовать модель {current_model} для структуры проекта, пробуем следующую")

        if project_structure is None:
            error_msg = "Не удалось сгенерировать структуру проекта ни с одной из доступных моделей"
            logger.error(error_msg)
            await progress_callback(error_msg)
            yield await process_callback("Ошибка создания структуры", error_msg)
            return

        # Шаг 3: Генерация кода для всего проекта
        await progress_callback("Генерация кода...")
        logger.debug("Начало генерации кода...")
        full_code = None
        for current_model in all_models:
            try:
                full_code = await generate_safe_content(f"""На основе следующего плана и структуры проекта, напиши полный код для всех файлов:

                План:
                {plan}

                Структура проекта:
                {project_structure}

                Пожалуйста, включи все необходимые импорты и создай полноценный, рабочий код для каждого файла. Разделяй файлы комментариями с названием файла.""",
                                                        current_model)
                if full_code:
                    await progress_callback(f"Полный код успешно сгенерирован с использованием модели {current_model}")
                    logger.info(f"Полный код сгенерирован: {full_code[:100]}...")
                    yield await process_callback("Генерация кода", full_code)
                    break
            except Exception as e:
                logger.warning(f"Не удалось использовать модель {current_model} для генерации кода: {str(e)}")
                await progress_callback(
                    f"Не удалось использовать модель {current_model} для генерации кода, пробуем следующую")

        if full_code is None:
            error_msg = "Не удалось сгенерировать код ни с одной из доступных моделей"
            logger.error(error_msg)
            await progress_callback(error_msg)
            yield await process_callback("Ошибка генерации кода", error_msg)
            return

        # Шаг 4: Обработка и исправление ошибок
        await progress_callback("Обработка и исправление ошибок...")
        blocked_functions = ["eval", "exec", "execfile", "compile"]
        code_interpreter = CodeInterpreter(blocked_functions)

        max_attempts = 5
        for attempt in range(max_attempts):
            code_result = code_interpreter.process_code(full_code)
            if code_result["verification_status"] == "CORRECT" and code_result["execution_status"] == "SUCCESS":
                await progress_callback(f"Код успешно выполнен на попытке {attempt + 1}")
                yield await process_callback(f"Успешное выполнение кода (попытка {attempt + 1})", full_code)
                break

            await progress_callback(f"Обнаружены ошибки. Попытка исправления {attempt + 1}/{max_attempts}...")
            error_message = code_result["verification_error"] or code_result["execution_result"]
            logger.warning(f"Ошибка выполнения кода: {error_message}")

            if attempt == max_attempts - 1:
                await progress_callback("Достигнуто максимальное количество попыток. Поиск решения в интернете...")
                search_query = f"Python error: {error_message}"
                search_results = await internet_search(search_query)
                await progress_callback(f"Результаты поиска в интернете:\n{search_results}")
                error_fix = await generate_safe_content(
                    f"Исправь следующие ошибки в коде, используя эту информацию:\n{search_results}\n\nОшибки:\n{error_message}\n\nТекущий код:\n{full_code}",
                    model)
            else:
                error_fix = await generate_safe_content(
                    f"Исправь следующие ошибки в коде, сохраняя его функциональность:\n{error_message}\n\nТекущий код:\n{full_code}",
                    model)

            if error_fix is not None:
                full_code = error_fix.strip('`')
            else:
                logger.error("Не удалось получить исправленный код")
                full_code = ""  # или другое значение по умолчанию

            yield await process_callback(f"Исправление ошибок (попытка {attempt + 1})", full_code)

        # Проверка на успешность выполнения кода
        final_code_result = code_interpreter.process_code(full_code)
        if final_code_result["verification_status"] != "CORRECT" or final_code_result["execution_status"] != "SUCCESS":
            await progress_callback("Внимание: Код все еще содержит ошибки после всех попыток исправления")
            yield await process_callback("Финальный код с ошибками", full_code)

        # Создание архива проекта
        await progress_callback("Создание архива проекта...")
        try:
            with tempfile.TemporaryDirectory() as project_dir:
                files = parse_file_structure(full_code)
                for file_name, file_content in files:
                    file_handler.write_file(file_name, file_content)
                archive_path = create_project_archive(project_dir)
            await progress_callback("Проект упакован и готов к скачиванию")
            yield await process_callback("Архив проекта", archive_path)
        except Exception as e:
            logger.error(f"Ошибка при создании архива проекта: {str(e)}")
            await progress_callback(f"Ошибка при создании архива проекта: {str(e)}")
            yield await process_callback("Ошибка архивации", str(e))

        # Pro Mode (если включен)
        if use_pro_mode:
            await progress_callback("Начало Pro Mode")
            try:
                pro_results = await pro_mode(full_code, user_request, model, progress_callback, process_callback)
                for pro_result in pro_results:
                    if isinstance(pro_result, str):
                        full_code = pro_result
                yield await process_callback("Результат Pro Mode", pro_results)

                # Обновление архива после Pro Mode
                with tempfile.TemporaryDirectory() as project_dir:
                    for file_name, file_content in parse_file_structure(full_code):
                        file_handler.write_file(file_name, file_content)
                    archive_path = create_project_archive(project_dir)
                await progress_callback("Обновленный проект упакован и готов к скачиванию")
                yield await process_callback("Обновленный архив проекта", archive_path)
            except Exception as e:
                logger.error(f"Ошибка в Pro Mode: {str(e)}")
                await progress_callback(f"Ошибка в Pro Mode: {str(e)}")
                yield await process_callback("Ошибка Pro Mode", str(e))

        # Поиск в Google и доработка кода
        await progress_callback("Поиск дополнительной информации в интернете...")
        try:
            search_results = await internet_search(user_request)
            if search_results:
                await progress_callback("Доработка кода с учетом найденной информации...")
                full_code = await generate_safe_content(
                    f"Улучши этот код, используя следующую информацию:\n{search_results}\n\nТекущий код:\n{full_code}",
                    model)
                yield await process_callback("Код после доработки", full_code)

                # Обновление архива после финальной доработки
                with tempfile.TemporaryDirectory() as project_dir:
                    for file_name, file_content in parse_file_structure(full_code):
                        file_handler.write_file(file_name, file_content)
                    archive_path = create_project_archive(project_dir)
                yield await process_callback("Финальный архив проекта", archive_path)
        except Exception as e:
            logger.error(f"Ошибка при поиске и доработке кода: {str(e)}")
            await progress_callback(f"Ошибка при поиске и доработке кода: {str(e)}")
            yield await process_callback("Ошибка финальной доработки", str(e))

        end_time = time.time()
        await progress_callback(f"### Завершено\nВремя выполнения: {end_time - start_time:.2f} секунд")

        yield full_code, archive_path
    except Exception as e:
        logger.error(f"Критическая ошибка в process_request: {str(e)}", exc_info=True)
        await progress_callback(f"Произошла критическая ошибка: {str(e)}")
        yield await process_callback("Критическая ошибка", str(e))
        yield None, None


@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    else:
        return "Файл не найден", 404


async def model_to_model_mode(user_request, model1, model2, progress_callback, process_callback):
    try:
        logger.debug("Начало Model to Model Mode")
        await progress_callback("Начало Model to Model Mode")

        # Генерация кода первой моделью
        logger.debug("Генерация кода первой моделью")
        code = await generate_safe_content(f"Напиши полный код для решения этой задачи: {user_request}", model1)
        yield await process_callback("Генерация кода (Модель 1)", code)

        max_iterations = 5
        for i in range(max_iterations):
            logger.debug(f"Итерация {i + 1}")
            # Проверка кода второй моделью
            logger.debug("Проверка кода второй моделью")
            review = await generate_safe_content(f"Проверь этот код на ошибки и предложи улучшения:\n{code}", model2)
            yield await process_callback(f"Проверка кода (Модель 2, итерация {i + 1})", review)

            if "ошибок не найдено" in review.lower():
                logger.debug("Код прошел проверку без ошибок")
                await progress_callback("Код прошел проверку без ошибок")
                break

            # Исправление кода первой моделью
            logger.debug("Исправление кода первой моделью")
            code = await generate_safe_content(
                f"Исправь код согласно этим замечаниям:\n{review}\n\nТекущий код:\n{code}", model1)
            yield await process_callback(f"Исправление кода (Модель 1, итерация {i + 1})", code)

        logger.debug("Завершение Model to Model Mode")
        yield gr.update(value="Model to Model Mode завершен"), gr.update(value=code), gr.update(value=code), gr.update(
            value=None)
    except Exception as e:
        logger.error(f"Ошибка в model_to_model_mode: {str(e)}", exc_info=True)
        await progress_callback(f"### Ошибка\n{str(e)}")
        yield gr.update(value=f"Ошибка: {str(e)}"), gr.update(), gr.update(), gr.update()


# Обновленная функция generate_response

async def generate_response(user_request, use_pro_mode, model, use_model_to_model=False):
    progress_messages = []
    process_steps = []
    code_output = ""
    file_path = None
    result_queue = asyncio.Queue()
    code_executor = CodeExecutor()

    async def progress_callback(message):
        logger.debug(f"Progress: {message}")
        progress_messages.append(message)
        await result_queue.put((
            gr.update(value="\n".join(progress_messages)),
            gr.update(),
            gr.update(),
            gr.update()
        ))

    async def process_callback(step_name, content):
        logger.debug(f"Process step: {step_name}")
        process_steps.append(f"### {step_name}\n{content}")
        nonlocal code_output
        if step_name.startswith("Код для") or step_name == "Итоговый код":
            code_output = content
        await result_queue.put((
            gr.update(),
            gr.update(value="\n\n".join(process_steps)),
            gr.update(value=code_output, language="python"),
            gr.update()
        ))

    async def execute_code_safely(code_executor, code, language):
        try:
            execution_result = code_executor.execute_code(code, language)
            return execution_result.output
        except SyntaxError as se:
            return f"Ошибка синтаксиса: {str(se)}"
        except Exception as e:
            return f"Ошибка выполнения кода: {str(e)}"

    try:
        logger.debug("Начало generate_response")
        await progress_callback("Начало обработки запроса")

        if use_model_to_model:
            logger.debug("Запуск режима Model to Model")
            async for result in model_to_model_mode(user_request, "gemini-1.5-pro-exp-0827",
                                                    "gemini-1.5-flash-8b-exp-0827", progress_callback,
                                                    process_callback):
                await result_queue.put(result)
        else:
            logger.debug("Запуск обычного режима")
            async for result in process_request(user_request, use_pro_mode, model, progress_callback, process_callback):
                if isinstance(result, tuple) and len(result) == 2:
                    code, archive_path = result
                    if code:
                        code_output = code
                        # Используем execute_code_safely
                        execution_result = await execute_code_safely(code_executor, code_output, 'python')
                        await process_callback("Результат выполнения кода", execution_result)
                    if archive_path and os.path.exists(archive_path):
                        file_path = archive_path
                    else:
                        file_path = None
                        await progress_callback(f"Файл архива не найден: {archive_path}")
                await result_queue.put((
                    gr.update(value="\n".join(progress_messages)),
                    gr.update(value="\n\n".join(process_steps)),
                    gr.update(value=code_output, language="python"),
                    gr.update(value=file_path) if file_path else gr.update()
                ))

        if use_pro_mode and code_output:
            logger.debug(f"Запуск Pro режима. Код для улучшения: {code_output[:100]}...")
            await progress_callback("Активирован Pro режим")
            try:
                async for improved_code in pro_mode(code_output, user_request, model, progress_callback,
                                                    process_callback):
                    if improved_code:
                        code_output = improved_code
                        # Используем execute_code_safely для улучшенного кода
                        execution_result = await execute_code_safely(code_executor, code_output, 'python')
                        await process_callback("Результат выполнения улучшенного кода", execution_result)
                        await result_queue.put((
                            gr.update(value="\n".join(progress_messages)),
                            gr.update(value="\n\n".join(process_steps)),
                            gr.update(value=code_output, language="python"),
                            gr.update(value=file_path) if file_path else gr.update()
                        ))
            except Exception as e:
                logger.error(f"Ошибка в Pro режиме: {str(e)}")
                await progress_callback(f"Ошибка в Pro режиме: {str(e)}")

        logger.debug("Завершение generate_response")
        await progress_callback("Обработка запроса завершена")
    except Exception as e:
        logger.error(f"Ошибка в generate_response: {str(e)}", exc_info=True)
        error_message = f"Произошла ошибка: {str(e)}"
        await result_queue.put((
            gr.update(value="\n".join(progress_messages + [error_message])),
            gr.update(value="\n\n".join(process_steps)),
            gr.update(value=code_output, language="python"),
            gr.update(value=file_path) if file_path else gr.update()
        ))
    finally:
        await result_queue.put(None)  # Сигнал о завершении

    return (
        gr.update(value="\n".join(progress_messages)),
        gr.update(value="\n\n".join(process_steps)),
        gr.update(value=code_output, language="python"),
        gr.update(value=file_path) if file_path else gr.update()
    )


# Остальной код остается без изменений
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# AI Code Generator")
        with gr.Row():
            with gr.Column(scale=1):
                user_request = gr.Textbox(label="Введите ваш запрос")
                use_pro_mode = gr.Checkbox(label="Использовать Pro Mode")
                use_model_to_model = gr.Checkbox(label="Использовать Model to Model Mode")
                model_choice = gr.Dropdown(choices=MODELS, label="Выберите модель")
                submit_btn = gr.Button("Сгенерировать код")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Прогресс"):
                        progress_output = gr.Markdown(label="Прогресс и логи")
                    with gr.TabItem("Процесс"):
                        process_output = gr.Markdown(label="Этапы процесса")
                    with gr.TabItem("Код"):
                        code_output = gr.Code(label="Сгенерированный код", language="python")
                    with gr.TabItem("Файл"):
                        file_output = gr.File(label="Скачать проект")

        submit_btn.click(
            generate_response,
            inputs=[user_request, use_pro_mode, model_choice, use_model_to_model],
            outputs=[progress_output, process_output, code_output, file_output],
            api_name="generate"
        )

    return demo


async def run_final_tests(code, file_id, model):
    try:
        test_code = await generate_safe_content(f"Напиши комплексные тесты для этого кода:\n{code}", model)

        test_file_data = await safe_request('POST', 'http://127.0.0.1:5000/create-file', json={
            "file_name": "final_test",
            "file_type": "py"
        })
        if test_file_data and 'file_id' in test_file_data:
            test_file_id = test_file_data['file_id']
            await safe_request('POST', 'http://127.0.0.1:5000/fill-file', json={
                "file_id": test_file_id,
                "content": test_code
            })

            stdout, stderr = await run_code(f"{test_file_id}/final_test.py")
            if stderr:
                logger.error(f"Ошибки при финальных тестах: {stderr}")
                return stderr
            return stdout
        return "Не удалось создать файл для финального тестирования"
    except Exception as e:
        logger.error(f"Ошибка при выполнении финальных тестов: {str(e)}")
        return str(e)


@app.route('/download_project/<path:filename>')
def download_project_file(filename):
    return send_file(os.path.join(tempfile.gettempdir(), filename), as_attachment=True)


def cleanup_temp_files():
    for filename in os.listdir(tempfile.gettempdir()):
        if filename.endswith('.zip'):
            os.unlink(os.path.join(tempfile.gettempdir(), filename))


atexit.register(cleanup_temp_files)


def execute_command(command: str) -> str:
    """Выполняет команду терминала после получения разрешения пользователя."""
    if not command:
        raise ValueError("Команда не предоставлена")

    user_confirmed = input(f"Вы хотите выполнить команду: {command}? (y/n): ").lower() in ['y', 'yes']
    if not user_confirmed:
        raise ValueError("Пользователь не подтвердил выполнение команды.")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)

    return result.stdout


def write_to_file(rel_path: str, content: str) -> None:
    """Записывает содержимое в файл после получения разрешения пользователя."""
    if not rel_path:
        raise ValueError("Путь к файлу не предоставлен.")
    if content is None:
        raise ValueError("Содержимое не предоставлено.")

    abs_path = os.path.abspath(rel_path)
    user_confirmed = input(f"Вы хотите записать в файл: {abs_path}? (y/n): ").lower() in ['y', 'yes']
    if not user_confirmed:
        raise ValueError("Пользователь не подтвердил запись в файл.")

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, 'w') as file:
        file.write(content)
    print(f"Содержимое успешно записано в {abs_path}")


def list_code_definition_names(rel_dir_path: str) -> dict:
    """Извлекает имена ключевых элементов из файлов исходного кода."""
    if not rel_dir_path:
        raise ValueError("Путь к директории не предоставлен.")

    abs_path = os.path.abspath(rel_dir_path)
    user_confirmed = input(f"Вы хотите перечислить определения кода в: {abs_path}? (y/n): ").lower() in ['y', 'yes']
    if not user_confirmed:
        raise ValueError("Пользователь не подтвердил операцию.")

    result = {}
    for file in os.listdir(abs_path):
        if file.endswith('.py'):
            file_path = os.path.join(abs_path, file)
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
                definitions = [
                    f"function {node.name}" if isinstance(node, ast.FunctionDef) else
                    f"class {node.name}" if isinstance(node, ast.ClassDef) else
                    f"variable {node.targets[0].id}" if isinstance(node, ast.Assign) and isinstance(node.targets[0],
                                                                                                    ast.Name) else
                    None
                    for node in ast.walk(tree)
                ]
                result[file] = [d for d in definitions if d]
    return result


def search_files(rel_dir_path: str, regex_pattern: str, file_pattern: str = None) -> list:
    """Ищет файлы в директории по регулярному выражению."""
    if not rel_dir_path or not regex_pattern:
        raise ValueError("Путь к директории или шаблон регулярного выражения не предоставлены.")

    abs_path = os.path.abspath(rel_dir_path)
    user_confirmed = input(f"Вы хотите искать файлы в: {abs_path}? (y/n): ").lower() in ['y', 'yes']
    if not user_confirmed:
        raise ValueError("Пользователь не подтвердил операцию.")

    regex = re.compile(regex_pattern)
    results = []

    for root, _, files in os.walk(abs_path):
        for file in files:
            if file_pattern and not re.match(file_pattern, file):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(f"File: {file_path}, Line: {i}, Match: {line.strip()}")
            except UnicodeDecodeError:
                logger.warning(f"Не удалось декодировать файл {file_path} как UTF-8, пробуем latin-1")
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(f"File: {file_path}, Line: {i}, Match: {line.strip()}")
                except UnicodeDecodeError:
                    logger.error(f"Не удалось декодировать файл {file_path} как UTF-8 или latin-1")
                except Exception as e:
                    logger.error(f"Неожиданная ошибка при чтении файла {file_path}: {e}")

    return results


def ask_followup_question(question: str) -> str:
    """Задает пользователю дополнительный вопрос."""
    if not question:
        raise ValueError("Вопрос не предоставлен.")
    return input(f"{question}\n")


def attempt_completion(result: str, command: str = None) -> None:
    """Представляет результат пользователю и опционально выполняет команду."""
    if not result:
        raise ValueError("Результат не предоставлен.")

    print("Результат:")
    print(result)

    if command:
        user_confirmed = input(f"Вы хотите выполнить команду: {command}? (y/n): ").lower() in ['y', 'yes']
        if user_confirmed:
            execute_command(command)


def execute_command(command: str) -> str:
    """Выполняет команду терминала после получения разрешения пользователя."""
    if not command:
        raise ValueError("Команда не предоставлена")

    user_confirmed = input(f"Вы хотите выполнить команду: {command}? (y/n): ").lower() in ['y', 'yes']
    if not user_confirmed:
        raise ValueError("Пользователь не подтвердил выполнение команды.")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)

    return result.stdout


def safe_file_operation(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"Файл не найден: {e}")
            return None
        except PermissionError as e:
            logger.error(f"Отказано в доступе: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при работе с файлом: {e}")
            return None

    return wrapper


@safe_file_operation
def read_file(rel_path: str) -> str:
    """Читает содержимое файла через FileHandler."""
    if not rel_path:
        raise ValueError("Путь к файлу не предоставлен.")

    filename = os.path.basename(rel_path)
    content = file_handler.read_file(filename)
    if content is None:
        raise FileNotFoundError(f"Файл {filename} не найден.")
    return content


@safe_file_operation
def write_to_file(rel_path: str, content: str) -> None:
    """Записывает содержимое в файл через FileHandler."""
    if not rel_path:
        raise ValueError("Путь к файлу не предоставлен.")
    if content is None:
        raise ValueError("Содержимое не предоставлено.")

    filename = os.path.basename(rel_path)
    file_handler.write_file(filename, content)


def read_file(rel_path: str) -> str:
    """Читает содержимое файла после получения разрешения пользователя."""
    if not rel_path:
        raise ValueError("Путь к файлу не предоставлен.")

    abs_path = os.path.abspath(rel_path)
    user_confirmed = input(f"Вы хотите прочитать файл: {abs_path}? (y/n): ").lower() in ['y', 'yes']
    if not user_confirmed:
        raise ValueError("Пользователь не подтвердил чтение файла.")

    with open(abs_path, 'r') as file:
        return file.read()


def test_functions():
    try:
        # Тест execute_command
        result = execute_command("echo 'Test command'")
        assert "Test command" in result, "execute_command failed"
        logger.info("execute_command test passed")

        # Тест read_file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("Test content")
            temp_file_path = temp_file.name
        content = read_file(temp_file_path)
        assert content == "Test content", "read_file failed"
        logger.info("read_file test passed")

        # Тест write_to_file
        test_content = "New test content"
        write_to_file(temp_file_path, test_content)
        content = read_file(temp_file_path)
        assert content == test_content, "write_to_file failed"
        logger.info("write_to_file test passed")

        # Очистка
        os.unlink(temp_file_path)

        logger.info("All function tests passed")
    except Exception as e:
        logger.error(f"Function test failed: {str(e)}")


def write_to_file(rel_path: str, content: str) -> None:
    """Записывает содержимое в файл после получения разрешения пользователя."""
    if not rel_path:
        raise ValueError("Путь к файлу не предоставлен.")
    if content is None:
        raise ValueError("Содержимое не предоставлено.")

    abs_path = os.path.abspath(rel_path)
    user_confirmed = input(f"Вы хотите записать в файл: {abs_path}? (y/n): ").lower() in ['y', 'yes']
    if not user_confirmed:
        raise ValueError("Пользователь не подтвердил запись в файл.")

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, 'w') as file:
        file.write(content)
    print(f"Содержимое успешно записано в {abs_path}")


def create_project_archive(project_dir: str) -> Optional[str]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_path = tmp_file.name

        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(project_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, project_dir)
                    zipf.write(file_path, arcname)

        return tmp_path
    except Exception as e:
        logger.error(f"Ошибка при создании архива проекта: {e}")
        return None


def list_files(rel_dir_path: str, recursive: bool = False) -> list:
    """Перечисляет файлы в директории, опционально рекурсивно."""
    if not rel_dir_path:
        raise ValueError("Путь к директории не предоставлен.")

    abs_path = os.path.abspath(rel_dir_path)
    user_confirmed = input(f"Вы хотите перечислить файлы в: {abs_path}? (y/n): ").lower() in ['y', 'yes']
    if not user_confirmed:
        raise ValueError("Пользователь не подтвердил перечисление файлов.")

    if recursive:
        return [os.path.join(root, file) for root, _, files in os.walk(abs_path) for file in files]
    else:
        return [os.path.join(abs_path, file) for file in os.listdir(abs_path) if
                os.path.isfile(os.path.join(abs_path, file))]


def list_code_definition_names(rel_dir_path: str) -> dict:
    """Извлекает имена ключевых элементов из файлов исходного кода."""
    if not rel_dir_path:
        raise ValueError("Путь к директории не предоставлен.")

    abs_path = os.path.abspath(rel_dir_path)
    user_confirmed = input(f"Вы хотите перечислить определения кода в: {abs_path}? (y/n): ").lower() in ['y', 'yes']
    if not user_confirmed:
        raise ValueError("Пользователь не подтвердил операцию.")

    result = {}
    for file in os.listdir(abs_path):
        if file.endswith('.py'):
            file_path = os.path.join(abs_path, file)
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
                definitions = [
                    f"function {node.name}" if isinstance(node, ast.FunctionDef) else
                    f"class {node.name}" if isinstance(node, ast.ClassDef) else
                    f"variable {node.targets[0].id}" if isinstance(node, ast.Assign) and isinstance(node.targets[0],
                                                                                                    ast.Name) else
                    None
                    for node in ast.walk(tree)
                ]
                result[file] = [d for d in definitions if d]
    return result


def search_files(rel_dir_path: str, regex_pattern: str, file_pattern: str = None) -> list:
    """Ищет файлы в директории по регулярному выражению."""
    if not rel_dir_path or not regex_pattern:
        raise ValueError("Путь к директории или шаблон регулярного выражения не предоставлены.")

    abs_path = os.path.abspath(rel_dir_path)
    user_confirmed = input(f"Вы хотите искать файлы в: {abs_path}? (y/n): ").lower() in ['y', 'yes']
    if not user_confirmed:
        raise ValueError("Пользователь не подтвердил операцию.")

    regex = re.compile(regex_pattern)
    results = []

    for root, _, files in os.walk(abs_path):
        for file in files:
            if file_pattern and not re.match(file_pattern, file):
                continue
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        results.append(f"File: {file_path}, Line: {i}, Match: {line.strip()}")

    return results


def ask_followup_question(question: str) -> str:
    """Задает пользователю дополнительный вопрос."""
    if not question:
        raise ValueError("Вопрос не предоставлен.")
    return input(f"{question}\n")


def attempt_completion(result: str, command: str = None) -> None:
    """Представляет результат пользователю и опционально выполняет команду."""
    if not result:
        raise ValueError("Результат не предоставлен.")

    print("Результат:")
    print(result)

    if command:
        user_confirmed = input(f"Вы хотите выполнить команду: {command}? (y/n): ").lower() in ['y', 'yes']
        if user_confirmed:
            execute_command(command)


async def run_flask():
    app.run(port=5000, threaded=True)


async def create_tunnel(port):
    process = await asyncio.create_subprocess_exec(
        'lt', '--port', str(port),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode == 0:
        output = stdout.decode().strip()
        logger.info(f"Вывод localtunnel: {output}")  # Добавлено логирование
        try:
            url = re.search(r'(https?://[^\s]+)', output)  # Изменено на поиск URL в выводе
            if url:
                return url.group(0)
            else:
                logger.error("URL не найден в выводе localtunnel")
                return None
        except Exception as e:
            logger.error(f"Ошибка при обработке вывода localtunnel: {str(e)}")
            return None
    else:
        logger.error(f"Ошибка при создании туннеля: {stderr.decode()}")
        return None


async def main():
    global running
    demo = gradio_interface()

    # Запускаем Flask в отдельном потоке
    flask_thread = threading.Thread(target=lambda: app.run(port=5000, debug=False, use_reloader=False))
    flask_thread.start()

    # Находим свободный порт для Gradio
    free_port = find_free_port()
    logger.info(f"Найден свободный порт для Gradio: {free_port}")

    # Запускаем Gradio
    gradio_thread = threading.Thread(target=lambda: demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=free_port,
        prevent_thread_lock=True
    ))
    gradio_thread.start()

    logger.info(f"Gradio интерфейс запущен на порту {free_port}. Используйте локальную ссылку для доступа.")
    print(f"Gradio интерфейс запущен на порту {free_port}. Используйте локальную ссылку для доступа.")

    while running:
        await asyncio.sleep(1)

    # Останавливаем Flask и Gradio при завершении
    flask_thread.join()
    gradio_thread.join()


import socket


def find_free_port(start_port=7860, max_port=7960):
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise OSError(f"Не удалось найти свободный порт в диапазоне {start_port}-{max_port}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Получено прерывание с клавиатуры")
    except TypeError as e:
        logger.error(f"Ошибка типа: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
    finally:
        logger.info("Программа завершена")