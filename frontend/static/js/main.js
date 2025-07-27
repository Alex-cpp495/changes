// 主JavaScript文件 - 东吴证券研报异常检测系统

// 全局配置
const CONFIG = {
    API_BASE_URL: '/api',
    POLLING_INTERVAL: 30000, // 30秒
    MAX_RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000 // 1秒
};

// 工具函数
const Utils = {
    // 格式化日期时间
    formatDateTime: function(dateStr) {
        const date = new Date(dateStr);
        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    },

    // 格式化数字为百分比
    formatPercentage: function(value, decimals = 1) {
        return (value * 100).toFixed(decimals) + '%';
    },

    // 格式化文件大小
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // 节流函数
    throttle: function(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    },

    // 防抖函数
    debounce: function(func, delay) {
        let timeoutId;
        return function() {
            const args = arguments;
            const context = this;
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(context, args), delay);
        }
    },

    // 显示Toast消息
    showToast: function(message, type = 'info', duration = 3000) {
        const toastContainer = this.getOrCreateToastContainer();
        const toast = this.createToastElement(message, type);
        
        toastContainer.appendChild(toast);
        
        // 显示动画
        setTimeout(() => toast.classList.add('show'), 100);
        
        // 自动隐藏
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, duration);
    },

    getOrCreateToastContainer: function() {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        return container;
    },

    createToastElement: function(message, type) {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        // 添加关闭按钮事件
        const closeBtn = toast.querySelector('.btn-close');
        closeBtn.addEventListener('click', () => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        });
        
        return toast;
    }
};

// API服务类
class ApiService {
    constructor() {
        this.baseURL = CONFIG.API_BASE_URL;
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };
    }

    // 通用请求方法
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: { ...this.defaultHeaders, ...options.headers },
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API请求失败:', error);
            throw error;
        }
    }

    // GET请求
    async get(endpoint, params = {}) {
        const url = new URL(`${this.baseURL}${endpoint}`, window.location.origin);
        Object.keys(params).forEach(key => {
            if (params[key] !== undefined && params[key] !== null) {
                url.searchParams.append(key, params[key]);
            }
        });

        return this.request(url.pathname + url.search);
    }

    // POST请求
    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    // PUT请求
    async put(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    // DELETE请求
    async delete(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE'
        });
    }

    // 带重试的请求
    async requestWithRetry(endpoint, options = {}, maxRetries = CONFIG.MAX_RETRY_ATTEMPTS) {
        let lastError;
        
        for (let i = 0; i <= maxRetries; i++) {
            try {
                return await this.request(endpoint, options);
            } catch (error) {
                lastError = error;
                
                if (i === maxRetries) {
                    throw error;
                }
                
                // 等待后重试
                await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY * Math.pow(2, i)));
            }
        }
        
        throw lastError;
    }
}

// 全局API实例
const api = new ApiService();

// 数据管理器
class DataManager {
    constructor() {
        this.cache = new Map();
        this.cacheExpiry = new Map();
        this.defaultTTL = 5 * 60 * 1000; // 5分钟
    }

    // 设置缓存
    set(key, value, ttl = this.defaultTTL) {
        this.cache.set(key, value);
        this.cacheExpiry.set(key, Date.now() + ttl);
    }

    // 获取缓存
    get(key) {
        const expiry = this.cacheExpiry.get(key);
        
        if (!expiry || Date.now() > expiry) {
            this.cache.delete(key);
            this.cacheExpiry.delete(key);
            return null;
        }
        
        return this.cache.get(key);
    }

    // 删除缓存
    delete(key) {
        this.cache.delete(key);
        this.cacheExpiry.delete(key);
    }

    // 清空缓存
    clear() {
        this.cache.clear();
        this.cacheExpiry.clear();
    }

    // 获取或设置数据
    async getOrFetch(key, fetchFunction, ttl = this.defaultTTL) {
        let data = this.get(key);
        
        if (data === null) {
            data = await fetchFunction();
            this.set(key, data, ttl);
        }
        
        return data;
    }
}

// 全局数据管理器实例
const dataManager = new DataManager();

// 事件总线
class EventBus {
    constructor() {
        this.events = {};
    }

    // 订阅事件
    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }

    // 发布事件
    emit(event, data) {
        if (this.events[event]) {
            this.events[event].forEach(callback => callback(data));
        }
    }

    // 取消订阅
    off(event, callback) {
        if (this.events[event]) {
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        }
    }
}

// 全局事件总线实例
const eventBus = new EventBus();

// 文件处理工具
class FileHandler {
    constructor() {
        this.allowedTypes = {
            'text/plain': ['.txt'],
            'application/pdf': ['.pdf'],
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
        };
        this.maxFileSize = 10 * 1024 * 1024; // 10MB
    }

    // 验证文件类型和大小
    validateFile(file) {
        const errors = [];

        // 检查文件大小
        if (file.size > this.maxFileSize) {
            errors.push(`文件大小不能超过 ${Utils.formatFileSize(this.maxFileSize)}`);
        }

        // 检查文件类型
        const isValidType = Object.keys(this.allowedTypes).includes(file.type) ||
                           Object.values(this.allowedTypes).flat().some(ext => 
                               file.name.toLowerCase().endsWith(ext));

        if (!isValidType) {
            errors.push('不支持的文件格式，请上传 TXT、PDF 或 DOCX 文件');
        }

        return errors;
    }

    // 读取文本文件
    async readTextFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = () => reject(new Error('文件读取失败'));
            
            reader.readAsText(file, 'UTF-8');
        });
    }

    // 处理文件上传
    async handleFileUpload(file, progressCallback) {
        const errors = this.validateFile(file);
        
        if (errors.length > 0) {
            throw new Error(errors.join(', '));
        }

        try {
            if (progressCallback) progressCallback(0);

            let content;
            if (file.type === 'text/plain') {
                content = await this.readTextFile(file);
            } else {
                // 对于PDF和DOCX文件，需要后端处理
                content = await this.uploadToServer(file, progressCallback);
            }

            if (progressCallback) progressCallback(100);
            
            return {
                name: file.name,
                size: file.size,
                type: file.type,
                content: content
            };
        } catch (error) {
            console.error('文件处理失败:', error);
            throw error;
        }
    }

    // 上传到服务器
    async uploadToServer(file, progressCallback) {
        const formData = new FormData();
        formData.append('file', file);

        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable && progressCallback) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    progressCallback(percentComplete);
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        resolve(response.content);
                    } catch (error) {
                        reject(new Error('服务器响应解析失败'));
                    }
                } else {
                    reject(new Error(`上传失败: ${xhr.statusText}`));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('网络错误'));
            });

            xhr.open('POST', `${CONFIG.API_BASE_URL}/upload`);
            xhr.send(formData);
        });
    }
}

// 全局文件处理器实例
const fileHandler = new FileHandler();

// 表单验证工具
class FormValidator {
    constructor() {
        this.rules = {};
    }

    // 添加验证规则
    addRule(fieldName, validator, message) {
        if (!this.rules[fieldName]) {
            this.rules[fieldName] = [];
        }
        this.rules[fieldName].push({ validator, message });
    }

    // 验证字段
    validateField(fieldName, value) {
        const fieldRules = this.rules[fieldName] || [];
        
        for (const rule of fieldRules) {
            if (!rule.validator(value)) {
                return rule.message;
            }
        }
        
        return null;
    }

    // 验证表单
    validateForm(formData) {
        const errors = {};
        
        Object.keys(this.rules).forEach(fieldName => {
            const error = this.validateField(fieldName, formData[fieldName]);
            if (error) {
                errors[fieldName] = error;
            }
        });
        
        return errors;
    }

    // 显示字段错误
    showFieldError(fieldElement, message) {
        this.clearFieldError(fieldElement);
        
        const errorElement = document.createElement('div');
        errorElement.className = 'invalid-feedback d-block';
        errorElement.textContent = message;
        
        fieldElement.classList.add('is-invalid');
        fieldElement.parentNode.appendChild(errorElement);
    }

    // 清除字段错误
    clearFieldError(fieldElement) {
        fieldElement.classList.remove('is-invalid');
        const errorElement = fieldElement.parentNode.querySelector('.invalid-feedback');
        if (errorElement) {
            errorElement.remove();
        }
    }

    // 清除所有错误
    clearAllErrors(formElement) {
        const invalidFields = formElement.querySelectorAll('.is-invalid');
        invalidFields.forEach(field => this.clearFieldError(field));
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化工具提示
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // 初始化弹出框
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // 全局错误处理
    window.addEventListener('error', function(e) {
        console.error('全局错误:', e.error);
        Utils.showToast('系统发生错误，请稍后重试', 'danger');
    });

    // 网络状态检测
    window.addEventListener('online', function() {
        Utils.showToast('网络连接已恢复', 'success');
    });

    window.addEventListener('offline', function() {
        Utils.showToast('网络连接已断开', 'warning');
    });

    // 监听路由变化
    eventBus.on('route-change', function(data) {
        console.log('路由变化:', data);
        // 可以在这里添加页面切换逻辑
    });

    console.log('东吴证券研报异常检测系统 - 前端已初始化');
});

// 导出全局对象
window.AnomalyDetectionSystem = {
    Utils,
    ApiService,
    api,
    DataManager,
    dataManager,
    EventBus,
    eventBus,
    FileHandler,
    fileHandler,
    FormValidator,
    CONFIG
}; 