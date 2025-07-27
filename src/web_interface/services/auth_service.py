"""
认证服务
负责处理身份验证和授权相关的业务逻辑
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib
import secrets

from ...utils.logger import get_logger

logger = get_logger(__name__)


class AuthService:
    """
    认证服务
    
    提供身份验证和授权功能：
    1. 用户认证
    2. 权限管理
    3. 会话管理
    4. API密钥管理
    """
    
    def __init__(self):
        """初始化认证服务"""
        # 简化的用户数据库（实际应用中应使用真实数据库）
        self.users = {
            "admin": {
                "password_hash": self._hash_password("admin123"),
                "role": "admin",
                "permissions": ["read", "write", "admin"]
            },
            "analyst": {
                "password_hash": self._hash_password("analyst123"),
                "role": "analyst", 
                "permissions": ["read", "write"]
            },
            "viewer": {
                "password_hash": self._hash_password("viewer123"),
                "role": "viewer",
                "permissions": ["read"]
            }
        }
        
        # 活跃会话
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # API密钥
        self.api_keys: Dict[str, Dict[str, Any]] = {
            "demo_key_123": {
                "user": "admin",
                "permissions": ["read", "write"],
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(days=365)
            }
        }
        
        logger.info("认证服务初始化完成")
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """用户认证"""
        try:
            if username not in self.users:
                return None
            
            user = self.users[username]
            password_hash = self._hash_password(password)
            
            if password_hash == user["password_hash"]:
                # 创建会话
                session_token = self._generate_session_token()
                session_data = {
                    "username": username,
                    "role": user["role"],
                    "permissions": user["permissions"],
                    "created_at": datetime.now(),
                    "expires_at": datetime.now() + timedelta(hours=8)
                }
                
                self.active_sessions[session_token] = session_data
                
                return {
                    "token": session_token,
                    "user": {
                        "username": username,
                        "role": user["role"],
                        "permissions": user["permissions"]
                    },
                    "expires_at": session_data["expires_at"].isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"用户认证失败: {e}")
            return None
    
    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        """验证会话"""
        try:
            if token not in self.active_sessions:
                return None
            
            session = self.active_sessions[token]
            
            # 检查会话是否过期
            if datetime.now() > session["expires_at"]:
                del self.active_sessions[token]
                return None
            
            return session
            
        except Exception as e:
            logger.error(f"会话验证失败: {e}")
            return None
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """验证API密钥"""
        try:
            if api_key not in self.api_keys:
                return None
            
            key_data = self.api_keys[api_key]
            
            # 检查密钥是否过期
            if datetime.now() > key_data["expires_at"]:
                return None
            
            return key_data
            
        except Exception as e:
            logger.error(f"API密钥验证失败: {e}")
            return None
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """检查权限"""
        return required_permission in user_permissions or "admin" in user_permissions
    
    def logout_user(self, token: str) -> bool:
        """用户登出"""
        try:
            if token in self.active_sessions:
                del self.active_sessions[token]
                return True
            return False
            
        except Exception as e:
            logger.error(f"用户登出失败: {e}")
            return False
    
    def refresh_session(self, token: str) -> Optional[str]:
        """刷新会话"""
        try:
            session = self.validate_session(token)
            if not session:
                return None
            
            # 删除旧会话
            del self.active_sessions[token]
            
            # 创建新会话
            new_token = self._generate_session_token()
            session["expires_at"] = datetime.now() + timedelta(hours=8)
            self.active_sessions[new_token] = session
            
            return new_token
            
        except Exception as e:
            logger.error(f"刷新会话失败: {e}")
            return None
    
    def generate_api_key(self, username: str, permissions: List[str], 
                        expires_days: int = 365) -> Optional[str]:
        """生成API密钥"""
        try:
            if username not in self.users:
                return None
            
            api_key = f"key_{secrets.token_urlsafe(32)}"
            
            self.api_keys[api_key] = {
                "user": username,
                "permissions": permissions,
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(days=expires_days)
            }
            
            return api_key
            
        except Exception as e:
            logger.error(f"生成API密钥失败: {e}")
            return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """撤销API密钥"""
        try:
            if api_key in self.api_keys:
                del self.api_keys[api_key]
                return True
            return False
            
        except Exception as e:
            logger.error(f"撤销API密钥失败: {e}")
            return False
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        try:
            if username not in self.users:
                return None
            
            user = self.users[username]
            return {
                "username": username,
                "role": user["role"],
                "permissions": user["permissions"]
            }
            
        except Exception as e:
            logger.error(f"获取用户信息失败: {e}")
            return None
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """获取活跃会话列表"""
        try:
            sessions = []
            current_time = datetime.now()
            
            for token, session in list(self.active_sessions.items()):
                # 清理过期会话
                if current_time > session["expires_at"]:
                    del self.active_sessions[token]
                    continue
                
                sessions.append({
                    "username": session["username"],
                    "role": session["role"],
                    "created_at": session["created_at"].isoformat(),
                    "expires_at": session["expires_at"].isoformat()
                })
            
            return sessions
            
        except Exception as e:
            logger.error(f"获取活跃会话失败: {e}")
            return []
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_session_token(self) -> str:
        """生成会话令牌"""
        return secrets.token_urlsafe(32)


def get_auth_service() -> AuthService:
    """获取认证服务实例"""
    return AuthService() 