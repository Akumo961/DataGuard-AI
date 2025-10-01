import os
import magic
from typing import Dict, List, Any
import logging


class FileHandler:
    """Handle file operations and validation"""

    def __init__(self):
        self.supported_types = {
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.m4a': 'audio/mp4'
        }
        self.logger = logging.getLogger(__name__)

    def is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported"""
        if not filename:
            return False

        file_ext = os.path.splitext(filename)[1].lower()
        return file_ext in self.supported_types

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate file integrity and type"""
        try:
            if not os.path.exists(file_path):
                return {'valid': False, 'error': 'File does not exist'}

            # Check file size (max 50MB)
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:
                return {'valid': False, 'error': 'File too large (max 50MB)'}

            # Verify file type using magic
            file_type = magic.from_file(file_path, mime=True)
            file_ext = os.path.splitext(file_path)[1].lower()

            expected_type = self.supported_types.get(file_ext)
            if expected_type and file_type.startswith(expected_type.split('/')[0]):
                return {
                    'valid': True,
                    'file_size': file_size,
                    'detected_type': file_type,
                    'expected_type': expected_type
                }
            else:
                return {'valid': False, 'error': f'File type mismatch: {file_type}'}

        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed file information"""
        try:
            stat_info = os.stat(file_path)
            return {
                'filename': os.path.basename(file_path),
                'file_size': stat_info.st_size,
                'created': stat_info.st_ctime,
                'modified': stat_info.st_mtime,
                'extension': os.path.splitext(file_path)[1].lower()
            }
        except Exception as e:
            self.logger.error(f"Error getting file info: {str(e)}")
            return {}