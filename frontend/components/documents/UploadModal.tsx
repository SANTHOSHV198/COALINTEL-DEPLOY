'use client';

import React, { useState, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Upload, X, FileText, CheckCircle2, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Badge } from '@/components/ui/Badge';
import { documentApi } from '@/lib/api/documentApi';
import { CIL_SUBSIDIARIES, FISCAL_YEARS } from '@/lib/constants';
import { DocumentItem } from '@/types/document';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess?: (doc: DocumentItem) => void;
}

const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB
  const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.csv', '.png', '.jpg', '.jpeg', '.tif', '.tiff'];

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onUploadSuccess,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [subsidiary, setSubsidiary] = useState('ECL');
  const [fiscalYear, setFiscalYear] = useState('2023-24');
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [duplicateDocInfo, setDuplicateDocInfo] = useState<string | null>(null);

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !isUploading) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, isUploading, onClose]);

  React.useEffect(() => {
    if (!isOpen || typeof window === 'undefined') return;

    // AppShell's page-enter animation applies a transform to <main>. Keep the
    // dashboard in place while portaling the modal to <body> so fixed layout
    // remains relative to the viewport.
    const scrollY = window.scrollY;
    const body = document.body;
    const previous = {
      overflow: body.style.overflow,
      position: body.style.position,
      top: body.style.top,
      width: body.style.width,
    };

    body.style.overflow = 'hidden';
    body.style.position = 'fixed';
    body.style.top = `-${scrollY}px`;
    body.style.width = '100%';

    return () => {
      body.style.overflow = previous.overflow;
      body.style.position = previous.position;
      body.style.top = previous.top;
      body.style.width = previous.width;
      window.scrollTo(0, scrollY);
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const validateFile = (file: File): boolean => {
    setErrorMessage(null);
    setDuplicateDocInfo(null);

    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setErrorMessage(`File type '${ext}' is not supported. Supported formats: ${ALLOWED_EXTENSIONS.join(', ')}`);
      return false;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
      setErrorMessage(`File size (${sizeMb} MB) exceeds maximum allowed limit of 50 MB.`);
      return false;
    }

    return true;
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadProgress(10);
    setErrorMessage(null);
    setDuplicateDocInfo(null);

    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 85) {
          clearInterval(interval);
          return 85;
        }
        return prev + 15;
      });
    }, 300);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('subsidiary', subsidiary);
      formData.append('fiscal_year', fiscalYear);

      const result = await documentApi.uploadDocument(formData);
      clearInterval(interval);
      setUploadProgress(100);

      setTimeout(() => {
        if (onUploadSuccess && result) {
          onUploadSuccess(result);
        }
        onClose();
      }, 500);
    } catch (err: unknown) {
      clearInterval(interval);
      setIsUploading(false);
      setUploadProgress(0);

      const error = err as { response?: { status?: number; data?: { detail?: string; error?: string; existing_document?: { id: number; filename: string } } }; message?: string };

      if (error.response?.status === 409) {
        const detail = error.response.data?.detail || error.response.data?.error || 'A document with identical content (SHA-256 hash) already exists.';
        setDuplicateDocInfo(detail);
      } else {
        setErrorMessage(
          error.response?.data?.detail ||
          error.response?.data?.error ||
          error.message ||
          'Failed to upload and parse document. Please check connection and try again.'
        );
      }
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} Bytes`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  if (typeof document === 'undefined') return null;

  return createPortal((
    <div
      className="fixed inset-0 z-50 flex min-h-full items-center justify-center overflow-hidden p-4 bg-[#0E1113]/80 backdrop-blur-sm animate-fade-in"
      onClick={() => !isUploading && onClose()}
      role="dialog"
      aria-modal="true"
      aria-label="Ingest Mining Document Modal"
    >
      <div
        className="relative w-full max-w-xl max-h-[90vh] overflow-y-auto overscroll-contain p-6 rounded-lg bg-[#1C2226] border border-[#30383D] shadow-2xl space-y-6 text-[#E8ECEB] animate-slide-up"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#30383D] pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[#C58B3A]/15 text-[#C58B3A] border border-[#C58B3A]/30">
              <Upload className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-[#E8ECEB]">Ingest Mining Document</h3>
                <p className="text-xs text-[#9BA5A8]">PDF, DOCX, XLSX, CSV, PNG, JPG, TIFF up to 50 MB</p>
            </div>
          </div>

          <button
            onClick={onClose}
            disabled={isUploading}
            className="p-1.5 rounded-lg text-[#9BA5A8] hover:text-[#E8ECEB] hover:bg-[#242C30] transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Banners */}
        {errorMessage && (
          <div className="p-3.5 rounded-lg bg-[#C94B45]/10 border border-[#C94B45]/40 text-[#C94B45] text-xs flex items-start gap-2.5">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-[#C94B45]" />
            <span>{errorMessage}</span>
          </div>
        )}

        {duplicateDocInfo && (
          <div className="p-3.5 rounded-lg bg-[#D6A23A]/10 border border-[#D6A23A]/40 text-[#D6A23A] text-xs flex items-start gap-2.5">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-[#D6A23A]" />
            <div className="space-y-1">
              <span className="font-semibold block">Duplicate SHA-256 Hash Detected</span>
              <span>{duplicateDocInfo}</span>
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Drag & Drop Zone */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative flex flex-col items-center justify-center p-8 rounded-lg border-2 border-dashed cursor-pointer transition-colors duration-150 text-center ${
              dragActive
                ? 'border-[#C58B3A] bg-[#C58B3A]/10'
                : selectedFile
                ? 'border-[#4F8A62]/50 bg-[#4F8A62]/10'
                : 'border-[#30383D] bg-[#151A1D] hover:border-[#C58B3A]/60 hover:bg-[#242C30]'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
                accept=".pdf,.docx,.xlsx,.csv,.png,.jpg,.jpeg,.tif,.tiff"
              onChange={handleFileChange}
              className="hidden"
            />

            {selectedFile ? (
              <div className="flex flex-col items-center space-y-2">
                <div className="p-3 rounded-lg bg-[#4F8A62]/20 text-[#4F8A62] border border-[#4F8A62]/30">
                  <CheckCircle2 className="h-8 w-8" />
                </div>
                <span className="text-sm font-semibold text-[#E8ECEB] max-w-xs truncate">
                  {selectedFile.name}
                </span>
                <Badge variant="amber" size="sm">
                  {formatFileSize(selectedFile.size)}
                </Badge>
                <span className="text-[11px] text-[#9BA5A8]">Click or drag another file to replace</span>
              </div>
            ) : (
              <div className="flex flex-col items-center space-y-2">
                <div className="p-3 rounded-lg bg-[#242C30] text-[#C58B3A] border border-[#30383D] shadow-sm">
                  <FileText className="h-8 w-8" />
                </div>
                <div className="text-xs text-[#9BA5A8]">
                  <span className="font-semibold text-[#C58B3A]">Click to browse</span> or drag and drop document here
                </div>
                <span className="text-[10px] text-[#9BA5A8] font-mono uppercase tracking-wider">
                   Supported formats: PDF, DOCX, XLSX, CSV, PNG, JPG, TIFF (Max 50 MB)
                </span>
              </div>
            )}
          </div>

          {/* Subsidiary & Fiscal Year Scope Selectors */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Select
              label="Target Subsidiary / Enterprise"
              value={subsidiary}
              onChange={(e) => setSubsidiary(e.target.value)}
              options={CIL_SUBSIDIARIES.filter((s) => s.value !== 'ALL')}
            />

            <Select
              label="Fiscal Year Scope"
              value={fiscalYear}
              onChange={(e) => setFiscalYear(e.target.value)}
              options={FISCAL_YEARS}
            />
          </div>

          {/* Upload Progress Bar */}
          {isUploading && (
            <div className="space-y-2 pt-2">
              <div className="flex items-center justify-between text-xs font-mono text-[#9BA5A8]">
                <span>Ingesting & Parsing Document...</span>
                <span className="text-[#C58B3A] font-semibold">{uploadProgress}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-[#151A1D] border border-[#30383D] overflow-hidden">
                <div
                  className="h-full bg-[#C58B3A] transition-all duration-300 rounded-full"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Footer Actions */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t border-[#30383D]">
            <Button variant="ghost" size="md" onClick={onClose} disabled={isUploading}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              size="md"
              disabled={!selectedFile || isUploading}
              isLoading={isUploading}
              leftIcon={<Upload className="h-4 w-4" />}
            >
              Upload & Parse Document
            </Button>
          </div>
        </form>
      </div>
    </div>
  ), document.body);
};
