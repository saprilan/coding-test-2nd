import React, { useState } from 'react';

interface FileUploadProps {
  onUploadComplete: (res: any) => void;
  onUploadError: (err: string) => void;
}

export default function FileUpload({ onUploadComplete, onUploadError }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      onUploadComplete(data);
    } catch (error: any) {
      onUploadError(error.message || 'An error occurred');
    }
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <input
        type="file"
        accept="application/pdf"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="border border-gray-300 rounded p-2"
      />
      <button
        onClick={handleUpload}
        disabled={!file}
        className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
      >
        Upload PDF
      </button>
    </div>
  );
}
