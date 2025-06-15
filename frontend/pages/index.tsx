import React, { useState } from 'react';
import Head from 'next/head';
import FileUpload from '@/components/FileUpload';
import ChatInterface from '@/components/ChatInterface';
// import DocumentViewer from '@/components/DocumentViewer'; // Optional

export default function Home() {
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-gray-100 text-gray-900">
      <Head>
        <title>RAG-based Financial Q&A System</title>
        <meta name="description" content="AI-powered Q&A system for financial documents" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="max-w-3xl mx-auto py-10 px-4 space-y-8">
        <h1 className="text-4xl font-bold text-center">
          RAG-based Financial Statement Q&A System
        </h1>

        <div className="text-center">
          <p className="mb-4 text-lg">Upload a financial statement (PDF) to get started.</p>
          <FileUpload
            onUploadComplete={(res) => {
              setUploadResult(res);
              setError(null);
            }}
            onUploadError={(err) => setError(err)}
          />
          {error && <p className="text-red-600 mt-2">{error}</p>}
        </div>

        {uploadResult && (
          <div className="mt-10">
            <h2 className="text-2xl font-semibold mb-4">Ask a Question</h2>
            <ChatInterface />
          </div>
        )}
      </main>
    </div>
  );
}
