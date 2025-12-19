'use client';

import { useState } from 'react';
import './globals.css';

interface Assessment {
    assessment_name: string;
    assessment_url: string;
    relevance_score?: number;
    test_type?: string;
}

interface RecommendResponse {
    query: string;
    recommendations: Assessment[];
    total_recommendations: number;
}

export default function Home() {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<RecommendResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    const exampleQueries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams.",
        "I want to hire new graduates for a sales role in my company.",
        "Looking for a COO who is culturally a right fit for our company.",
        "Need QA Engineer with automation testing skills."
    ];

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_URL}/recommend`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    input_text: query,
                    max_recommendations: 10
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'API request failed');
            }

            const data: RecommendResponse = await response.json();
            setResults(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
            console.error('Error:', err);
        } finally {
            setLoading(false);
        }
    };

    const loadExample = (example: string) => {
        setQuery(example);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
            <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
                {/* Header */}
                <div className="text-center mb-12">
                    <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                        SHL Assessment Recommender
                    </h1>
                    <p className="text-lg text-gray-600">
                        AI-powered recommendations for the perfect assessment match
                    </p>
                </div>

                {/* Input Section */}
                <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-100">
                    <form onSubmit={handleSubmit}>
                        <label className="block text-sm font-semibold text-gray-700 mb-3">
                            Enter Job Description or Query
                        </label>
                        <textarea
                            className="w-full border-2 border-gray-200 rounded-xl p-4 h-40 mb-4 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all duration-200 resize-none"
                            placeholder="E.g., I am hiring for Java developers who can collaborate effectively with business teams..."
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            required
                        />

                        {/* Example Queries */}
                        <div className="mb-4">
                            <p className="text-sm text-gray-600 mb-2">Quick examples:</p>
                            <div className="flex flex-wrap gap-2">
                                {exampleQueries.map((example, idx) => (
                                    <button
                                        key={idx}
                                        type="button"
                                        onClick={() => loadExample(example)}
                                        className="text-xs bg-blue-50 hover:bg-blue-100 text-blue-700 px-3 py-1.5 rounded-lg transition-colors duration-200"
                                    >
                                        {example.substring(0, 40)}...
                                    </button>
                                ))}
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={loading || !query.trim()}
                            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
                        >
                            {loading ? (
                                <span className="flex items-center justify-center">
                                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Analyzing...
                                </span>
                            ) : (
                                'Get Recommendations'
                            )}
                        </button>
                    </form>

                    {/* Backend Note */}
                    <p className="text-sm text-gray-500 text-center mt-3">
                        Note: First query may take 30-60 seconds as the backend (Render free tier) wakes up.
                    </p>
                </div>

                {/* Error Display */}
                {error && (
                    <div className="bg-red-50 border-2 border-red-200 text-red-700 px-6 py-4 rounded-xl mb-8 shadow-sm">
                        <p className="font-semibold">⚠️ Error</p>
                        <p className="text-sm mt-1">{error}</p>
                    </div>
                )}

                {/* Results Display */}
                {results && (
                    <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-3xl font-bold text-gray-900">
                                Recommended Assessments
                            </h2>
                            <span className="bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm font-semibold">
                                {results.total_recommendations} Results
                            </span>
                        </div>

                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200">
                                <thead className="bg-gradient-to-r from-gray-50 to-blue-50">
                                    <tr>
                                        <th className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider">
                                            #
                                        </th>
                                        <th className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider">
                                            Assessment Name
                                        </th>
                                        <th className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider">
                                            Test Type
                                        </th>
                                        <th className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider">
                                            Relevance
                                        </th>
                                        <th className="px-6 py-4 text-left text-xs font-bold text-gray-700 uppercase tracking-wider">
                                            Link
                                        </th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-100">
                                    {results.recommendations.map((assessment, index) => (
                                        <tr key={index} className="hover:bg-blue-50 transition-colors duration-150">
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-900">
                                                {index + 1}
                                            </td>
                                            <td className="px-6 py-4 text-sm text-gray-900 font-medium">
                                                {assessment.assessment_name}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                                                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${assessment.test_type === 'K' ? 'bg-blue-100 text-blue-800' :
                                                    assessment.test_type === 'P' ? 'bg-green-100 text-green-800' :
                                                        'bg-gray-100 text-gray-800'
                                                    }`}>
                                                    {assessment.test_type === 'K' ? 'Knowledge' :
                                                        assessment.test_type === 'P' ? 'Personality' :
                                                            assessment.test_type || 'Other'}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                                {assessment.relevance_score ? (
                                                    <div className="flex items-center">
                                                        <div className="w-24 bg-gray-200 rounded-full h-2 mr-2">
                                                            <div
                                                                className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                                                                style={{ width: `${Math.round(assessment.relevance_score * 100)}%` }}
                                                            ></div>
                                                        </div>
                                                        <span className="font-semibold">{Math.round(assessment.relevance_score * 100)}%</span>
                                                    </div>
                                                ) : 'N/A'}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                                                <a
                                                    href={assessment.assessment_url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="text-blue-600 hover:text-blue-800 font-semibold hover:underline inline-flex items-center"
                                                >
                                                    View Details
                                                    <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                                    </svg>
                                                </a>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* Footer */}
                <div className="mt-12 text-center text-sm text-gray-500">
                    <p>Powered by AI • Built with Next.js and FastAPI</p>
                </div>
            </div>
        </div>
    );
}
