import type { Metadata } from 'next'

export const metadata: Metadata = {
    title: 'SHL Assessment Recommender',
    description: 'AI-powered recommendations for SHL assessments based on job descriptions',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    )
}
