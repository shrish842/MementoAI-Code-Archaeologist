export interface Commit {
  sha: string;
  message: string;
  author: string;
  date: string;
  additions: number;
  deletions: number;
}

export interface AnalysisResult {
  summary: string;
  keyFindings: string[];
  codeChanges: {
    file: string;
    oldCode: string;
    newCode: string;
    explanation: string;
    diffText: string;
  }[];
}

export type Stage = "input" | "commits" | "question" | "results";
