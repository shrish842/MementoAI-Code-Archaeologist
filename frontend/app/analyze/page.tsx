"use client";

import { Stage } from "@/types/analysis";
import React, { useState } from "react";
import {
  GitBranch,
  GitCommit,
  Calendar,
  User,
  MessageSquare,
  Search,
  Code,
  ArrowLeft,
  Home,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
// import { ThemeToggle } from "@/components/theme-toggle";

import { Diff, Hunk, parseDiff } from "react-diff-view";
import "react-diff-view/style/index.css";

import { mockAnalysis, mockCommits } from "../data";

const Page = () => {
  const [repoUrl, setRepoUrl] = useState("");
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [stage, setStage] = useState<Stage>("input");

  const handleRepoSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!repoUrl.trim()) return;

    setIsLoading(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setIsLoading(false);
    setStage("commits");
  };

  const handleQuestionSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setIsLoading(true);
    // Simulate AI analysis
    await new Promise((resolve) => setTimeout(resolve, 3000));
    setIsLoading(false);
    setStage("results");
  };

  const resetToStart = () => {
    setStage("input");
    setRepoUrl("");
    setQuestion("");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="text-center flex-1">
            <h1 className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-2">
              MementoAI: Codebase Archaeologist
            </h1>
            <p className="text-slate-600 dark:text-slate-400 text-lg">
              Uncover the hidden stories in your code repository
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" className="cursor-pointer py-2">
              <Home className="h-4 w-4" />
            </Button>
            {/* <ThemeToggle /> */}
          </div>
        </div>

        {/* Progress Indicator */}
        <div className="flex justify-center mb-8">
          <div className="flex items-center space-x-4">
            {["input", "commits", "question", "results"].map((s, index) => (
              <div key={s} className="flex items-center">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    stage === s
                      ? "bg-blue-600 text-white"
                      : ["input", "commits", "question", "results"].indexOf(
                          stage
                        ) > index
                      ? "bg-green-600 text-white"
                      : "bg-slate-300 dark:bg-slate-600 text-slate-600 dark:text-slate-300"
                  }`}
                >
                  {index + 1}
                </div>
                {index < 3 && (
                  <div className="w-12 h-0.5 bg-slate-300 dark:bg-slate-600 mx-2" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Stage 1: Repository Input */}
        {stage === "input" && (
          <Card className="max-w-2xl mx-auto">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GitBranch className="w-5 h-5" />
                Enter Repository URL
              </CardTitle>
              <CardDescription>
                Provide a GitHub repository URL to begin the archaeological dig
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleRepoSubmit} className="space-y-4">
                <div>
                  <Label htmlFor="repo-url">GitHub Repository URL</Label>
                  <Input
                    id="repo-url"
                    type="url"
                    placeholder="https://github.com/username/repository"
                    value={repoUrl}
                    onChange={(e) => setRepoUrl(e.target.value)}
                    className="mt-1"
                    required
                  />
                </div>
                <Button type="submit" className="w-full cursor-pointer" disabled={isLoading}>
                  {isLoading ? "Analyzing Repository..." : "Start Analysis"}
                </Button>
              </form>
            </CardContent>
          </Card>
        )}

        {/* Stage 2: Commit Information */}
        {stage === "commits" && (
          <div className="max-w-4xl mx-auto space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <GitCommit className="w-5 h-5" />
                  Recent Commit History
                </CardTitle>
                <CardDescription>Repository: {repoUrl}</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-96">
                  <div className="space-y-4">
                    {mockCommits.map((commit) => (
                      <div
                        key={commit.sha}
                        className="border rounded-lg p-4 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <h3 className="font-medium text-slate-900 dark:text-slate-100">
                            {commit.message}
                          </h3>
                          <Badge variant="outline" className="ml-2">
                            {commit.sha}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-4 text-sm text-slate-600 dark:text-slate-400">
                          <div className="flex items-center gap-1">
                            <User className="w-4 h-4" />
                            {commit.author}
                          </div>
                          <div className="flex items-center gap-1">
                            <Calendar className="w-4 h-4" />
                            {commit.date}
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-green-600">
                              +{commit.additions}
                            </span>
                            <span className="text-red-600">
                              -{commit.deletions}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
                <div className="mt-6">
                  <Button
                    onClick={() => setStage("question")}
                    className="w-full cursor-pointer"
                  >
                    Continue to Questions
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Stage 3: Question Input */}
        {stage === "question" && (
          <Card className="max-w-2xl mx-auto">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="w-5 h-5" />
                Ask About the Repository
              </CardTitle>
              <CardDescription>
                What would you like to know about this codebase? Ask about
                patterns, changes, or specific aspects.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleQuestionSubmit} className="space-y-4">
                <div>
                  <Label htmlFor="question">Your Question</Label>
                  <Textarea
                    id="question"
                    placeholder="e.g., What are the main security improvements made recently? What patterns show technical debt?"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    className="mt-1 min-h-[100px]"
                    required
                  />
                </div>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setStage("commits")}
                    className="flex items-center gap-2 cursor-pointer"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    Back
                  </Button>
                  <Button type="submit" className="flex-1 cursor-pointer" disabled={isLoading}>
                    {isLoading ? "Analyzing..." : "Analyze Repository"}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        )}

        {/* Stage 4: Results */}
        {stage === "results" && (
          <div className="max-w-6xl mx-auto space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Search className="w-5 h-5" />
                  Analysis Results
                </CardTitle>
                <CardDescription>
                  Question: &quot;{question}&quot;
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* Summary */}
                  <div>
                    <h3 className="text-lg font-semibold mb-3">Summary</h3>
                    <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                      {mockAnalysis.summary}
                    </p>
                  </div>

                  <Separator />

                  {/* Key Findings */}
                  <div>
                    <h3 className="text-lg font-semibold mb-3">Key Findings</h3>
                    <div className="grid gap-3">
                      {mockAnalysis.keyFindings.map((finding, index) => (
                        <div
                          key={index}
                          className="flex items-start gap-3 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg"
                        >
                          <div className="w-6 h-6 bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full flex items-center justify-center text-sm font-medium mt-0.5">
                            {index + 1}
                          </div>
                          <p className="text-slate-700 dark:text-slate-300">
                            {finding}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <Separator />

                  {/* Code Changes with react-diff-view */}
                  <div>
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                      <Code className="w-5 h-5" />
                      Code Changes Analysis
                    </h3>
                    {mockAnalysis.codeChanges.map((change, index) => {
                      const files = parseDiff(change.diffText);

                      return (
                        <div
                          key={index}
                          className="border rounded-lg overflow-hidden mb-6"
                        >
                          <div className="bg-slate-100 dark:bg-slate-800 px-4 py-2 border-b">
                            <h4 className="font-medium">{change.file}</h4>
                            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                              {change.explanation}
                            </p>
                          </div>
                          <div className="bg-white dark:bg-slate-900">
                            {files.map((file, fileIndex) => (
                              <Diff
                                key={fileIndex}
                                viewType="split"
                                diffType={file.type}
                                hunks={file.hunks}
                              >
                                {(hunks) =>
                                  hunks.map((hunk) => (
                                    <Hunk key={hunk.content} hunk={hunk} />
                                  ))
                                }
                              </Diff>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2 pt-4">
                    <Button
                      variant="outline"
                      onClick={() => setStage("question")}
                      className="flex items-center gap-2 cursor-pointer"
                    >
                      <ArrowLeft className="w-4 h-4" />
                      Ask Another Question
                    </Button>
                    <Button variant="outline" onClick={resetToStart} className="cursor-pointer">
                      Analyze New Repository
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};

export default Page;
