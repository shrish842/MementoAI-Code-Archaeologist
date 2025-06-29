import {
  GitBranch,
  Search,
  Code,
  Zap,
  Shield,
  Clock,
  ArrowRight,
  Github,
  MessageSquare,
  BarChart3,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
// import { ThemeToggle } from "@/components/theme-toggle";
import Link from "next/link";

export default function Page() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-800">
      <header className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
              <Search className="w-4 h-4 text-white" />
            </div>
            <span className="text-xl font-bold text-slate-900 dark:text-slate-100">
              MementoAI
            </span>
          </div>
          {/* <ThemeToggle /> */}
        </div>
      </header>

      <section className="container mx-auto px-4 py-16 text-center">
        <div className="max-w-4xl mx-auto">
          <Badge variant="secondary" className="mb-4">
            <Zap className="w-3 h-3 mr-1" />
            AI-Powered Code Analysis
          </Badge>

          <h1 className="text-5xl md:text-6xl font-bold text-slate-900 dark:text-slate-100 mb-6 leading-tight">
            Uncover the Hidden Stories in Your{" "}
            <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Codebase
            </span>
          </h1>

          <p className="text-xl text-slate-600 dark:text-slate-400 mb-8 leading-relaxed max-w-2xl mx-auto">
            MementoAI: Codebase Archaeologist uses advanced AI to analyze your
            GitHub repositories, revealing patterns, technical debt, and
            evolutionary insights that traditional tools miss.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
            <Link href="/analyze">
              <Button
                size="lg"
                className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-3 text-lg cursor-pointer"
              >
                Start Archaeological Dig
                <ArrowRight className="ml-2 w-5 h-5" />
              </Button>
            </Link>
            <Button
              variant="outline"
              size="lg"
              className="px-8 py-3 text-lg bg-transparent"
            >
              <Github className="mr-2 w-5 h-5" />
              View on GitHub
            </Button>
          </div>

          <div className="relative max-w-4xl mx-auto">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-indigo-600/20 blur-3xl rounded-3xl"></div>
            <Card className="relative border-2 border-slate-200 dark:border-slate-700 shadow-2xl">
              <CardContent className="p-0">
                <div className="bg-slate-100 dark:bg-slate-800 px-4 py-2 border-b flex items-center gap-2">
                  <div className="flex gap-1">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  </div>
                  <span className="text-sm text-slate-600 dark:text-slate-400 ml-2">
                    MementoAI Analysis Dashboard
                  </span>
                </div>
                <div className="p-8 bg-white dark:bg-slate-900">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-3/4"></div>
                      <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-1/2"></div>
                      <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-5/6"></div>
                    </div>
                    <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Code className="w-4 h-4 text-blue-600" />
                        <span className="text-sm font-medium">
                          Code Changes
                        </span>
                      </div>
                      <div className="space-y-2">
                        <div className="h-2 bg-green-200 dark:bg-green-800 rounded w-full"></div>
                        <div className="h-2 bg-red-200 dark:bg-red-800 rounded w-2/3"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      <section className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            Powerful Analysis Features
          </h2>
          <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Discover insights that help you understand your codebase evolution,
            identify technical debt, and make informed architectural decisions.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <Card className="border-2 hover:border-blue-200 dark:hover:border-blue-800 transition-colors">
            <CardContent className="p-6">
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center mb-4">
                <GitBranch className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
                Commit Analysis
              </h3>
              <p className="text-slate-600 dark:text-slate-400">
                Deep dive into commit history to understand development
                patterns, contributor behavior, and code evolution over time.
              </p>
            </CardContent>
          </Card>

          <Card className="border-2 hover:border-indigo-200 dark:hover:border-indigo-800 transition-colors">
            <CardContent className="p-6">
              <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900 rounded-lg flex items-center justify-center mb-4">
                <MessageSquare className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
                AI-Powered Q&A
              </h3>
              <p className="text-slate-600 dark:text-slate-400">
                Ask natural language questions about your codebase and get
                intelligent answers backed by comprehensive code analysis.
              </p>
            </CardContent>
          </Card>

          <Card className="border-2 hover:border-purple-200 dark:hover:border-purple-800 transition-colors">
            <CardContent className="p-6">
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded-lg flex items-center justify-center mb-4">
                <BarChart3 className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
                Visual Diff Viewer
              </h3>
              <p className="text-slate-600 dark:text-slate-400">
                Advanced diff visualization shows exactly what changed, when,
                and why, with intelligent highlighting and explanations.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      <section className="bg-slate-100 dark:bg-slate-800/50 py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-8">
              Why Choose MementoAI?
            </h2>

            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="w-16 h-16 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Clock className="w-8 h-8 text-green-600 dark:text-green-400" />
                </div>
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                  Save Time
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                  Instantly understand complex codebases without spending hours
                  reading through commits
                </p>
              </div>

              <div className="text-center">
                <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Shield className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                </div>
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                  Identify Risks
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                  Spot technical debt, security issues, and maintenance problems
                  before they become critical
                </p>
              </div>

              <div className="text-center">
                <div className="w-16 h-16 bg-purple-100 dark:bg-purple-900 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Zap className="w-8 h-8 text-purple-600 dark:text-purple-400" />
                </div>
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                  Make Better Decisions
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                  Data-driven insights help you plan refactoring, architecture
                  changes, and team allocation
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="container mx-auto px-4 py-16 text-center">
        <div className="max-w-2xl mx-auto">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            Ready to Explore Your Codebase?
          </h2>
          <p className="text-lg text-slate-600 dark:text-slate-400 mb-8">
            Start your archaeological journey today and uncover the hidden
            stories in your code.
          </p>
          <Link href="/analyze">
            <Button
              size="lg"
              // onClick={onGetStarted}
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-3 text-lg cursor-pointer"
            >
              Begin Analysis
              <ArrowRight className="ml-2 w-5 h-5" />
            </Button>
          </Link>
        </div>
      </section>

      <footer className="border-t border-slate-200 dark:border-slate-700 py-8">
        <div className="container mx-auto px-4 text-center text-slate-600 dark:text-slate-400">
          <p>
            &copy; 2024 MementoAI. Uncover the stories your code wants to tell.
          </p>
        </div>
      </footer>
    </div>
  );
}
