import type { AnalysisResult, Commit } from "@/types/analysis";

export const mockCommits: Commit[] = [
  {
    sha: "a1b2c3d",
    message: "Add user authentication system",
    author: "john.doe",
    date: "2024-01-15",
    additions: 245,
    deletions: 12,
  },
  {
    sha: "e4f5g6h",
    message: "Refactor database connection logic",
    author: "jane.smith",
    date: "2024-01-14",
    additions: 89,
    deletions: 156,
  },
  {
    sha: "i7j8k9l",
    message: "Fix memory leak in data processing",
    author: "bob.wilson",
    date: "2024-01-13",
    additions: 34,
    deletions: 67,
  },
  {
    sha: "m0n1o2p",
    message: "Update dependencies and security patches",
    author: "alice.brown",
    date: "2024-01-12",
    additions: 178,
    deletions: 203,
  },
];

export const mockAnalysis: AnalysisResult = {
  summary:
    "The repository shows a well-maintained codebase with recent focus on security improvements and code refactoring. The authentication system was recently overhauled, and there's evidence of proactive maintenance through dependency updates.",
  keyFindings: [
    "Major authentication system overhaul in recent commits",
    "Database connection logic was refactored for better performance",
    "Memory leak issues were identified and resolved",
    "Regular security updates and dependency maintenance",
    "Code quality improvements through systematic refactoring",
  ],
  codeChanges: [
    {
      file: "src/auth/login.ts",
      oldCode: `function login(username: string, password: string) {
  // Basic authentication
  if (users[username] === password) {
    return { success: true };
  }
  return { success: false };
}`,
      newCode: `async function login(username: string, password: string) {
  // Enhanced authentication with hashing
  const hashedPassword = await bcrypt.hash(password, 10);
  const user = await User.findOne({ username });
  
  if (user && await bcrypt.compare(password, user.password)) {
    const token = jwt.sign({ userId: user.id }, JWT_SECRET);
    return { success: true, token };
  }
  return { success: false, error: 'Invalid credentials' };
}`,
      explanation:
        "Authentication was upgraded from plain text comparison to secure password hashing with JWT token generation.",
      diffText: `--- a/src/auth/login.ts
+++ b/src/auth/login.ts
@@ -1,7 +1,12 @@
-function login(username: string, password: string) {
-  // Basic authentication
-  if (users[username] === password) {
-    return { success: true };
+async function login(username: string, password: string) {
+  // Enhanced authentication with hashing
+  const hashedPassword = await bcrypt.hash(password, 10);
+  const user = await User.findOne({ username });
+  
+  if (user && await bcrypt.compare(password, user.password)) {
+    const token = jwt.sign({ userId: user.id }, JWT_SECRET);
+    return { success: true, token };
   }
-  return { success: false };
+  return { success: false, error: 'Invalid credentials' };
 }`,
    },
  ],
};
