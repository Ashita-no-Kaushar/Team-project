import { lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
const SignupFormDemo = lazy(() => import('./components/SignUp'));
const Login = lazy(() => import('./components/Login'));
const HistoryPage = lazy(() => import('./components/History'));
const LandingPage = lazy(() => import('./components/LandingPage'));
const ProfilePage = lazy(() => import('./components/ProfilePage'));
const DocumentationPage = lazy(() => import('./components/DocumentationPage'));
const PredictionPage = lazy(() => import('./components/PredictionPage'));
const EncryptionPage = lazy(() => import('./components/EncryptionPage'));
const PrivateRoute = lazy(() => import('./components/PrivateRoute'));
const Layout = lazy(() => import('./components/Layout'));

function RouteLoadingFallback() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center">
      Loading page...
    </div>
  );
}


function App() {

 
  return (
    <Router>
      <Toaster position="top-right" reverseOrder={false} />

      <Suspense fallback={<RouteLoadingFallback />}>
        <Routes>
          <Route path="/signup" element={<SignupFormDemo />} />
          <Route path="/login" element={<Login />} />
          <Route
            path="/"
            element={
              <Layout>
                <LandingPage />
              </Layout>
            }
          />
          <Route
            path="/history"
            element={
              <PrivateRoute>
                <Layout>
                  <HistoryPage />
                </Layout>
              </PrivateRoute>
            }
          />
          <Route
            path="/profile"
            element={
              <PrivateRoute>
                <Layout>
                  <ProfilePage />
                </Layout>
              </PrivateRoute>
            }
          />
          <Route
            path="/docs"
            element={
              <Layout>
                <DocumentationPage />
              </Layout>
            }
          />
          <Route
            path="/prediction"
            element={
              <PrivateRoute>
                <Layout>
                  <PredictionPage />
                </Layout>
              </PrivateRoute>
            }
          />
          <Route
            path="/encry"
            element={
              <PrivateRoute>
                <Layout>
                  <EncryptionPage />
                </Layout>
              </PrivateRoute>
            }
          />
        </Routes>
      </Suspense>
    </Router>
  );
}

export default App;


