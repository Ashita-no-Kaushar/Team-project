import { Navigate, useLocation } from "react-router-dom";
import { useSelector } from "react-redux";
import type { RootState } from "../redux/store";

const normalizeToken = (value: string | null | undefined): string | null => {
  if (!value) {
    return null;
  }
  if (value === "undefined" || value === "null") {
    return null;
  }
  return value;
};

const PrivateRoute = ({ children }: { children: React.ReactNode }) => {
  const location = useLocation();
  const reduxToken = useSelector((state: RootState) => state.auth.token);
  const storageToken = normalizeToken(localStorage.getItem("accessToken"))
    || normalizeToken(localStorage.getItem("token"));
  const token = normalizeToken(reduxToken) || storageToken;

  console.log("Current Token:", token ? "present" : "missing");

  return token ? children : <Navigate to="/login" replace state={{ from: location.pathname }} />;
};

export default PrivateRoute;
