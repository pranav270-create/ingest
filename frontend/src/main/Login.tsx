import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiService } from '@/api/axios';


type LoginResponse = {
	status: string;
	message: string;
	token?: string;
}

const Login: React.FC = () => {
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");
	const navigate = useNavigate();

	const handleLogin = async (event: React.FormEvent<HTMLFormElement>) => {
		event.preventDefault();

		try {
			const response = await apiService.post<LoginResponse>('/user/login', {
				username: email,
				password: password
			});
			
			if (response.status === "success" && response.token) {
				localStorage.setItem('isAuthenticated', 'true');
				localStorage.setItem('token', response.token);
				navigate("/internal");
			} else {
				alert("Invalid credentials");
			}
		} catch (error) {
			console.error("Login error:", error);
			alert("Login failed");
		}
	};

	return (
		<div className="flex flex-col min-h-screen font-aliance">
			<div className="flex-grow flex items-center justify-center bg-gray-100">
				<div className="w-full max-w-md p-8 bg-white rounded-lg shadow-md">
					<h2 className="text-2xl font-bold mb-6 text-center">Log in</h2>
					<form className="space-y-4" onSubmit={handleLogin}>
						<input
							type="email"
							value={email}
							onChange={(e) => setEmail(e.target.value)}
							placeholder="Email *"
							required
							className="w-full p-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
						<input
							type="password"
							value={password}
							onChange={(e) => setPassword(e.target.value)}
							placeholder="Password *"
							required
							className="w-full p-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
						<button
							type="submit"
							className="w-full p-3 text-white bg-black rounded hover:bg-gray-800 transition duration-200"
						>
							Log in
						</button>
					</form>
				</div>
			</div>
		</div>
	);
};

export default Login;
