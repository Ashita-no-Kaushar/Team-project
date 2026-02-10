package com.project.backend.Filters;


import com.project.backend.Exceptions.BearerTokenNotFoundException;
import com.project.backend.Exceptions.InvalidJwtAccessToken;
import com.project.backend.Services.JwtService;
import com.project.backend.Services.UserDetailsServiceImpl;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

@Component
public class JwtAuthFilter extends OncePerRequestFilter {

    private final JwtService jwtService;
    private final UserDetailsServiceImpl userDetailsService;

    //logger for logging, debugging and other stuff.
    private static final Logger logger = LoggerFactory.getLogger(JwtAuthFilter.class);

    @Autowired
    public JwtAuthFilter(JwtService jwtService, UserDetailsServiceImpl userDetailsService) {
        this.jwtService = jwtService;
        this.userDetailsService = userDetailsService;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        logger.info("Request Received in doFilterInternal : {}", request.getRequestURI());
        String authHeader = request.getHeader("Authorization"); //get authorization header.
        if (authHeader == null) { //handling cases
            authHeader = request.getHeader("authorization"); // Handle lowercase header
        }
        logger.info("Auth Header in doFilterInternal: {}", authHeader);
        String token = null;
        String username = null;

        if (authHeader != null && authHeader.startsWith("Bearer ")) {//Accounting for bearer token.
            logger.info("Authorization header is not null and also starts with Bearer ");
            token = authHeader.substring(7);
            try {
                username = jwtService.extractUsername(token);
            } catch (Exception e) {
                logger.warn("Failed to extract username from token: {}", e.getMessage());
                // Let the request proceed without authentication — public routes will still work
            }
        }
//        else{ //in case no header is found. dont do this .
//            throw new BearerTokenNotFoundException("Bearer token or Authentication token not found.");
//        }

        logger.info("Username in doFilterInternal : {}", username);
        if (username != null && (SecurityContextHolder.getContext().getAuthentication() == null
                || !SecurityContextHolder.getContext().getAuthentication().isAuthenticated())) {
            try {
                logger.info("Looking up user: {}", username);
                UserDetails userDetails = userDetailsService.loadUserByUsername(username);

                if (jwtService.validateToken(token, username)) {
                    logger.info("Inside the jwt validation of doInternalFilter");
                    UsernamePasswordAuthenticationToken authToken =
                            new UsernamePasswordAuthenticationToken(userDetails, null, userDetails.getAuthorities());
                    authToken.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                    SecurityContextHolder.getContext().setAuthentication(authToken);
                    logger.info("User authenticated: {}", username);
                }
            } catch (Exception e) {
                logger.warn("Could not authenticate user '{}': {}", username, e.getMessage());
                // Let the request proceed without authentication — public routes will still work
            }
        }
        logger.info("Returning request received.");
        filterChain.doFilter(request, response);
    }
}
