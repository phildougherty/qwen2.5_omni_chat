FROM nginx:alpine

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create directory for SSL certificates
RUN mkdir -p /etc/nginx/certs

# Create directory for frontend files
RUN mkdir -p /usr/share/nginx/html

# Copy frontend files (will be mounted via volume in docker-compose)

# Expose ports
EXPOSE 80
EXPOSE 443

# Start nginx
CMD ["nginx", "-g", "daemon off;"]