package com.paperpilot.bff.controller;

import com.paperpilot.bff.config.AppProperties;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.*;
import org.springframework.util.StreamUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartHttpServletRequest;
import org.springframework.web.multipart.support.StandardServletMultipartResolver;

import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Enumeration;

@RestController
public class ProxyController {
    private final AppProperties properties;
    private final RestTemplate restTemplate = new RestTemplate();
    private final StandardServletMultipartResolver multipartResolver = new StandardServletMultipartResolver();

    public ProxyController(AppProperties properties) {
        this.properties = properties;
    }

    @RequestMapping({
            "/api/sessions/**",
            "/api/chat/**",
            "/api/documents/**"
    })
    public ResponseEntity<byte[]> proxy(HttpServletRequest request) throws IOException {
        String targetUrl = properties.getDjangoBaseUrl()
                + request.getRequestURI()
                + (request.getQueryString() != null ? "?" + request.getQueryString() : "");

        HttpMethod method = HttpMethod.valueOf(request.getMethod());
        HttpHeaders headers = copyHeaders(request);
        headers.remove(HttpHeaders.HOST);

        byte[] body = readBody(request);
        HttpEntity<byte[]> entity = new HttpEntity<>(body, headers);

        ResponseEntity<byte[]> response = restTemplate.exchange(
                URI.create(targetUrl),
                method,
                entity,
                byte[].class
        );

        HttpHeaders responseHeaders = new HttpHeaders();
        responseHeaders.putAll(response.getHeaders());
        responseHeaders.remove(HttpHeaders.TRANSFER_ENCODING);

        return new ResponseEntity<>(response.getBody(), responseHeaders, response.getStatusCode());
    }

    private HttpHeaders copyHeaders(HttpServletRequest request) {
        HttpHeaders headers = new HttpHeaders();
        Enumeration<String> names = request.getHeaderNames();
        while (names.hasMoreElements()) {
            String name = names.nextElement();
            if ("host".equalsIgnoreCase(name) || "content-length".equalsIgnoreCase(name)) {
                continue;
            }
            headers.put(name, Collections.list(request.getHeaders(name)));
        }
        if (!headers.containsKey(HttpHeaders.CONTENT_TYPE) && request.getContentType() != null) {
            headers.set(HttpHeaders.CONTENT_TYPE, request.getContentType());
        }
        return headers;
    }

    private byte[] readBody(HttpServletRequest request) throws IOException {
        if (multipartResolver.isMultipart(request) && request instanceof MultipartHttpServletRequest multipart) {
            // Forward multipart by rebuilding is complex; upload goes direct to Django in dev.
            // For proxy path, read raw stream if available.
            return StreamUtils.copyToByteArray(request.getInputStream());
        }
        return StreamUtils.copyToByteArray(request.getInputStream());
    }
}
