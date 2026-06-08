package com.paperpilot.bff.controller;

import com.paperpilot.bff.config.AppProperties;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@RestController
public class HealthController {
    private final AppProperties properties;
    private final RestTemplate restTemplate = new RestTemplate();

    public HealthController(AppProperties properties) {
        this.properties = properties;
    }

    @GetMapping("/api/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> body = new HashMap<>();
        body.put("status", "ok");
        body.put("bff", "online");

        try {
            @SuppressWarnings("unchecked")
            Map<String, Object> django = restTemplate.getForObject(
                    properties.getDjangoBaseUrl() + "/api/health/",
                    Map.class
            );
            body.put("django", django);
        } catch (Exception ex) {
            body.put("django", Map.of("status", "offline", "error", ex.getMessage()));
        }

        return ResponseEntity.ok(body);
    }
}
