package com.paperpilot.bff.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix = "paperpilot")
public class AppProperties {
    private String djangoBaseUrl = "http://127.0.0.1:8001";

    public String getDjangoBaseUrl() {
        return djangoBaseUrl;
    }

    public void setDjangoBaseUrl(String djangoBaseUrl) {
        this.djangoBaseUrl = djangoBaseUrl;
    }
}
