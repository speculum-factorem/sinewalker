package com.stivensigal.walking.storage;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stivensigal.walking.api.GenomeDTO;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Optional;

@Component
public class ModelStorage {

  private final Path storageDir;
  private final ObjectMapper objectMapper;

  public ModelStorage(
      @Value("${app.storageDir:./backend/data/models}") String storageDir,
      ObjectMapper objectMapper
  ) {
    this.storageDir = Path.of(storageDir);
    this.objectMapper = objectMapper;
  }

  public ModelEntry save(String id, String name, GenomeDTO genome, Instant createdAt) {
    try {
      Files.createDirectories(storageDir);
      var entry = new ModelEntry(id, name, genome, createdAt);
      var filePath = storageDir.resolve(id + ".json");
      objectMapper.writeValue(filePath.toFile(), entry);
      return entry;
    } catch (IOException e) {
      throw new RuntimeException("Failed to save model", e);
    }
  }

  public ModelEntry get(String id) {
    var filePath = storageDir.resolve(id + ".json");
    if (!Files.exists(filePath)) return null;

    try {
      return objectMapper.readValue(filePath.toFile(), ModelEntry.class);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model", e);
    }
  }

  public record ModelEntry(
      String id,
      String name,
      GenomeDTO genome,
      Instant createdAt
  ) {}
}

