package com.stivensigal.walking.api;

import com.stivensigal.walking.storage.ModelStorage;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.util.UUID;

@RestController
@RequestMapping("/models")
@CrossOrigin(origins = "*")
public class ModelController {

  private final ModelStorage storage;

  public ModelController(ModelStorage storage) {
    this.storage = storage;
  }

  @PostMapping
  public ResponseEntity<ModelCreateResponse> createModel(@Valid @RequestBody ModelCreateRequest request) {
    var id = UUID.randomUUID().toString();
    var createdAt = Instant.now();
    var entry = storage.save(id, request.name(), request.genome(), createdAt);
    return ResponseEntity.ok(new ModelCreateResponse(entry.id(), entry.name(), entry.createdAt()));
  }

  @GetMapping("/{id}")
  public ResponseEntity<ModelGetResponse> getModel(@PathVariable String id) {
    var entry = storage.get(id);
    if (entry == null) {
      return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
    }
    return ResponseEntity.ok(new ModelGetResponse(entry.id(), entry.name(), entry.genome(), entry.createdAt()));
  }

  public record ModelCreateRequest(
      @NotBlank(message = "name must not be blank")
      String name,
      @Valid GenomeDTO genome
  ) {}

  public record ModelCreateResponse(String id, String name, Instant createdAt) {}

  public record ModelGetResponse(String id, String name, GenomeDTO genome, Instant createdAt) {}
}

