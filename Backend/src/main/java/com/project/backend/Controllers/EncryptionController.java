package com.project.backend.Controllers;

import com.project.backend.Services.GeneratedSampleRegistryService;
import com.project.backend.Services.EncryptionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/encryption")
public class EncryptionController {

    private final EncryptionService encryptionService;
    private final GeneratedSampleRegistryService generatedSampleRegistryService;

    public EncryptionController(EncryptionService encryptionService,
                                GeneratedSampleRegistryService generatedSampleRegistryService) {
        this.encryptionService = encryptionService;
        this.generatedSampleRegistryService = generatedSampleRegistryService;
    }

    private ResponseEntity<String> registerAndReturn(String payload, String algorithm) {
        generatedSampleRegistryService.registerSample(payload, algorithm);
        return ResponseEntity.ok(payload);
    }

    @GetMapping("/aes")
    public ResponseEntity<String> aesEncrypt() {
        try {
            String encryptedData = encryptionService.aesEncrypt();
            return registerAndReturn(encryptedData, "AES");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error during AES encryption: " + e.getMessage());
        }
    }


    @GetMapping("/des")
    public ResponseEntity<String> desEncrypt() throws Exception {
        try {
            String encryptedData = encryptionService.desEncrypt();
            return registerAndReturn(encryptedData, "DES");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error during DES encryption: " + e.getMessage());
        }
    }


    @GetMapping("/3des")
    public ResponseEntity<String> encryptWithTripleDes() {
        try {
            String encryptedData = encryptionService.tripleDesEncrypt();
            return registerAndReturn(encryptedData, "TripleDES");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error during triple des encryption: " + e.getMessage());
        }
    }

    @GetMapping("/blowfish")
    public ResponseEntity<String> encryptWithBlowfish() {
        try {
            String encryptedData = encryptionService.blowfishEncrypt();
            return registerAndReturn(encryptedData, "Blowfish");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error during encryption: " + e.getMessage());
        }
    }

    @GetMapping("/rc2")
    public ResponseEntity<String> encryptWithRC2() {
        try {
            String encryptedData = encryptionService.rc2Encrypt();
            return registerAndReturn(encryptedData, "RC2");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error during RC2 encryption: " + e.getMessage());
        }
    }


    @GetMapping("/rc4")
    public ResponseEntity<String> encryptWithRC4(@RequestParam(required = false) String plaintext) {
        try {
            String encryptedData = encryptionService.rc4Encrypt(plaintext);
            return registerAndReturn(encryptedData, "RC4");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error during encryption: " + e.getMessage());
        }
    }

    @GetMapping("/chacha20")
    public ResponseEntity<String> encryptWithChaCha20() {
        try {
            String encryptedData = encryptionService.chacha20Encrypt();
            return registerAndReturn(encryptedData, "ChaCha20");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error during encryption: " + e.getMessage());
        }
    }

    @GetMapping("/rsa")
    public ResponseEntity<String> encryptWithRSA() {
        try {
            String encryptedData = encryptionService.rsaEncrypt();
            return registerAndReturn(encryptedData, "RSA");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error during encryption: " + e.getMessage());
        }
    }

    @GetMapping("/dsa")
    public ResponseEntity<String> generateSignature() {
        try {
            String signature = encryptionService.generateDsaSignature();
            return registerAndReturn(signature, "DSA");
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error generating DSA signature: " + e.getMessage());
        }
    }

    @GetMapping("/ecdsa")
    public ResponseEntity<String> generateSignaturee() {
        try {
            String signature = encryptionService.generateEcdsaSignature();
            return registerAndReturn(signature, "ECDSA");
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error generating ECDSA signature: " + e.getMessage());
        }
    }

    @GetMapping("/diffe")
    public ResponseEntity<String> exchangeKeys() {
        try {
            String derivedKey = encryptionService.performDiffieHellmanKeyExchange();
            return registerAndReturn(derivedKey, "Diffie-Hellman");
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error during key exchange: " + e.getMessage());
        }
    }

    @GetMapping("/ecdh")
    public ResponseEntity<String> exchangeKeysEcdh() {
        try {
            String derivedKey = encryptionService.performECDHKeyExchange();
            return registerAndReturn(derivedKey, "ECDH");
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error during ECDH key exchange: " + e.getMessage());
        }
    }

    @GetMapping("/md5")
    public ResponseEntity<String> getMD5Hash() {
        try {
            String md5Hash = encryptionService.generateMD5Hash();
            return registerAndReturn(md5Hash, "MD5");
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error generating MD5 hash: " + e.getMessage());
        }
    }

    @GetMapping("/sha1")
    public ResponseEntity<String> getSHA1Hash() {
        try {
            String sha1Hash = encryptionService.generateSHA1Hash();
            return registerAndReturn(sha1Hash, "SHA-1");
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error generating SHA-1 hash: " + e.getMessage());
        }
    }


    @GetMapping("/sha256")
    public ResponseEntity<String> getSHA256Hash() {
        try {
            String sha256Hash = encryptionService.generateSHA256Hash();
            return registerAndReturn(sha256Hash, "SHA-256");
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error generating SHA-256 hash: " + e.getMessage());
        }
    }

    @GetMapping("/sha512")
    public ResponseEntity<String> getSHA512Hash() {
        try {
            String sha512Hash = encryptionService.generateSHA512Hash();
            return registerAndReturn(sha512Hash, "SHA-512");
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error generating SHA-512 hash: " + e.getMessage());
        }
    }


    @GetMapping("/sha3-256")
    public ResponseEntity<String> getSHA3_256Hash() {
        try {
            String sha3Hash = encryptionService.generateSHA3_256Hash();
            return registerAndReturn(sha3Hash, "SHA-3-256");
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error generating SHA-3-256 hash: " + e.getMessage());
        }
    }
}
