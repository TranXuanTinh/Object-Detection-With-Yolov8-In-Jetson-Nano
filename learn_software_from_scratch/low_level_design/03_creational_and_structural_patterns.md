# 03. Creational & Structural Design Patterns

> All C++17 examples strictly use explicit types (no `auto` keyword).

---

## 1. Creational Patterns (Object Creation Mechanisms)

| Pattern | Core Intent | When to Use in LLD Interviews |
|---|---|---|
| **Singleton** | Ensure a class has **only 1 instance** and provide a global access point. | `Logger`, `ConfigManager`, `ParkingLot` root coordinator, connection pool. |
| **Factory Method** | Delegate instantiation to subclasses or a creator method behind an interface. | Creating `Vehicle` (`Car`, `Bike`, `Truck`), `Notification` (`Email`, `SMS`), `Payment`. |
| **Abstract Factory** | Create **families of related objects** without specifying concrete classes. | Cross-platform UI kits (`WinButton` + `WinCheckbox` vs. `MacButton` + `MacCheckbox`). |
| **Builder** | Construct complex objects **step-by-step** cleanly (avoid telescoping constructors). | `HttpRequest`, `Pizza`, `SQLQuery`, `UserProfile` with many optional fields. |
| **Prototype** | Clone existing pre-configured objects instead of building from scratch. | Expensive object initialization (game NPCs, document/config templates). |

### Key C++ Examples (No `auto`)

#### 1.1. Thread-Safe Singleton (Meyers' Singleton — C++11/17)
```cpp
#include <iostream>

class ConfigManager {
public:
    static ConfigManager& getInstance() {
        static ConfigManager instance; // Guaranteed thread-safe initialization in C++11+
        return instance;
    }

    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    void setPort(int port) { port_ = port; }
    int getPort() const { return port_; }

private:
    ConfigManager() : port_(8080) {}
    int port_;
};
```

#### 1.2. Factory Method & Abstract Factory
```cpp
#include <memory>
#include <string>

class INotification {
public:
    virtual ~INotification() = default;
    virtual void send(const std::string& msg) = 0;
};

class EmailNotification : public INotification {
public:
    void send(const std::string& msg) override { std::cout << "[Email] " << msg << "\n"; }
};

class SmsNotification : public INotification {
public:
    void send(const std::string& msg) override { std::cout << "[SMS] " << msg << "\n"; }
};

class NotificationFactory {
public:
    static std::unique_ptr<INotification> create(const std::string& channel) {
        if (channel == "email") return std::make_unique<EmailNotification>();
        if (channel == "sms")   return std::make_unique<SmsNotification>();
        return nullptr;
    }
};
```

#### 1.3. Builder Pattern (Fluent API)
```cpp
#include <string>

class HttpRequest {
public:
    class Builder {
    public:
        explicit Builder(const std::string& url) : url_(url) {}
        Builder& method(const std::string& m) { method_ = m; return *this; }
        Builder& body(const std::string& b)   { body_ = b;   return *this; }
        Builder& timeoutMs(int ms)            { timeout_ms_ = ms; return *this; }

        HttpRequest build() const {
            return HttpRequest(url_, method_, body_, timeout_ms_);
        }
    private:
        std::string url_;
        std::string method_{"GET"};
        std::string body_;
        int timeout_ms_{3000};
    };

private:
    HttpRequest(const std::string& u, const std::string& m, const std::string& b, int t)
        : url_(u), method_(m), body_(b), timeout_ms_(t) {}

    std::string url_;
    std::string method_;
    std::string body_;
    int timeout_ms_;
};

// Usage: HttpRequest req = HttpRequest::Builder("https://api.example.com").method("POST").timeoutMs(5000).build();
```

#### 1.4. Prototype Pattern
```cpp
#include <memory>
#include <string>

class IShape {
public:
    virtual ~IShape() = default;
    virtual std::unique_ptr<IShape> clone() const = 0;
};

class Circle : public IShape {
public:
    explicit Circle(int radius) : radius_(radius) {}
    std::unique_ptr<IShape> clone() const override {
        return std::make_unique<Circle>(*this);
    }
private:
    int radius_;
};
```

---

## 2. Structural Patterns (Object Composition & Interfaces)

| Pattern | Core Intent | Classic Example / Use Case |
|---|---|---|
| **Adapter** | Convert an incompatible interface into the interface clients expect. | Wrapping a 3rd-party `StripeSDK` behind your internal `IPaymentGateway`. |
| **Decorator** | Dynamically wrap an object to add responsibilities **without subclassing**. | `Coffee` + `MilkDecorator` + `SugarDecorator`; File stream + Compression + Encryption. |
| **Proxy** | Control access to another object (lazy loading, auth, caching, logging). | Virtual Proxy for heavy images, Protection Proxy checking user roles. |
| **Bridge** | Decouple an **abstraction** from its **implementation** so both vary independently. | `RemoteControl` (Basic, Advanced) bridged to `IDevice` (`TV`, `Radio`) $\implies$ avoids $M \times N$ subclasses. |
| **Composite** | Compose objects into **tree structures** and treat leaves and containers uniformly. | File system (`File` + `Folder`), UI containers, Arithmetic expression trees. |
| **Facade** | Provide a **single simplified interface** to a complex subsystem. | `HomeTheaterFacade::watchMovie()` orchestrating `Amp`, `Projector`, `Lights`. |
| **Flyweight** | Share **intrinsic (immutable)** state across thousands of objects to save RAM. | Text editor storing 1 `FontGlyph` per character `'A'` instead of duplicating bitmap/font per position. |

### Key C++ Examples (No `auto`)

#### 2.1. Adapter Pattern
```cpp
#include <iostream>
#include <string>

// Target interface expected by our application
class IPaymentProcessor {
public:
    virtual ~IPaymentProcessor() = default;
    virtual void processPayment(double dollars) = 0;
};

// Incompatible 3rd-party legacy library
class LegacyBankApi {
public:
    void makeTransactionInCents(long cents) {
        std::cout << "Bank charged " << cents << " cents.\n";
    }
};

// Adapter translates dollars -> cents
class BankPaymentAdapter : public IPaymentProcessor {
public:
    explicit BankPaymentAdapter(LegacyBankApi& api) : api_(api) {}
    void processPayment(double dollars) override {
        api_.makeTransactionInCents(static_cast<long>(dollars * 100.0));
    }
private:
    LegacyBankApi& api_;
};
```

#### 2.2. Decorator Pattern
```cpp
#include <memory>
#include <string>

class IDataSource {
public:
    virtual ~IDataSource() = default;
    virtual std::string read() const = 0;
};

class RawFileSource : public IDataSource {
public:
    explicit RawFileSource(const std::string& content) : content_(content) {}
    std::string read() const override { return content_; }
private:
    std::string content_;
};

class BaseDecorator : public IDataSource {
public:
    explicit BaseDecorator(std::unique_ptr<IDataSource> wrappee)
        : wrappee_(std::move(wrappee)) {}
protected:
    std::unique_ptr<IDataSource> wrappee_;
};

class DecryptDecorator : public BaseDecorator {
public:
    explicit DecryptDecorator(std::unique_ptr<IDataSource> wrappee)
        : BaseDecorator(std::move(wrappee)) {}
    std::string read() const override {
        return "[Decrypted] " + wrappee_->read();
    }
};
```

#### 2.3. Composite Pattern
```cpp
#include <memory>
#include <string>
#include <vector>

class IFileSystemNode {
public:
    virtual ~IFileSystemNode() = default;
    virtual int getSizeKb() const = 0;
};

class FileLeaf : public IFileSystemNode {
public:
    explicit FileLeaf(int size_kb) : size_kb_(size_kb) {}
    int getSizeKb() const override { return size_kb_; }
private:
    int size_kb_;
};

class FolderComposite : public IFileSystemNode {
public:
    void add(std::unique_ptr<IFileSystemNode> node) {
        children_.push_back(std::move(node));
    }
    int getSizeKb() const override {
        int total = 0;
        for (const std::unique_ptr<IFileSystemNode>& child : children_) {
            total += child->getSizeKb();
        }
        return total;
    }
private:
    std::vector<std::unique_ptr<IFileSystemNode>> children_;
};
```
