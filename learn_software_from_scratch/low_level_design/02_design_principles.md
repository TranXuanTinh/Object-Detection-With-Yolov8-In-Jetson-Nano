# 02. Design Principles (SOLID & Pragmatic Rules)

---

## 1. The SOLID Principles

| Letter | Principle | One-Line Rule | Violation Smell | Clean Fix |
|---|---|---|---|---|
| **S** | **Single Responsibility (SRP)** | A class should have **only one reason to change**. | "God class" handling business logic + DB + email/logging. | Split into focused classes (`Invoice`, `InvoiceRepository`, `InvoicePrinter`). |
| **O** | **Open/Closed (OCP)** | **Open for extension, closed for modification**. | Giant `if / else if / switch` on type enums whenever a new type is added. | Extract an abstract interface and add new subclasses. |
| **L** | **Liskov Substitution (LSP)** | Subclasses must be substitutable for their base class without breaking contracts. | Subclass throws `NotSupportedException` or breaks postconditions (`Square` inheriting `Rectangle`). | Model distinct abstractions (`IShape`) instead of forcing broken inheritance. |
| **I** | **Interface Segregation (ISP)** | Clients should **not** depend on methods they do not use. | Fat interface (`IMachine` forcing `OldPrinter` to stub `scan()` & `fax()`). | Split into small, role-specific interfaces (`IPrinter`, `IScanner`). |
| **D** | **Dependency Inversion (DIP)** | High-level modules depend on **abstractions**, not concrete low-level details. | Business service calls `new MySQLDatabase()` directly inside constructor. | Inject abstract interface (`IDatabase&`) via constructor. |

### Concise C++ Examples (No `auto`)

#### OCP + DIP Example (Discount & Notification System)
```cpp
#include <iostream>
#include <string>

// Abstraction (DIP & OCP)
class IDiscountStrategy {
public:
    virtual ~IDiscountStrategy() = default;
    virtual double apply(double price) const = 0;
};

class NoDiscount : public IDiscountStrategy {
public:
    double apply(double price) const override { return price; }
};

class SeasonalDiscount : public IDiscountStrategy {
public:
    explicit SeasonalDiscount(double pct) : pct_(pct) {}
    double apply(double price) const override { return price * (1.0 - pct_); }
private:
    double pct_;
};

// High-level class depends ONLY on IDiscountStrategy abstraction (DIP)
// Adding VIPDiscount requires zero edits to CheckoutService (OCP)
class CheckoutService {
public:
    explicit CheckoutService(const IDiscountStrategy& discount) : discount_(discount) {}

    double calculateTotal(double raw_price) const {
        return discount_.apply(raw_price);
    }
private:
    const IDiscountStrategy& discount_;
};
```

#### ISP + LSP Fix Example
```cpp
// Bad: Fat IMachine forces SimplePrinter to throw on scan() (violates ISP & LSP)
// Good: Segregated small interfaces
class IPrinter {
public:
    virtual ~IPrinter() = default;
    virtual void print(const std::string& doc) = 0;
};

class IScanner {
public:
    virtual ~IScanner() = default;
    virtual void scan() = 0;
};

class SimplePrinter : public IPrinter {
public:
    void print(const std::string& doc) override {
        std::cout << "Printing: " << doc << "\n";
    }
};

class MultiFunctionPrinter : public IPrinter, public IScanner {
public:
    void print(const std::string& doc) override { std::cout << "Printing: " << doc << "\n"; }
    void scan() override { std::cout << "Scanning...\n"; }
};
```

---

## 2. Core Pragmatic Design Principles

| Principle | Meaning | Practical Rule of Thumb |
|---|---|---|
| **DRY** *(Don't Repeat Yourself)* | Every piece of knowledge/logic should have a single, unambiguous source of truth. | Extract duplicated validation, formatting, or formulas into a shared helper — but don't prematurely merge things that only *look* similar by coincidence. |
| **KISS** *(Keep It Simple, Stupid)* | Simplicity should be a key goal; avoid unnecessary complexity. | Write clear, readable code over clever tricks or over-engineered abstractions. |
| **YAGNI** *(You Aren't Gonna Need It)* | Don't build features, hooks, or abstractions until there is an actual requirement. | Solve today's requirements cleanly; use interfaces only at genuine extension points. |
| **Law of Demeter** *(Principle of Least Knowledge)* | Talk only to your immediate friends, not strangers (`a.getB().getC().doX()`). | Wrap delegation cleanly (`a.doX()`) so callers aren't coupled to internal object chains. |
| **Composition Over Inheritance** | Favor assembling behaviors by holding component interfaces (*has-a*) rather than subclassing (*is-a*). | Inheritance is static at compile time and explodes with combinations (`FlyingFireSwimDuck`). Composition lets you swap behaviors dynamically at runtime. |
