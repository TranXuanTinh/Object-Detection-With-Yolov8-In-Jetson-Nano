# 04. Behavioral Design Patterns

Behavioral patterns govern **communication, algorithms, and responsibility assignment** between objects. They are the most frequently tested patterns in LLD interviews.

> All C++17 examples strictly use explicit types (no `auto` keyword).

---

## 1. Quick-Reference Matrix

| Pattern | Core Intent | Interview Trigger Keywords |
|---|---|---|
| **Strategy** | Encapsulate interchangeable algorithms and swap them at runtime. | *"Multiple pricing rules / sorting algorithms / routing / payment modes"* |
| **Observer** | Define a 1-to-many Pub/Sub dependency so listeners auto-update on state changes. | *"Notify users / dashboard / email / inventory when event occurs"* |
| **State** | Alter an object's behavior cleanly when its internal state changes (avoid giant `switch`). | *"Vending machine, Elevator, Traffic Light, Order lifecycle"* |
| **Command** | Encapsulate a request as an object to support queuing, logging, and **Undo/Redo**. | *"Text editor undo/redo, task queue, smart home remote"* |
| **Chain of Responsibility** | Pass a request along a chain of handlers until one handles or rejects it. | *"ATM cash dispenser, HTTP auth/rate-limit middleware, logging levels"* |
| **Template Method** | Define the skeleton of an algorithm in a base class; subclasses override specific steps. | *"Data parser pipeline (CSV/JSON/XML), Build/Test runner"* |
| **Iterator** | Traverse elements of a collection sequentially without exposing its internal structure. | *"Custom tree/graph traversal, paginated API iterator"* |
| **Mediator** | Centralize complex many-to-many communication through a hub object. | *"Chat room, Air Traffic Control, UI dialog coordinating widgets"* |
| **Memento** | Capture and restore an object's internal state without breaking encapsulation. | *"Save/load game checkpoints, snapshot rollback"* |
| **Visitor** | Separate an algorithm from the object structure it operates on (double dispatch). | *"AST compiler pass, exporting shapes to XML/JSON"* |

---

## 2. Essential C++17 Implementations (No `auto`)

### 2.1. Strategy Pattern (Pricing / Routing)
```cpp
#include <iostream>
#include <memory>

class IPricingStrategy {
public:
    virtual ~IPricingStrategy() = default;
    virtual double calculateFee(int hours) const = 0;
};

class HourlyPricing : public IPricingStrategy {
public:
    double calculateFee(int hours) const override { return hours * 2.0; }
};

class FlatDailyPricing : public IPricingStrategy {
public:
    double calculateFee(int /*hours*/) const override { return 15.0; }
};

class FeeCalculator {
public:
    void setStrategy(std::unique_ptr<IPricingStrategy> strategy) {
        strategy_ = std::move(strategy);
    }
    double compute(int hours) const {
        return strategy_ ? strategy_->calculateFee(hours) : 0.0;
    }
private:
    std::unique_ptr<IPricingStrategy> strategy_;
};
```

### 2.2. Observer Pattern (Pub/Sub Notifications)
```cpp
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

class IObserver {
public:
    virtual ~IObserver() = default;
    virtual void onUpdate(const std::string& event) = 0;
};

class Subject {
public:
    void subscribe(IObserver* obs) { observers_.push_back(obs); }

    void unsubscribe(IObserver* obs) {
        observers_.erase(std::remove(observers_.begin(), observers_.end(), obs),
                         observers_.end());
    }

    void notify(const std::string& event) {
        for (IObserver* obs : observers_) {
            if (obs != nullptr) obs->onUpdate(event);
        }
    }
private:
    std::vector<IObserver*> observers_;
};
```

### 2.3. State Pattern (Vending Machine / Order Lifecycle)
```cpp
#include <iostream>
#include <memory>

class MachineContext;

class IMachineState {
public:
    virtual ~IMachineState() = default;
    virtual void insertCoin(MachineContext& ctx) = 0;
    virtual void dispense(MachineContext& ctx) = 0;
};

class MachineContext {
public:
    explicit MachineContext(std::unique_ptr<IMachineState> initial)
        : state_(std::move(initial)) {}

    void setState(std::unique_ptr<IMachineState> next) { state_ = std::move(next); }
    void insertCoin() { state_->insertCoin(*this); }
    void dispense()   { state_->dispense(*this); }

private:
    std::unique_ptr<IMachineState> state_;
};

class HasCoinState : public IMachineState {
public:
    void insertCoin(MachineContext& /*ctx*/) override {
        std::cout << "Coin already inserted.\n";
    }
    void dispense(MachineContext& ctx) override;
};

class IdleState : public IMachineState {
public:
    void insertCoin(MachineContext& ctx) override {
        std::cout << "Coin accepted -> switching to HasCoinState.\n";
        ctx.setState(std::make_unique<HasCoinState>());
    }
    void dispense(MachineContext& /*ctx*/) override {
        std::cout << "Insert coin first.\n";
    }
};

inline void HasCoinState::dispense(MachineContext& ctx) {
    std::cout << "Item dispensed -> returning to IdleState.\n";
    ctx.setState(std::make_unique<IdleState>());
}
```

### 2.4. Command Pattern (With Undo History)
```cpp
#include <memory>
#include <stack>
#include <string>

class ICommand {
public:
    virtual ~ICommand() = default;
    virtual void execute() = 0;
    virtual void undo() = 0;
};

class AppendTextCommand : public ICommand {
public:
    AppendTextCommand(std::string& doc, const std::string& text)
        : doc_(doc), text_(text) {}

    void execute() override { doc_ += text_; }
    void undo() override {
        if (doc_.size() >= text_.size()) {
            doc_.erase(doc_.size() - text_.size());
        }
    }
private:
    std::string& doc_;
    std::string text_;
};

class CommandInvoker {
public:
    void run(std::unique_ptr<ICommand> cmd) {
        cmd->execute();
        history_.push(std::move(cmd));
    }
    void undoLast() {
        if (!history_.empty()) {
            history_.top()->undo();
            history_.pop();
        }
    }
private:
    std::stack<std::unique_ptr<ICommand>> history_;
};
```

### 2.5. Chain of Responsibility (ATM Dispenser / Middleware)
```cpp
#include <iostream>
#include <memory>

class CashHandler {
public:
    virtual ~CashHandler() = default;
    void setNext(std::unique_ptr<CashHandler> next) { next_ = std::move(next); }

    virtual void dispense(int amount) {
        if (next_) next_->dispense(amount);
    }
protected:
    std::unique_ptr<CashHandler> next_;
};

class NoteHandler : public CashHandler {
public:
    explicit NoteHandler(int denom) : denom_(denom) {}

    void dispense(int amount) override {
        int count = amount / denom_;
        int remainder = amount % denom_;
        if (count > 0) {
            std::cout << "Dispensing " << count << " x $" << denom_ << "\n";
        }
        if (remainder > 0 && next_) {
            next_->dispense(remainder);
        }
    }
private:
    int denom_;
};
```

### 2.6. Template Method Pattern
```cpp
#include <iostream>

class DataMiner {
public:
    virtual ~DataMiner() = default;
    // Fixed algorithm skeleton (non-virtual)
    void mine() {
        openFile();
        extractData();
        closeFile();
    }
protected:
    virtual void openFile() = 0;
    virtual void extractData() = 0;
    void closeFile() { std::cout << "Closing file handle.\n"; }
};
```
