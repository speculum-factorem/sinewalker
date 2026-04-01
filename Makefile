SHELL := /bin/bash

ROOT_DIR := $(abspath .)
RUN_DIR := $(ROOT_DIR)/.run
LOG_DIR := $(RUN_DIR)/logs

FRONTEND_DIR := $(ROOT_DIR)/frontend
BACKEND_DIR := $(ROOT_DIR)/backend
ROS2_DIR := $(ROOT_DIR)/robotics/ros2_bridge

FRONTEND_PID := $(RUN_DIR)/frontend.pid
BACKEND_PID := $(RUN_DIR)/backend.pid
BRIDGE_PID := $(RUN_DIR)/bridge.pid
SAFETY_PID := $(RUN_DIR)/safety.pid
DRIVER_PID := $(RUN_DIR)/driver.pid

FRONTEND_LOG := $(LOG_DIR)/frontend.log
BACKEND_LOG := $(LOG_DIR)/backend.log
BRIDGE_LOG := $(LOG_DIR)/bridge.log
SAFETY_LOG := $(LOG_DIR)/safety.log
DRIVER_LOG := $(LOG_DIR)/driver.log

.PHONY: help init up down status logs frontend backend \
	robotics robotics-down bridge safety driver \
	bridge-down safety-down driver-down \
	frontend-down backend-down

help:
	@echo "Usage:"
	@echo "  make up            - start frontend + backend in background"
	@echo "  make down          - stop frontend + backend"
	@echo "  make status        - show running services"
	@echo "  make logs          - tail all logs"
	@echo "  make robotics      - start ROS2 bridge+safety+driver"
	@echo "  make robotics-down - stop ROS2 bridge+safety+driver"
	@echo "  make frontend      - start frontend only"
	@echo "  make backend       - start backend only"

init:
	@mkdir -p "$(RUN_DIR)" "$(LOG_DIR)"

up: init backend frontend
	@echo "Application stack started."

down: frontend-down backend-down
	@echo "Application stack stopped."

status:
	@for svc in frontend backend bridge safety driver; do \
		pid_file="$(RUN_DIR)/$$svc.pid"; \
		if [[ -f "$$pid_file" ]]; then \
			pid="$$(cat "$$pid_file")"; \
			if kill -0 "$$pid" >/dev/null 2>&1; then \
				echo "$$svc: running (pid=$$pid)"; \
			else \
				echo "$$svc: stale pid file (pid=$$pid)"; \
			fi; \
		else \
			echo "$$svc: stopped"; \
		fi; \
	done

logs:
	@touch "$(FRONTEND_LOG)" "$(BACKEND_LOG)" "$(BRIDGE_LOG)" "$(SAFETY_LOG)" "$(DRIVER_LOG)"
	@tail -f "$(FRONTEND_LOG)" "$(BACKEND_LOG)" "$(BRIDGE_LOG)" "$(SAFETY_LOG)" "$(DRIVER_LOG)"

frontend: init
	@if [[ -f "$(FRONTEND_PID)" ]] && kill -0 "$$(cat "$(FRONTEND_PID)")" >/dev/null 2>&1; then \
		echo "frontend already running (pid=$$(cat "$(FRONTEND_PID)"))"; \
	else \
		echo "Starting frontend..."; \
		nohup bash -lc 'cd "$(FRONTEND_DIR)" && npm run dev' >"$(FRONTEND_LOG)" 2>&1 & \
		echo $$! >"$(FRONTEND_PID)"; \
		echo "frontend started (pid=$$(cat "$(FRONTEND_PID)"))"; \
	fi

backend: init
	@if [[ -f "$(BACKEND_PID)" ]] && kill -0 "$$(cat "$(BACKEND_PID)")" >/dev/null 2>&1; then \
		echo "backend already running (pid=$$(cat "$(BACKEND_PID)"))"; \
	else \
		echo "Starting backend..."; \
		nohup bash -lc 'cd "$(BACKEND_DIR)" && mvn spring-boot:run' >"$(BACKEND_LOG)" 2>&1 & \
		echo $$! >"$(BACKEND_PID)"; \
		echo "backend started (pid=$$(cat "$(BACKEND_PID)"))"; \
	fi

frontend-down:
	@if [[ -f "$(FRONTEND_PID)" ]]; then \
		pid="$$(cat "$(FRONTEND_PID)")"; \
		if kill -0 "$$pid" >/dev/null 2>&1; then kill "$$pid" || true; fi; \
		rm -f "$(FRONTEND_PID)"; \
		echo "frontend stopped"; \
	fi

backend-down:
	@if [[ -f "$(BACKEND_PID)" ]]; then \
		pid="$$(cat "$(BACKEND_PID)")"; \
		if kill -0 "$$pid" >/dev/null 2>&1; then kill "$$pid" || true; fi; \
		rm -f "$(BACKEND_PID)"; \
		echo "backend stopped"; \
	fi

robotics: init bridge safety driver
	@echo "ROS2 bridge stack started."

robotics-down: bridge-down safety-down driver-down
	@echo "ROS2 bridge stack stopped."

bridge: init
	@if [[ -f "$(BRIDGE_PID)" ]] && kill -0 "$$(cat "$(BRIDGE_PID)")" >/dev/null 2>&1; then \
		echo "bridge already running (pid=$$(cat "$(BRIDGE_PID)"))"; \
	else \
		echo "Starting ROS2 bridge..."; \
		nohup bash -lc 'cd "$(ROOT_DIR)" && python3 "$(ROS2_DIR)/bridge_node.py" --ws-host 0.0.0.0 --ws-port 8765' >"$(BRIDGE_LOG)" 2>&1 & \
		echo $$! >"$(BRIDGE_PID)"; \
		echo "bridge started (pid=$$(cat "$(BRIDGE_PID)"))"; \
	fi

safety: init
	@if [[ -f "$(SAFETY_PID)" ]] && kill -0 "$$(cat "$(SAFETY_PID)")" >/dev/null 2>&1; then \
		echo "safety already running (pid=$$(cat "$(SAFETY_PID)"))"; \
	else \
		echo "Starting safety controller..."; \
		nohup bash -lc 'cd "$(ROOT_DIR)" && python3 "$(ROS2_DIR)/safety_controller_node.py" --mapping "$(ROS2_DIR)/joint_mapping.example.yaml"' >"$(SAFETY_LOG)" 2>&1 & \
		echo $$! >"$(SAFETY_PID)"; \
		echo "safety started (pid=$$(cat "$(SAFETY_PID)"))"; \
	fi

driver: init
	@if [[ -f "$(DRIVER_PID)" ]] && kill -0 "$$(cat "$(DRIVER_PID)")" >/dev/null 2>&1; then \
		echo "driver already running (pid=$$(cat "$(DRIVER_PID)"))"; \
	else \
		echo "Starting driver adapter..."; \
		nohup bash -lc 'cd "$(ROOT_DIR)" && python3 "$(ROS2_DIR)/driver_adapter_node.py" --config "$(ROS2_DIR)/driver_adapter.example.yaml"' >"$(DRIVER_LOG)" 2>&1 & \
		echo $$! >"$(DRIVER_PID)"; \
		echo "driver started (pid=$$(cat "$(DRIVER_PID)"))"; \
	fi

bridge-down:
	@if [[ -f "$(BRIDGE_PID)" ]]; then \
		pid="$$(cat "$(BRIDGE_PID)")"; \
		if kill -0 "$$pid" >/dev/null 2>&1; then kill "$$pid" || true; fi; \
		rm -f "$(BRIDGE_PID)"; \
		echo "bridge stopped"; \
	fi

safety-down:
	@if [[ -f "$(SAFETY_PID)" ]]; then \
		pid="$$(cat "$(SAFETY_PID)")"; \
		if kill -0 "$$pid" >/dev/null 2>&1; then kill "$$pid" || true; fi; \
		rm -f "$(SAFETY_PID)"; \
		echo "safety stopped"; \
	fi

driver-down:
	@if [[ -f "$(DRIVER_PID)" ]]; then \
		pid="$$(cat "$(DRIVER_PID)")"; \
		if kill -0 "$$pid" >/dev/null 2>&1; then kill "$$pid" || true; fi; \
		rm -f "$(DRIVER_PID)"; \
		echo "driver stopped"; \
	fi
