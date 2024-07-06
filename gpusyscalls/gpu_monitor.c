#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <sys/ioctl.h>
#include <asm/ioctl.h>

#define SYSCALL_OPEN 2
#define SYSCALL_IOCTL 16
#define SYSCALL_MMAP 9
#define SYSCALL_WRITE 1

const char* NVIDIA_DEVICE = "/dev/nvidia";
const char* AMD_DEVICE = "/dev/dri/renderD";

#define MAX_KERNEL_LAUNCHES 1000
#define KERNEL_LAUNCH_THRESHOLD 5 // seconds

typedef struct {
    time_t timestamp;
    char description[256];
} KernelLaunch;

KernelLaunch kernel_launches[MAX_KERNEL_LAUNCHES];
int kernel_launch_count = 0;

pid_t child_pid = 0;

// NVIDIA IOCTL command decoding
#define NVRM_IOCTL_MAGIC      'F'
#define NVRM_IOCTL_ESC_BASE   200

const char* nvidia_ioctl_names[] = {
    "NVRM_IOCTL_CHECK_VERSION_STR",
    "NVRM_IOCTL_XFER_CMD",
    "NVRM_IOCTL_SYSCTL",
    "NVRM_IOCTL_STATUS",
    "NVRM_IOCTL_ATTACH",
    "NVRM_IOCTL_DETACH",
    "NVRM_IOCTL_CARD_INFO",
    "NVRM_IOCTL_ATTACH_ZOMBIES",
    "NVRM_IOCTL_SET_NUMA_STATUS",
    "NVRM_IOCTL_EXPORT_OBJECT",
    "NVRM_IOCTL_ALLOC_OS_EVENT",
    "NVRM_IOCTL_FREE_OS_EVENT",
    "NVRM_IOCTL_STATUS_CODE",
    // Add more as needed
};

const char* decode_nvidia_ioctl(unsigned long cmd) {
    static char buffer[256];
    
    if (_IOC_TYPE(cmd) != NVRM_IOCTL_MAGIC) {
        snprintf(buffer, sizeof(buffer), "Unknown IOCTL (not NVIDIA)");
        return buffer;
    }

    int nr = _IOC_NR(cmd) - NVRM_IOCTL_ESC_BASE;
    if (nr >= 0 && nr < sizeof(nvidia_ioctl_names)/sizeof(char*)) {
        snprintf(buffer, sizeof(buffer), "%s (size: %d, dir: %s)", 
                 nvidia_ioctl_names[nr], 
                 _IOC_SIZE(cmd),
                 (_IOC_DIR(cmd) == _IOC_NONE) ? "none" :
                 (_IOC_DIR(cmd) == _IOC_READ) ? "read" :
                 (_IOC_DIR(cmd) == _IOC_WRITE) ? "write" : "read/write");
    } else {
        snprintf(buffer, sizeof(buffer), "Unknown NVIDIA IOCTL (cmd: 0x%lx)", cmd);
    }
    
    return buffer;
}

void handle_signal(int sig) {
    if (child_pid != 0) {
        kill(child_pid, SIGTERM);
    }
    exit(0);
}

int is_gpu_call(struct user_regs_struct *regs, pid_t pid) {
    if (regs->orig_rax == SYSCALL_OPEN) {
        char path[256];
        memset(path, 0, sizeof(path));
        struct user_regs_struct regs_after;
        
        ptrace(PTRACE_SYSCALL, pid, 0, 0);
        waitpid(pid, NULL, 0);
        ptrace(PTRACE_GETREGS, pid, 0, &regs_after);
        
        long data = 0;
        for (int i = 0; i < sizeof(path); i += sizeof(long)) {
            data = ptrace(PTRACE_PEEKDATA, pid, regs_after.rdi + i, NULL);
            memcpy(path + i, &data, sizeof(long));
            if (strchr((char *)&data, 0)) {
                break;
            }
        }
        
        return (strncmp(path, NVIDIA_DEVICE, strlen(NVIDIA_DEVICE)) == 0) ||
               (strncmp(path, AMD_DEVICE, strlen(AMD_DEVICE)) == 0);
    }
    return 0;
}

void log_kernel_launch(const char* description) {
    if (kernel_launch_count < MAX_KERNEL_LAUNCHES) {
        time(&kernel_launches[kernel_launch_count].timestamp);
        strncpy(kernel_launches[kernel_launch_count].description, description, 255);
        kernel_launches[kernel_launch_count].description[255] = '\0';
        kernel_launch_count++;
    }
}

void analyze_kernel_launches() {
    printf("\nAnalyzing kernel launches:\n");
    for (int i = 0; i < kernel_launch_count; i++) {
        time_t duration = (i < kernel_launch_count - 1) ? 
            (kernel_launches[i+1].timestamp - kernel_launches[i].timestamp) : 
            (time(NULL) - kernel_launches[i].timestamp);
        
        printf("Kernel %d: %s\n", i+1, kernel_launches[i].description);
        printf("  Duration: %ld seconds\n", duration);
        
        if (duration > KERNEL_LAUNCH_THRESHOLD) {
            printf("  WARNING: This kernel took longer than %d seconds to complete!\n", KERNEL_LAUNCH_THRESHOLD);
            printf("  This might indicate a long-running or stuck kernel.\n");
        }
    }
}

void log_gpu_call(struct user_regs_struct *regs, pid_t pid) {
    switch(regs->orig_rax) {
        case SYSCALL_IOCTL: {
            int fd = regs->rdi;
            unsigned long request = regs->rsi;
            char path[256] = {0};
            char fd_path[256] = {0};
            snprintf(fd_path, sizeof(fd_path), "/proc/%d/fd/%d", pid, fd);
            readlink(fd_path, path, sizeof(path));
            
            if (strstr(path, "nvidia")) {
                const char* ioctl_description = decode_nvidia_ioctl(request);
                char description[256];
                snprintf(description, sizeof(description), "NVIDIA: %s", ioctl_description);
                log_kernel_launch(description);
                printf("GPU: %s\n", description);
            } else if (strstr(path, "dri")) {
                char description[256];
                snprintf(description, sizeof(description), "AMD: IOCTL call (cmd: 0x%lx)", request);
                log_kernel_launch(description);
                printf("GPU: %s\n", description);
            }
            break;
        }
        case SYSCALL_MMAP:
            printf("GPU: MMAP call (length: %llu, prot: 0x%llx)\n", regs->rsi, regs->rdx);
            break;
        case SYSCALL_WRITE: {
            int fd = regs->rdi;
            char fd_path[256] = {0};
            char path[256] = {0};
            snprintf(fd_path, sizeof(fd_path), "/proc/%d/fd/%d", pid, fd);
            readlink(fd_path, path, sizeof(path));
            
            if (strstr(path, "nvidia") || strstr(path, "dri")) {
                printf("GPU: Write operation to GPU device\n");
            }
            break;
        }
    }
}

void monitor_process(pid_t pid) {
    int status;
    struct user_regs_struct regs;
    time_t last_output_time = time(NULL);

    while(1) {
        ptrace(PTRACE_SYSCALL, pid, 0, 0);
        waitpid(pid, &status, 0);

        if(WIFEXITED(status) || WIFSIGNALED(status)) {
            break;
        }

        ptrace(PTRACE_GETREGS, pid, 0, &regs);
        
        if (is_gpu_call(&regs, pid) || 
            regs.orig_rax == SYSCALL_IOCTL || 
            regs.orig_rax == SYSCALL_MMAP ||
            regs.orig_rax == SYSCALL_WRITE) {
            log_gpu_call(&regs, pid);
            last_output_time = time(NULL);
        }

        // Check for long periods of inactivity
        if (difftime(time(NULL), last_output_time) > 10) {
            printf("WARNING: No GPU activity detected for over 10 seconds. Possible deadlock or long-running kernel.\n");
            last_output_time = time(NULL);
        }
    }

    analyze_kernel_launches();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <command>\n", argv[0]);
        exit(1);
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    child_pid = fork();

    if (child_pid == 0) {
        // Child process
        ptrace(PTRACE_TRACEME, 0, 0, 0);
        execvp(argv[1], &argv[1]);
    } else if (child_pid > 0) {
        // Parent process
        monitor_process(child_pid);
    } else {
        perror("fork");
        exit(1);
    }

    return 0;
}
