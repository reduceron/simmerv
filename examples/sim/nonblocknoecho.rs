use std::io::Read;
use std::io::Stdin;
use std::io::{self};

pub struct NonblockNoEcho {
    stdin: i32,
    orig_termios: termios::Termios,
    reader: Stdin,
}

impl NonblockNoEcho {
    #[allow(clippy::expect_used, clippy::unwrap_used)]
    pub fn new(ctrlc_breaks: bool) -> Self {
        use std::os::unix::io::AsRawFd;
        use termios::ECHO;
        use termios::ICANON;
        use termios::ISIG;
        use termios::TCSANOW;
        use termios::Termios;
        use termios::tcsetattr;

        let stdin: i32 = std::io::stdin().as_raw_fd();
        assert_eq!(stdin, 0);

        unsafe {
            let flags = libc::fcntl(stdin, libc::F_GETFL);
            libc::fcntl(stdin, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }

        let orig_termios = Termios::from_fd(stdin).expect("Termio::from_fd(stdin)");

        let mut termios = orig_termios;
        termios.c_lflag &= !(ECHO | ICANON); // no echo and canonical mode
        if !ctrlc_breaks {
            termios.c_lflag &= !ISIG; // Don't break on Ctrl-C
        }

        termios.c_iflag &= !(termios::IGNBRK
            | termios::BRKINT
            | termios::PARMRK
            | termios::ISTRIP
            | termios::INLCR
            | termios::IGNCR
            | termios::ISIG
            | termios::ICRNL
            | termios::IXON);
        termios.c_oflag |= termios::OPOST;
        termios.c_cflag &= !(termios::CSIZE | termios::PARENB);
        termios.c_cflag |= termios::CS8;
        termios.c_cc[termios::VMIN] = 1;
        termios.c_cc[termios::VTIME] = 0;

        tcsetattr(stdin, TCSANOW, &termios).unwrap();

        Self {
            stdin,
            orig_termios,
            reader: io::stdin(),
        }
    }

    pub fn get_key(&mut self) -> Option<u8> {
        let mut buffer = [0; 1]; // read exactly one byte
        self.reader.read(&mut buffer).map_or(None, |n| {
            assert!(n == 1);
            Some(buffer[0])
        })
    }
}

impl Drop for NonblockNoEcho {
    #[allow(clippy::expect_used, clippy::unwrap_used)]
    fn drop(&mut self) {
        // reset the stdin to original termios data
        termios::tcsetattr(self.stdin, termios::TCSANOW, &self.orig_termios).unwrap();
        unsafe {
            let flags = libc::fcntl(self.stdin, libc::F_GETFL);
            libc::fcntl(self.stdin, libc::F_SETFL, flags & !libc::O_NONBLOCK);
        }
    }
}
