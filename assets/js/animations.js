// Smooth fade-in animations on page load and scroll
document.addEventListener('DOMContentLoaded', function() {
    // Add fade-in classes to elements that should animate on scroll
    const elementsToAnimate = document.querySelectorAll('h2, h3, h4, p, ul, ol, .abstract-section, blockquote, pre');
    
    elementsToAnimate.forEach((element, index) => {
        // Skip elements that are already animated on load (hero section elements)
        if (!element.closest('.hero-section')) {
            element.classList.add('fade-in-on-scroll');
            
            // Add staggered delay for list items
            if (element.tagName === 'LI') {
                const staggerIndex = index % 4;
                element.classList.add(`stagger-${staggerIndex + 1}`);
            }
        }
    });
    
    // Intersection Observer for scroll-triggered animations
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    // Observe all elements with fade-in-on-scroll class
    document.querySelectorAll('.fade-in-on-scroll').forEach(el => {
        observer.observe(el);
    });
    
    // Smooth scroll for table of contents links with offset
    document.querySelectorAll('.table-of-contents a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const headerHeight = document.querySelector('.site-header').offsetHeight;
                const targetPosition = target.offsetTop - headerHeight - 20;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Add loading animation completion
    setTimeout(() => {
        document.body.classList.add('loaded');
    }, 100);
});

// Collapsible sections functionality
function initCollapsibleSections() {
    // Find all colored span elements that should be toggleable
    const toggleSpans = document.querySelectorAll('span[style*="color:#A0AF85"], span[style*="color:#A95FD0"], span[style*="color:#7F9161"]');
    
    toggleSpans.forEach(span => {
        // Add toggle class and start collapsed
        span.classList.add('section-toggle', 'collapsed');
        
        // Find the next elements until the next toggle or major heading
        let nextElement = span.parentElement.nextElementSibling;
        const contentElements = [];
        
        while (nextElement) {
            // Stop if we hit another toggle span or an h2
            const hasToggleSpan = nextElement.querySelector('span[style*="color:#A0AF85"], span[style*="color:#A95FD0"], span[style*="color:#7F9161"]');
            if (hasToggleSpan || nextElement.tagName === 'H2') {
                break;
            }
            contentElements.push(nextElement);
            nextElement = nextElement.nextElementSibling;
        }
        
        // Wrap content in a collapsible container
        if (contentElements.length > 0) {
            const wrapper = document.createElement('div');
            wrapper.classList.add('collapsible-content', 'collapsed'); // Start collapsed
            
            const parent = span.parentElement.parentNode;
            const insertBefore = contentElements[0];
            parent.insertBefore(wrapper, insertBefore);
            
            contentElements.forEach(el => {
                wrapper.appendChild(el);
            });
            
            // Add click handler
            span.addEventListener('click', function(e) {
                e.preventDefault();
                this.classList.toggle('collapsed');
                wrapper.classList.toggle('collapsed');
            });
        }
    });
}

// Call the function after DOM is ready
initCollapsibleSections();