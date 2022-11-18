// console.log("Login modal");

let links = document.querySelectorAll(".link");
let loginModal = document.querySelector(".login-modal");
let loginClose = document.querySelector(".login-close");


function showLoginModal(ev){
    loginModal.style.display = 'block';
    loginClose.addEventListener('click', closeLoginModal);
    
}
function closeLoginModal(){
    loginModal.style.display = 'none';
}



links.forEach((each_link)=>{
    if(each_link.innerText === 'Log In'){
        // console.log(each_link);
        each_link.addEventListener('click', showLoginModal);
    }
})

window.onclick = function(event) {
  if (event.target == loginModal) {
    loginModal.style.display = "none";
  }
}


let all_close_btn = document.querySelectorAll('.close-btn');
all_close_btn.forEach((item)=>{
  // console.log(item)
  item.addEventListener("click", (ev)=>{
    ev.target.parentElement.parentElement.style.display = 'none';
  })
})

// if(window.location.pathname == '/search_data'){

//   let searchBtn = document.querySelector(".submit-btn-search");
//   let loader = document.querySelector(".loader");

//   searchBtn.addEventListener("click", (ev)=>{
//     loader.style.display = 'block';
//   })

// }





if(window.location.pathname == '/running'){

  let intervalID = setInterval(()=>{

    fetchStatus().then((stat)=>{
      if (stat['status'] == 'completed'){
        clearInterval(intervalID);
        // document.querySelector(".view-results").style.display = 'inline';
        // let text = document.querySelector(".go-to-res p");
        // text.innerText = 'Results are up please check !'
        window.location = "/results"
      }
      else if (stat['status'] == 'error'){
        clearInterval(intervalID);
        window.location = "/search_data"
      }

    })

  }, 15000)
}

async function fetchStatus(){
  let val = await fetch("/checking");
  let res = await val.json();
  return res;
}