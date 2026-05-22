import { createRouter, createWebHistory } from "vue-router";

import UploadPage from "../pages/UploadPage.vue";
import ResultPage from "../pages/ResultPage.vue";

const routes = [
  {
    path: "/",
    redirect: "/upload"
  },
  {
    path: "/upload",
    name: "upload",
    component: UploadPage
  },
  {
    path: "/result",
    name: "result",
    component: ResultPage
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

router.beforeEach((to) => {
  if (to.name === "result") {
    const rawResult = sessionStorage.getItem("nsfw-demo-last-result");
    if (!rawResult) {
      return { name: "upload" };
    }
  }
  return true;
});

export default router;
