using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

public class CameraMovement : MonoBehaviour
{
    public GameObject targetObj;
    public int resWidth = 2550; 
    public int resHeight = 3300;  
    private IEnumerator cameraCoroutine; 
    public String resultsDir = "";
    public float speed = 1.0f;

    public static int innerZ = 16;
    public static int outerZ = 20;
    public static Vector3 A = new Vector3(0,2,innerZ); //center
    public static Vector3 B = new Vector3(-3,2,innerZ); //center right
    public static Vector3 C = new Vector3(-3,4,innerZ); //upper right
    public static Vector3 D = new Vector3(3,4,innerZ); //upper left
    public static Vector3 E = new Vector3(3,0,innerZ); //lower left
    public static Vector3 F = new Vector3(-3,0,innerZ); //lower right
    public static Vector3 G = new Vector3(0,2,outerZ); //center back
    public static Vector3 H = new Vector3(-3,2,outerZ); //center right back
    public static Vector3 I = new Vector3(-3,4,outerZ); //upper right back
    public static Vector3 J = new Vector3(3,4,outerZ); //upper left back
    public static Vector3 K = new Vector3(3,0,outerZ); //lower left back
    public static Vector3 L = new Vector3(-3,0,outerZ); //lower right back  
    public static Vector3 Target = new Vector3(3,4,outerZ);

    Camera mainCam;
    public static Vector3[] movementSequence = {A, B, C, D, E, F, A, G, H, I, J, K, L};
    static int movementIndex = 0;
    public Vector3 target = movementSequence[movementIndex];

    //public Animator anim;
    //int jumpHash = Animator.StringToHash("Jump");

    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Working!!!@#$");
        mainCam = Camera.main;
        targetObj = GameObject.Find("BeaglePrefab");
        //anim = targetObj.GetComponent<Animator>();


        String todaysDate = DateTime.Now.ToString("dd-MMM-yyyy");
        resultsDir = "ImageData_" + todaysDate;

        if(Directory.Exists(resultsDir))
        {
            Debug.Log("Directory " + resultsDir + " already exists, so not making it!");
        }
        else
        {
            Directory.CreateDirectory(resultsDir); 
        }

        //Make a directory for every object type that we'll take pictures of
        String[] objects = {"Dog"};
        foreach (String objName in objects)
        {
            String newPath = resultsDir + "/" + objName;
            if(!Directory.Exists(newPath))
            {
                Directory.CreateDirectory(newPath); 
                Debug.Log("Making new folder: " + newPath);
            }
        }

        cameraCoroutine = TakeAndSavePicture(0.4f);
        StartCoroutine(cameraCoroutine);            
    }

    // Update is called once per frame
    void Update()
    {
        movementIndex = (movementIndex + 1) % (movementSequence.Length);
        target = movementSequence[movementIndex];
        Debug.Log("Moving towards: " + target);
        
        transform.position = Vector3.MoveTowards(transform.position, target, Time.deltaTime * speed);
        transform.LookAt (targetObj.transform);
    }

    private IEnumerator TakeAndSavePicture(float waitTime)
    {
        while (true)
        {
            //anim.Play("Jump");
            //anim.set
            //wait the wait time
            yield return new WaitForSeconds(waitTime);

            String fileName = GetImageName("Dog"); //get object type

            RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
            mainCam.targetTexture = rt;
            Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
            mainCam.Render();
            RenderTexture.active = rt;
            screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
            mainCam.targetTexture = null;
            RenderTexture.active = null; // JC: added to avoid errors
            Destroy(rt);
            byte[] bytes = screenShot.EncodeToPNG();
            System.IO.File.WriteAllBytes(fileName, bytes);
        }
    }

    String GetImageName(String objectType)
    {
        String fileName = string.Format("{0}/{1}/screen_{2}x{3}_{4}.png", 
                                    resultsDir,
                                    objectType,
                                    resWidth, resHeight, 
                                    System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));   
        Debug.Log("Making new file: " + fileName);
        return fileName;      
    }      
}
