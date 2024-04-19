from utils import predict_exercise, draw_text, get_landmark_array, calculate_angle, get_landmark, calculate_vertical_angle
import cv2
import pandas as pd

class ProcessFrame:
  def __init__(self, data, form_data) -> None:
    # Colors in BGR format.
    self.COLORS = {
                    'blue'       : (0, 127, 255),
                    'red'        : (255, 50, 50),
                    'green'      : (0, 255, 127),
                    'light_green': (100, 233, 127),
                    'yellow'     : (255, 255, 0),
                    'magenta'    : (255, 0, 255),
                    'white'      : (255,255,255),
                    'cyan'       : (0, 255, 255),
                    'light_blue' : (102, 204, 255)
                  }
    # Font type.
    self.font = cv2.FONT_HERSHEY_SIMPLEX

    # line type
    self.linetype = cv2.LINE_AA

    # set radius to draw arc
    self.radius = 20
    
    self.current_exercise = None
    self.stage_seq = []
    self.stage = ""
    self.correct_exercise_reps = {}
    self.incorrect_exercise_reps = {}
    self.exercise_pred_list = []
    self.data = data
    self.form_data = form_data
    
    self.side_tracked = ''
    
    self.incorrect_action = False
    
    self.feedback_strings = {}
  
  def _get_data_item(self, key):
    return self.current_exercise[key].item()
  
  def _get_exercise_form_data(self, key):
    return self.form_data[self.form_data.feddback_id == key]
  
  def _get_form_data_item(self, row, key):
    return row[key].item()
  
  def process(self, frame, pose):
    frame_height, frame_width, _ = frame.shape

    # ================================
    # Predict exercise being perfomed
    # ================================

    exercise_pred =  predict_exercise(frame)

    self.exercise_pred_list.append(int(exercise_pred))

    if(len(self.exercise_pred_list) > 25):
      self.exercise_pred_list.pop(0)

    exercise_index = max(set(self.exercise_pred_list), key=self.exercise_pred_list.count)
    print(exercise_index)

    self.current_exercise = self.data[self.data['index']==exercise_index]

    # ================================
    # if exercise is not supported
    # ================================
    is_supported = exercise_index in self.data['index'].unique()

    if not is_supported:
      draw_text(
                    frame, 
                    "Sorry, the exercise is not supported yet! ", 
                    pos=(int(frame_width*0.05), 130),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                )  
    else:
      
      # ================================
      # Get current exercise info
      # ================================

      exercise_label = str(self._get_data_item('label'))
      exercise_required_joints = str(self._get_data_item('required_cv_joints')).split(',')
      exercise_offset_range = self._get_data_item('offset_range')
      exercise_rep_joint_a = self._get_data_item('angle_joint_1_a')
      exercise_rep_joint_b = self._get_data_item('angle_joint_1_b')
      exercise_rep_joint_c = self._get_data_item('angle_joint_1_c')
      
      exercise_form_checks = self._get_data_item('feedback_points')
      
      stage_1_condition = str(self._get_data_item('joint_1_stage_1'))
      stage_2_condition = str(self._get_data_item('joint_1_stage_2'))
      stage_3_condition = str(self._get_data_item('joint_1_stage_3'))


      draw_text(
                      frame, 
                      "Exercise: " + exercise_label , 
                      pos=(int(frame_width*0.05), 30),
                      text_color=(255, 255, 230),
                      font_scale=0.7,
                      text_color_bg=(18, 185, 0)
                  )  

      # ================================
      # Initialize rep counters
      # ================================

      if exercise_label not in self.correct_exercise_reps:
        self.correct_exercise_reps[exercise_label] = 0
      if exercise_label not in self.incorrect_exercise_reps:
        self.incorrect_exercise_reps[exercise_label] = 0
      if exercise_label not in self.feedback_strings:
        self.feedback_strings[exercise_label] = []
        
      draw_text(
        frame, 
        "CORRECT: " + str(self.correct_exercise_reps[exercise_label]), 
        pos=(int(frame_width*0.05), 80),
        text_color=(255, 255, 230),
        font_scale=0.7,
        text_color_bg=(18, 185, 0)
      )  

      draw_text(
        frame, 
        "INCORRECT: " + str(self.incorrect_exercise_reps[exercise_label]), 
        pos=(int(frame_width*0.05), 130),
        text_color=(255, 255, 230),
        font_scale=0.7,
        text_color_bg=(221, 0, 0),
      )  
      # ================================
      # Get mediapipe pose estimations
      # ================================
      # convert image coloring from BGR to RGB
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame.flags.writeable = False
  
      # Make pose estimations
      pose_results = pose.process(frame)
  
      # convert image coloring from RGB to BGR
      frame.flags.writeable = True
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  

      try:
        # ================================
        # Calculate camera horizontal offset
        # =================================

        landmarks = pose_results.pose_landmarks.landmark

        left_shoulder = get_landmark_array(landmarks, 'LEFT_SHOULDER', frame_width, frame_height)

        right_shoulder = get_landmark_array(landmarks, 'RIGHT_SHOULDER', frame_width, frame_height)
        
        nose = get_landmark_array(landmarks, 'NOSE', frame_width, frame_height)

        offset_angle = calculate_angle(left_shoulder, nose, right_shoulder)

        # ================================
        # Check if offset matches requirements
        # =================================

        if not pd.isna(exercise_offset_range) and not eval(str(offset_angle)+str(exercise_offset_range)):
          
          # ================================
          # if offset is outside recommended range
          # =================================
          draw_text(
                      frame, 
                      'CAMERA NOT ALIGNED PROPERLY!!!', 
                      pos=(30, frame_height-60),
                      text_color=(255, 255, 230),
                      font_scale=0.65,
                      text_color_bg=(255, 153, 0),
                  ) 
                  
                  
          draw_text(
              frame, 
              'OFFSET ANGLE: '+str(offset_angle), 
              pos=(30, frame_height-30),
              text_color=(255, 255, 230),
              font_scale=0.65,
              text_color_bg=(255, 153, 0),
          ) 
        
        else:
          
          # ================================
          # Get required joint landmarks
          # =================================
          
          left_joint = get_landmark(landmarks,'LEFT_' + str(exercise_required_joints[0]))
          right_joint = get_landmark(landmarks, 'RIGHT_' + str(exercise_required_joints[0]))

          if left_joint.visibility > right_joint.visibility:
            self.side_tracked = 'LEFT_'
            rep_joint_a = get_landmark_array(landmarks, 'LEFT_' + str(exercise_rep_joint_a), frame_width, frame_height)
            rep_joint_b = get_landmark_array(landmarks, 'LEFT_' + str(exercise_rep_joint_b), frame_width, frame_height)
            rep_joint_c = get_landmark_array(landmarks, 'LEFT_' + str(exercise_rep_joint_c), frame_width, frame_height)
            
            text_coordinate_x = rep_joint_b[0]-10
          else:
            self.side_tracked = 'RIGHT_'
            rep_joint_a = get_landmark_array(landmarks, 'RIGHT_' + str(exercise_rep_joint_a), frame_width, frame_height)
            rep_joint_b = get_landmark_array(landmarks, 'RIGHT_' + str(exercise_rep_joint_b), frame_width, frame_height)
            rep_joint_c = get_landmark_array(landmarks, 'RIGHT_' + str(exercise_rep_joint_c), frame_width, frame_height)
            
            text_coordinate_x = rep_joint_b[0]+10
            
          rep_angle = calculate_angle(rep_joint_a, rep_joint_b, rep_joint_c)
          
          cv2.putText(frame, str(int(rep_angle)), (text_coordinate_x, rep_joint_b[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
          
          # ================================
          # Get current exercise rep state
          # =================================
          
          if eval(stage_1_condition.replace('x', str(rep_angle))):
            self.stage = "s1"
          elif eval(stage_2_condition.replace('x', str(rep_angle))):
            self.stage = "s2"
          elif eval(stage_3_condition.replace('x', str(rep_angle))):
            self.stage = "s3"
            
                  
          self.incorrect_action = False
          
          # ================================
          # Compute feedback
          # =================================
          print("all checks " + exercise_form_checks)
          
          form_check_list = exercise_form_checks.split(',')
          
          for form_check in form_check_list:
            print(form_check)
            
            check = self._get_exercise_form_data(form_check)
            
            metric = self._get_form_data_item(check, 'metric')
            joints = self._get_form_data_item(check, 'joints')
            range = self._get_form_data_item(check, 'range')
            feedback_text = self._get_form_data_item(check, 'feedback_text')
            
            joints_list = list(joints.split(','))
            joints_coord = [ get_landmark_array(landmarks, self.side_tracked + str(joint), frame_width, frame_height) for joint in joints_list ]
            
            if metric == 'joint_angle_threshold':
              angle = calculate_angle(joints_coord[0], joints_coord[1], joints_coord[2])
            elif metric == 'joint_vertical_angle':
              angle = calculate_vertical_angle(joints_coord[0], joints_coord[1])
                
            if not eval(range.replace('x', str(angle))):
              self.incorrect_action = True
              print("feedback: " + feedback_text)
              if feedback_text not in self.feedback_strings[exercise_label]:
                self.feedback_strings[exercise_label].append(feedback_text)
          
          for feedback in self.feedback_strings[exercise_label]:
            idx = self.feedback_strings[exercise_label].index(feedback)
            print(str(idx), feedback)
            draw_text(
              frame, 
              str(feedback), 
              pos=(int(frame_width*0.05), (170 + (50*(idx))) ),
              text_color=(255, 255, 255),
              font_scale=0.7,
              text_color_bg=(221, 0, 0)
            )
            
          # ================================
          # Update counters
          # =================================
          
          if self.stage == 's1':
            if len(self.stage_seq) == 3 and not self.incorrect_action:
                self.correct_exercise_reps[exercise_label]+=1
                # print(str(self.correct_exercise_reps[exercise_label]))
                
            elif len(self.stage_seq) == 3 and self.incorrect_action:
              self.incorrect_exercise_reps[exercise_label]+=1
                
            elif 's2' in self.stage_seq and len(self.stage_seq)==1:
                self.incorrect_exercise_reps[exercise_label]+=1
                feedback_text = "Try using full range of motion"
                print(feedback_text)
                if feedback_text not in self.feedback_strings[exercise_label]:
                  self.feedback_strings[exercise_label].append(feedback_text)
                # str(self.incorrect_exercise_reps[exercise_label])


            self.stage_seq = []
            self.incorrect_action = False
            
        
          elif self.stage == 's2':
              if (('s3' not in self.stage_seq) and (self.stage_seq.count('s2'))==0) or \
                  (('s3' in self.stage_seq) and (self.stage_seq.count('s2')==1)):
                  self.stage_seq.append(self.stage)
                  
          elif self.stage == 's3':
              if (self.stage not in self.stage_seq) and 's2' in self.stage_seq: 
                  self.stage_seq.append(self.stage)
            
            
      except:
        pass
    return frame, _
        
        
        
        
          