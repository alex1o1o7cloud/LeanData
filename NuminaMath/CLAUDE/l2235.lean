import Mathlib

namespace inscribed_circles_radius_l2235_223550

/-- Two circles inscribed in a 60-degree angle -/
structure InscribedCircles :=
  (r1 : ℝ) -- radius of smaller circle
  (r2 : ℝ) -- radius of larger circle
  (angle : ℝ) -- angle in which circles are inscribed
  (touch : Prop) -- circles touch each other

/-- Theorem: Given two circles inscribed in a 60-degree angle, touching each other, 
    with the smaller circle having a radius of 24, the radius of the larger circle is 72. -/
theorem inscribed_circles_radius 
  (circles : InscribedCircles) 
  (h1 : circles.r1 = 24) 
  (h2 : circles.r2 > circles.r1) 
  (h3 : circles.angle = 60) 
  (h4 : circles.touch) : 
  circles.r2 = 72 := by
  sorry

end inscribed_circles_radius_l2235_223550


namespace only_345_right_triangle_l2235_223509

/-- A function that checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that only (3, 4, 5) forms a right-angled triangle among the given sets --/
theorem only_345_right_triangle :
  ¬ is_right_triangle 2 4 4 ∧
  ¬ is_right_triangle (Real.sqrt 3) 2 2 ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 5 12 14 :=
by sorry

end only_345_right_triangle_l2235_223509


namespace geometric_sequence_S24_l2235_223558

/-- A geometric sequence with partial sums S_n -/
def geometric_sequence (S : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, S (n + 1) - S n = r * (S n - S (n - 1))

/-- Theorem: For a geometric sequence with given partial sums, S_24 can be determined -/
theorem geometric_sequence_S24 (S : ℕ → ℚ) 
  (h_geom : geometric_sequence S) 
  (h_S6 : S 6 = 48)
  (h_S12 : S 12 = 60) : 
  S 24 = 255 / 4 := by
  sorry

end geometric_sequence_S24_l2235_223558


namespace security_breach_likely_and_measures_needed_l2235_223566

/-- Represents the security level of an online transaction -/
inductive SecurityLevel
| Low
| Medium
| High

/-- Represents the actions taken by the user -/
structure UserActions where
  clickedSuspiciousEmail : Bool
  enteredSensitiveInfo : Bool
  usedUnofficialWebsite : Bool
  enteredSMSPassword : Bool

/-- Represents additional security measures -/
structure SecurityMeasures where
  useSecureNetworks : Bool
  useAntivirus : Bool
  updateApplications : Bool
  checkAddressBar : Bool
  useStrongPasswords : Bool
  use2FA : Bool

/-- Determines the security level based on user actions -/
def determineSecurityLevel (actions : UserActions) : SecurityLevel :=
  if actions.clickedSuspiciousEmail && actions.enteredSensitiveInfo && 
     actions.usedUnofficialWebsite && actions.enteredSMSPassword then
    SecurityLevel.Low
  else if actions.clickedSuspiciousEmail || actions.enteredSensitiveInfo || 
          actions.usedUnofficialWebsite || actions.enteredSMSPassword then
    SecurityLevel.Medium
  else
    SecurityLevel.High

/-- Checks if additional security measures are sufficient -/
def areMeasuresSufficient (measures : SecurityMeasures) : Bool :=
  measures.useSecureNetworks && measures.useAntivirus && measures.updateApplications &&
  measures.checkAddressBar && measures.useStrongPasswords && measures.use2FA

/-- Theorem: Given the user's actions, the security level is low and additional measures are necessary -/
theorem security_breach_likely_and_measures_needed 
  (actions : UserActions)
  (measures : SecurityMeasures)
  (h1 : actions.clickedSuspiciousEmail = true)
  (h2 : actions.enteredSensitiveInfo = true)
  (h3 : actions.usedUnofficialWebsite = true)
  (h4 : actions.enteredSMSPassword = true) :
  determineSecurityLevel actions = SecurityLevel.Low ∧ 
  areMeasuresSufficient measures = true :=
by sorry


end security_breach_likely_and_measures_needed_l2235_223566


namespace driving_meeting_problem_l2235_223569

/-- A problem about two people driving and meeting on the road. -/
theorem driving_meeting_problem (wife_delay : Real) (wife_speed : Real) (meeting_time : Real) :
  wife_delay = 0.5 →
  wife_speed = 50 →
  meeting_time = 2 →
  ∃ man_speed : Real, man_speed = 40 := by
  sorry

end driving_meeting_problem_l2235_223569


namespace traffic_light_change_probability_l2235_223576

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total duration of the traffic light cycle -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the total duration of intervals where a color change can be observed -/
def changeWindowDuration (viewingTime : ℕ) : ℕ :=
  3 * viewingTime

/-- Theorem: The probability of observing a color change in a 4-second interval is 2/15 -/
theorem traffic_light_change_probability 
  (cycle : TrafficLightCycle) 
  (h1 : cycle.green = 40)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 45)
  (viewingTime : ℕ)
  (h4 : viewingTime = 4) :
  (changeWindowDuration viewingTime : ℚ) / (cycleDuration cycle : ℚ) = 2 / 15 := by
  sorry

end traffic_light_change_probability_l2235_223576


namespace largest_possible_b_l2235_223568

theorem largest_possible_b (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) → (c < b) → (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 12 := by sorry

end largest_possible_b_l2235_223568


namespace complex_quadrant_l2235_223519

theorem complex_quadrant (z : ℂ) (h : z * (1 + Complex.I) = 5 + Complex.I) :
  Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end complex_quadrant_l2235_223519


namespace largest_integer_dividing_factorial_l2235_223520

theorem largest_integer_dividing_factorial (n : ℕ) : 
  (∀ k : ℕ, k ≤ 9 → (2007 : ℕ).factorial % (2007 ^ k) = 0) ∧ 
  ((2007 : ℕ).factorial % (2007 ^ 10) ≠ 0) := by
  sorry

end largest_integer_dividing_factorial_l2235_223520


namespace certain_number_value_l2235_223542

/-- Custom operation # -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem stating that if 3 # x = 48, then x = 6 -/
theorem certain_number_value (x : ℝ) : hash 3 x = 48 → x = 6 := by
  sorry

end certain_number_value_l2235_223542


namespace exists_divisible_by_digit_sum_l2235_223516

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among any 18 consecutive positive integers less than or equal to 2005,
    there exists at least one integer that is divisible by the sum of its digits -/
theorem exists_divisible_by_digit_sum (start : ℕ) (h : start + 17 ≤ 2005) :
  ∃ k : ℕ, k ∈ Finset.range 18 ∧ (start + k) % sum_of_digits (start + k) = 0 := by
  sorry

end exists_divisible_by_digit_sum_l2235_223516


namespace complex_expression_simplification_l2235_223567

theorem complex_expression_simplification (a b : ℂ) (ha : a = 3 + 2*I) (hb : b = 2 - 3*I) :
  3*a + 4*b = 17 - 6*I :=
by sorry

end complex_expression_simplification_l2235_223567


namespace first_car_speed_l2235_223538

/-- Proves that the speed of the first car is 50 km/h given the specified conditions. -/
theorem first_car_speed (time1 : ℝ) (speed2 distance_ratio : ℝ) 
  (h1 : time1 = 6)
  (h2 : speed2 = 100)
  (h3 : distance_ratio = 3)
  (h4 : speed2 * 1 = distance_ratio * (speed2 * 1)) :
  ∃ (speed1 : ℝ), speed1 * time1 = distance_ratio * (speed2 * 1) ∧ speed1 = 50 := by
  sorry

end first_car_speed_l2235_223538


namespace sandwich_price_calculation_l2235_223572

/-- The price of a single sandwich -/
def sandwich_price : ℝ := 5

/-- The number of sandwiches ordered -/
def num_sandwiches : ℕ := 18

/-- The delivery fee -/
def delivery_fee : ℝ := 20

/-- The tip percentage -/
def tip_percent : ℝ := 0.1

/-- The total amount received -/
def total_received : ℝ := 121

theorem sandwich_price_calculation :
  sandwich_price * num_sandwiches + delivery_fee +
  (sandwich_price * num_sandwiches + delivery_fee) * tip_percent = total_received :=
by sorry

end sandwich_price_calculation_l2235_223572


namespace necessary_not_sufficient_condition_l2235_223563

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, x^2 - 2*x - 3 < 0 → -2 < x ∧ x < 3) ∧
  ¬(∀ x : ℝ, -2 < x ∧ x < 3 → x^2 - 2*x - 3 < 0) :=
by sorry

end necessary_not_sufficient_condition_l2235_223563


namespace intersection_line_circle_l2235_223527

/-- Given a line x + y = a intersecting a circle x² + y² = 1 at points A and B,
    if |OA + OB| = |OA - OB|, then a = ±1 -/
theorem intersection_line_circle (a : ℝ) (A B : ℝ × ℝ) : 
  (∀ (x y : ℝ), x + y = a → x^2 + y^2 = 1 → (x, y) = A ∨ (x, y) = B) → 
  ‖(A.1, A.2)‖ = 1 →
  ‖(B.1, B.2)‖ = 1 →
  ‖(A.1 + B.1, A.2 + B.2)‖ = ‖(A.1 - B.1, A.2 - B.2)‖ →
  a = 1 ∨ a = -1 := by
  sorry

end intersection_line_circle_l2235_223527


namespace journey_remaining_distance_l2235_223577

/-- Given a total journey distance and the distance already driven, 
    calculate the remaining distance to be driven. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem: For a journey of 1200 miles where 768 miles have been driven,
    the remaining distance is 432 miles. -/
theorem journey_remaining_distance :
  remaining_distance 1200 768 = 432 := by
  sorry

end journey_remaining_distance_l2235_223577


namespace square_area_l2235_223557

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + 4*x + 3

/-- The line function -/
def g (x : ℝ) : ℝ := 8

/-- The square's side length -/
def side_length : ℝ := 6

theorem square_area : 
  (∃ (x₁ x₂ : ℝ), 
    f x₁ = g x₁ ∧ 
    f x₂ = g x₂ ∧ 
    x₂ - x₁ = side_length) →
  side_length^2 = 36 := by
sorry

end square_area_l2235_223557


namespace curve_parameter_value_l2235_223589

/-- Given a curve C with parametric equations x = 1 + 3t and y = at² + 2,
    where t is the parameter and a is a real number,
    prove that if the point (4,3) lies on C, then a = 1. -/
theorem curve_parameter_value (a : ℝ) :
  (∃ t : ℝ, 1 + 3 * t = 4 ∧ a * t^2 + 2 = 3) → a = 1 := by
  sorry

end curve_parameter_value_l2235_223589


namespace marathon_water_bottles_l2235_223504

theorem marathon_water_bottles (runners : ℕ) (bottles_per_runner : ℕ) (available_bottles : ℕ) : 
  runners = 14 → bottles_per_runner = 5 → available_bottles = 68 → 
  (runners * bottles_per_runner - available_bottles) = 2 := by
sorry

end marathon_water_bottles_l2235_223504


namespace no_integer_solution_l2235_223529

theorem no_integer_solution (m n p : ℤ) :
  m + n * Real.sqrt 2 + p * Real.sqrt 3 = 0 → m = 0 ∧ n = 0 ∧ p = 0 := by
  sorry

end no_integer_solution_l2235_223529


namespace f_min_max_on_interval_l2235_223561

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem f_min_max_on_interval :
  let a : ℝ := 0
  let b : ℝ := 2 * Real.pi
  ∃ (x_min x_max : ℝ), a ≤ x_min ∧ x_min ≤ b ∧ a ≤ x_max ∧ x_max ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    f x_min = -3 * Real.pi / 2 ∧
    f x_max = Real.pi / 2 + 2 :=
  sorry

end f_min_max_on_interval_l2235_223561


namespace balls_in_boxes_count_l2235_223584

/-- The number of ways to place 4 different balls into 4 numbered boxes with exactly one empty box -/
def placeBallsInBoxes : ℕ :=
  -- We define this as a natural number, but don't provide the implementation
  sorry

/-- The theorem stating that there are 144 ways to place 4 different balls into 4 numbered boxes
    such that exactly one box is empty -/
theorem balls_in_boxes_count : placeBallsInBoxes = 144 := by
  sorry

end balls_in_boxes_count_l2235_223584


namespace triangle_inequality_l2235_223502

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (a ≥ b ∧ b ≥ c → A ≥ B ∧ B ≥ C) →
  (a * A + b * B + c * C) / (a + b + c) ≥ π / 3 := by
  sorry

end triangle_inequality_l2235_223502


namespace tank_emptying_time_specific_tank_emptying_time_l2235_223565

/-- Proves that a tank with given volume and flow rates empties in the specified time -/
theorem tank_emptying_time (tank_volume_cubic_feet : ℝ) 
                            (inlet_rate : ℝ) 
                            (outlet_rate_1 outlet_rate_2 outlet_rate_3 outlet_rate_4 : ℝ) 
                            (inches_per_foot : ℝ) : ℝ :=
  let tank_volume_cubic_inches := tank_volume_cubic_feet * (inches_per_foot^3)
  let total_outflow_rate := outlet_rate_1 + outlet_rate_2 + outlet_rate_3 + outlet_rate_4
  let net_outflow_rate := total_outflow_rate - inlet_rate
  tank_volume_cubic_inches / net_outflow_rate

/-- The specific instance of the tank emptying problem -/
theorem specific_tank_emptying_time : 
  tank_emptying_time 60 3 12 6 18 9 12 = 2468.57 := by
  sorry

end tank_emptying_time_specific_tank_emptying_time_l2235_223565


namespace tommy_bike_ride_l2235_223591

/-- Tommy's bike riding problem -/
theorem tommy_bike_ride (tommy_width : ℕ) (tommy_north : ℕ) (friend_area : ℕ) 
  (h1 : tommy_width = 1)
  (h2 : tommy_north = 2)
  (h3 : friend_area = 80)
  (h4 : ∃ s : ℕ, 4 * (tommy_width * (tommy_north + s)) = friend_area) :
  ∃ s : ℕ, s = 18 ∧ 4 * (tommy_width * (tommy_north + s)) = friend_area := by
sorry

end tommy_bike_ride_l2235_223591


namespace column_products_sign_l2235_223590

/-- Represents a 3x3 matrix with elements of type α -/
def Matrix3x3 (α : Type*) := Fin 3 → Fin 3 → α

/-- Given a 3x3 matrix where the product of numbers in each row is negative,
    the products of numbers in the columns must be either
    negative in one column and positive in two columns,
    or negative in all three columns. -/
theorem column_products_sign
  (α : Type*) [LinearOrderedField α]
  (A : Matrix3x3 α)
  (row_products_negative : ∀ i : Fin 3, (A i 0) * (A i 1) * (A i 2) < 0) :
  (∃ j : Fin 3, (A 0 j) * (A 1 j) * (A 2 j) < 0 ∧
    ∀ k : Fin 3, k ≠ j → (A 0 k) * (A 1 k) * (A 2 k) > 0) ∨
  (∀ j : Fin 3, (A 0 j) * (A 1 j) * (A 2 j) < 0) :=
by sorry

end column_products_sign_l2235_223590


namespace other_number_proof_l2235_223574

theorem other_number_proof (x : ℤ) (h : x + 2001 = 3016) : x = 1015 := by
  sorry

end other_number_proof_l2235_223574


namespace room_length_is_twelve_l2235_223518

/-- Represents the dimensions and carpet placement of a rectangular room. -/
structure RoomWithCarpet where
  length : ℝ
  width : ℝ
  borderWidth : ℝ

/-- Calculates the area of the border given room dimensions and border width. -/
def borderArea (room : RoomWithCarpet) : ℝ :=
  room.length * room.width - (room.length - 2 * room.borderWidth) * (room.width - 2 * room.borderWidth)

/-- Theorem: If a rectangular room has width 10 feet, a carpet is placed leaving a 2-foot 
    wide border all around, and the area of the border is 72 square feet, then the length 
    of the room is 12 feet. -/
theorem room_length_is_twelve (room : RoomWithCarpet) 
    (h1 : room.width = 10)
    (h2 : room.borderWidth = 2)
    (h3 : borderArea room = 72) : 
  room.length = 12 := by
  sorry


end room_length_is_twelve_l2235_223518


namespace tv_screen_height_l2235_223552

theorem tv_screen_height (area : ℝ) (base1 : ℝ) (base2 : ℝ) (h : area = 21 ∧ base1 = 3 ∧ base2 = 5) :
  ∃ height : ℝ, area = (1/2) * (base1 + base2) * height ∧ height = 5.25 := by
  sorry

end tv_screen_height_l2235_223552


namespace line_through_two_points_l2235_223585

/-- 
Given a line with equation x = 8y + 5 that passes through points (m, n) and (m + 2, n + p),
prove that p = 1/4.
-/
theorem line_through_two_points (m n p : ℝ) : 
  (m = 8 * n + 5) ∧ (m + 2 = 8 * (n + p) + 5) → p = 1/4 := by
  sorry

end line_through_two_points_l2235_223585


namespace johns_running_distance_l2235_223555

/-- The distance John ran each morning -/
def daily_distance (total_distance : ℕ) (days : ℕ) : ℚ :=
  total_distance / days

theorem johns_running_distance :
  daily_distance 10200 6 = 1700 := by
  sorry

end johns_running_distance_l2235_223555


namespace max_sphere_volume_in_prism_max_sphere_volume_in_specific_prism_l2235_223515

/-- The maximum volume of a sphere inscribed in a right triangular prism -/
theorem max_sphere_volume_in_prism (a b h : ℝ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) :
  let r := min (a * b / (a + b)) (h / 2)
  (4 / 3) * Real.pi * r^3 ≤ (4 / 3) * Real.pi * ((3 : ℝ) / 2)^3 :=
by sorry

/-- The specific case for the given prism dimensions -/
theorem max_sphere_volume_in_specific_prism :
  let max_volume := (4 / 3) * Real.pi * ((3 : ℝ) / 2)^3
  max_volume = 9 * Real.pi / 2 :=
by sorry

end max_sphere_volume_in_prism_max_sphere_volume_in_specific_prism_l2235_223515


namespace quadratic_equation_pairs_l2235_223500

theorem quadratic_equation_pairs : 
  let count := Finset.filter (fun p : ℕ × ℕ => 
    let b := p.1
    let c := p.2
    b > 0 ∧ c > 0 ∧ 
    b + c ≤ 10 ∧
    b^2 - 4*c = 0 ∧
    c^2 - 4*b ≤ 0) (Finset.product (Finset.range 11) (Finset.range 11))
  count.card = 2 := by sorry

end quadratic_equation_pairs_l2235_223500


namespace ghost_castle_windows_l2235_223505

theorem ghost_castle_windows (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end ghost_castle_windows_l2235_223505


namespace highest_throw_l2235_223596

def christine_throw_1 : ℕ := 20
def janice_throw_1 : ℕ := christine_throw_1 - 4
def christine_throw_2 : ℕ := christine_throw_1 + 10
def janice_throw_2 : ℕ := janice_throw_1 * 2
def christine_throw_3 : ℕ := christine_throw_2 + 4
def janice_throw_3 : ℕ := christine_throw_1 + 17

theorem highest_throw :
  max christine_throw_1 (max christine_throw_2 (max christine_throw_3 (max janice_throw_1 (max janice_throw_2 janice_throw_3)))) = 37 := by
  sorry

end highest_throw_l2235_223596


namespace circle_center_correct_l2235_223581

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, return its center -/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 1 (-4) 1 (-6) (-12)
  findCircleCenter eq = CircleCenter.mk 2 3 := by sorry

end circle_center_correct_l2235_223581


namespace perpendicular_vectors_condition_l2235_223583

/-- Given two vectors in R², prove that if they satisfy certain conditions, then a specific component of one vector equals -1. -/
theorem perpendicular_vectors_condition (m : ℝ) : 
  let a : Fin 2 → ℝ := ![m, 3]
  let b : Fin 2 → ℝ := ![-2, 2]
  (∀ i, (a - b) i * b i = 0) → m = -1 := by
  sorry

end perpendicular_vectors_condition_l2235_223583


namespace door_opening_probability_l2235_223526

theorem door_opening_probability (total_keys : ℕ) (opening_keys : ℕ) : 
  total_keys = 4 → 
  opening_keys = 2 → 
  (opening_keys : ℚ) * (total_keys - opening_keys) / (total_keys * (total_keys - 1)) = 1/3 := by
  sorry

end door_opening_probability_l2235_223526


namespace factor_tree_value_l2235_223546

/-- Represents a node in the modified factor tree -/
inductive TreeNode
| Prime (value : ℕ)
| Composite (value : ℕ) (left : TreeNode) (middle : TreeNode) (right : TreeNode)

/-- Calculates the value of a node in the modified factor tree -/
def nodeValue : TreeNode → ℕ
| TreeNode.Prime n => n
| TreeNode.Composite _ left middle right => nodeValue left * nodeValue middle * nodeValue right

/-- The modified factor tree structure -/
def factorTree : TreeNode :=
  TreeNode.Composite 0  -- A
    (TreeNode.Composite 0  -- B
      (TreeNode.Prime 3)
      (TreeNode.Composite 0  -- D
        (TreeNode.Prime 3)
        (TreeNode.Prime 2)
        (TreeNode.Prime 2))
      (TreeNode.Prime 3))
    (TreeNode.Prime 3)
    (TreeNode.Composite 0  -- C
      (TreeNode.Prime 5)
      (TreeNode.Composite 0  -- E
        (TreeNode.Prime 5)
        (TreeNode.Prime 2)
        (TreeNode.Prime 1))  -- Using 1 as a placeholder for the missing third child
      (TreeNode.Prime 5))

theorem factor_tree_value :
  nodeValue factorTree = 1800 := by sorry

end factor_tree_value_l2235_223546


namespace similar_polygons_ratio_l2235_223544

theorem similar_polygons_ratio (A₁ A₂ P₁ P₂ : ℝ) (h_positive : A₁ > 0 ∧ A₂ > 0 ∧ P₁ > 0 ∧ P₂ > 0) :
  A₁ / A₂ = 5 → P₁ / P₂ = m → 5 / m = Real.sqrt 5 :=
by sorry

end similar_polygons_ratio_l2235_223544


namespace final_score_proof_l2235_223580

def game_score (initial : ℕ) (penalty : ℕ) (additional : ℕ) : ℕ :=
  initial - penalty + additional

theorem final_score_proof :
  game_score 92 15 3 = 80 := by
  sorry

end final_score_proof_l2235_223580


namespace students_from_other_communities_l2235_223578

/-- Given a school with 1000 students and the percentages of students belonging to different communities,
    prove that the number of students from other communities is 90. -/
theorem students_from_other_communities
  (total_students : ℕ)
  (muslim_percent : ℚ)
  (hindu_percent : ℚ)
  (sikh_percent : ℚ)
  (christian_percent : ℚ)
  (buddhist_percent : ℚ)
  (h1 : total_students = 1000)
  (h2 : muslim_percent = 36 / 100)
  (h3 : hindu_percent = 24 / 100)
  (h4 : sikh_percent = 15 / 100)
  (h5 : christian_percent = 10 / 100)
  (h6 : buddhist_percent = 6 / 100) :
  ↑total_students * (1 - (muslim_percent + hindu_percent + sikh_percent + christian_percent + buddhist_percent)) = 90 :=
by sorry

end students_from_other_communities_l2235_223578


namespace students_in_class_l2235_223556

/-- The number of students in a class that needs to earn a certain number of points for eating vegetables. -/
def number_of_students (total_points : ℕ) (points_per_vegetable : ℕ) (num_weeks : ℕ) (vegetables_per_week : ℕ) : ℕ :=
  total_points / (points_per_vegetable * num_weeks * vegetables_per_week)

/-- Theorem stating that there are 25 students in the class given the problem conditions. -/
theorem students_in_class : 
  number_of_students 200 2 2 2 = 25 := by
  sorry

end students_in_class_l2235_223556


namespace sum_of_roots_l2235_223532

theorem sum_of_roots (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 6 * x₁ - 1 = 0) → 
  (2 * x₂^2 + 6 * x₂ - 1 = 0) → 
  x₁ + x₂ = -3 := by
sorry

end sum_of_roots_l2235_223532


namespace triangle_properties_l2235_223570

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.A * Real.sin t.B + (Real.sin t.C)^2 = (Real.sin t.A)^2 + (Real.sin t.B)^2)
  (h2 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)
  (h3 : t.A + t.B + t.C = π)
  (h4 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0)
  (h5 : t.a / Real.sin t.A = t.b / Real.sin t.B)
  (h6 : t.b / Real.sin t.B = t.c / Real.sin t.C) :
  -- Part 1: A, C, B form an arithmetic sequence
  ∃ d : Real, t.B = t.C + d ∧ t.C = t.A + d ∧
  -- Part 2: If c = 2, the maximum area is √3
  (t.c = 2 → ∀ (s : Real), s = 1/2 * t.a * t.b * Real.sin t.C → s ≤ Real.sqrt 3) :=
sorry


end triangle_properties_l2235_223570


namespace length_breadth_difference_is_32_l2235_223510

/-- Represents a rectangular plot with given dimensions and fencing costs. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- The difference between length and breadth of a rectangular plot. -/
def length_breadth_difference (plot : RectangularPlot) : ℝ :=
  plot.length - plot.breadth

/-- The perimeter of a rectangular plot. -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

theorem length_breadth_difference_is_32 (plot : RectangularPlot)
  (h1 : plot.length = 66)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : perimeter plot = plot.total_fencing_cost / plot.fencing_cost_per_meter) :
  length_breadth_difference plot = 32 := by
  sorry


end length_breadth_difference_is_32_l2235_223510


namespace marcos_salary_calculation_l2235_223507

theorem marcos_salary_calculation (initial_salary : ℝ) : 
  initial_salary = 2500 →
  let salary_after_first_raise := initial_salary * 1.15
  let salary_after_second_raise := salary_after_first_raise * 1.10
  let final_salary := salary_after_second_raise * 0.85
  final_salary = 2688.125 := by
sorry

end marcos_salary_calculation_l2235_223507


namespace union_is_reals_intersect_complement_l2235_223564

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 ≥ 0}
def B : Set ℝ := {x | x < 5}

-- Theorem for the union of A and B
theorem union_is_reals : A ∪ B = Set.univ := by sorry

-- Theorem for the intersection of complement of A and B
theorem intersect_complement : (Set.univ \ A) ∩ B = {x : ℝ | x < 2} := by sorry

end union_is_reals_intersect_complement_l2235_223564


namespace polyhedron_volume_is_twenty_thirds_l2235_223531

/-- Represents a polygon in the geometric arrangement --/
inductive Polygon
| IsoscelesRightTriangle : Polygon
| Square : Polygon
| RegularHexagon : Polygon

/-- The geometric arrangement of polygons --/
structure GeometricArrangement where
  triangles : Fin 3 → Polygon
  squares : Fin 3 → Polygon
  hexagon : Polygon
  triangles_are_isosceles_right : ∀ i, triangles i = Polygon.IsoscelesRightTriangle
  squares_are_squares : ∀ i, squares i = Polygon.Square
  hexagon_is_hexagon : hexagon = Polygon.RegularHexagon

/-- The side length of the squares --/
def square_side_length : ℝ := 2

/-- The volume of the polyhedron formed by folding the geometric arrangement --/
noncomputable def polyhedron_volume (arrangement : GeometricArrangement) : ℝ := 20/3

/-- Theorem stating that the volume of the polyhedron is 20/3 --/
theorem polyhedron_volume_is_twenty_thirds (arrangement : GeometricArrangement) :
  polyhedron_volume arrangement = 20/3 := by
  sorry

end polyhedron_volume_is_twenty_thirds_l2235_223531


namespace overall_speed_theorem_l2235_223588

/-- Given three points A, B, C on a line with AB = BC, and a car traveling from A to B at 40 km/h
    and from B to C at 60 km/h without stopping, the overall speed of the trip from A to C is 48 km/h. -/
theorem overall_speed_theorem (A B C : ℝ) (h1 : A < B) (h2 : B < C) (h3 : B - A = C - B) : 
  let d := B - A
  let t1 := d / 40
  let t2 := d / 60
  let total_time := t1 + t2
  let total_distance := 2 * d
  total_distance / total_time = 48 := by
  sorry

end overall_speed_theorem_l2235_223588


namespace billy_age_l2235_223598

theorem billy_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 64) : 
  billy = 48 := by
sorry

end billy_age_l2235_223598


namespace f_2_equals_4_l2235_223553

def f (n : ℕ) : ℕ := 
  (List.range n).sum + n + (List.range n).sum

theorem f_2_equals_4 : f 2 = 4 := by
  sorry

end f_2_equals_4_l2235_223553


namespace expand_product_l2235_223521

theorem expand_product (x : ℝ) : (x + 3)^2 * (x - 5) = x^3 + x^2 - 21*x - 45 := by
  sorry

end expand_product_l2235_223521


namespace a_is_positive_l2235_223582

theorem a_is_positive (x y a : ℝ) (h1 : x < y) (h2 : a * x < a * y) : a > 0 := by
  sorry

end a_is_positive_l2235_223582


namespace parallel_line_through_point_in_plane_l2235_223545

/-- A plane in 3D space -/
structure Plane where
  -- Define plane properties here (omitted for brevity)

/-- A line in 3D space -/
structure Line where
  -- Define line properties here (omitted for brevity)

/-- A point in 3D space -/
structure Point where
  -- Define point properties here (omitted for brevity)

/-- Predicate to check if a line is parallel to a plane -/
def isParallelToPlane (l : Line) (α : Plane) : Prop :=
  sorry

/-- Predicate to check if a point lies on a plane -/
def isOnPlane (P : Point) (α : Plane) : Prop :=
  sorry

/-- Predicate to check if two lines are parallel -/
def areLinesParallel (l1 l2 : Line) : Prop :=
  sorry

/-- Predicate to check if a line passes through a point -/
def linePassesThroughPoint (l : Line) (P : Point) : Prop :=
  sorry

/-- Predicate to check if a line lies entirely in a plane -/
def lineInPlane (l : Line) (α : Plane) : Prop :=
  sorry

theorem parallel_line_through_point_in_plane 
  (l : Line) (α : Plane) (P : Point)
  (h1 : isParallelToPlane l α)
  (h2 : isOnPlane P α) :
  ∃! m : Line, 
    linePassesThroughPoint m P ∧ 
    areLinesParallel m l ∧ 
    lineInPlane m α :=
  sorry

end parallel_line_through_point_in_plane_l2235_223545


namespace actual_plot_area_l2235_223575

/-- Represents the scale of the map --/
def scale : ℝ := 3

/-- Represents the conversion factor from square miles to acres --/
def sq_mile_to_acre : ℝ := 640

/-- Represents the length of the rectangle on the map in cm --/
def map_length : ℝ := 20

/-- Represents the width of the rectangle on the map in cm --/
def map_width : ℝ := 12

/-- Theorem stating that the area of the actual plot is 1,382,400 acres --/
theorem actual_plot_area :
  (map_length * scale) * (map_width * scale) * sq_mile_to_acre = 1382400 := by
  sorry

end actual_plot_area_l2235_223575


namespace compound_interest_rate_exists_l2235_223530

theorem compound_interest_rate_exists : ∃! r : ℝ, 0 < r ∧ r < 1 ∧ (1 + r)^15 = 9/5 := by
  sorry

end compound_interest_rate_exists_l2235_223530


namespace exists_meaningful_sqrt_l2235_223512

theorem exists_meaningful_sqrt : ∃ x : ℝ, x - 1 ≥ 0 := by
  sorry

end exists_meaningful_sqrt_l2235_223512


namespace complement_intersection_theorem_l2235_223524

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 5}

-- Define set A
def A : Set ℝ := {x | -3 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {x | -5 ≤ x ∧ x ≤ -3} :=
by sorry

end complement_intersection_theorem_l2235_223524


namespace andy_diana_weight_l2235_223501

theorem andy_diana_weight (a b c d : ℝ) 
  (h1 : a + b = 300)
  (h2 : b + c = 280)
  (h3 : c + d = 310) :
  a + d = 330 := by
  sorry

end andy_diana_weight_l2235_223501


namespace rachel_homework_difference_l2235_223594

def rachel_homework (math_pages reading_pages biology_pages : ℕ) : Prop :=
  math_pages - reading_pages = 7

theorem rachel_homework_difference :
  rachel_homework 9 2 96 := by
  sorry

end rachel_homework_difference_l2235_223594


namespace tower_heights_count_l2235_223597

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of different tower heights achievable -/
def countTowerHeights (numBricks : ℕ) (brickDim : BrickDimensions) : ℕ :=
  sorry

/-- The main theorem stating the number of different tower heights -/
theorem tower_heights_count :
  let numBricks : ℕ := 200
  let brickDim : BrickDimensions := ⟨3, 8, 20⟩
  countTowerHeights numBricks brickDim = 680 :=
by sorry

end tower_heights_count_l2235_223597


namespace evaluate_expression_l2235_223554

theorem evaluate_expression : -25 - 5 * (8 / 4) = -35 := by
  sorry

end evaluate_expression_l2235_223554


namespace gcd_from_lcm_and_ratio_l2235_223503

theorem gcd_from_lcm_and_ratio (C D : ℕ+) 
  (h_lcm : Nat.lcm C D = 250)
  (h_ratio : C * 5 = D * 2) :
  Nat.gcd C D = 5 := by
sorry

end gcd_from_lcm_and_ratio_l2235_223503


namespace book_selection_count_l2235_223523

theorem book_selection_count (A B : Type) [Fintype A] [Fintype B] 
  (h1 : Fintype.card A = 4) (h2 : Fintype.card B = 5) : 
  Fintype.card (A × B) = 20 := by
  sorry

end book_selection_count_l2235_223523


namespace triangle_bisector_inequality_l2235_223586

/-- In any triangle ABC, the product of the lengths of its internal angle bisectors 
    is less than or equal to (3√3 / 8) times the product of its side lengths. -/
theorem triangle_bisector_inequality (a b c t_a t_b t_c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < t_a ∧ 0 < t_b ∧ 0 < t_c →  -- Positive bisector lengths
  t_a + t_b > c ∧ t_b + t_c > a ∧ t_c + t_a > b →  -- Triangle inequality for bisectors
  t_a * t_b * t_c ≤ (3 * Real.sqrt 3 / 8) * a * b * c := by
  sorry

end triangle_bisector_inequality_l2235_223586


namespace max_min_product_of_three_l2235_223517

def S : Finset Int := {-1, -2, 3, 4}

theorem max_min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
    x * y * z ≤ 8 ∧ x * y * z ≥ -24) ∧ 
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 8) ∧
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = -24) :=
by sorry

end max_min_product_of_three_l2235_223517


namespace community_event_earnings_sharing_l2235_223540

theorem community_event_earnings_sharing (earnings : Fin 3 → ℕ) 
  (h1 : earnings 0 = 18)
  (h2 : earnings 1 = 24)
  (h3 : earnings 2 = 36) :
  36 - (earnings 0 + earnings 1 + earnings 2) / 3 = 10 := by
  sorry

end community_event_earnings_sharing_l2235_223540


namespace sin_alpha_for_point_l2235_223511

/-- Given an angle α whose terminal side passes through the point (-a, 2a) where a < 0,
    prove that sin α = -2√5/5 -/
theorem sin_alpha_for_point (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : ∃ k : ℝ, k > 0 ∧ k * Real.cos α = -a ∧ k * Real.sin α = 2*a) : 
  Real.sin α = -2 * Real.sqrt 5 / 5 := by
  sorry

end sin_alpha_for_point_l2235_223511


namespace complex_multiplication_l2235_223593

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The property that i^2 = -1 -/
axiom i_squared : i ^ 2 = -1

/-- Theorem stating that (1+2i)(2+i) = 5i -/
theorem complex_multiplication : (1 + 2 * i) * (2 + i) = 5 * i := by sorry

end complex_multiplication_l2235_223593


namespace triangle_structure_pieces_l2235_223537

/-- Calculates the sum of arithmetic sequence -/
def arithmeticSum (a₁ n d : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

/-- Represents the structure of the triangle -/
structure TriangleStructure where
  rows : ℕ
  firstRowRods : ℕ
  rodIncrement : ℕ

/-- Calculates the total number of pieces in the triangle structure with base -/
def totalPieces (t : TriangleStructure) : ℕ :=
  let topRods := arithmeticSum t.firstRowRods t.rows t.rodIncrement
  let topConnectors := arithmeticSum 1 (t.rows + 1) 1
  let baseRods := 2 * (t.firstRowRods + (t.rows - 1) * t.rodIncrement)
  let basePieces := 2 * baseRods
  topRods + topConnectors + basePieces

/-- The main theorem to prove -/
theorem triangle_structure_pieces :
  let t : TriangleStructure := { rows := 10, firstRowRods := 3, rodIncrement := 3 }
  totalPieces t = 351 := by
  sorry

end triangle_structure_pieces_l2235_223537


namespace robotics_club_mentors_average_age_l2235_223599

theorem robotics_club_mentors_average_age 
  (total_members : ℕ) 
  (avg_age_all : ℕ) 
  (num_girls : ℕ) 
  (num_boys : ℕ) 
  (num_mentors : ℕ) 
  (avg_age_girls : ℕ) 
  (avg_age_boys : ℕ) 
  (h1 : total_members = 50)
  (h2 : avg_age_all = 20)
  (h3 : num_girls = 25)
  (h4 : num_boys = 20)
  (h5 : num_mentors = 5)
  (h6 : avg_age_girls = 18)
  (h7 : avg_age_boys = 19)
  (h8 : total_members = num_girls + num_boys + num_mentors) :
  (total_members * avg_age_all - num_girls * avg_age_girls - num_boys * avg_age_boys) / num_mentors = 34 := by
  sorry

end robotics_club_mentors_average_age_l2235_223599


namespace rectangle_area_twice_perimeter_l2235_223522

theorem rectangle_area_twice_perimeter (x : ℝ) : 
  (4 * x) * (x + 7) = 2 * (2 * (4 * x) + 2 * (x + 7)) → x = 1 := by
  sorry

end rectangle_area_twice_perimeter_l2235_223522


namespace magazines_per_box_l2235_223513

theorem magazines_per_box (total_magazines : ℕ) (num_boxes : ℕ) (magazines_per_box : ℕ) 
  (h1 : total_magazines = 63)
  (h2 : num_boxes = 7)
  (h3 : total_magazines = num_boxes * magazines_per_box) :
  magazines_per_box = 9 := by
  sorry

end magazines_per_box_l2235_223513


namespace largest_c_value_l2235_223508

theorem largest_c_value (c : ℝ) : (3 * c + 4) * (c - 2) = 9 * c → c ≤ 4 := by
  sorry

end largest_c_value_l2235_223508


namespace smallest_k_no_real_roots_l2235_223559

theorem smallest_k_no_real_roots : 
  ∃ (k : ℤ), (∀ (x : ℝ), 3 * x * (k * x - 5) - 2 * x^2 + 8 ≠ 0) ∧ 
  (∀ (j : ℤ), j < k → ∃ (x : ℝ), 3 * x * (j * x - 5) - 2 * x^2 + 8 = 0) := by
  sorry

#check smallest_k_no_real_roots

end smallest_k_no_real_roots_l2235_223559


namespace mans_downstream_speed_l2235_223514

/-- Given a man's upstream speed and the stream speed, calculates his downstream speed -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  upstream_speed + 2 * stream_speed

/-- Theorem: The man's downstream speed is 12 kmph given the conditions -/
theorem mans_downstream_speed :
  let upstream_speed := 8
  let stream_speed := 2
  downstream_speed upstream_speed stream_speed = 12 := by
  sorry

end mans_downstream_speed_l2235_223514


namespace similar_triangles_perimeter_l2235_223571

/-- Given two similar right triangles, where the smaller triangle has legs of 5 and 12,
    and the larger triangle has a hypotenuse of 39, prove that the perimeter of the larger triangle is 90. -/
theorem similar_triangles_perimeter (small_leg1 small_leg2 large_hypotenuse : ℝ)
    (h1 : small_leg1 = 5)
    (h2 : small_leg2 = 12)
    (h3 : large_hypotenuse = 39)
    (h4 : small_leg1^2 + small_leg2^2 = (small_leg1^2 + small_leg2^2).sqrt^2) -- Pythagorean theorem for smaller triangle
    (h5 : ∃ k : ℝ, k * (small_leg1^2 + small_leg2^2).sqrt = large_hypotenuse) -- Similarity condition
    : ∃ large_leg1 large_leg2 : ℝ,
      large_leg1^2 + large_leg2^2 = large_hypotenuse^2 ∧ -- Pythagorean theorem for larger triangle
      large_leg1 + large_leg2 + large_hypotenuse = 90 := by
  sorry

end similar_triangles_perimeter_l2235_223571


namespace fermat_like_contradiction_l2235_223547

theorem fermat_like_contradiction (a b c : ℝ) (m n : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hm : 0 < m) (hn : 0 < n) (hmn : m ≠ n) :
  ¬(a^m + b^m = c^m ∧ a^n + b^n = c^n) :=
sorry

end fermat_like_contradiction_l2235_223547


namespace square_perimeter_from_area_l2235_223539

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (h1 : area = 36) (h2 : side * side = area) :
  4 * side = 24 := by
  sorry

end square_perimeter_from_area_l2235_223539


namespace bridget_apples_l2235_223541

/-- The number of apples Bridget bought -/
def total_apples : ℕ := 26

/-- The number of apples Bridget gave to Cassie -/
def apples_to_cassie : ℕ := 5

/-- The number of apples Bridget gave to Dan -/
def apples_to_dan : ℕ := 2

/-- The number of apples Bridget kept for herself -/
def apples_kept : ℕ := 6

theorem bridget_apples : 
  total_apples / 2 - apples_to_cassie - apples_to_dan = apples_kept :=
by sorry

end bridget_apples_l2235_223541


namespace graph_is_two_lines_factored_is_two_lines_graph_consists_of_two_intersecting_lines_l2235_223587

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop := x * y - 2 * x + 3 * y - 6 = 0

/-- The factored form of the equation -/
def factored_equation (x y : ℝ) : Prop := (x + 3) * (y - 2) = 0

/-- Theorem stating that the graph equation is equivalent to the factored equation -/
theorem graph_is_two_lines :
  ∀ x y : ℝ, graph_equation x y ↔ factored_equation x y :=
by sorry

/-- Theorem stating that the factored equation represents two intersecting lines -/
theorem factored_is_two_lines :
  ∃ a b : ℝ, ∀ x y : ℝ, factored_equation x y ↔ (x = a ∨ y = b) :=
by sorry

/-- Main theorem proving that the graph consists of two intersecting lines -/
theorem graph_consists_of_two_intersecting_lines :
  ∃ a b : ℝ, ∀ x y : ℝ, graph_equation x y ↔ (x = a ∨ y = b) :=
by sorry

end graph_is_two_lines_factored_is_two_lines_graph_consists_of_two_intersecting_lines_l2235_223587


namespace intersection_equals_interval_l2235_223535

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Define the interval (1, 2]
def interval_one_two : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_equals_interval : A ∩ B = interval_one_two := by
  sorry

end intersection_equals_interval_l2235_223535


namespace quadratic_inequality_range_l2235_223592

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a + 2) * x^2 + 2 * (a + 2) * x + 4 > 0) ↔ a ∈ Set.Ici (-2) ∩ Set.Iio 2 :=
by sorry

end quadratic_inequality_range_l2235_223592


namespace vector_sum_magnitude_l2235_223543

/-- Given vectors a and b in ℝ², prove that ‖a + 2b‖ = 2√3 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  (a.1 = 2 ∧ a.2 = 0) →  -- a = (2, 0)
  ‖b‖ = 1 →  -- ‖b‖ = 1
  (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) = 1/2 →  -- angle between a and b is π/3 (cos(π/3) = 1/2)
  ‖a + 2 • b‖ = 2 * Real.sqrt 3 := by
sorry

end vector_sum_magnitude_l2235_223543


namespace binary_calculation_l2235_223560

def binary_to_decimal (b : ℕ) : ℕ := sorry

theorem binary_calculation : 
  (binary_to_decimal 0b111111111 + binary_to_decimal 0b11111) * binary_to_decimal 0b11 = 1626 := by
  sorry

end binary_calculation_l2235_223560


namespace algebraic_expression_value_l2235_223536

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2*x = -1) : 5 + x*(x + 2) = 4 := by
  sorry

end algebraic_expression_value_l2235_223536


namespace gcd_1729_867_l2235_223525

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 := by
  sorry

end gcd_1729_867_l2235_223525


namespace house_store_transaction_l2235_223549

theorem house_store_transaction : 
  ∀ (house_cost store_cost : ℝ),
  house_cost * 0.9 = 9000 →
  store_cost * 1.3 = 13000 →
  (9000 + 13000) - (house_cost + store_cost) = 2000 := by
  sorry

end house_store_transaction_l2235_223549


namespace geometric_series_ratio_l2235_223551

theorem geometric_series_ratio (a : ℝ) (r : ℝ) (h : r ≠ 1) :
  (a / (1 - r)) = 16 * (a * r^5 / (1 - r)) → r = 1/2 := by
  sorry

end geometric_series_ratio_l2235_223551


namespace football_season_games_james_football_season_l2235_223534

/-- Calculates the number of games in a football season based on a player's performance -/
theorem football_season_games (touchdowns_per_game : ℕ) (points_per_touchdown : ℕ) 
  (two_point_conversions : ℕ) (old_record : ℕ) (points_above_record : ℕ) : ℕ :=
  let total_points := old_record + points_above_record
  let points_from_conversions := two_point_conversions * 2
  let points_from_touchdowns := total_points - points_from_conversions
  let points_per_game := touchdowns_per_game * points_per_touchdown
  points_from_touchdowns / points_per_game

/-- The number of games in James' football season -/
theorem james_football_season : 
  football_season_games 4 6 6 300 72 = 15 := by
  sorry

end football_season_games_james_football_season_l2235_223534


namespace min_value_of_M_l2235_223562

theorem min_value_of_M (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  let M := (a^2 + b^2) / (a - b)
  ∀ x, M ≥ x → x ≥ 2 * Real.sqrt 2 :=
by sorry

end min_value_of_M_l2235_223562


namespace notebook_buyers_difference_l2235_223595

theorem notebook_buyers_difference : ∃ (price : ℕ) (eighth_buyers fifth_buyers : ℕ),
  price > 0 ∧
  price * eighth_buyers = 210 ∧
  price * fifth_buyers = 240 ∧
  fifth_buyers = 25 ∧
  fifth_buyers - eighth_buyers = 2 :=
by sorry

end notebook_buyers_difference_l2235_223595


namespace gcd_98_63_l2235_223533

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l2235_223533


namespace jose_bottle_caps_l2235_223548

theorem jose_bottle_caps (initial : ℝ) (given_away : ℝ) (remaining : ℝ) : 
  initial = 7.0 → given_away = 2.0 → remaining = initial - given_away → remaining = 5.0 := by
  sorry

end jose_bottle_caps_l2235_223548


namespace complex_equation_solution_l2235_223573

theorem complex_equation_solution (z : ℂ) : z + Complex.abs z = 2 + I → z = 1 + I := by
  sorry

end complex_equation_solution_l2235_223573


namespace binomial_coefficient_16_11_l2235_223506

theorem binomial_coefficient_16_11 
  (h1 : Nat.choose 15 9 = 5005)
  (h2 : Nat.choose 15 10 = 3003)
  (h3 : Nat.choose 17 11 = 12376) :
  Nat.choose 16 11 = 4368 := by
sorry

end binomial_coefficient_16_11_l2235_223506


namespace necessary_not_sufficient_condition_l2235_223528

/-- The curve C is defined by the equation x^2 / (k - 5) + y^2 / (3 - k) = -1 -/
def curve_C (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (k - 5) + p.2^2 / (3 - k) = -1}

/-- Predicate to check if a curve represents an ellipse with foci on the y-axis -/
def is_ellipse_y_foci (C : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ C = {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}

/-- The main theorem stating that 4 ≤ k < 5 is a necessary but not sufficient condition -/
theorem necessary_not_sufficient_condition (k : ℝ) :
  (is_ellipse_y_foci (curve_C k) → 4 ≤ k ∧ k < 5) ∧
  ¬(4 ≤ k ∧ k < 5 → is_ellipse_y_foci (curve_C k)) :=
sorry

end necessary_not_sufficient_condition_l2235_223528


namespace product_sum_fractions_l2235_223579

theorem product_sum_fractions : (3 * 4 * 5 * 6) * (1/3 + 1/4 + 1/5 + 1/6) = 342 := by
  sorry

end product_sum_fractions_l2235_223579
