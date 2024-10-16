import Mathlib

namespace NUMINAMATH_CALUDE_meeting_participants_ratio_l1746_174676

/-- Given information about participants in a meeting, prove the ratio of female democrats to total female participants -/
theorem meeting_participants_ratio :
  let total_participants : ℕ := 810
  let female_democrats : ℕ := 135
  let male_democrat_ratio : ℚ := 1/4
  let total_democrat_ratio : ℚ := 1/3
  ∃ (female_participants male_participants : ℕ),
    female_participants + male_participants = total_participants ∧
    female_democrats + male_democrat_ratio * male_participants = total_democrat_ratio * total_participants ∧
    female_democrats / female_participants = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_meeting_participants_ratio_l1746_174676


namespace NUMINAMATH_CALUDE_max_fold_length_less_than_eight_l1746_174658

theorem max_fold_length_less_than_eight (length width : ℝ) 
  (h_length : length = 6) (h_width : width = 5) : 
  Real.sqrt (length^2 + width^2) < 8 := by
  sorry

end NUMINAMATH_CALUDE_max_fold_length_less_than_eight_l1746_174658


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1746_174667

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) * Real.sqrt (x + 3) ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {-3} ∪ Set.Ici 2

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1746_174667


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1746_174654

theorem partial_fraction_decomposition (a b c d : ℤ) (h : a * d ≠ b * c) :
  ∃ (r s : ℚ), ∀ (x : ℝ), 
    1 / ((a * x + b) * (c * x + d)) = r / (a * x + b) + s / (c * x + d) ∧
    r = a / (a * d - b * c) ∧
    s = -c / (a * d - b * c) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1746_174654


namespace NUMINAMATH_CALUDE_temperature_calculation_l1746_174684

/-- Given the average temperatures for two sets of four consecutive days and the temperature of the first day, calculate the temperature of the last day. -/
theorem temperature_calculation (temp_mon tues wed thurs fri : ℝ) :
  (temp_mon + tues + wed + thurs) / 4 = 48 →
  (tues + wed + thurs + fri) / 4 = 46 →
  temp_mon = 39 →
  fri = 31 := by
  sorry

end NUMINAMATH_CALUDE_temperature_calculation_l1746_174684


namespace NUMINAMATH_CALUDE_square_field_area_l1746_174621

/-- The area of a square field with side length 20 meters is 400 square meters. -/
theorem square_field_area : ∀ (side_length : ℝ), side_length = 20 → side_length * side_length = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l1746_174621


namespace NUMINAMATH_CALUDE_max_value_a_l1746_174686

theorem max_value_a (a b c d : ℕ+) 
  (hab : a < 2 * b)
  (hbc : b < 3 * c)
  (hcd : c < 4 * d)
  (hd : d < 100) :
  a ≤ 2367 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 2367 ∧ 
    a' < 2 * b' ∧ 
    b' < 3 * c' ∧ 
    c' < 4 * d' ∧ 
    d' < 100 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l1746_174686


namespace NUMINAMATH_CALUDE_car_a_speed_l1746_174671

/-- Proves that Car A's speed is 58 mph given the problem conditions -/
theorem car_a_speed (initial_distance : ℝ) (car_b_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 30 →
  car_b_speed = 50 →
  time = 4.75 →
  final_distance = 8 →
  ∃ (car_a_speed : ℝ),
    car_a_speed * time = car_b_speed * time + initial_distance + final_distance ∧
    car_a_speed = 58 := by
  sorry

end NUMINAMATH_CALUDE_car_a_speed_l1746_174671


namespace NUMINAMATH_CALUDE_ellipse_parameters_sum_l1746_174664

-- Define the foci
def F₁ : ℝ × ℝ := (1, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the constant sum of distances
def distance_sum : ℝ := 10

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = distance_sum

-- Define the general form of the ellipse equation
def ellipse_equation (h k a b : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - h)^2 / a^2 + (P.2 - k)^2 / b^2 = 1

-- Theorem statement
theorem ellipse_parameters_sum :
  ∃ (h k a b : ℝ),
    (∀ P, is_on_ellipse P ↔ ellipse_equation h k a b P) ∧
    h + k + a + b = 8 + Real.sqrt 21 :=
sorry

end NUMINAMATH_CALUDE_ellipse_parameters_sum_l1746_174664


namespace NUMINAMATH_CALUDE_complex_cube_root_unity_l1746_174607

/-- Given that i is the imaginary unit and z = -1/2 + (√3/2)i, prove that z^2 + z + 1 = 0 -/
theorem complex_cube_root_unity (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = -1/2 + (Real.sqrt 3 / 2) * i →
  z^2 + z + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_unity_l1746_174607


namespace NUMINAMATH_CALUDE_min_cost_14000_l1746_174665

/-- Represents the number of soup + main course combinations -/
def x : ℕ := 15

/-- Represents the number of salad + main course combinations -/
def y : ℕ := 0

/-- Represents the number of all three dish combinations -/
def z : ℕ := 0

/-- Represents the number of standalone main courses -/
def q : ℕ := 35

/-- The cost of a salad -/
def salad_cost : ℕ := 200

/-- The cost of soup + main course -/
def soup_main_cost : ℕ := 350

/-- The cost of salad + main course -/
def salad_main_cost : ℕ := 350

/-- The cost of soup + salad + main course -/
def all_three_cost : ℕ := 500

/-- The total number of main courses required -/
def total_main : ℕ := 50

/-- The total number of salads required -/
def total_salad : ℕ := 30

/-- The total number of soups required -/
def total_soup : ℕ := 15

theorem min_cost_14000 :
  (x + y + z + q = total_main) ∧
  (y + z = total_salad) ∧
  (x + z = total_soup) ∧
  (∀ x' y' z' q' : ℕ,
    (x' + y' + z' + q' = total_main) →
    (y' + z' = total_salad) →
    (x' + z' = total_soup) →
    soup_main_cost * x + salad_main_cost * y + all_three_cost * z + salad_cost * q ≤
    soup_main_cost * x' + salad_main_cost * y' + all_three_cost * z' + salad_cost * q') →
  soup_main_cost * x + salad_main_cost * y + all_three_cost * z + salad_cost * q = 14000 :=
sorry

end NUMINAMATH_CALUDE_min_cost_14000_l1746_174665


namespace NUMINAMATH_CALUDE_unique_fraction_decomposition_l1746_174652

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (n m : ℕ), n ≠ m ∧ 2 / p = 1 / n + 1 / m ∧
  ((n = (p + 1) / 2 ∧ m = p * (p + 1) / 2) ∨
   (m = (p + 1) / 2 ∧ n = p * (p + 1) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_fraction_decomposition_l1746_174652


namespace NUMINAMATH_CALUDE_calculate_principal_l1746_174619

/-- Given simple interest, rate, and time, calculate the principal amount --/
theorem calculate_principal
  (simple_interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : simple_interest = 4016.25)
  (h2 : rate = 3)
  (h3 : time = 5)
  : (simple_interest * 100) / (rate * time) = 26775 := by
  sorry

end NUMINAMATH_CALUDE_calculate_principal_l1746_174619


namespace NUMINAMATH_CALUDE_candy_division_l1746_174678

theorem candy_division (total_candy : ℚ) (portions : ℚ) (ana_portions : ℕ) :
  total_candy = 75 / 4 ∧ 
  portions = 7 / 2 ∧ 
  ana_portions = 2 →
  ana_portions * (total_candy / portions) = 75 / 7 :=
by sorry

end NUMINAMATH_CALUDE_candy_division_l1746_174678


namespace NUMINAMATH_CALUDE_complex_quadrant_l1746_174647

theorem complex_quadrant (z : ℂ) (h : z * (1 + Complex.I) = 5 + Complex.I) :
  Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l1746_174647


namespace NUMINAMATH_CALUDE_oliver_money_problem_l1746_174690

theorem oliver_money_problem (X : ℤ) :
  X + 5 - 4 - 3 + 8 = 15 → X = 13 := by
  sorry

end NUMINAMATH_CALUDE_oliver_money_problem_l1746_174690


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l1746_174612

/-- Given a right triangle OAB with O at the origin, this structure represents the hyperbola
    y = k/x passing through the midpoint of OB and intersecting AB at C. -/
structure Hyperbola_Triangle :=
  (a b : ℝ)  -- Coordinates of point B (a, b)
  (k : ℝ)    -- Parameter of the hyperbola y = k/x
  (h_k_pos : k > 0)  -- k is positive
  (h_right_triangle : a * b = 2 * 3)  -- Area of OAB is 3, so a * b / 2 = 3
  (h_midpoint : k / (a/2) = b/2)  -- Hyperbola passes through midpoint of OB
  (c : ℝ)    -- x-coordinate of point C
  (h_c_on_ab : 0 < c ∧ c < a)  -- C is between O and B on AB
  (h_c_on_hyperbola : k / c = b * (1 - c/a))  -- C is on the hyperbola

/-- The main theorem: if the area of OBC is 3, then k = 2 -/
theorem hyperbola_triangle_area (ht : Hyperbola_Triangle) 
  (h_area_obc : ht.a * ht.b * (1 - ht.c/ht.a) / 2 = 3) : ht.k = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l1746_174612


namespace NUMINAMATH_CALUDE_cubic_monotone_increasing_l1746_174657

/-- A cubic function f(x) = ax³ - x² + x - 5 is monotonically increasing on ℝ if and only if a ≥ 1/3 -/
theorem cubic_monotone_increasing (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => a * x^3 - x^2 + x - 5) (3 * a * x^2 - 2 * x + 1) x) →
  (∀ x y : ℝ, x < y → (a * x^3 - x^2 + x - 5) < (a * y^3 - y^2 + y - 5)) ↔
  a ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_monotone_increasing_l1746_174657


namespace NUMINAMATH_CALUDE_min_sum_squares_l1746_174682

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3 * x₂ + 4 * x₃ = 120) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 7200 / 13 ∧ 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 
    y₁ + 3 * y₂ + 4 * y₃ = 120 ∧ 
    y₁^2 + y₂^2 + y₃^2 = 7200 / 13 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1746_174682


namespace NUMINAMATH_CALUDE_points_per_correct_answer_l1746_174638

theorem points_per_correct_answer 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (final_score : ℚ) 
  (incorrect_penalty : ℚ) 
  (h1 : total_questions = 120)
  (h2 : correct_answers = 104)
  (h3 : final_score = 100)
  (h4 : incorrect_penalty = -1/4)
  (h5 : correct_answers ≤ total_questions) :
  ∃ (points_per_correct : ℚ), 
    points_per_correct * correct_answers + 
    incorrect_penalty * (total_questions - correct_answers) = final_score ∧
    points_per_correct = 1 :=
by sorry

end NUMINAMATH_CALUDE_points_per_correct_answer_l1746_174638


namespace NUMINAMATH_CALUDE_square_value_proof_l1746_174680

theorem square_value_proof (square : Real) : 
  ((11.2 - 1.2 * square) / 4 + 51.2 * square) * 0.1 = 9.1 → square = 1.568 := by
  sorry

end NUMINAMATH_CALUDE_square_value_proof_l1746_174680


namespace NUMINAMATH_CALUDE_waiter_tips_l1746_174668

/-- Calculates the total tips earned by a waiter given the number of customers, 
    number of non-tipping customers, and the tip amount from each tipping customer. -/
def calculate_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Theorem stating that under the given conditions, the waiter earns $32 in tips. -/
theorem waiter_tips : 
  calculate_tips 9 5 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_l1746_174668


namespace NUMINAMATH_CALUDE_inscribed_circles_radius_l1746_174659

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

end NUMINAMATH_CALUDE_inscribed_circles_radius_l1746_174659


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1746_174640

-- Define the probabilities for each ring
def P_10 : ℝ := 0.24
def P_9 : ℝ := 0.28
def P_8 : ℝ := 0.19
def P_7 : ℝ := 0.16
def P_below_7 : ℝ := 0.13

-- Theorem for the three probability calculations
theorem shooting_probabilities :
  (P_10 + P_9 = 0.52) ∧
  (P_10 + P_9 + P_8 + P_7 = 0.87) ∧
  (P_7 + P_below_7 = 0.29) := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l1746_174640


namespace NUMINAMATH_CALUDE_compound_interest_rate_exists_l1746_174645

theorem compound_interest_rate_exists : ∃! r : ℝ, 0 < r ∧ r < 1 ∧ (1 + r)^15 = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_exists_l1746_174645


namespace NUMINAMATH_CALUDE_deans_vacation_cost_l1746_174666

/-- The total cost of a group vacation given the number of people and individual costs -/
def vacation_cost (num_people : ℕ) (rent transport food activities : ℚ) : ℚ :=
  num_people * (rent + transport + food + activities)

/-- Theorem stating the total cost for Dean's group vacation -/
theorem deans_vacation_cost :
  vacation_cost 7 70 25 55 40 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_deans_vacation_cost_l1746_174666


namespace NUMINAMATH_CALUDE_randy_third_quiz_score_l1746_174697

theorem randy_third_quiz_score 
  (first_quiz : ℕ) 
  (second_quiz : ℕ) 
  (fifth_quiz : ℕ) 
  (desired_average : ℕ) 
  (total_quizzes : ℕ) 
  (third_fourth_sum : ℕ) :
  first_quiz = 90 →
  second_quiz = 98 →
  fifth_quiz = 96 →
  desired_average = 94 →
  total_quizzes = 5 →
  third_fourth_sum = 186 →
  ∃ (fourth_quiz : ℕ), 
    (first_quiz + second_quiz + 94 + fourth_quiz + fifth_quiz) / total_quizzes = desired_average :=
by
  sorry


end NUMINAMATH_CALUDE_randy_third_quiz_score_l1746_174697


namespace NUMINAMATH_CALUDE_valerie_light_bulb_purchase_l1746_174689

/-- Valerie's light bulb purchase problem -/
theorem valerie_light_bulb_purchase (small_bulb_cost large_bulb_cost small_bulb_count large_bulb_count leftover_money : ℕ) :
  small_bulb_cost = 8 →
  large_bulb_cost = 12 →
  small_bulb_count = 3 →
  large_bulb_count = 1 →
  leftover_money = 24 →
  small_bulb_cost * small_bulb_count + large_bulb_cost * large_bulb_count + leftover_money = 60 :=
by sorry

end NUMINAMATH_CALUDE_valerie_light_bulb_purchase_l1746_174689


namespace NUMINAMATH_CALUDE_no_member_divisible_by_4_or_5_l1746_174624

def T : Set Int := {x | ∃ n : Int, x = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2}

theorem no_member_divisible_by_4_or_5 : ∀ x ∈ T, ¬(x % 4 = 0 ∨ x % 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_member_divisible_by_4_or_5_l1746_174624


namespace NUMINAMATH_CALUDE_line_not_in_first_quadrant_l1746_174636

/-- A line that does not pass through the first quadrant has a non-positive slope -/
def not_in_first_quadrant (t : ℝ) : Prop :=
  3 - 2 * t ≤ 0

/-- The range of t for which the line (2t-3)x + y + 6 = 0 does not pass through the first quadrant -/
def t_range : Set ℝ :=
  {t : ℝ | t ≥ 3/2}

theorem line_not_in_first_quadrant :
  ∀ t : ℝ, not_in_first_quadrant t ↔ t ∈ t_range :=
sorry

end NUMINAMATH_CALUDE_line_not_in_first_quadrant_l1746_174636


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l1746_174674

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℕ) :
  ∃ k : ℤ, (n^3 + (n+1)^3 + (n+2)^3 : ℤ) = 9 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l1746_174674


namespace NUMINAMATH_CALUDE_quadrilateral_formation_l1746_174699

/-- A function that checks if four line segments can form a quadrilateral --/
def can_form_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a

/-- The theorem stating which set of line segments can form a quadrilateral with length 5 --/
theorem quadrilateral_formation :
  ¬(can_form_quadrilateral 1 1 1 5) ∧
  ¬(can_form_quadrilateral 1 1 8 5) ∧
  ¬(can_form_quadrilateral 1 2 2 5) ∧
  can_form_quadrilateral 3 3 3 5 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_formation_l1746_174699


namespace NUMINAMATH_CALUDE_g_forms_l1746_174672

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the property for g
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9*x^2 - 6*x + 1

-- Theorem statement
theorem g_forms {g : ℝ → ℝ} (h : g_property g) :
  (∀ x, g x = 3*x - 1) ∨ (∀ x, g x = -3*x + 1) := by
  sorry

end NUMINAMATH_CALUDE_g_forms_l1746_174672


namespace NUMINAMATH_CALUDE_value_of_expression_l1746_174615

theorem value_of_expression (α : Real) (h : 4 * Real.sin α - 3 * Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin (2 * α)) = 25 / 64 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1746_174615


namespace NUMINAMATH_CALUDE_largest_integer_dividing_factorial_l1746_174648

theorem largest_integer_dividing_factorial (n : ℕ) : 
  (∀ k : ℕ, k ≤ 9 → (2007 : ℕ).factorial % (2007 ^ k) = 0) ∧ 
  ((2007 : ℕ).factorial % (2007 ^ 10) ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_dividing_factorial_l1746_174648


namespace NUMINAMATH_CALUDE_average_string_length_l1746_174688

theorem average_string_length (s1 s2 s3 : ℝ) (h1 : s1 = 1) (h2 : s2 = 3) (h3 : s3 = 5) :
  (s1 + s2 + s3) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_string_length_l1746_174688


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l1746_174655

theorem fruit_seller_apples : ∀ (original : ℕ),
  (original : ℝ) * 0.6 = 420 → original = 700 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l1746_174655


namespace NUMINAMATH_CALUDE_problem_1_l1746_174683

theorem problem_1 : Real.sqrt 9 + (-2)^3 - Real.cos (π / 3) = -11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1746_174683


namespace NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l1746_174602

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Type := Nat

/-- The number of vertices in a decagon -/
def num_vertices : Nat := 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def num_adjacent : Nat := 2

/-- The probability of choosing two distinct adjacent vertices in a decagon -/
def prob_adjacent_vertices (d : Decagon) : Rat :=
  num_adjacent / (num_vertices - 1)

theorem decagon_adjacent_vertices_probability :
  ∀ d : Decagon, prob_adjacent_vertices d = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l1746_174602


namespace NUMINAMATH_CALUDE_line_solution_l1746_174630

/-- Given a line y = ax + b (a ≠ 0) passing through points (0,4) and (-3,0),
    the solution to ax + b = 0 is x = -3. -/
theorem line_solution (a b : ℝ) (ha : a ≠ 0) :
  (4 = b) →                        -- Line passes through (0,4)
  (0 = -3*a + b) →                 -- Line passes through (-3,0)
  (∀ x, a*x + b = 0 ↔ x = -3) :=   -- Solution to ax + b = 0 is x = -3
by
  sorry

end NUMINAMATH_CALUDE_line_solution_l1746_174630


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1746_174635

theorem geometric_series_ratio (a : ℝ) (r : ℝ) (h : r ≠ 1) :
  (a / (1 - r)) = 16 * (a * r^5 / (1 - r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1746_174635


namespace NUMINAMATH_CALUDE_unique_solution_in_p_arithmetic_l1746_174677

-- Define p-arithmetic structure
structure PArithmetic (p : ℕ) where
  carrier : Type
  add : carrier → carrier → carrier
  mul : carrier → carrier → carrier
  zero : carrier
  one : carrier
  -- Add necessary axioms for p-arithmetic

-- Define the theorem
theorem unique_solution_in_p_arithmetic {p : ℕ} (P : PArithmetic p) :
  ∀ (a b : P.carrier), a ≠ P.zero → ∃! x : P.carrier, P.mul a x = b :=
sorry

end NUMINAMATH_CALUDE_unique_solution_in_p_arithmetic_l1746_174677


namespace NUMINAMATH_CALUDE_rectangle_area_twice_perimeter_l1746_174600

theorem rectangle_area_twice_perimeter (x : ℝ) : 
  (4 * x) * (x + 7) = 2 * (2 * (4 * x) + 2 * (x + 7)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_twice_perimeter_l1746_174600


namespace NUMINAMATH_CALUDE_sum_of_distances_is_12_sqrt_2_l1746_174653

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define point P
def P : ℝ × ℝ := (-2, -4)

-- Define the intersection points M and N (existence assumed)
axiom M_exists : ∃ M : ℝ × ℝ, C M.1 M.2 ∧ l M.1 M.2
axiom N_exists : ∃ N : ℝ × ℝ, C N.1 N.2 ∧ l N.1 N.2
axiom M_ne_N : ∀ M N : ℝ × ℝ, C M.1 M.2 ∧ l M.1 M.2 ∧ C N.1 N.2 ∧ l N.1 N.2 → M ≠ N

-- Theorem statement
theorem sum_of_distances_is_12_sqrt_2 :
  ∃ M N : ℝ × ℝ, C M.1 M.2 ∧ l M.1 M.2 ∧ C N.1 N.2 ∧ l N.1 N.2 ∧ M ≠ N ∧
  Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) + Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2) = 12 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_is_12_sqrt_2_l1746_174653


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1746_174633

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1746_174633


namespace NUMINAMATH_CALUDE_no_primes_in_range_l1746_174696

theorem no_primes_in_range (n : ℕ) (hn : n > 2) :
  ∀ p, Prime p → ¬(n.factorial + 2 < p ∧ p < n.factorial + 2*n) :=
sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l1746_174696


namespace NUMINAMATH_CALUDE_max_sum_of_four_numbers_l1746_174631

theorem max_sum_of_four_numbers (a b c d : ℕ) : 
  a < b → b < c → c < d → 
  (b + d) + (c + d) + (a + b + c) + (a + b + d) = 2017 →
  a + b + c + d ≤ 1006 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_four_numbers_l1746_174631


namespace NUMINAMATH_CALUDE_prime_diff_perfect_square_pairs_l1746_174626

theorem prime_diff_perfect_square_pairs (m n : ℕ+) (p : ℕ) :
  p.Prime →
  m - n = p →
  ∃ k : ℕ, m * n = k^2 →
  p % 2 = 1 ∧ m = ((p + 1)^2 / 4 : ℕ) ∧ n = ((p - 1)^2 / 4 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_prime_diff_perfect_square_pairs_l1746_174626


namespace NUMINAMATH_CALUDE_solomon_collected_66_cans_l1746_174639

/-- The number of cans collected by Solomon, Juwan, and Levi -/
structure CanCollection where
  solomon : ℕ
  juwan : ℕ
  levi : ℕ

/-- The conditions of the can collection problem -/
def validCollection (c : CanCollection) : Prop :=
  c.solomon = 3 * c.juwan ∧
  c.levi = c.juwan / 2 ∧
  c.solomon + c.juwan + c.levi = 99

/-- Theorem stating that Solomon collected 66 cans -/
theorem solomon_collected_66_cans :
  ∃ (c : CanCollection), validCollection c ∧ c.solomon = 66 := by
  sorry

end NUMINAMATH_CALUDE_solomon_collected_66_cans_l1746_174639


namespace NUMINAMATH_CALUDE_problem_statement_l1746_174611

theorem problem_statement (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  let expr1 := (1 + a*b)/(a - b) * (1 + b*c)/(b - c) + 
               (1 + b*c)/(b - c) * (1 + c*a)/(c - a) + 
               (1 + c*a)/(c - a) * (1 + a*b)/(a - b)
  let expr2 := (1 - a*b)/(a - b) * (1 - b*c)/(b - c) + 
               (1 - b*c)/(b - c) * (1 - c*a)/(c - a) + 
               (1 - c*a)/(c - a) * (1 - a*b)/(a - b)
  let expr3 := (1 + a^2*b^2)/(a - b)^2 + (1 + b^2*c^2)/(b - c)^2 + (1 + c^2*a^2)/(c - a)^2
  (expr1 = 1) ∧ 
  (expr2 = -1) ∧ 
  (expr3 ≥ (3/2)) ∧ 
  (expr3 = (3/2) ↔ a = b ∨ b = c ∨ c = a) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1746_174611


namespace NUMINAMATH_CALUDE_beckys_necklaces_l1746_174637

theorem beckys_necklaces (initial_count : ℕ) (broken : ℕ) (new_purchases : ℕ) (final_count : ℕ)
  (h1 : initial_count = 50)
  (h2 : broken = 3)
  (h3 : new_purchases = 5)
  (h4 : final_count = 37) :
  initial_count - broken + new_purchases - final_count = 15 := by
  sorry

end NUMINAMATH_CALUDE_beckys_necklaces_l1746_174637


namespace NUMINAMATH_CALUDE_correct_factorization_l1746_174622

theorem correct_factorization (x y : ℝ) : x^2 - 2*x*y + x = x*(x - 2*y + 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1746_174622


namespace NUMINAMATH_CALUDE_tanya_plums_l1746_174642

/-- The number of plums Tanya bought at the grocery store -/
def plums : ℕ := 6

/-- The total number of pears, apples, and pineapples Tanya bought -/
def other_fruits : ℕ := 12

/-- The number of fruits remaining in the bag after half fell out -/
def remaining_fruits : ℕ := 9

theorem tanya_plums :
  plums = remaining_fruits * 2 - other_fruits :=
by sorry

end NUMINAMATH_CALUDE_tanya_plums_l1746_174642


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l1746_174601

theorem quadratic_root_implies_a_value (a : ℝ) : 
  (∃ (z : ℂ), z = 2 + I ∧ z^2 - 4*z + a = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l1746_174601


namespace NUMINAMATH_CALUDE_parentheses_removal_l1746_174605

theorem parentheses_removal (a b c : ℝ) : a - (b - c) = a - b + c := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l1746_174605


namespace NUMINAMATH_CALUDE_lost_revenue_calculation_l1746_174695

/-- Represents the revenue calculation for a movie theater --/
def theater_revenue (capacity : ℕ) (general_price : ℚ) (child_price : ℚ) (senior_price : ℚ) 
  (veteran_discount : ℚ) (general_sold : ℕ) (child_sold : ℕ) (senior_sold : ℕ) (veteran_sold : ℕ) : ℚ :=
  let actual_revenue := general_sold * general_price + child_sold * child_price + 
                        senior_sold * senior_price + veteran_sold * (general_price - veteran_discount)
  let max_potential_revenue := capacity * general_price
  max_potential_revenue - actual_revenue

/-- Theorem stating the lost revenue for the given scenario --/
theorem lost_revenue_calculation : 
  theater_revenue 50 10 6 8 2 20 3 4 2 = 234 := by sorry

end NUMINAMATH_CALUDE_lost_revenue_calculation_l1746_174695


namespace NUMINAMATH_CALUDE_five_ps_high_gpa_l1746_174603

/-- Represents the number of applicants satisfying various criteria in a law school application process. -/
structure Applicants where
  total : ℕ
  political_science : ℕ
  high_gpa : ℕ
  not_ps_low_gpa : ℕ

/-- Calculates the number of applicants who majored in political science and had a GPA higher than 3.0. -/
def political_science_and_high_gpa (a : Applicants) : ℕ :=
  a.high_gpa - (a.total - a.political_science - a.not_ps_low_gpa)

/-- Theorem stating that for the given applicant data, 5 applicants majored in political science and had a GPA higher than 3.0. -/
theorem five_ps_high_gpa (a : Applicants) 
    (h_total : a.total = 40)
    (h_ps : a.political_science = 15)
    (h_high_gpa : a.high_gpa = 20)
    (h_not_ps_low_gpa : a.not_ps_low_gpa = 10) :
    political_science_and_high_gpa a = 5 := by
  sorry

#eval political_science_and_high_gpa ⟨40, 15, 20, 10⟩

end NUMINAMATH_CALUDE_five_ps_high_gpa_l1746_174603


namespace NUMINAMATH_CALUDE_sequence_formula_l1746_174625

theorem sequence_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, S n = (1/3) * (a n - 1)) :
  ∀ n : ℕ+, a n = n + 1 := by sorry

end NUMINAMATH_CALUDE_sequence_formula_l1746_174625


namespace NUMINAMATH_CALUDE_min_sum_of_dimensions_l1746_174679

def is_valid_box (l w h : ℕ) : Prop :=
  l > 0 ∧ w > 0 ∧ h > 0 ∧ l * w * h = 2310

theorem min_sum_of_dimensions :
  ∃ (l w h : ℕ), is_valid_box l w h ∧
  ∀ (a b c : ℕ), is_valid_box a b c → l + w + h ≤ a + b + c ∧
  l + w + h = 48 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_dimensions_l1746_174679


namespace NUMINAMATH_CALUDE_no_integer_solution_l1746_174644

theorem no_integer_solution (m n p : ℤ) :
  m + n * Real.sqrt 2 + p * Real.sqrt 3 = 0 → m = 0 ∧ n = 0 ∧ p = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1746_174644


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l1746_174681

/-- Represents a rectangular field with given area and width-length relationship -/
structure RectangularField where
  area : ℕ
  width_length_diff : ℕ

/-- Checks if the given length and width satisfy the conditions for a rectangular field -/
def is_valid_dimensions (field : RectangularField) (length width : ℕ) : Prop :=
  length * width = field.area ∧ length = width + field.width_length_diff

theorem rectangular_field_dimensions (field : RectangularField) 
  (h : field.area = 864 ∧ field.width_length_diff = 12) :
  ∃ (length width : ℕ), is_valid_dimensions field length width ∧ length = 36 ∧ width = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l1746_174681


namespace NUMINAMATH_CALUDE_equation_solution_exists_l1746_174608

theorem equation_solution_exists (a : Real) : 
  a ∈ Set.Icc 0.5 1.5 →
  ∃ t ∈ Set.Icc 0 (Real.pi / 2), 
    (abs (Real.cos t - 0.5) + abs (Real.sin t) - a) / (Real.sqrt 3 * Real.sin t - Real.cos t) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l1746_174608


namespace NUMINAMATH_CALUDE_polynomial_solution_set_l1746_174628

theorem polynomial_solution_set : ∃ (S : Set ℂ), 
  S = {z : ℂ | z^4 + 2*z^3 + 2*z^2 + 2*z + 1 = 0} ∧ 
  S = {-1, Complex.I, -Complex.I} := by
  sorry

end NUMINAMATH_CALUDE_polynomial_solution_set_l1746_174628


namespace NUMINAMATH_CALUDE_root_value_l1746_174693

/-- Given a quadratic equation kx^2 + 2x + 5 = 0 with roots C and D, prove that if C = 10, then D = -2 -/
theorem root_value (k : ℝ) (C D : ℝ) : 
  k * C^2 + 2 * C + 5 = 0 →
  k * D^2 + 2 * D + 5 = 0 →
  C = 10 →
  D = -2 := by
sorry

end NUMINAMATH_CALUDE_root_value_l1746_174693


namespace NUMINAMATH_CALUDE_price_of_car_is_five_l1746_174656

/-- Calculates the price of one little car given the total earnings, cost of Legos, and number of cars sold. -/
def price_of_one_car (total_earnings : ℕ) (legos_cost : ℕ) (num_cars : ℕ) : ℚ :=
  (total_earnings - legos_cost : ℚ) / num_cars

/-- Theorem stating that the price of one little car is $5 given the problem conditions. -/
theorem price_of_car_is_five :
  price_of_one_car 45 30 3 = 5 := by
  sorry

#eval price_of_one_car 45 30 3

end NUMINAMATH_CALUDE_price_of_car_is_five_l1746_174656


namespace NUMINAMATH_CALUDE_cat_bowl_refill_days_l1746_174662

theorem cat_bowl_refill_days (empty_bowl_weight : ℝ) (daily_food : ℝ) (weight_after_eating : ℝ) (eaten_amount : ℝ) :
  empty_bowl_weight = 420 →
  daily_food = 60 →
  weight_after_eating = 586 →
  eaten_amount = 14 →
  (weight_after_eating + eaten_amount - empty_bowl_weight) / daily_food = 3 := by
  sorry

end NUMINAMATH_CALUDE_cat_bowl_refill_days_l1746_174662


namespace NUMINAMATH_CALUDE_membership_fee_increase_l1746_174660

/-- Proves that the yearly increase in membership fee is $10 given the initial and final fees -/
theorem membership_fee_increase
  (initial_fee : ℕ)
  (final_fee : ℕ)
  (initial_year : ℕ)
  (final_year : ℕ)
  (h1 : initial_fee = 80)
  (h2 : final_fee = 130)
  (h3 : initial_year = 1)
  (h4 : final_year = 6)
  (h5 : final_fee = initial_fee + (final_year - initial_year) * (yearly_increase : ℕ)) :
  yearly_increase = 10 := by
  sorry

end NUMINAMATH_CALUDE_membership_fee_increase_l1746_174660


namespace NUMINAMATH_CALUDE_evaluate_expression_max_value_function_max_value_function_achievable_l1746_174609

-- Part 1
theorem evaluate_expression : 
  Real.sqrt 3 * Real.cos (π / 12) - Real.sin (π / 12) = Real.sqrt 2 := by sorry

-- Part 2
theorem max_value_function : 
  ∀ θ : ℝ, Real.sqrt 3 * Real.cos θ - Real.sin θ ≤ 2 := by sorry

theorem max_value_function_achievable : 
  ∃ θ : ℝ, Real.sqrt 3 * Real.cos θ - Real.sin θ = 2 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_max_value_function_max_value_function_achievable_l1746_174609


namespace NUMINAMATH_CALUDE_line_slope_equals_k_l1746_174675

/-- 
Given a line passing through points (-1, -4) and (4, k),
if the slope of the line is equal to k, then k = 1.
-/
theorem line_slope_equals_k (k : ℝ) : 
  (k - (-4)) / (4 - (-1)) = k → k = 1 := by sorry

end NUMINAMATH_CALUDE_line_slope_equals_k_l1746_174675


namespace NUMINAMATH_CALUDE_clives_box_balls_l1746_174620

/-- The number of balls in Clive's box -/
def total_balls (blue red green yellow : ℕ) : ℕ := blue + red + green + yellow

/-- Theorem: The total number of balls in Clive's box is 36 -/
theorem clives_box_balls : 
  ∃ (blue red green yellow : ℕ),
    blue = 6 ∧ 
    red = 4 ∧ 
    green = 3 * blue ∧ 
    yellow = 2 * red ∧ 
    total_balls blue red green yellow = 36 := by
  sorry

end NUMINAMATH_CALUDE_clives_box_balls_l1746_174620


namespace NUMINAMATH_CALUDE_polar_equation_is_circle_l1746_174618

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 1 / (1 - Real.sin θ)

-- Define the Cartesian equation derived from the polar equation
def cartesian_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 0

-- Theorem stating that the equation represents a circle (point)
theorem polar_equation_is_circle :
  ∃! (x y : ℝ), cartesian_equation x y ∧ x = 0 ∧ y = 1 :=
sorry

end NUMINAMATH_CALUDE_polar_equation_is_circle_l1746_174618


namespace NUMINAMATH_CALUDE_point_opposite_sides_range_l1746_174692

/-- Determines if two points are on opposite sides of a line -/
def oppositeSides (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  (a * x1 + b * y1 + c) * (a * x2 + b * y2 + c) < 0

theorem point_opposite_sides_range (a : ℝ) :
  oppositeSides 3 1 (-4) 6 3 (-2) a ↔ -7 < a ∧ a < 24 := by
  sorry

end NUMINAMATH_CALUDE_point_opposite_sides_range_l1746_174692


namespace NUMINAMATH_CALUDE_combined_work_time_l1746_174663

/-- The time taken for three people to complete a task together, given their individual rates --/
theorem combined_work_time (rate_shawn rate_karen rate_alex : ℚ) 
  (h_shawn : rate_shawn = 1 / 18)
  (h_karen : rate_karen = 1 / 12)
  (h_alex : rate_alex = 1 / 15) :
  1 / (rate_shawn + rate_karen + rate_alex) = 180 / 37 :=
by sorry

end NUMINAMATH_CALUDE_combined_work_time_l1746_174663


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l1746_174632

theorem smallest_n_divisible (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(24 ∣ m^2 ∧ 864 ∣ m^3)) ∧ 
  (24 ∣ n^2) ∧ (864 ∣ n^3) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l1746_174632


namespace NUMINAMATH_CALUDE_eliminate_cycles_in_complete_digraph_l1746_174685

/-- A complete directed graph with 32 vertices -/
def CompleteDigraph : Type := Fin 32 → Fin 32 → Prop

/-- The property that a graph contains no directed cycles -/
def NoCycles (g : CompleteDigraph) : Prop := sorry

/-- A step that changes the direction of a single edge -/
def Step (g₁ g₂ : CompleteDigraph) : Prop := sorry

/-- The theorem stating that it's possible to eliminate all cycles in at most 208 steps -/
theorem eliminate_cycles_in_complete_digraph :
  ∃ (sequence : Fin 209 → CompleteDigraph),
    (∀ i : Fin 208, Step (sequence i) (sequence (i + 1))) ∧
    NoCycles (sequence 208) :=
  sorry

end NUMINAMATH_CALUDE_eliminate_cycles_in_complete_digraph_l1746_174685


namespace NUMINAMATH_CALUDE_matrix_commutation_result_l1746_174614

/-- Given two 2x2 matrices A and B, where A is fixed and B has variable entries,
    prove that if AB = BA and 4y ≠ z, then (x - w) / (z - 4y) = 0. -/
theorem matrix_commutation_result (x y z w : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (A * B = B * A) → (4 * y ≠ z) → (x - w) / (z - 4 * y) = 0 := by
  sorry


end NUMINAMATH_CALUDE_matrix_commutation_result_l1746_174614


namespace NUMINAMATH_CALUDE_magnitude_of_b_is_one_l1746_174661

/-- Given two vectors a and b in ℝ², prove that the magnitude of b is 1 -/
theorem magnitude_of_b_is_one (a b : ℝ × ℝ) : 
  (Real.cos (60 * π / 180) = a.fst * b.fst + a.snd * b.snd) →  -- angle between a and b is 60°
  (a.fst^2 + a.snd^2 = 1) →  -- |a| = 1
  ((2*a.fst - b.fst)^2 + (2*a.snd - b.snd)^2 = 3) →  -- |2a - b| = √3
  (b.fst^2 + b.snd^2 = 1) :=  -- |b| = 1
by sorry

end NUMINAMATH_CALUDE_magnitude_of_b_is_one_l1746_174661


namespace NUMINAMATH_CALUDE_cube_volume_problem_l1746_174650

theorem cube_volume_problem (x : ℝ) (h : x > 0) :
  (x - 2) * x * (x + 2) = x^3 - 10 → x^3 = 15.625 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l1746_174650


namespace NUMINAMATH_CALUDE_translation_of_complex_plane_l1746_174651

theorem translation_of_complex_plane (t : ℂ → ℂ) :
  (t (1 + 3*I) = 4 - 2*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = z + w) →
  (t (2 - I) = 5 - 6*I) := by
sorry

end NUMINAMATH_CALUDE_translation_of_complex_plane_l1746_174651


namespace NUMINAMATH_CALUDE_hypotenuse_increase_bound_l1746_174669

theorem hypotenuse_increase_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt ((x + 1)^2 + (y + 1)^2) ≤ Real.sqrt (x^2 + y^2) + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_increase_bound_l1746_174669


namespace NUMINAMATH_CALUDE_wage_payment_days_l1746_174691

theorem wage_payment_days (S : ℝ) (hX : S > 0) (hY : S > 0) : 
  (∃ (wX wY : ℝ), wX > 0 ∧ wY > 0 ∧ S = 36 * wX ∧ S = 45 * wY) →
  ∃ (d : ℝ), d = 20 ∧ S = d * (S / 36 + S / 45) :=
by sorry

end NUMINAMATH_CALUDE_wage_payment_days_l1746_174691


namespace NUMINAMATH_CALUDE_slower_speed_percentage_l1746_174629

theorem slower_speed_percentage (usual_time slower_time : ℝ) 
  (h1 : usual_time = 8)
  (h2 : slower_time = usual_time + 24) :
  (usual_time / slower_time) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_slower_speed_percentage_l1746_174629


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1746_174604

theorem min_value_on_circle (x y : ℝ) : 
  (x - 1)^2 + (y - 2)^2 = 9 → y ≥ 2 → x + Real.sqrt 3 * y ≥ 2 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1746_174604


namespace NUMINAMATH_CALUDE_probability_all_white_balls_l1746_174623

def white_balls : ℕ := 7
def black_balls : ℕ := 7
def total_balls : ℕ := white_balls + black_balls
def num_draws : ℕ := 6

theorem probability_all_white_balls :
  (white_balls : ℚ) / total_balls ^ num_draws = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_all_white_balls_l1746_174623


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1746_174673

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1746_174673


namespace NUMINAMATH_CALUDE_point_coordinates_on_horizontal_line_l1746_174617

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line parallel to the x-axis -/
structure HorizontalLine where
  y : ℝ

def Point.liesOn (p : Point) (l : HorizontalLine) : Prop :=
  p.y = l.y

theorem point_coordinates_on_horizontal_line 
  (m : ℝ)
  (P : Point)
  (A : Point)
  (l : HorizontalLine)
  (h1 : P = ⟨2*m + 4, m - 1⟩)
  (h2 : A = ⟨2, -4⟩)
  (h3 : l.y = A.y)
  (h4 : P.liesOn l) :
  P = ⟨-2, -4⟩ :=
sorry

end NUMINAMATH_CALUDE_point_coordinates_on_horizontal_line_l1746_174617


namespace NUMINAMATH_CALUDE_sum_representation_exists_l1746_174698

/-- Regular 15-gon inscribed in a circle -/
structure RegularPolygon :=
  (n : ℕ)
  (radius : ℝ)
  (is_regular : n = 15)
  (is_inscribed : radius = 15)

/-- Sum of lengths of all sides and diagonals -/
def sum_lengths (p : RegularPolygon) : ℝ := sorry

/-- Representation of the sum in the required form -/
structure SumRepresentation :=
  (a b c d : ℕ)
  (sum : ℝ)
  (eq : sum = a + b * Real.sqrt 2 + c * Real.sqrt 3 + d * Real.sqrt 5)

/-- Theorem stating the existence of the representation -/
theorem sum_representation_exists (p : RegularPolygon) :
  ∃ (rep : SumRepresentation), sum_lengths p = rep.sum :=
sorry

end NUMINAMATH_CALUDE_sum_representation_exists_l1746_174698


namespace NUMINAMATH_CALUDE_inequalities_always_hold_l1746_174627

theorem inequalities_always_hold :
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → c / a < c / b) ∧
  (∀ a b : ℝ, (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2) ∧
  (∀ a b : ℝ, a + b ≤ Real.sqrt (2 * (a^2 + b^2))) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_always_hold_l1746_174627


namespace NUMINAMATH_CALUDE_sin_x_bounds_l1746_174649

theorem sin_x_bounds (x : ℝ) (h : 0 < x) (h' : x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := by
  sorry

end NUMINAMATH_CALUDE_sin_x_bounds_l1746_174649


namespace NUMINAMATH_CALUDE_triangle_inequality_with_circumradius_l1746_174634

-- Define a structure for a triangle with its circumradius
structure Triangle :=
  (a b c : ℝ)
  (R : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hR : R > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (circumradius : R = (a * b * c) / (4 * area))
  (area : ℝ)
  (area_positive : area > 0)

-- State the theorem
theorem triangle_inequality_with_circumradius (t : Triangle) :
  1 / (t.a * t.b) + 1 / (t.b * t.c) + 1 / (t.c * t.a) ≥ 1 / (t.R ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_circumradius_l1746_174634


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l1746_174694

/-- Given a circle with center (4, 2) and one endpoint of its diameter at (7, 5),
    prove that the other endpoint of the diameter is at (1, -1). -/
theorem circle_diameter_endpoint (center : ℝ × ℝ) (endpoint_a : ℝ × ℝ) (endpoint_b : ℝ × ℝ) :
  center = (4, 2) →
  endpoint_a = (7, 5) →
  (center.1 - endpoint_a.1 = endpoint_b.1 - center.1 ∧
   center.2 - endpoint_a.2 = endpoint_b.2 - center.2) →
  endpoint_b = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l1746_174694


namespace NUMINAMATH_CALUDE_square_area_l1746_174643

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

end NUMINAMATH_CALUDE_square_area_l1746_174643


namespace NUMINAMATH_CALUDE_four_digit_permutations_eq_six_l1746_174616

/-- The number of different positive, four-digit integers that can be formed using the digits 3, 3, 8, and 8 -/
def four_digit_permutations : ℕ :=
  Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive, four-digit integers
    that can be formed using the digits 3, 3, 8, and 8 is equal to 6 -/
theorem four_digit_permutations_eq_six :
  four_digit_permutations = 6 := by
  sorry

#eval four_digit_permutations

end NUMINAMATH_CALUDE_four_digit_permutations_eq_six_l1746_174616


namespace NUMINAMATH_CALUDE_factorization_problems_l1746_174613

theorem factorization_problems (a x y : ℝ) : 
  (a * (a - 2) + 2 * (a - 2) = (a - 2) * (a + 2)) ∧ 
  (3 * x^2 - 6 * x * y + 3 * y^2 = 3 * (x - y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1746_174613


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l1746_174610

def A (a b c d e : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![a, 1, 0, b;
     0, 3, 2, 0;
     c, 4, d, 5;
     6, 0, 7, e]

def B (f g h : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![-7, f,  0, -15;
      g, -20, h,   0;
      0,  2,  5,   0;
      3,  0,  8,   6]

theorem inverse_matrices_sum (a b c d e f g h : ℝ) :
  A a b c d e * B f g h = 1 →
  a + b + c + d + e + f + g + h = 27 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l1746_174610


namespace NUMINAMATH_CALUDE_total_pamphlets_printed_prove_total_pamphlets_l1746_174641

/-- Calculates the total number of pamphlets printed by Mike and Leo -/
theorem total_pamphlets_printed (mike_initial_speed : ℕ) (mike_initial_hours : ℕ) 
  (mike_additional_hours : ℕ) (leo_speed_multiplier : ℕ) : ℕ :=
  let mike_initial_pamphlets := mike_initial_speed * mike_initial_hours
  let mike_reduced_speed := mike_initial_speed / 3
  let mike_additional_pamphlets := mike_reduced_speed * mike_additional_hours
  let leo_hours := mike_initial_hours / 3
  let leo_speed := mike_initial_speed * leo_speed_multiplier
  let leo_pamphlets := leo_speed * leo_hours
  mike_initial_pamphlets + mike_additional_pamphlets + leo_pamphlets

/-- Proves that Mike and Leo print 9400 pamphlets in total -/
theorem prove_total_pamphlets : total_pamphlets_printed 600 9 2 2 = 9400 := by
  sorry

end NUMINAMATH_CALUDE_total_pamphlets_printed_prove_total_pamphlets_l1746_174641


namespace NUMINAMATH_CALUDE_polyhedron_volume_is_twenty_thirds_l1746_174646

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

end NUMINAMATH_CALUDE_polyhedron_volume_is_twenty_thirds_l1746_174646


namespace NUMINAMATH_CALUDE_triangle_inequality_l1746_174606

theorem triangle_inequality (a b c : ℝ) (h : |((a^2 + b^2 - c^2) / (a*b))| < 2) :
  |((b^2 + c^2 - a^2) / (b*c))| < 2 ∧ |((c^2 + a^2 - b^2) / (c*a))| < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1746_174606


namespace NUMINAMATH_CALUDE_min_folders_required_l1746_174670

/-- Represents the types of files --/
inductive FileType
  | PDF
  | Word
  | PPT

/-- Represents the initial file counts --/
structure InitialFiles where
  pdf : Nat
  word : Nat
  ppt : Nat

/-- Represents the deleted file counts --/
structure DeletedFiles where
  pdf : Nat
  ppt : Nat

/-- Calculates the remaining files after deletion --/
def remainingFiles (initial : InitialFiles) (deleted : DeletedFiles) : Nat :=
  initial.pdf + initial.word + initial.ppt - deleted.pdf - deleted.ppt

/-- Represents the folder allocation problem --/
structure FolderAllocationProblem where
  initial : InitialFiles
  deleted : DeletedFiles
  folderCapacity : Nat
  wordImportance : Nat

/-- Theorem: The minimum number of folders required is 6 --/
theorem min_folders_required (problem : FolderAllocationProblem)
  (h1 : problem.initial = ⟨43, 30, 30⟩)
  (h2 : problem.deleted = ⟨33, 30⟩)
  (h3 : problem.folderCapacity = 7)
  (h4 : problem.wordImportance = 2) :
  let remainingWordFiles := problem.initial.word
  let remainingPDFFiles := problem.initial.pdf - problem.deleted.pdf
  let totalRemainingFiles := remainingFiles problem.initial problem.deleted
  let minFolders := 
    (remainingWordFiles / problem.folderCapacity) +
    ((remainingWordFiles % problem.folderCapacity + remainingPDFFiles + problem.folderCapacity - 1) / problem.folderCapacity)
  minFolders = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_folders_required_l1746_174670


namespace NUMINAMATH_CALUDE_cosine_ratio_sum_bound_l1746_174687

/-- For an acute triangle ABC with angles α, β, and γ, 
    the sum of certain cosine ratios is at least 3/2 -/
theorem cosine_ratio_sum_bound (α β γ : Real) 
  (acute_triangle : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) :
  (Real.cos (β - γ) / Real.cos α) + 
  (Real.cos (γ - α) / Real.cos β) + 
  (Real.cos (α - β) / Real.cos γ) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_ratio_sum_bound_l1746_174687
