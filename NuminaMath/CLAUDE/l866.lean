import Mathlib

namespace yellow_block_weight_l866_86696

theorem yellow_block_weight (green_weight : ℝ) (weight_difference : ℝ) 
  (h1 : green_weight = 0.4)
  (h2 : weight_difference = 0.2) : 
  green_weight + weight_difference = 0.6 :=
by sorry

end yellow_block_weight_l866_86696


namespace plane_sphere_intersection_ratio_sum_l866_86623

/-- The theorem states that for a plane intersecting the coordinate axes and a sphere passing through these intersection points and the origin, the sum of the ratios of a point on the plane to the sphere's center coordinates is 2. -/
theorem plane_sphere_intersection_ratio_sum (k : ℝ) (a b c p q r : ℝ) : 
  k ≠ 0 → -- k is a non-zero constant
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 → -- p, q, r are non-zero (as they are denominators)
  (∃ (α β γ : ℝ), α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0 ∧ -- A, B, C exist and are distinct from O
    (k*a/α + k*b/β + k*c/γ = 1) ∧ -- plane equation
    (p^2 + q^2 + r^2 = (p - α)^2 + q^2 + r^2) ∧ -- sphere equation for A
    (p^2 + q^2 + r^2 = p^2 + (q - β)^2 + r^2) ∧ -- sphere equation for B
    (p^2 + q^2 + r^2 = p^2 + q^2 + (r - γ)^2)) → -- sphere equation for C
  k*a/p + k*b/q + k*c/r = 2 := by
sorry

end plane_sphere_intersection_ratio_sum_l866_86623


namespace sleeping_passenger_journey_l866_86628

theorem sleeping_passenger_journey (total_journey : ℝ) (sleeping_distance : ℝ) :
  (sleeping_distance = total_journey / 3) ∧
  (total_journey / 2 = sleeping_distance + sleeping_distance / 2) →
  sleeping_distance / total_journey = 1 / 3 :=
by sorry

end sleeping_passenger_journey_l866_86628


namespace train_passing_time_train_passing_time_specific_l866_86616

/-- Time for a train to pass a tree -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (wind_speed : ℝ) : ℝ :=
  let train_speed_ms := train_speed * 1000 / 3600
  let wind_speed_ms := wind_speed * 1000 / 3600
  let effective_speed := train_speed_ms - wind_speed_ms
  train_length / effective_speed

/-- Proof that the time for a train of length 850 m, traveling at 85 km/hr against a 5 km/hr wind, to pass a tree is approximately 38.25 seconds -/
theorem train_passing_time_specific : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_passing_time 850 85 5 - 38.25| < ε :=
sorry

end train_passing_time_train_passing_time_specific_l866_86616


namespace pascal_triangle_prob_one_l866_86690

/-- The number of rows in Pascal's Triangle we're considering -/
def n : ℕ := 20

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def total_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1s in the first n rows of Pascal's Triangle -/
def ones_count (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def prob_one (n : ℕ) : ℚ := (ones_count n : ℚ) / (total_elements n : ℚ)

theorem pascal_triangle_prob_one : 
  prob_one n = 39 / 210 := by sorry

end pascal_triangle_prob_one_l866_86690


namespace solve_equation_one_solve_equation_two_l866_86645

-- Problem 1
theorem solve_equation_one (x : ℝ) : 4 * x^2 = 25 ↔ x = 5/2 ∨ x = -5/2 := by sorry

-- Problem 2
theorem solve_equation_two (x : ℝ) : (x + 1)^3 - 8 = 56 ↔ x = 3 := by sorry

end solve_equation_one_solve_equation_two_l866_86645


namespace percent_of_x_l866_86608

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 5 + x / 25) / x * 100 = 24 := by
  sorry

end percent_of_x_l866_86608


namespace cyclic_inequality_with_powers_l866_86640

theorem cyclic_inequality_with_powers (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) (h₆ : x₆ > 0) :
  (x₂/x₁)^5 + (x₄/x₂)^5 + (x₆/x₃)^5 + (x₁/x₄)^5 + (x₃/x₅)^5 + (x₅/x₆)^5 ≥ 
  x₁/x₂ + x₂/x₄ + x₃/x₆ + x₄/x₁ + x₅/x₃ + x₆/x₅ := by
  sorry

end cyclic_inequality_with_powers_l866_86640


namespace det_sum_of_matrices_l866_86651

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 6; 2, 3]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 1, 0]

theorem det_sum_of_matrices : Matrix.det (A + B) = -3 := by sorry

end det_sum_of_matrices_l866_86651


namespace acute_angles_equation_solution_l866_86699

theorem acute_angles_equation_solution (A B : Real) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  Real.sin A * Real.cos B + Real.sqrt (2 * Real.sin A) * Real.sin B = (3 * Real.sin A + 1) / Real.sqrt 5 →
  A = π/6 ∧ B = π/2 - Real.arcsin (Real.sqrt 5 / 5) := by
  sorry

end acute_angles_equation_solution_l866_86699


namespace simplify_expression_l866_86606

theorem simplify_expression : 
  2 - (2 / (2 + Real.sqrt 5)) + (2 / (2 - Real.sqrt 5)) = 2 - 4 * Real.sqrt 5 := by
  sorry

end simplify_expression_l866_86606


namespace system_solution_unique_l866_86607

theorem system_solution_unique :
  ∃! (x y z : ℝ),
    x^2 - 2*y + 1 = 0 ∧
    y^2 - 4*z + 7 = 0 ∧
    z^2 + 2*x - 2 = 0 ∧
    x = -1 ∧ y = 1 ∧ z = 2 := by
  sorry

end system_solution_unique_l866_86607


namespace dogs_accessible_area_l866_86615

theorem dogs_accessible_area (s : ℝ) (s_pos : s > 0) :
  let square_area := (2 * s) ^ 2
  let circle_area := π * s ^ 2
  circle_area / square_area = π / 4 := by
  sorry

#check dogs_accessible_area

end dogs_accessible_area_l866_86615


namespace general_term_formula_l866_86659

/-- The sequence term for a given positive integer n -/
def a (n : ℕ+) : ℚ :=
  (4 * n^2 + n - 1) / (2 * n + 1)

/-- The first part of each term in the sequence -/
def b (n : ℕ+) : ℕ :=
  2 * n - 1

/-- The second part of each term in the sequence -/
def c (n : ℕ+) : ℚ :=
  n / (2 * n + 1)

/-- Theorem stating that the general term formula is correct -/
theorem general_term_formula (n : ℕ+) : 
  a n = (b n : ℚ) + c n :=
sorry

end general_term_formula_l866_86659


namespace contradiction_proof_l866_86622

theorem contradiction_proof (a b : ℕ) : a < 2 → b < 2 → a + b < 3 := by
  sorry

end contradiction_proof_l866_86622


namespace average_marks_second_class_l866_86681

/-- Theorem: Average marks of second class --/
theorem average_marks_second_class 
  (n₁ : ℕ) (n₂ : ℕ) (avg₁ : ℝ) (avg_total : ℝ) :
  n₁ = 55 →
  n₂ = 48 →
  avg₁ = 60 →
  avg_total = 59.067961165048544 →
  let avg₂ := ((n₁ + n₂ : ℝ) * avg_total - n₁ * avg₁) / n₂
  ∃ ε > 0, |avg₂ - 57.92| < ε :=
by sorry

end average_marks_second_class_l866_86681


namespace smallest_n_for_integer_S_l866_86680

def b : ℕ := 8

-- S_n is the sum of reciprocals of non-zero digits of integers from 1 to b^n
def S (n : ℕ) : ℚ :=
  -- We don't implement the actual sum here, just define its signature
  sorry

-- Predicate to check if a number is an integer
def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

-- Main theorem
theorem smallest_n_for_integer_S :
  ∀ n : ℕ, n > 0 → is_integer (S n) → n ≥ 105 :=
sorry

end smallest_n_for_integer_S_l866_86680


namespace solve_cubic_equation_l866_86674

theorem solve_cubic_equation (t p s : ℝ) : 
  t = 3 * s^3 + 2 * p → t = 29 → p = 3 → s = (23/3)^(1/3) :=
by
  sorry

end solve_cubic_equation_l866_86674


namespace geometric_sequence_common_ratio_l866_86644

/-- A geometric sequence with first four terms 25, -50, 100, -200 has a common ratio of -2. -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℚ), 
    (a 0 = 25) → 
    (a 1 = -50) → 
    (a 2 = 100) → 
    (a 3 = -200) → 
    (∀ n : ℕ, a (n + 1) = a n * (-2)) → 
    (∀ n : ℕ, a (n + 1) / a n = -2) :=
by sorry

end geometric_sequence_common_ratio_l866_86644


namespace problem_1_problem_2_l866_86639

-- Problem 1
theorem problem_1 : (-1)^4 - 2 * Real.tan (60 * π / 180) + (Real.sqrt 3 - Real.sqrt 2)^0 + Real.sqrt 12 = 2 := by
  sorry

-- Problem 2
theorem problem_2 : ∀ x : ℝ, (x - 1) / 3 ≥ x / 2 - 2 ↔ x ≤ 10 := by
  sorry

end problem_1_problem_2_l866_86639


namespace cos_negative_300_degrees_l866_86613

theorem cos_negative_300_degrees :
  Real.cos ((-300 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end cos_negative_300_degrees_l866_86613


namespace new_person_weight_is_75_l866_86626

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (initial_count : ℕ) (average_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the new person is 75 kg -/
theorem new_person_weight_is_75 :
  new_person_weight 8 (5/2) 55 = 75 := by
  sorry

end new_person_weight_is_75_l866_86626


namespace arithmetic_sequence_nth_term_l866_86661

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2_a5 : a 2 + a 5 = 12)
  (h_an : ∃ n : ℕ, a n = 25) :
  ∃ n : ℕ, n = 13 ∧ a n = 25 :=
sorry

end arithmetic_sequence_nth_term_l866_86661


namespace hockey_team_selection_l866_86624

def number_of_players : ℕ := 18
def players_to_select : ℕ := 8

theorem hockey_team_selection :
  Nat.choose number_of_players players_to_select = 43758 := by
  sorry

end hockey_team_selection_l866_86624


namespace circles_intersection_condition_l866_86685

/-- Two circles in the xy-plane -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y + 1 = 0

def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - m = 0

/-- The circles intersect -/
def circles_intersect (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle1 x y ∧ circle2 x y m

/-- Theorem stating the condition for the circles to intersect -/
theorem circles_intersection_condition :
  ∀ m : ℝ, circles_intersect m ↔ -1 < m ∧ m < 79 :=
sorry

end circles_intersection_condition_l866_86685


namespace hyperbola_range_l866_86653

-- Define the equation
def equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + (2 - m) * y^2 = 1

-- Define what it means for the equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  m ≠ 0 ∧ m ≠ 2 ∧ m * (2 - m) < 0

-- Theorem statement
theorem hyperbola_range (m : ℝ) :
  is_hyperbola m ↔ m < 0 ∨ m > 2 :=
by sorry


end hyperbola_range_l866_86653


namespace boxes_per_day_calculation_l866_86688

/-- The number of apples packed in a box -/
def apples_per_box : ℕ := 40

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The total number of apples packed in two weeks -/
def total_apples : ℕ := 24500

/-- The number of fewer apples packed per day in the second week -/
def fewer_apples_per_day : ℕ := 500

/-- The number of full boxes produced per day -/
def boxes_per_day : ℕ := 50

theorem boxes_per_day_calculation :
  boxes_per_day * apples_per_box * days_per_week +
  (boxes_per_day * apples_per_box - fewer_apples_per_day) * days_per_week = total_apples :=
by sorry

end boxes_per_day_calculation_l866_86688


namespace initial_ratio_problem_l866_86643

theorem initial_ratio_problem (a b : ℕ) : 
  b = 6 → 
  (a + 2 : ℚ) / (b + 2 : ℚ) = 3 / 2 → 
  (a : ℚ) / b = 5 / 3 := by
sorry

end initial_ratio_problem_l866_86643


namespace ralph_peanuts_l866_86666

/-- Represents the number of peanuts Ralph starts with -/
def initial_peanuts : ℕ := sorry

/-- Represents the number of peanuts Ralph loses -/
def lost_peanuts : ℕ := 59

/-- Represents the number of peanuts Ralph ends up with -/
def final_peanuts : ℕ := 15

/-- Theorem stating that Ralph started with 74 peanuts -/
theorem ralph_peanuts : initial_peanuts = 74 :=
by
  sorry

end ralph_peanuts_l866_86666


namespace a_5_equals_10_l866_86676

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

theorem a_5_equals_10 (a : ℕ → ℕ) (h1 : arithmetic_sequence a) (h2 : a 1 = 2) :
  a 5 = 10 := by
  sorry

end a_5_equals_10_l866_86676


namespace problem_solution_l866_86618

def circle_times (a b : ℚ) : ℚ := (a + b) / (a - b)

def circle_plus (a b : ℚ) : ℚ := 2 * (circle_times a b)

theorem problem_solution : circle_plus (circle_plus 8 6) 2 = 8 / 3 := by
  sorry

end problem_solution_l866_86618


namespace peter_initial_erasers_l866_86683

-- Define the variables
def initial_erasers : ℕ := sorry
def received_erasers : ℕ := 3
def final_erasers : ℕ := 11

-- State the theorem
theorem peter_initial_erasers : 
  initial_erasers + received_erasers = final_erasers → initial_erasers = 8 := by
  sorry

end peter_initial_erasers_l866_86683


namespace intersection_of_A_and_B_l866_86657

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | (-1 < x ∧ x < 1) ∨ (3 < x ∧ x < 4)} := by sorry

end intersection_of_A_and_B_l866_86657


namespace article_cost_price_l866_86611

theorem article_cost_price (loss_percentage : Real) (gain_percentage : Real) (price_increase : Real) 
  (cost_price : Real) :
  loss_percentage = 0.15 →
  gain_percentage = 0.125 →
  price_increase = 72.50 →
  (1 - loss_percentage) * cost_price + price_increase = (1 + gain_percentage) * cost_price →
  cost_price = 263.64 := by
sorry

end article_cost_price_l866_86611


namespace text_pages_count_l866_86641

theorem text_pages_count (total_pages : ℕ) (image_pages : ℕ) (intro_pages : ℕ) : 
  total_pages = 98 →
  image_pages = total_pages / 2 →
  intro_pages = 11 →
  (total_pages - image_pages - intro_pages) % 2 = 0 →
  (total_pages - image_pages - intro_pages) / 2 = 19 :=
by sorry

end text_pages_count_l866_86641


namespace line_perp_two_planes_implies_parallel_l866_86698

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_two_planes_implies_parallel 
  (l : Line) (α β : Plane) (h1 : α ≠ β) (h2 : perpendicular l α) (h3 : perpendicular l β) :
  parallel α β := by sorry

end line_perp_two_planes_implies_parallel_l866_86698


namespace square_plus_inverse_square_l866_86621

theorem square_plus_inverse_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) : x^2 + 1/x^2 = 7 := by
  sorry

end square_plus_inverse_square_l866_86621


namespace xyz_congruence_l866_86663

theorem xyz_congruence (x y z : ℕ) : 
  x < 10 → y < 10 → z < 10 → x > 0 → y > 0 → z > 0 →
  (x * y * z) % 9 = 1 →
  (7 * z) % 9 = 4 →
  (8 * y) % 9 = (5 + y) % 9 →
  (x + y + z) % 9 = 2 := by sorry

end xyz_congruence_l866_86663


namespace sum_of_six_consecutive_odd_integers_l866_86631

theorem sum_of_six_consecutive_odd_integers (S : ℤ) :
  (∃ n : ℤ, S = 6*n + 30 ∧ Odd n) ↔ (∃ k : ℤ, S - 30 = 6*k ∧ Odd k) :=
sorry

end sum_of_six_consecutive_odd_integers_l866_86631


namespace cylinder_volume_l866_86669

/-- The volume of a cylinder with specific geometric conditions -/
theorem cylinder_volume (l α β : ℝ) (hl : l > 0) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) :
  ∃ V : ℝ, V = (π * l^3 * Real.sin (2*α) * Real.cos α^3) / (8 * Real.cos (α + β) * Real.cos (α - β)) :=
by sorry

end cylinder_volume_l866_86669


namespace mod_product_equivalence_l866_86687

theorem mod_product_equivalence : ∃ m : ℕ, 
  198 * 955 ≡ m [ZMOD 50] ∧ 0 ≤ m ∧ m < 50 ∧ m = 40 := by
  sorry

end mod_product_equivalence_l866_86687


namespace equation_solutions_l866_86679

def equation (n : ℕ) (x : ℝ) : Prop :=
  (((x + 1)^2)^(1/n : ℝ)) + (((x - 1)^2)^(1/n : ℝ)) = 4 * ((x^2 - 1)^(1/n : ℝ))

theorem equation_solutions :
  (∀ x : ℝ, equation 2 x ↔ x = 2 / Real.sqrt 3 ∨ x = -2 / Real.sqrt 3) ∧
  (∀ x : ℝ, equation 3 x ↔ x = 3 * Real.sqrt 3 / 5 ∨ x = -3 * Real.sqrt 3 / 5) ∧
  (∀ x : ℝ, equation 4 x ↔ x = 7 / (4 * Real.sqrt 3) ∨ x = -7 / (4 * Real.sqrt 3)) :=
by sorry

end equation_solutions_l866_86679


namespace min_value_problem1_l866_86614

theorem min_value_problem1 (x : ℝ) (h : x > 3) : 4 / (x - 3) + x ≥ 7 := by
  sorry

end min_value_problem1_l866_86614


namespace product_remainder_mod_17_l866_86638

theorem product_remainder_mod_17 : (2011 * 2012 * 2013 * 2014 * 2015) % 17 = 7 := by
  sorry

end product_remainder_mod_17_l866_86638


namespace systematic_sampling_first_number_l866_86610

/-- Systematic sampling function -/
def systematicSample (firstSelected : ℕ) (groupSize : ℕ) (groupNumber : ℕ) : ℕ :=
  firstSelected + groupSize * (groupNumber - 1)

theorem systematic_sampling_first_number 
  (totalStudents : ℕ) 
  (sampleSize : ℕ) 
  (selectedNumber : ℕ) 
  (selectedGroup : ℕ) 
  (h1 : totalStudents = 800) 
  (h2 : sampleSize = 50) 
  (h3 : selectedNumber = 503) 
  (h4 : selectedGroup = 32) :
  ∃ (firstSelected : ℕ), 
    firstSelected = 7 ∧ 
    systematicSample firstSelected (totalStudents / sampleSize) selectedGroup = selectedNumber :=
by
  sorry

end systematic_sampling_first_number_l866_86610


namespace no_solution_for_sock_problem_l866_86619

theorem no_solution_for_sock_problem : ¬∃ (n m : ℕ), 
  n + m = 2009 ∧ (n^2 - 2*n*m + m^2) = (n + m) := by
  sorry

end no_solution_for_sock_problem_l866_86619


namespace vectors_opposite_directions_l866_86652

def a (x : ℝ) : ℝ × ℝ := (1, -x)
def b (x : ℝ) : ℝ × ℝ := (x, -16)

theorem vectors_opposite_directions :
  ∃ (k : ℝ), k ≠ 0 ∧ a (-5) = k • b (-5) :=
by sorry

end vectors_opposite_directions_l866_86652


namespace arithmetic_progression_equality_l866_86601

theorem arithmetic_progression_equality (n : ℕ) 
  (hn : n ≥ 2018) 
  (a b : Fin n → ℕ) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → (a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j))
  (h_bound : ∀ i : Fin n, a i ≤ 5*n ∧ b i ≤ 5*n)
  (h_positive : ∀ i : Fin n, a i > 0 ∧ b i > 0)
  (h_arithmetic : ∃ d : ℚ, ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) - (a j : ℚ) / (b j : ℚ) = (i.val - j.val : ℚ) * d) :
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) :=
sorry

end arithmetic_progression_equality_l866_86601


namespace quadratic_inequality_solution_set_l866_86625

theorem quadratic_inequality_solution_set : 
  {x : ℝ | 3 * x^2 - 5 * x - 2 < 0} = {x : ℝ | -1/3 < x ∧ x < 2} := by sorry

end quadratic_inequality_solution_set_l866_86625


namespace problem_statement_l866_86632

theorem problem_statement (x y z a : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : x^2 - 1/y = y^2 - 1/z ∧ y^2 - 1/z = z^2 - 1/x ∧ z^2 - 1/x = a) :
  (x + y + z) * x * y * z = -a^2 := by
  sorry

end problem_statement_l866_86632


namespace pats_calculation_l866_86648

theorem pats_calculation (x : ℝ) : 
  (x / 8) - 20 = 12 → 
  1800 < (x * 8) + 20 ∧ (x * 8) + 20 < 2200 :=
by
  sorry

end pats_calculation_l866_86648


namespace sum_of_circle_areas_l866_86650

/-- Given a triangle with sides 6, 8, and 10 units, formed by the centers of
    three mutually externally tangent circles, the sum of the areas of these
    circles is 56π. -/
theorem sum_of_circle_areas (r s t : ℝ) : 
  r + s = 6 →
  r + t = 8 →
  s + t = 10 →
  π * (r^2 + s^2 + t^2) = 56 * π :=
by sorry

end sum_of_circle_areas_l866_86650


namespace grooming_time_calculation_l866_86604

/-- Proves that the time to clip each claw is 10 seconds given the grooming conditions --/
theorem grooming_time_calculation (total_time : ℕ) (num_claws : ℕ) (ear_cleaning_time : ℕ) (shampoo_time_minutes : ℕ) :
  total_time = 640 →
  num_claws = 16 →
  ear_cleaning_time = 90 →
  shampoo_time_minutes = 5 →
  ∃ (claw_clip_time : ℕ),
    claw_clip_time = 10 ∧
    total_time = num_claws * claw_clip_time + 2 * ear_cleaning_time + shampoo_time_minutes * 60 :=
by sorry

end grooming_time_calculation_l866_86604


namespace units_digit_of_57_to_57_l866_86673

theorem units_digit_of_57_to_57 : (57^57) % 10 = 7 := by
  sorry

end units_digit_of_57_to_57_l866_86673


namespace cos_is_semi_odd_tan_is_semi_odd_l866_86677

-- Definition of a semi-odd function
def is_semi_odd (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = -f (2*a - x)

-- Statement for cos(x+1)
theorem cos_is_semi_odd :
  is_semi_odd (λ x => Real.cos (x + 1)) :=
sorry

-- Statement for tan(x)
theorem tan_is_semi_odd :
  is_semi_odd Real.tan :=
sorry

end cos_is_semi_odd_tan_is_semi_odd_l866_86677


namespace overestimation_correct_l866_86609

/-- The overestimation in cents when y quarters are miscounted as half-dollars and y pennies are miscounted as nickels -/
def overestimation (y : ℕ) : ℕ := 29 * y

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a half-dollar in cents -/
def half_dollar_value : ℕ := 50

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

theorem overestimation_correct (y : ℕ) : 
  overestimation y = 
    y * (half_dollar_value - quarter_value) + 
    y * (nickel_value - penny_value) := by
  sorry

end overestimation_correct_l866_86609


namespace other_root_of_quadratic_l866_86693

/-- If x^2 + 3x + a = 0 has -1 as one of its roots, then the other root is -2 -/
theorem other_root_of_quadratic (a : ℝ) : 
  ((-1 : ℝ)^2 + 3*(-1) + a = 0) → 
  (∃ x : ℝ, x ≠ -1 ∧ x^2 + 3*x + a = 0 ∧ x = -2) :=
by sorry

end other_root_of_quadratic_l866_86693


namespace negative_one_times_negative_three_equals_three_l866_86664

theorem negative_one_times_negative_three_equals_three :
  (-1 : ℤ) * (-3 : ℤ) = (3 : ℤ) := by sorry

end negative_one_times_negative_three_equals_three_l866_86664


namespace simplify_and_sum_coefficients_l866_86671

theorem simplify_and_sum_coefficients (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ), (15*d + 11 + 18*d^2) + (3*d + 2) = a*d^2 + b*d + c ∧ a + b + c = 49 := by
  sorry

end simplify_and_sum_coefficients_l866_86671


namespace cistern_leak_emptying_time_l866_86660

/-- Given a cistern that fills in 8 hours without a leak and 9 hours with a leak,
    the time it takes for the leak to empty a full cistern is 72 hours. -/
theorem cistern_leak_emptying_time :
  ∀ (fill_rate_no_leak : ℝ) (fill_rate_with_leak : ℝ) (leak_rate : ℝ),
    fill_rate_no_leak = 1 / 8 →
    fill_rate_with_leak = 1 / 9 →
    fill_rate_with_leak = fill_rate_no_leak - leak_rate →
    (1 / leak_rate : ℝ) = 72 := by
  sorry


end cistern_leak_emptying_time_l866_86660


namespace sum_of_roots_l866_86692

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 18*c^2 + 27*c - 100 = 0)
  (hd : 9*d^3 - 81*d^2 - 324*d + 3969 = 0) : 
  c + d = 9 := by
  sorry

end sum_of_roots_l866_86692


namespace shaded_area_theorem_l866_86627

/-- Represents a rectangular grid --/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a rectangular shaded region within the grid --/
structure ShadedRegion :=
  (start_x : ℕ)
  (start_y : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Calculates the area of a shaded region --/
def area_of_region (region : ShadedRegion) : ℕ :=
  region.width * region.height

/-- Calculates the total area of multiple shaded regions --/
def total_shaded_area (regions : List ShadedRegion) : ℕ :=
  regions.map area_of_region |>.sum

theorem shaded_area_theorem (grid : Grid) (regions : List ShadedRegion) : 
  grid.width = 15 → 
  grid.height = 5 → 
  regions = [
    { start_x := 0, start_y := 0, width := 6, height := 3 },
    { start_x := 6, start_y := 3, width := 9, height := 2 }
  ] → 
  total_shaded_area regions = 36 := by
  sorry

#check shaded_area_theorem

end shaded_area_theorem_l866_86627


namespace toy_price_difference_is_250_l866_86603

def toy_price_difference : ℝ → Prop :=
  λ price_diff : ℝ =>
    ∃ (a b : ℝ),
      a > 150 ∧ b > 150 ∧
      (∀ p : ℝ, a ≤ p ∧ p ≤ b →
        (0.2 * p ≥ 40 ∧ 0.2 * p ≥ 0.3 * (p - 150))) ∧
      (∀ p : ℝ, p < a ∨ p > b →
        (0.2 * p < 40 ∨ 0.2 * p < 0.3 * (p - 150))) ∧
      price_diff = b - a

theorem toy_price_difference_is_250 :
  toy_price_difference 250 :=
sorry

end toy_price_difference_is_250_l866_86603


namespace min_value_two_over_x_plus_x_over_two_min_value_achievable_l866_86665

theorem min_value_two_over_x_plus_x_over_two (x : ℝ) (hx : x > 0) :
  2/x + x/2 ≥ 2 :=
sorry

theorem min_value_achievable :
  ∃ x : ℝ, x > 0 ∧ 2/x + x/2 = 2 :=
sorry

end min_value_two_over_x_plus_x_over_two_min_value_achievable_l866_86665


namespace sum_of_seven_consecutive_integers_l866_86686

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
  sorry

end sum_of_seven_consecutive_integers_l866_86686


namespace triangle_relations_l866_86602

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  let ⟨A, B, C, D⟩ := t
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 -- B is right angle

def BC_equals_2AB (t : Triangle) : Prop :=
  let ⟨A, B, C, D⟩ := t
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 * ((B.1 - A.1)^2 + (B.2 - A.2)^2)

def D_on_angle_bisector (t : Triangle) : Prop :=
  let ⟨A, B, C, D⟩ := t
  ∃ k : ℝ, 0 < k ∧ k < 1 ∧
    D.1 = A.1 + k * (C.1 - A.1) ∧
    D.2 = A.2 + k * (C.2 - A.2)

-- Theorem statement
theorem triangle_relations (t : Triangle) 
  (h1 : is_right_triangle t) 
  (h2 : BC_equals_2AB t) 
  (h3 : D_on_angle_bisector t) :
  let ⟨A, B, C, D⟩ := t
  (D.1 - B.1)^2 + (D.2 - B.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) * (Real.sin (18 * π / 180))^2 ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) * (Real.sin (36 * π / 180))^2 :=
by sorry

end triangle_relations_l866_86602


namespace expression_evaluation_l866_86600

theorem expression_evaluation : (5 ^ 2 : ℤ) + 15 / 3 - (3 * 2) ^ 2 = -6 := by
  sorry

end expression_evaluation_l866_86600


namespace square_side_length_l866_86670

/-- Given a regular triangle and a square with specific perimeter conditions, 
    prove that the side length of the square is 8 cm. -/
theorem square_side_length 
  (triangle_perimeter : ℝ) 
  (total_perimeter : ℝ) 
  (h1 : triangle_perimeter = 46) 
  (h2 : total_perimeter = 78) : 
  (total_perimeter - triangle_perimeter) / 4 = 8 := by
  sorry

end square_side_length_l866_86670


namespace prob_shortest_diagonal_nonagon_l866_86605

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular polygon with n sides -/
def num_shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal in a regular polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  (num_shortest_diagonals n : ℚ) / (num_diagonals n : ℚ)

theorem prob_shortest_diagonal_nonagon :
  prob_shortest_diagonal 9 = 1/3 := by sorry

end prob_shortest_diagonal_nonagon_l866_86605


namespace six_digit_square_reverse_square_exists_l866_86656

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem six_digit_square_reverse_square_exists : ∃ n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧
  is_perfect_square n ∧
  is_perfect_square (reverse_digits n) := by
  sorry

end six_digit_square_reverse_square_exists_l866_86656


namespace angle_aoc_in_regular_octagon_l866_86695

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The center of a regular octagon -/
def center (octagon : RegularOctagon) : ℝ × ℝ := sorry

/-- Angle between two points and the center -/
def angle_with_center (octagon : RegularOctagon) (p1 p2 : Fin 8) : ℝ := sorry

theorem angle_aoc_in_regular_octagon (octagon : RegularOctagon) :
  angle_with_center octagon 0 2 = 45 := by sorry

end angle_aoc_in_regular_octagon_l866_86695


namespace total_mileage_scientific_notation_l866_86662

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The total mileage of national expressways -/
def totalMileage : ℕ := 108000

/-- Theorem: The scientific notation of the total mileage is 1.08 × 10^5 -/
theorem total_mileage_scientific_notation :
  ∃ (sn : ScientificNotation), sn.coefficient = 1.08 ∧ sn.exponent = 5 ∧ (sn.coefficient * (10 : ℝ) ^ sn.exponent = totalMileage) :=
sorry

end total_mileage_scientific_notation_l866_86662


namespace vincent_train_books_l866_86694

theorem vincent_train_books (animal_books : ℕ) (space_books : ℕ) (book_cost : ℕ) (total_spent : ℕ) :
  animal_books = 10 →
  space_books = 1 →
  book_cost = 16 →
  total_spent = 224 →
  ∃ (train_books : ℕ), train_books = 3 ∧ total_spent = book_cost * (animal_books + space_books + train_books) :=
by sorry

end vincent_train_books_l866_86694


namespace square_sum_value_l866_86629

theorem square_sum_value (x y : ℝ) :
  (x^2 + y^2 + 1) * (x^2 + y^2 + 2) = 6 → x^2 + y^2 = 1 := by
  sorry

end square_sum_value_l866_86629


namespace journey_speed_proof_l866_86691

/-- Proves that given a journey of 108 miles completed in 90 minutes, 
    where the average speed for the first 30 minutes was 65 mph and 
    for the second 30 minutes was 70 mph, the average speed for the 
    last 30 minutes was 81 mph. -/
theorem journey_speed_proof 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_first_segment : ℝ) 
  (speed_second_segment : ℝ) 
  (h1 : total_distance = 108) 
  (h2 : total_time = 90 / 60) 
  (h3 : speed_first_segment = 65) 
  (h4 : speed_second_segment = 70) : 
  ∃ (speed_last_segment : ℝ), 
    speed_last_segment = 81 ∧ 
    (speed_first_segment + speed_second_segment + speed_last_segment) / 3 = 
      total_distance / total_time := by
  sorry

end journey_speed_proof_l866_86691


namespace smallest_group_size_l866_86646

theorem smallest_group_size : ∃ n : ℕ, n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 6 = 5 ∧ 
  n % 8 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 
    (m % 3 = 2 ∧ m % 6 = 5 ∧ m % 8 = 7) → 
    n ≤ m :=
by sorry

end smallest_group_size_l866_86646


namespace inequality_proof_l866_86612

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * Real.sqrt (a * b)) / (Real.sqrt a + Real.sqrt b) ≤ (a * b) ^ (1/4) := by
  sorry

end inequality_proof_l866_86612


namespace subsidy_scheme_2_maximizes_profit_l866_86647

-- Define the daily processing capacity
def x : ℝ := 100

-- Define the constraints on x
axiom x_lower_bound : 70 ≤ x
axiom x_upper_bound : x ≤ 100

-- Define the total daily processing cost function
def total_cost (x : ℝ) : ℝ := 0.5 * x^2 + 40 * x + 3200

-- Define the selling price per ton
def selling_price : ℝ := 110

-- Define the two subsidy schemes
def subsidy_scheme_1 : ℝ := 2300
def subsidy_scheme_2 (x : ℝ) : ℝ := 30 * x

-- Define the profit functions for each subsidy scheme
def profit_scheme_1 (x : ℝ) : ℝ := selling_price * x - total_cost x + subsidy_scheme_1
def profit_scheme_2 (x : ℝ) : ℝ := selling_price * x - total_cost x + subsidy_scheme_2 x

-- Theorem: Subsidy scheme 2 maximizes profit
theorem subsidy_scheme_2_maximizes_profit :
  profit_scheme_2 x > profit_scheme_1 x :=
sorry

end subsidy_scheme_2_maximizes_profit_l866_86647


namespace quadratic_root_implies_a_l866_86630

theorem quadratic_root_implies_a (a : ℝ) :
  (3^2 + a*3 + a - 1 = 0) → a = -2 := by
  sorry

end quadratic_root_implies_a_l866_86630


namespace original_denominator_proof_l866_86667

theorem original_denominator_proof (d : ℚ) : 
  (2 : ℚ) / d ≠ 0 →    -- Ensure the fraction is well-defined
  (6 : ℚ) / (3 * d) = (2 : ℚ) / 3 → 
  d = 3 := by
sorry

end original_denominator_proof_l866_86667


namespace second_number_value_l866_86636

theorem second_number_value (x y z : ℝ) : 
  z = 4.5 * y →
  y = 2.5 * x →
  (x + y + z) / 3 = 165 →
  y = 82.5 := by
sorry

end second_number_value_l866_86636


namespace full_spots_is_186_l866_86620

/-- Represents a parking garage with open spots on each level -/
structure ParkingGarage where
  levels : Nat
  spotsPerLevel : Nat
  openSpotsFirstLevel : Nat
  openSpotsSecondLevel : Nat
  openSpotsThirdLevel : Nat
  openSpotsFourthLevel : Nat

/-- Calculates the number of full parking spots in the garage -/
def fullParkingSpots (garage : ParkingGarage) : Nat :=
  garage.levels * garage.spotsPerLevel -
  (garage.openSpotsFirstLevel + garage.openSpotsSecondLevel +
   garage.openSpotsThirdLevel + garage.openSpotsFourthLevel)

/-- Theorem stating that the number of full parking spots is 186 -/
theorem full_spots_is_186 (garage : ParkingGarage)
  (h1 : garage.levels = 4)
  (h2 : garage.spotsPerLevel = 100)
  (h3 : garage.openSpotsFirstLevel = 58)
  (h4 : garage.openSpotsSecondLevel = garage.openSpotsFirstLevel + 2)
  (h5 : garage.openSpotsThirdLevel = garage.openSpotsSecondLevel + 5)
  (h6 : garage.openSpotsFourthLevel = 31) :
  fullParkingSpots garage = 186 := by
  sorry

#eval fullParkingSpots {
  levels := 4,
  spotsPerLevel := 100,
  openSpotsFirstLevel := 58,
  openSpotsSecondLevel := 60,
  openSpotsThirdLevel := 65,
  openSpotsFourthLevel := 31
}

end full_spots_is_186_l866_86620


namespace books_in_childrens_section_l866_86635

theorem books_in_childrens_section
  (initial_books : ℕ)
  (books_left : ℕ)
  (history_books : ℕ)
  (fiction_books : ℕ)
  (misplaced_books : ℕ)
  (h1 : initial_books = 51)
  (h2 : books_left = 16)
  (h3 : history_books = 12)
  (h4 : fiction_books = 19)
  (h5 : misplaced_books = 4) :
  initial_books + misplaced_books - history_books - fiction_books - books_left = 8 :=
by sorry

end books_in_childrens_section_l866_86635


namespace rationalize_and_simplify_l866_86684

theorem rationalize_and_simplify :
  ∃ (A B C D E : ℤ),
    (B < D) ∧
    (3 : ℝ) / (4 * Real.sqrt 5 + 3 * Real.sqrt 7) = 
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    A = 12 ∧ B = 5 ∧ C = -9 ∧ D = 7 ∧ E = 17 :=
by sorry

end rationalize_and_simplify_l866_86684


namespace zero_subset_A_l866_86637

def A : Set ℕ := {x | x < 4}

theorem zero_subset_A : {0} ⊆ A := by sorry

end zero_subset_A_l866_86637


namespace max_score_15_cards_l866_86658

/-- The score of a hand of cards -/
def score (R B Y : ℕ) : ℕ :=
  R + 2 * R * B + 3 * B * Y

/-- The theorem stating the maximum score achievable with 15 cards -/
theorem max_score_15_cards :
  ∃ R B Y : ℕ,
    R + B + Y = 15 ∧
    ∀ R' B' Y' : ℕ, R' + B' + Y' = 15 →
      score R' B' Y' ≤ score R B Y ∧
      score R B Y = 168 :=
sorry

end max_score_15_cards_l866_86658


namespace circle_center_coordinates_sum_l866_86689

/-- Given a circle with equation x² + y² = -4x + 6y - 12, 
    the sum of the x and y coordinates of its center is 1. -/
theorem circle_center_coordinates_sum : 
  ∀ (x y : ℝ), x^2 + y^2 = -4*x + 6*y - 12 → 
  ∃ (h k : ℝ), (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = 1) ∧ h + k = 1 := by
sorry

end circle_center_coordinates_sum_l866_86689


namespace prob_at_least_two_hits_eq_81_125_l866_86634

/-- The probability of hitting a target in one shot. -/
def p : ℝ := 0.6

/-- The number of shots taken. -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in n shots. -/
def prob_at_least_two_hits : ℝ := 
  Finset.sum (Finset.range (n + 1) \ Finset.range 2) (λ k => 
    (n.choose k : ℝ) * p^k * (1 - p)^(n - k))

theorem prob_at_least_two_hits_eq_81_125 : 
  prob_at_least_two_hits = 81 / 125 := by
  sorry

end prob_at_least_two_hits_eq_81_125_l866_86634


namespace stating_calculate_downstream_speed_l866_86649

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- 
Theorem stating that given a man's upstream and still water speeds, 
his downstream speed can be calculated.
-/
theorem calculate_downstream_speed (speed : RowingSpeed) 
  (h1 : speed.upstream = 15)
  (h2 : speed.stillWater = 20) :
  speed.downstream = 25 := by
  sorry

#check calculate_downstream_speed

end stating_calculate_downstream_speed_l866_86649


namespace afternoon_session_count_l866_86655

/-- Represents the number of kids in each session for a sport -/
structure SportSessions :=
  (morning : ℕ)
  (afternoon : ℕ)
  (evening : ℕ)
  (undecided : ℕ)

/-- Calculates the total number of kids in afternoon sessions across all sports -/
def total_afternoon_kids (soccer : SportSessions) (basketball : SportSessions) (swimming : SportSessions) : ℕ :=
  soccer.afternoon + basketball.afternoon + swimming.afternoon

theorem afternoon_session_count :
  ∀ (total_kids : ℕ) 
    (soccer basketball swimming : SportSessions),
  total_kids = 2000 →
  soccer.morning + soccer.afternoon + soccer.evening + soccer.undecided = 400 →
  basketball.morning + basketball.afternoon + basketball.evening = 300 →
  swimming.morning + swimming.afternoon + swimming.evening = 300 →
  soccer.morning = 100 →
  soccer.afternoon = 280 →
  soccer.undecided = 20 →
  basketball.evening = 180 →
  basketball.morning = basketball.afternoon →
  swimming.morning = swimming.afternoon →
  swimming.afternoon = swimming.evening →
  ∃ (soccer_new basketball_new swimming_new : SportSessions),
    soccer_new.morning = soccer.morning + 30 →
    soccer_new.afternoon = soccer.afternoon - 30 →
    soccer_new.evening = soccer.evening →
    soccer_new.undecided = soccer.undecided →
    basketball_new = basketball →
    swimming_new.morning = swimming.morning + 15 →
    swimming_new.afternoon = swimming.afternoon - 15 →
    swimming_new.evening = swimming.evening →
    total_afternoon_kids soccer_new basketball_new swimming_new = 395 :=
by sorry

end afternoon_session_count_l866_86655


namespace senate_subcommittee_count_l866_86642

/-- The number of ways to form a subcommittee from a Senate committee -/
def subcommittee_ways (total_republicans : ℕ) (total_democrats : ℕ) 
  (subcommittee_republicans : ℕ) (min_subcommittee_democrats : ℕ) 
  (max_subcommittee_size : ℕ) : ℕ :=
  sorry

theorem senate_subcommittee_count : 
  subcommittee_ways 10 8 3 2 5 = 10080 := by sorry

end senate_subcommittee_count_l866_86642


namespace purple_socks_probability_l866_86617

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer where
  green : ℕ
  purple : ℕ
  orange : ℕ

/-- Calculates the total number of socks in the drawer -/
def SockDrawer.total (d : SockDrawer) : ℕ :=
  d.green + d.purple + d.orange

/-- Calculates the probability of selecting a purple sock -/
def purpleProbability (d : SockDrawer) : ℚ :=
  d.purple / d.total

/-- The initial state of the sock drawer -/
def initialDrawer : SockDrawer :=
  { green := 6, purple := 18, orange := 12 }

/-- The number of purple socks added -/
def addedPurpleSocks : ℕ := 9

/-- The final state of the sock drawer after adding purple socks -/
def finalDrawer : SockDrawer :=
  { green := initialDrawer.green,
    purple := initialDrawer.purple + addedPurpleSocks,
    orange := initialDrawer.orange }

theorem purple_socks_probability :
  purpleProbability finalDrawer = 3/5 := by
  sorry

end purple_socks_probability_l866_86617


namespace cheese_cost_is_50_l866_86697

/-- The cost of a sandwich in cents -/
def sandwich_cost : ℕ := 90

/-- The cost of a slice of bread in cents -/
def bread_cost : ℕ := 15

/-- The cost of a slice of ham in cents -/
def ham_cost : ℕ := 25

/-- The cost of a slice of cheese in cents -/
def cheese_cost : ℕ := sandwich_cost - bread_cost - ham_cost

theorem cheese_cost_is_50 : cheese_cost = 50 := by
  sorry

end cheese_cost_is_50_l866_86697


namespace shipping_cost_per_pound_l866_86678

/-- Shipping cost calculation -/
theorem shipping_cost_per_pound 
  (flat_fee : ℝ) 
  (weight : ℝ) 
  (total_cost : ℝ) 
  (h1 : flat_fee = 5)
  (h2 : weight = 5)
  (h3 : total_cost = 9)
  (h4 : total_cost = flat_fee + weight * (total_cost - flat_fee) / weight) :
  (total_cost - flat_fee) / weight = 0.8 := by
  sorry

end shipping_cost_per_pound_l866_86678


namespace triangle_circles_tangency_l866_86654

theorem triangle_circles_tangency (DE DF EF : ℝ) (R S : ℝ) :
  DE = 120 →
  DF = 120 →
  EF = 70 →
  R = 20 →
  S > 0 →
  S + R > EF / 2 →
  S < DE - R →
  (S + R)^2 + (S - R)^2 = ((130 - 4*S) / 3)^2 →
  S = 55 - 5 * Real.sqrt 41 :=
by sorry

end triangle_circles_tangency_l866_86654


namespace point_satisfies_constraint_local_maximum_at_point_main_theorem_l866_86633

/-- The constraint function g(x₁, x₂) = x₁ - 2x₂ + 3 -/
def g (x₁ x₂ : ℝ) : ℝ := x₁ - 2*x₂ + 3

/-- The objective function f(x₁, x₂) = x₂² - x₁² -/
def f (x₁ x₂ : ℝ) : ℝ := x₂^2 - x₁^2

/-- The point (1, 2) satisfies the constraint -/
theorem point_satisfies_constraint : g 1 2 = 0 := by sorry

/-- The function f has a local maximum at (1, 2) under the constraint g(x₁, x₂) = 0 -/
theorem local_maximum_at_point :
  ∃ ε > 0, ∀ x₁ x₂ : ℝ, 
    g x₁ x₂ = 0 → 
    (x₁ - 1)^2 + (x₂ - 2)^2 < ε^2 → 
    f x₁ x₂ ≤ f 1 2 := by sorry

/-- The main theorem: f has a local maximum at (1, 2) under the constraint g(x₁, x₂) = 0 -/
theorem main_theorem : 
  ∃ (x₁ x₂ : ℝ), g x₁ x₂ = 0 ∧ 
  ∃ ε > 0, ∀ y₁ y₂ : ℝ, 
    g y₁ y₂ = 0 → 
    (y₁ - x₁)^2 + (y₂ - x₂)^2 < ε^2 → 
    f y₁ y₂ ≤ f x₁ x₂ :=
by
  use 1, 2
  constructor
  · exact point_satisfies_constraint
  · exact local_maximum_at_point

end point_satisfies_constraint_local_maximum_at_point_main_theorem_l866_86633


namespace min_value_and_inequality_l866_86672

theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 4 * a + b = a * b) :
  (∃ (min : ℝ), min = 9 ∧ ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = x * y → x + y ≥ min) ∧
  (∀ x t : ℝ, t ∈ Set.Icc (-1) 3 → |x - a| + |x - b| ≥ t^2 - 2*t) :=
by sorry

end min_value_and_inequality_l866_86672


namespace daniels_age_l866_86682

theorem daniels_age (ishaan_age : ℕ) (years_until_4x : ℕ) (daniel_age : ℕ) : 
  ishaan_age = 6 →
  years_until_4x = 15 →
  daniel_age + years_until_4x = 4 * (ishaan_age + years_until_4x) →
  daniel_age = 69 := by
sorry

end daniels_age_l866_86682


namespace cutlery_theorem_l866_86675

/-- Calculates the total number of cutlery pieces after purchases -/
def totalCutlery (initialKnives : ℕ) : ℕ :=
  let initialTeaspoons := 2 * initialKnives
  let additionalKnives := initialKnives / 3
  let additionalTeaspoons := (2 * initialTeaspoons) / 3
  let totalKnives := initialKnives + additionalKnives
  let totalTeaspoons := initialTeaspoons + additionalTeaspoons
  totalKnives + totalTeaspoons

/-- Theorem stating that given 24 initial knives, the total cutlery after purchases is 112 -/
theorem cutlery_theorem : totalCutlery 24 = 112 := by
  sorry

end cutlery_theorem_l866_86675


namespace min_games_to_satisfy_condition_l866_86668

/-- The number of teams in the tournament -/
def num_teams : ℕ := 20

/-- The total number of possible games between all teams -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- The maximum number of unplayed games while satisfying the condition -/
def max_unplayed_games : ℕ := (num_teams / 2) ^ 2

/-- A function that checks if the number of played games satisfies the condition -/
def satisfies_condition (played_games : ℕ) : Prop :=
  played_games ≥ total_games - max_unplayed_games

/-- The theorem stating the minimum number of games that must be played -/
theorem min_games_to_satisfy_condition :
  ∃ (min_games : ℕ), satisfies_condition min_games ∧
  ∀ (n : ℕ), n < min_games → ¬satisfies_condition n :=
sorry

end min_games_to_satisfy_condition_l866_86668
