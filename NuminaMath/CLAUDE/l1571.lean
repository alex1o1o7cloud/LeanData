import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1571_157145

/-- Given that 15 is the arithmetic mean of the set {7, 12, 19, 8, 10, y}, prove that y = 34 -/
theorem arithmetic_mean_problem (y : ℝ) : 
  (7 + 12 + 19 + 8 + 10 + y) / 6 = 15 → y = 34 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1571_157145


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1571_157131

theorem cubic_equation_solution (x y z a : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ z ≠ x →
  x^3 + a = -3*(y + z) →
  y^3 + a = -3*(x + z) →
  z^3 + a = -3*(x + y) →
  a ∈ Set.Ioo (-2 : ℝ) 2 \ {0} := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1571_157131


namespace NUMINAMATH_CALUDE_unique_solution_modular_equation_l1571_157134

theorem unique_solution_modular_equation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 107 ∧ (103 * n) % 107 = 56 % 107 ∧ n = 85 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_modular_equation_l1571_157134


namespace NUMINAMATH_CALUDE_ball_max_height_l1571_157151

/-- The height function of the ball -/
def height_function (t : ℝ) : ℝ := 180 * t - 20 * t^2

/-- The maximum height reached by the ball -/
def max_height : ℝ := 405

theorem ball_max_height : 
  ∃ t : ℝ, height_function t = max_height ∧ 
  ∀ u : ℝ, height_function u ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l1571_157151


namespace NUMINAMATH_CALUDE_christines_dog_weight_l1571_157112

/-- Theorem: Christine's dog weight calculation -/
theorem christines_dog_weight (cat1_weight cat2_weight : ℕ) (dog_weight : ℕ) : 
  cat1_weight = 7 →
  cat2_weight = 10 →
  dog_weight = 2 * (cat1_weight + cat2_weight) →
  dog_weight = 34 := by
  sorry

end NUMINAMATH_CALUDE_christines_dog_weight_l1571_157112


namespace NUMINAMATH_CALUDE_min_value_ab_l1571_157196

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = Real.sqrt (a*b)) :
  a * b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 4/b₀ = Real.sqrt (a₀*b₀) ∧ a₀ * b₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l1571_157196


namespace NUMINAMATH_CALUDE_max_sum_of_endpoints_l1571_157101

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Define the theorem
theorem max_sum_of_endpoints
  (m n : ℝ)
  (h1 : n > m)
  (h2 : ∀ x ∈ Set.Icc m n, -5 ≤ f x ∧ f x ≤ 4)
  (h3 : ∃ x ∈ Set.Icc m n, f x = -5)
  (h4 : ∃ x ∈ Set.Icc m n, f x = 4) :
  ∃ k : ℝ, (∀ a b : ℝ, a ≥ m ∧ b ≤ n ∧ b > a ∧
    (∀ x ∈ Set.Icc a b, -5 ≤ f x ∧ f x ≤ 4) ∧
    (∃ x ∈ Set.Icc a b, f x = -5) ∧
    (∃ x ∈ Set.Icc a b, f x = 4) →
    a + b ≤ k) ∧
  k = n + m ∧ k = 7 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_endpoints_l1571_157101


namespace NUMINAMATH_CALUDE_protest_jail_time_protest_jail_time_result_l1571_157161

/-- Calculate the total combined weeks of jail time given protest and arrest conditions -/
theorem protest_jail_time (days_of_protest : ℕ) (num_cities : ℕ) (arrests_per_day : ℕ) 
  (pre_trial_days : ℕ) (sentence_weeks : ℕ) : ℕ :=
  let total_arrests := days_of_protest * num_cities * arrests_per_day
  let jail_days_per_person := pre_trial_days + (sentence_weeks / 2) * 7
  let total_jail_days := total_arrests * jail_days_per_person
  total_jail_days / 7

/-- The total combined weeks of jail time is 9900 weeks -/
theorem protest_jail_time_result : 
  protest_jail_time 30 21 10 4 2 = 9900 := by sorry

end NUMINAMATH_CALUDE_protest_jail_time_protest_jail_time_result_l1571_157161


namespace NUMINAMATH_CALUDE_pear_difference_is_five_l1571_157170

/-- The number of bags of pears Austin picked fewer than Dallas -/
def pear_difference (dallas_apples dallas_pears austin_total : ℕ) (austin_apple_diff : ℕ) : ℕ :=
  dallas_pears - (austin_total - (dallas_apples + austin_apple_diff))

theorem pear_difference_is_five :
  pear_difference 14 9 24 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_pear_difference_is_five_l1571_157170


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_quadratic_inequality_solution_set_l1571_157142

-- Part 1
theorem quadratic_inequality_range (a : ℝ) : 
  (a > 0 ∧ ∃ x, a * x^2 - 3 * x + 2 < 0) ↔ (0 < a ∧ a < 9/8) :=
sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x < 1}
  else if a < 0 then {x | 3/a < x ∧ x < 1}
  else if 0 < a ∧ a < 3 then {x | x < 3/a ∨ x > 1}
  else if a = 3 then {x | x ≠ 1}
  else {x | x < 1 ∨ x > 3/a}

theorem quadratic_inequality_solution_set (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 - 3 * x + 2 > a * x - 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_quadratic_inequality_solution_set_l1571_157142


namespace NUMINAMATH_CALUDE_special_rectangle_sides_l1571_157175

/-- A rectangle with special properties -/
structure SpecialRectangle where
  -- The length of the rectangle
  l : ℝ
  -- The width of the rectangle
  w : ℝ
  -- The perimeter of the rectangle is 24
  perimeter : l + w = 12
  -- M is the midpoint of BC
  midpoint : w / 2 = w / 2
  -- MA is perpendicular to MD
  perpendicular : l ^ 2 + (w / 2) ^ 2 = l ^ 2 + (w / 2) ^ 2

/-- The sides of a special rectangle are 4 and 8 -/
theorem special_rectangle_sides (r : SpecialRectangle) : r.l = 4 ∧ r.w = 8 := by
  sorry

#check special_rectangle_sides

end NUMINAMATH_CALUDE_special_rectangle_sides_l1571_157175


namespace NUMINAMATH_CALUDE_scaled_vector_is_monomial_l1571_157141

/-- A vector in ℝ² -/
def vector : ℝ × ℝ := (1, 5)

/-- The scalar multiple of the vector -/
def scaled_vector : ℝ × ℝ := (-3 * vector.1, -3 * vector.2)

/-- Definition of a monomial in this context -/
def is_monomial (v : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ) (n : ℕ × ℕ), v = (c * n.1, c * n.2)

theorem scaled_vector_is_monomial : is_monomial scaled_vector := by
  sorry

end NUMINAMATH_CALUDE_scaled_vector_is_monomial_l1571_157141


namespace NUMINAMATH_CALUDE_script_lines_proof_l1571_157121

theorem script_lines_proof :
  ∀ (lines1 lines2 lines3 : ℕ),
  lines1 = 20 →
  lines1 = lines2 + 8 →
  lines2 = 3 * lines3 + 6 →
  lines3 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_script_lines_proof_l1571_157121


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1155_l1571_157100

def is_prime_factor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ n % p = 0

theorem sum_of_largest_and_smallest_prime_factors_of_1155 :
  ∃ (smallest largest : ℕ),
    is_prime_factor smallest 1155 ∧
    is_prime_factor largest 1155 ∧
    (∀ p, is_prime_factor p 1155 → smallest ≤ p) ∧
    (∀ p, is_prime_factor p 1155 → p ≤ largest) ∧
    smallest + largest = 14 :=
sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1155_l1571_157100


namespace NUMINAMATH_CALUDE_spencer_jumps_per_minute_l1571_157143

-- Define the parameters
def minutes_per_session : ℕ := 10
def sessions_per_day : ℕ := 2
def total_jumps : ℕ := 400
def total_days : ℕ := 5

-- Theorem to prove
theorem spencer_jumps_per_minute :
  (total_jumps : ℚ) / ((minutes_per_session * sessions_per_day * total_days) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_spencer_jumps_per_minute_l1571_157143


namespace NUMINAMATH_CALUDE_bus_car_speed_problem_l1571_157171

theorem bus_car_speed_problem : ∀ (v_bus v_car : ℝ),
  -- Given conditions
  (1.5 * v_bus + 1.5 * v_car = 180) →
  (2.5 * v_bus + v_car = 180) →
  -- Conclusion
  (v_bus = 40 ∧ v_car = 80) :=
by
  sorry

end NUMINAMATH_CALUDE_bus_car_speed_problem_l1571_157171


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1571_157105

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : A = 64 * Real.pi → d = 16 → A = Real.pi * (d / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1571_157105


namespace NUMINAMATH_CALUDE_intersection_M_N_l1571_157120

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x < 2}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1571_157120


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l1571_157136

/-- A circle with center on the line y = x and tangent to lines x + y = 0 and x + y + 4 = 0 -/
structure TangentCircle where
  a : ℝ
  center_on_diagonal : a = a
  tangent_to_first_line : |2 * a| / Real.sqrt 2 = |0 - 0| / Real.sqrt 2
  tangent_to_second_line : |2 * a| / Real.sqrt 2 = |4| / Real.sqrt 2

/-- The equation of the circle described by TangentCircle is (x+1)² + (y+1)² = 2 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∀ x y : ℝ, (x + 1)^2 + (y + 1)^2 = 2 ↔ 
  (x - (-1))^2 + (y - (-1))^2 = (Real.sqrt 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l1571_157136


namespace NUMINAMATH_CALUDE_album_ratio_l1571_157156

theorem album_ratio (adele katrina bridget miriam : ℕ) 
  (h1 : ∃ s : ℕ, miriam = s * katrina)
  (h2 : katrina = 6 * bridget)
  (h3 : bridget = adele - 15)
  (h4 : adele = 30)
  (h5 : miriam + katrina + bridget + adele = 585) :
  miriam = 5 * katrina := by
sorry

end NUMINAMATH_CALUDE_album_ratio_l1571_157156


namespace NUMINAMATH_CALUDE_play_area_size_l1571_157138

/-- Represents the configuration of a rectangular play area with fence posts -/
structure PlayArea where
  total_posts : ℕ
  post_spacing : ℕ
  shorter_side_posts : ℕ
  longer_side_posts : ℕ

/-- Calculates the area of the play area given its configuration -/
def calculate_area (pa : PlayArea) : ℕ :=
  (pa.shorter_side_posts - 1) * pa.post_spacing * ((pa.longer_side_posts - 1) * pa.post_spacing)

/-- Theorem stating that the play area with given specifications has an area of 324 square yards -/
theorem play_area_size (pa : PlayArea) :
  pa.total_posts = 24 ∧
  pa.post_spacing = 3 ∧
  pa.longer_side_posts = 2 * pa.shorter_side_posts ∧
  pa.total_posts = 2 * pa.shorter_side_posts + 2 * pa.longer_side_posts - 4 →
  calculate_area pa = 324 := by
  sorry

end NUMINAMATH_CALUDE_play_area_size_l1571_157138


namespace NUMINAMATH_CALUDE_min_value_of_function_l1571_157153

theorem min_value_of_function (x : ℝ) (h : x ∈ Set.Ico 1 2) : 
  (1 / x + 1 / (2 - x)) ≥ 2 ∧ 
  (1 / x + 1 / (2 - x) = 2 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1571_157153


namespace NUMINAMATH_CALUDE_triangular_prism_volume_l1571_157167

/-- The volume of a triangular prism given the area of a lateral face and the distance to the opposite edge -/
theorem triangular_prism_volume (A_face : ℝ) (d : ℝ) (h_pos_A : A_face > 0) (h_pos_d : d > 0) :
  ∃ (V : ℝ), V = (1 / 2) * A_face * d ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_volume_l1571_157167


namespace NUMINAMATH_CALUDE_evaluate_M_l1571_157114

theorem evaluate_M : 
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) + Real.sqrt (4 - 2 * Real.sqrt 3)
  M = 7/4 := by
sorry

end NUMINAMATH_CALUDE_evaluate_M_l1571_157114


namespace NUMINAMATH_CALUDE_negation_equivalence_l1571_157159

theorem negation_equivalence : 
  (¬∃ x : ℝ, (2 / x) + Real.log x ≤ 0) ↔ (∀ x : ℝ, (2 / x) + Real.log x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1571_157159


namespace NUMINAMATH_CALUDE_horner_v3_value_l1571_157178

/-- The polynomial f(x) = 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

/-- The value of v_3 in Horner's method -/
def v_3 (x : ℝ) : ℝ := (((7*x + 6)*x + 5)*x + 4)

/-- Theorem: The value of v_3 is 262 when x = 3 -/
theorem horner_v3_value : v_3 3 = 262 := by
  sorry

end NUMINAMATH_CALUDE_horner_v3_value_l1571_157178


namespace NUMINAMATH_CALUDE_zoo_bus_seats_l1571_157127

theorem zoo_bus_seats (total_children : ℕ) (children_per_seat : ℕ) (seats_needed : ℕ) : 
  total_children = 58 → children_per_seat = 2 → seats_needed = total_children / children_per_seat → 
  seats_needed = 29 := by
sorry

end NUMINAMATH_CALUDE_zoo_bus_seats_l1571_157127


namespace NUMINAMATH_CALUDE_percent_relation_l1571_157166

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.1 * b) 
  (h2 : b = 2 * a) : 
  c = 0.2 * a := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l1571_157166


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l1571_157148

/-- The line passing through points (2, 10) and (5, 16) intersects the y-axis at (0, 6) -/
theorem line_intersection_y_axis :
  let p₁ : ℝ × ℝ := (2, 10)
  let p₂ : ℝ × ℝ := (5, 16)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  let line (x : ℝ) : ℝ := m * x + b
  (0, line 0) = (0, 6) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l1571_157148


namespace NUMINAMATH_CALUDE_opposite_of_seven_l1571_157103

theorem opposite_of_seven :
  ∀ x : ℤ, (7 + x = 0) → x = -7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_seven_l1571_157103


namespace NUMINAMATH_CALUDE_mart_vegetable_count_l1571_157174

/-- The number of cucumbers in the mart -/
def cucumbers : ℕ := 58

/-- The number of carrots in the mart -/
def carrots : ℕ := cucumbers - 24

/-- The number of tomatoes in the mart -/
def tomatoes : ℕ := cucumbers + 49

/-- The number of radishes in the mart -/
def radishes : ℕ := carrots

/-- The total number of vegetables in the mart -/
def total_vegetables : ℕ := cucumbers + carrots + tomatoes + radishes

/-- Theorem stating the total number of vegetables in the mart -/
theorem mart_vegetable_count : total_vegetables = 233 := by sorry

end NUMINAMATH_CALUDE_mart_vegetable_count_l1571_157174


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1571_157107

theorem fraction_equivalence : 
  ∀ (n : ℚ), n = 1/2 → (4 - n) / (7 - n) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1571_157107


namespace NUMINAMATH_CALUDE_negation_equivalence_l1571_157160

theorem negation_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1571_157160


namespace NUMINAMATH_CALUDE_min_value_and_monotonicity_l1571_157177

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x ^ 2 - a * x

theorem min_value_and_monotonicity (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ f (-2 * Real.exp 1) x = 3 ∧ 
    ∀ (y : ℝ), y > 0 → f (-2 * Real.exp 1) y ≥ f (-2 * Real.exp 1) x) ∧
  (∀ (x y : ℝ), 0 < x ∧ x < y → (f a x ≥ f a y ↔ a ≥ 2 / Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_monotonicity_l1571_157177


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_T_l1571_157172

/-- Sum of reciprocals of prime digits -/
def P : ℚ := 1/2 + 1/3 + 1/5 + 1/7

/-- T_n is the sum of the reciprocals of the prime digits of integers from 1 to 5^n inclusive -/
def T (n : ℕ) : ℚ := n * (5^(n-1) : ℚ) * P

/-- 42 is the smallest positive integer n for which T_n is an integer -/
theorem smallest_n_for_integer_T : ∀ k : ℕ, k > 0 → (∃ m : ℤ, T k = m) → k ≥ 42 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_T_l1571_157172


namespace NUMINAMATH_CALUDE_expression_value_l1571_157125

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (3 * x^4 + 4 * y^2) / 12 = 25.5833 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1571_157125


namespace NUMINAMATH_CALUDE_bottom_row_is_2143_l1571_157140

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Fin 4

-- Define a valid grid
def is_valid_grid (g : Grid) : Prop :=
  -- Each number appears exactly once per row and column
  (∀ i j k, i ≠ k → g i j ≠ g k j) ∧
  (∀ i j k, j ≠ k → g i j ≠ g i k) ∧
  -- L-shaped sum constraints
  g 0 0 + g 0 1 = 3 ∧
  g 0 3 + g 1 3 = 6 ∧
  g 2 1 + g 3 1 = 5

-- Define the bottom row
def bottom_row (g : Grid) : Fin 4 → Fin 4 := g 3

-- Theorem stating the bottom row forms 2143
theorem bottom_row_is_2143 (g : Grid) (h : is_valid_grid g) :
  (bottom_row g 0, bottom_row g 1, bottom_row g 2, bottom_row g 3) = (2, 1, 4, 3) :=
sorry

end NUMINAMATH_CALUDE_bottom_row_is_2143_l1571_157140


namespace NUMINAMATH_CALUDE_pamelas_initial_skittles_l1571_157188

/-- The number of Skittles Pamela gave away -/
def skittles_given : ℕ := 7

/-- The number of Skittles Pamela has now -/
def skittles_remaining : ℕ := 43

/-- Pamela's initial number of Skittles -/
def initial_skittles : ℕ := skittles_given + skittles_remaining

theorem pamelas_initial_skittles : initial_skittles = 50 := by
  sorry

end NUMINAMATH_CALUDE_pamelas_initial_skittles_l1571_157188


namespace NUMINAMATH_CALUDE_squirrel_survey_l1571_157152

theorem squirrel_survey (total : ℕ) 
  (harmful_belief_rate : ℚ) 
  (attack_belief_rate : ℚ) 
  (wrong_believers : ℕ) 
  (h1 : harmful_belief_rate = 883 / 1000) 
  (h2 : attack_belief_rate = 538 / 1000) 
  (h3 : wrong_believers = 28) :
  (↑wrong_believers / (harmful_belief_rate * attack_belief_rate) : ℚ).ceil = total → 
  total = 59 := by
sorry

end NUMINAMATH_CALUDE_squirrel_survey_l1571_157152


namespace NUMINAMATH_CALUDE_f_f_7_equals_0_l1571_157185

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_4 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

theorem f_f_7_equals_0 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_periodic : is_periodic_4 f)
  (h_f_1 : f 1 = 4) :
  f (f 7) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_f_7_equals_0_l1571_157185


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l1571_157186

theorem quadratic_equation_range (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x, 2 * x^2 - 2 * x + 3 * m - 1 = 0) →
  (x₁ * x₂ > x₁ + x₂ - 4) →
  (-5/3 < m ∧ m ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l1571_157186


namespace NUMINAMATH_CALUDE_cannot_form_triangle_l1571_157108

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the third side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that line segments 4, 5, and 9 cannot form a triangle -/
theorem cannot_form_triangle : ¬(can_form_triangle 4 5 9) := by
  sorry


end NUMINAMATH_CALUDE_cannot_form_triangle_l1571_157108


namespace NUMINAMATH_CALUDE_cross_pollinated_percentage_l1571_157168

/-- Represents the apple orchard with Fuji and Gala trees -/
structure Orchard where
  totalTrees : ℕ
  pureFuji : ℕ
  pureGala : ℕ
  crossPollinated : ℕ

/-- The percentage of cross-pollinated trees in the orchard is 2/3 -/
theorem cross_pollinated_percentage (o : Orchard) : 
  o.totalTrees = o.pureFuji + o.pureGala + o.crossPollinated →
  o.pureFuji + o.crossPollinated = 170 →
  o.pureFuji = 3 * o.totalTrees / 4 →
  o.pureGala = 30 →
  o.crossPollinated * 3 = o.totalTrees * 2 := by
  sorry

#check cross_pollinated_percentage

end NUMINAMATH_CALUDE_cross_pollinated_percentage_l1571_157168


namespace NUMINAMATH_CALUDE_slope_negative_one_implies_y_coordinate_l1571_157183

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is -1, then the y-coordinate of Q is -3. -/
theorem slope_negative_one_implies_y_coordinate (x1 y1 x2 y2 : ℝ) :
  x1 = -3 →
  y1 = 5 →
  x2 = 5 →
  (y2 - y1) / (x2 - x1) = -1 →
  y2 = -3 := by
sorry

end NUMINAMATH_CALUDE_slope_negative_one_implies_y_coordinate_l1571_157183


namespace NUMINAMATH_CALUDE_seventh_observation_l1571_157184

theorem seventh_observation (n : Nat) (initial_avg : ℝ) (new_avg : ℝ) : 
  n = 6 →
  initial_avg = 15 →
  new_avg = initial_avg - 1 →
  (n * initial_avg + (n + 1) * new_avg) / (n + 1) = new_avg →
  (n + 1) * new_avg - n * initial_avg = 8 :=
by sorry

end NUMINAMATH_CALUDE_seventh_observation_l1571_157184


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1571_157180

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  is_arithmetic_sequence b →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 4 * b 5 = 10 →
  b 2 * b 7 = -224 ∨ b 2 * b 7 = -44 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1571_157180


namespace NUMINAMATH_CALUDE_shaded_area_is_75_l1571_157122

-- Define the side lengths of the squares
def larger_side : ℝ := 10
def smaller_side : ℝ := 5

-- Define the areas of the squares
def larger_area : ℝ := larger_side ^ 2
def smaller_area : ℝ := smaller_side ^ 2

-- Define the shaded area
def shaded_area : ℝ := larger_area - smaller_area

-- Theorem to prove
theorem shaded_area_is_75 : shaded_area = 75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_75_l1571_157122


namespace NUMINAMATH_CALUDE_compound_interest_multiple_l1571_157137

theorem compound_interest_multiple (P r m : ℝ) 
  (h1 : P * r^2 = 40)
  (h2 : P * (m * r)^2 = 360) :
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_multiple_l1571_157137


namespace NUMINAMATH_CALUDE_quadratic_root_l1571_157187

theorem quadratic_root (a b k : ℝ) : 
  (∃ x : ℝ, x^2 - (a+b)*x + a*b*(1-k) = 0 ∧ x = 1) →
  (∃ y : ℝ, y^2 - (a+b)*y + a*b*(1-k) = 0 ∧ y = a + b - 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_l1571_157187


namespace NUMINAMATH_CALUDE_temperature_difference_l1571_157164

theorem temperature_difference (lowest highest : ℤ) 
  (h_lowest : lowest = -11)
  (h_highest : highest = -3) :
  highest - lowest = 8 := by
sorry

end NUMINAMATH_CALUDE_temperature_difference_l1571_157164


namespace NUMINAMATH_CALUDE_not_right_triangle_4_6_11_l1571_157158

/-- Checks if three line segments can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem: The line segments 4, 6, and 11 cannot form a right triangle -/
theorem not_right_triangle_4_6_11 : ¬ is_right_triangle 4 6 11 := by
  sorry

#check not_right_triangle_4_6_11

end NUMINAMATH_CALUDE_not_right_triangle_4_6_11_l1571_157158


namespace NUMINAMATH_CALUDE_pens_per_student_is_five_l1571_157124

-- Define the given constants
def num_students : ℕ := 30
def notebooks_per_student : ℕ := 3
def binders_per_student : ℕ := 1
def highlighters_per_student : ℕ := 2
def pen_cost : ℚ := 0.5
def notebook_cost : ℚ := 1.25
def binder_cost : ℚ := 4.25
def highlighter_cost : ℚ := 0.75
def teacher_discount : ℚ := 100
def total_spent : ℚ := 260

-- Define the theorem
theorem pens_per_student_is_five :
  let cost_per_student_excl_pens := notebooks_per_student * notebook_cost + 
                                    binders_per_student * binder_cost + 
                                    highlighters_per_student * highlighter_cost
  let total_cost_excl_pens := num_students * cost_per_student_excl_pens
  let total_spent_before_discount := total_spent + teacher_discount
  let total_spent_on_pens := total_spent_before_discount - total_cost_excl_pens
  let total_pens := total_spent_on_pens / pen_cost
  let pens_per_student := total_pens / num_students
  pens_per_student = 5 := by sorry

end NUMINAMATH_CALUDE_pens_per_student_is_five_l1571_157124


namespace NUMINAMATH_CALUDE_tangent_line_equation_range_of_a_l1571_157144

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 10

-- Theorem for the tangent line equation
theorem tangent_line_equation (a : ℝ) :
  a = 1 →
  ∃ (m b : ℝ), m = 8 ∧ b = -2 ∧
  ∀ (x y : ℝ), y = f a x → (x = 2 → m*x - y + b = 0) :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ (a : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f a x < 0) →
  a > 9/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_range_of_a_l1571_157144


namespace NUMINAMATH_CALUDE_large_mean_small_variance_reflects_common_prosperity_l1571_157135

/-- Represents a personal income distribution --/
structure IncomeDistribution where
  mean : ℝ
  variance : ℝ
  mean_nonneg : 0 ≤ mean
  variance_nonneg : 0 ≤ variance

/-- Defines the concept of common prosperity --/
def common_prosperity (id : IncomeDistribution) : Prop :=
  id.mean > 0 ∧ id.variance < 1 -- Arbitrary thresholds for illustration

/-- Defines universal prosperity --/
def universal_prosperity (id : IncomeDistribution) : Prop :=
  id.mean > 0

/-- Defines elimination of polarization and poverty --/
def no_polarization_poverty (id : IncomeDistribution) : Prop :=
  id.variance < 1 -- Arbitrary threshold for illustration

/-- Theorem stating that large mean and small variance best reflect common prosperity --/
theorem large_mean_small_variance_reflects_common_prosperity
  (id : IncomeDistribution)
  (h1 : universal_prosperity id → common_prosperity id)
  (h2 : no_polarization_poverty id → common_prosperity id) :
  common_prosperity id ↔ (id.mean > 0 ∧ id.variance < 1) := by
  sorry

#check large_mean_small_variance_reflects_common_prosperity

end NUMINAMATH_CALUDE_large_mean_small_variance_reflects_common_prosperity_l1571_157135


namespace NUMINAMATH_CALUDE_blue_sock_pairs_l1571_157195

theorem blue_sock_pairs (n : ℕ) (k : ℕ) : n = 4 ∧ k = 2 → Nat.choose n k = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_sock_pairs_l1571_157195


namespace NUMINAMATH_CALUDE_parabola_equation_l1571_157193

-- Define the parabola
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => x^2 = 2 * p * y

-- Define points on the parabola
def Point := ℝ × ℝ

-- Define the problem setup
structure ParabolaProblem where
  parabola : Parabola
  F : Point
  A : Point
  B : Point
  C : Point
  D : Point
  l : Point → Prop

-- Define the conditions
def satisfies_conditions (prob : ParabolaProblem) : Prop :=
  let (xf, yf) := prob.F
  let (xa, ya) := prob.A
  let (xb, yb) := prob.B
  let (xc, yc) := prob.C
  let (xd, yd) := prob.D
  xf = 0 ∧ yf > 0 ∧
  prob.parabola.eq xa ya ∧
  prob.parabola.eq xb yb ∧
  prob.l prob.A ∧ prob.l prob.B ∧
  xc = xa ∧ xd = xb ∧
  (ya - yf)^2 + xa^2 = 4 * ((yf - yb)^2 + xb^2) ∧
  (xd - xc) * (xa - xb) + (yd - yc) * (ya - yb) = 72

-- Theorem statement
theorem parabola_equation (prob : ParabolaProblem) 
  (h : satisfies_conditions prob) : 
  prob.parabola.p = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1571_157193


namespace NUMINAMATH_CALUDE_sum_of_squares_perfect_square_two_even_l1571_157182

theorem sum_of_squares_perfect_square_two_even (x y z : ℤ) :
  ∃ (u : ℤ), x^2 + y^2 + z^2 = u^2 →
  (Even x ∧ Even y) ∨ (Even x ∧ Even z) ∨ (Even y ∧ Even z) :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_perfect_square_two_even_l1571_157182


namespace NUMINAMATH_CALUDE_power_of_power_l1571_157181

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1571_157181


namespace NUMINAMATH_CALUDE_no_rational_roots_l1571_157173

/-- The polynomial we're investigating -/
def p (x : ℚ) : ℚ := 3 * x^4 + 2 * x^3 - 8 * x^2 - x + 1

/-- Theorem stating that the polynomial has no rational roots -/
theorem no_rational_roots : ∀ x : ℚ, p x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l1571_157173


namespace NUMINAMATH_CALUDE_fundraiser_final_day_amount_l1571_157104

def fundraiser (goal : ℕ) (bronze_donation silver_donation gold_donation : ℕ) 
  (bronze_families silver_families gold_families : ℕ) : ℕ :=
  let total_raised := bronze_donation * bronze_families + 
                      silver_donation * silver_families + 
                      gold_donation * gold_families
  goal - total_raised

theorem fundraiser_final_day_amount : 
  fundraiser 750 25 50 100 10 7 1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_final_day_amount_l1571_157104


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_l1571_157129

theorem max_value_x_plus_y (x y : ℝ) (h : x - 3 * Real.sqrt (x + 1) = 3 * Real.sqrt (y + 2) - y) :
  (∀ a b : ℝ, a - 3 * Real.sqrt (a + 1) = 3 * Real.sqrt (b + 2) - b → x + y ≥ a + b) ∧
  x + y ≤ 9 + 3 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_l1571_157129


namespace NUMINAMATH_CALUDE_replaced_sailor_weight_l1571_157176

/-- Given 8 sailors, if replacing one sailor with a new 64 kg sailor increases
    the average weight by 1 kg, then the replaced sailor's weight was 56 kg. -/
theorem replaced_sailor_weight
  (num_sailors : ℕ)
  (new_sailor_weight : ℕ)
  (avg_weight_increase : ℚ)
  (h1 : num_sailors = 8)
  (h2 : new_sailor_weight = 64)
  (h3 : avg_weight_increase = 1)
  : ℕ :=
by
  sorry

#check replaced_sailor_weight

end NUMINAMATH_CALUDE_replaced_sailor_weight_l1571_157176


namespace NUMINAMATH_CALUDE_largest_n_for_product_l1571_157115

/-- Arithmetic sequence (a_n) -/
def a (n : ℕ) (d : ℤ) : ℤ := 2 + (n - 1 : ℤ) * d

/-- Arithmetic sequence (b_n) -/
def b (n : ℕ) (e : ℤ) : ℤ := 3 + (n - 1 : ℤ) * e

theorem largest_n_for_product (d e : ℤ) (h1 : a 2 d ≤ b 2 e) :
  (∃ n : ℕ, a n d * b n e = 2728) →
  (∀ m : ℕ, a m d * b m e = 2728 → m ≤ 52) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_product_l1571_157115


namespace NUMINAMATH_CALUDE_units_digit_of_13_power_2003_l1571_157162

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to compute the units digit of 3^n
def unitsDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

-- State the theorem
theorem units_digit_of_13_power_2003 :
  unitsDigit (13^2003) = unitsDigitOf3Power 2003 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_13_power_2003_l1571_157162


namespace NUMINAMATH_CALUDE_bank_interest_rate_determination_l1571_157189

/-- Proves that given two equal deposits with the same interest rate but different time periods, 
    if the difference in interest is known, then the interest rate can be determined. -/
theorem bank_interest_rate_determination 
  (principal : ℝ) 
  (time1 time2 : ℝ) 
  (interest_difference : ℝ) : 
  principal = 640 →
  time1 = 3.5 →
  time2 = 5 →
  interest_difference = 144 →
  ∃ (rate : ℝ), 
    (principal * time2 * rate / 100 - principal * time1 * rate / 100 = interest_difference) ∧
    rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_bank_interest_rate_determination_l1571_157189


namespace NUMINAMATH_CALUDE_log_relationship_l1571_157150

theorem log_relationship (c d : ℝ) (hc : c = Real.log 625 / Real.log 4) (hd : d = Real.log 25 / Real.log 5) :
  c = 2 * d := by
  sorry

end NUMINAMATH_CALUDE_log_relationship_l1571_157150


namespace NUMINAMATH_CALUDE_song_storage_size_l1571_157199

-- Define the given values
def total_storage : ℕ := 16  -- in GB
def used_storage : ℕ := 4    -- in GB
def num_songs : ℕ := 400
def mb_per_gb : ℕ := 1000

-- Define the theorem
theorem song_storage_size :
  let available_storage : ℕ := total_storage - used_storage
  let available_storage_mb : ℕ := available_storage * mb_per_gb
  available_storage_mb / num_songs = 30 := by sorry

end NUMINAMATH_CALUDE_song_storage_size_l1571_157199


namespace NUMINAMATH_CALUDE_arithmetic_sequence_14th_term_l1571_157102

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_14th_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_2 : a 2 = 5)
  (h_6 : a 6 = 17) :
  a 14 = 41 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_14th_term_l1571_157102


namespace NUMINAMATH_CALUDE_smallest_3digit_base6_divisible_by_7_l1571_157116

/-- Converts a base 6 number to decimal --/
def base6ToDecimal (n : ℕ) : ℕ :=
  sorry

/-- Converts a decimal number to base 6 --/
def decimalToBase6 (n : ℕ) : ℕ :=
  sorry

/-- Checks if a number is a 3-digit base 6 number --/
def isThreeDigitBase6 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem smallest_3digit_base6_divisible_by_7 :
  ∃ (n : ℕ), isThreeDigitBase6 n ∧ 
             base6ToDecimal n % 7 = 0 ∧
             decimalToBase6 (base6ToDecimal n) = 110 ∧
             ∀ (m : ℕ), isThreeDigitBase6 m ∧ base6ToDecimal m % 7 = 0 → base6ToDecimal n ≤ base6ToDecimal m :=
by sorry

end NUMINAMATH_CALUDE_smallest_3digit_base6_divisible_by_7_l1571_157116


namespace NUMINAMATH_CALUDE_books_divided_l1571_157154

theorem books_divided (num_girls num_boys : ℕ) (total_girls_books : ℕ) : 
  num_girls = 15 →
  num_boys = 10 →
  total_girls_books = 225 →
  (total_girls_books / num_girls) * (num_girls + num_boys) = 375 :=
by sorry

end NUMINAMATH_CALUDE_books_divided_l1571_157154


namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l1571_157126

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l1571_157126


namespace NUMINAMATH_CALUDE_square_difference_of_roots_l1571_157117

theorem square_difference_of_roots (α β : ℝ) : 
  α ≠ β ∧ α^2 - 3*α + 1 = 0 ∧ β^2 - 3*β + 1 = 0 → (α - β)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_roots_l1571_157117


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1571_157133

theorem opposite_of_2023 : 
  ∀ y : ℤ, (2023 + y = 0) ↔ (y = -2023) := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1571_157133


namespace NUMINAMATH_CALUDE_dice_sum_possibilities_l1571_157157

/-- The number of dice being rolled -/
def num_dice : ℕ := 4

/-- The minimum value on a die face -/
def min_face : ℕ := 1

/-- The maximum value on a die face -/
def max_face : ℕ := 6

/-- The minimum possible sum when rolling the dice -/
def min_sum : ℕ := num_dice * min_face

/-- The maximum possible sum when rolling the dice -/
def max_sum : ℕ := num_dice * max_face

/-- The number of distinct possible sums when rolling the dice -/
def num_distinct_sums : ℕ := max_sum - min_sum + 1

theorem dice_sum_possibilities : num_distinct_sums = 21 := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_possibilities_l1571_157157


namespace NUMINAMATH_CALUDE_fifth_patient_cure_rate_l1571_157192

theorem fifth_patient_cure_rate 
  (cure_rate : ℝ) 
  (h_cure_rate : cure_rate = 1/5) 
  (first_four_patients : Fin 4 → Bool) 
  : ℝ :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_patient_cure_rate_l1571_157192


namespace NUMINAMATH_CALUDE_state_selection_difference_l1571_157113

theorem state_selection_difference (total_candidates : ℕ) 
  (selection_rate_A selection_rate_B : ℚ) : 
  total_candidates = 8000 →
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B * total_candidates : ℚ) - (selection_rate_A * total_candidates : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_state_selection_difference_l1571_157113


namespace NUMINAMATH_CALUDE_triangle_function_properties_l1571_157132

open Real

theorem triangle_function_properties (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sine_law : sin A / a = sin B / b ∧ sin B / b = sin C / c)
  (f : ℝ → ℝ)
  (h_f_def : ∀ x, f x = sin (2*x + B) + sqrt 3 * cos (2*x + B))
  (h_f_odd : ∀ x, f (x - π/3) = -f (-(x - π/3))) :
  B = π/3 ∧
  (∀ x, f x = 2 * sin (2*x + 2*π/3)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k*π - 7*π/12) (k*π - π/12), 
    ∀ y ∈ Set.Icc (k*π - 7*π/12) (k*π - π/12), 
    x < y → f x < f y) ∧
  (a = 1 → b = f 0 → (1/2) * a * b * sin C = sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_function_properties_l1571_157132


namespace NUMINAMATH_CALUDE_train_speed_l1571_157130

theorem train_speed (train_length : Real) (man_speed : Real) (passing_time : Real) :
  train_length = 220 →
  man_speed = 6 →
  passing_time = 12 →
  (train_length / 1000) / (passing_time / 3600) - man_speed = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1571_157130


namespace NUMINAMATH_CALUDE_bandwidth_calculation_correct_l1571_157110

/-- Represents the parameters for an audio channel --/
structure AudioChannelParams where
  sessionDurationMinutes : ℕ
  samplingRate : ℕ
  samplingDepth : ℕ
  metadataBytes : ℕ
  metadataPerAudioKilobits : ℕ

/-- Calculates the required bandwidth for a stereo audio channel --/
def calculateBandwidth (params : AudioChannelParams) : ℚ :=
  let sessionDurationSeconds := params.sessionDurationMinutes * 60
  let dataVolume := params.samplingRate * params.samplingDepth * sessionDurationSeconds
  let metadataVolume := params.metadataBytes * 8 * dataVolume / (params.metadataPerAudioKilobits * 1024)
  let totalDataVolume := (dataVolume + metadataVolume) * 2
  totalDataVolume / (sessionDurationSeconds * 1024)

/-- Theorem stating that the calculated bandwidth matches the expected result --/
theorem bandwidth_calculation_correct (params : AudioChannelParams) 
  (h1 : params.sessionDurationMinutes = 51)
  (h2 : params.samplingRate = 63)
  (h3 : params.samplingDepth = 17)
  (h4 : params.metadataBytes = 47)
  (h5 : params.metadataPerAudioKilobits = 5) :
  calculateBandwidth params = 2.25 := by
  sorry

#eval calculateBandwidth {
  sessionDurationMinutes := 51,
  samplingRate := 63,
  samplingDepth := 17,
  metadataBytes := 47,
  metadataPerAudioKilobits := 5
}

end NUMINAMATH_CALUDE_bandwidth_calculation_correct_l1571_157110


namespace NUMINAMATH_CALUDE_felix_tree_chopping_l1571_157198

/-- Given that Felix needs to resharpen his axe every 13 trees, each sharpening costs $5,
    and he has spent $35 on sharpening, prove that he has chopped down at least 91 trees. -/
theorem felix_tree_chopping (trees_per_sharpening : ℕ) (cost_per_sharpening : ℕ) (total_spent : ℕ) 
    (h1 : trees_per_sharpening = 13)
    (h2 : cost_per_sharpening = 5)
    (h3 : total_spent = 35) :
  trees_per_sharpening * (total_spent / cost_per_sharpening) ≥ 91 := by
  sorry

#check felix_tree_chopping

end NUMINAMATH_CALUDE_felix_tree_chopping_l1571_157198


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1571_157118

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a + b + c = 40 →
  (1/2) * a * b = 24 →
  a^2 + b^2 = c^2 →
  c = 18.8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1571_157118


namespace NUMINAMATH_CALUDE_awards_assignment_count_l1571_157146

/-- The number of different types of awards -/
def num_awards : ℕ := 4

/-- The number of students -/
def num_students : ℕ := 8

/-- The total number of ways to assign awards -/
def total_assignments : ℕ := num_awards ^ num_students

/-- Theorem stating that the total number of assignments is 65536 -/
theorem awards_assignment_count :
  total_assignments = 65536 := by
  sorry

end NUMINAMATH_CALUDE_awards_assignment_count_l1571_157146


namespace NUMINAMATH_CALUDE_julia_bought_399_balls_l1571_157149

/-- The number of balls Julia bought -/
def total_balls (red_packs yellow_packs green_packs balls_per_pack : ℕ) : ℕ :=
  (red_packs + yellow_packs + green_packs) * balls_per_pack

/-- Theorem stating that Julia bought 399 balls in total -/
theorem julia_bought_399_balls :
  total_balls 3 10 8 19 = 399 := by
  sorry

end NUMINAMATH_CALUDE_julia_bought_399_balls_l1571_157149


namespace NUMINAMATH_CALUDE_james_pages_per_year_l1571_157123

/-- Calculates the number of pages James writes in a year -/
def pages_per_year (pages_per_letter : ℕ) (friends : ℕ) (times_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  pages_per_letter * friends * times_per_week * weeks_per_year

/-- Proves that James writes 624 pages in a year -/
theorem james_pages_per_year :
  pages_per_year 3 2 2 52 = 624 := by
  sorry

end NUMINAMATH_CALUDE_james_pages_per_year_l1571_157123


namespace NUMINAMATH_CALUDE_cosh_inequality_l1571_157163

theorem cosh_inequality (c : ℝ) :
  (∀ x : ℝ, (Real.exp x + Real.exp (-x)) / 2 ≤ Real.exp (c * x^2)) ↔ c ≥ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_cosh_inequality_l1571_157163


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1571_157191

theorem negation_of_proposition :
  (¬(∀ x : ℝ, x ≥ 1 → x^2 - 4*x + 2 ≥ -1)) ↔ (∃ x : ℝ, x < 1 ∧ x^2 - 4*x + 2 < -1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1571_157191


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1571_157147

theorem scientific_notation_equivalence : 0.0000036 = 3.6 * 10^(-6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1571_157147


namespace NUMINAMATH_CALUDE_principal_calculation_l1571_157155

/-- Proves that given specific conditions, the principal amount is 1400 --/
theorem principal_calculation (rate : ℝ) (time : ℝ) (amount : ℝ) :
  rate = 0.05 →
  time = 2.4 →
  amount = 1568 →
  (∃ (principal : ℝ), principal * (1 + rate * time) = amount ∧ principal = 1400) :=
by sorry

end NUMINAMATH_CALUDE_principal_calculation_l1571_157155


namespace NUMINAMATH_CALUDE_sum_of_factorials_1_to_10_l1571_157165

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_of_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => sum_of_factorials n + factorial (n + 1)

theorem sum_of_factorials_1_to_10 : sum_of_factorials 10 = 4037913 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factorials_1_to_10_l1571_157165


namespace NUMINAMATH_CALUDE_amusement_park_running_cost_l1571_157169

/-- The amusement park problem -/
theorem amusement_park_running_cost 
  (initial_cost : ℝ) 
  (daily_tickets : ℕ) 
  (ticket_price : ℝ) 
  (days_to_breakeven : ℕ) 
  (h1 : initial_cost = 100000)
  (h2 : daily_tickets = 150)
  (h3 : ticket_price = 10)
  (h4 : days_to_breakeven = 200) :
  let daily_revenue := daily_tickets * ticket_price
  let total_revenue := daily_revenue * days_to_breakeven
  let daily_running_cost_percentage := 
    (total_revenue - initial_cost) / (initial_cost * days_to_breakeven) * 100
  daily_running_cost_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_amusement_park_running_cost_l1571_157169


namespace NUMINAMATH_CALUDE_bag_contents_theorem_l1571_157190

/-- Represents the contents of a bag of colored balls. -/
structure BagContents where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of selecting two red balls. -/
def probTwoRed (bag : BagContents) : ℚ :=
  (bag.red.choose 2 : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ)

/-- Calculates the probability of selecting one red and one yellow ball. -/
def probRedYellow (bag : BagContents) : ℚ :=
  ((bag.red.choose 1 * bag.yellow.choose 1) : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ)

/-- Calculates the expected value of the number of red balls selected. -/
def expectedRedBalls (bag : BagContents) : ℚ :=
  0 * ((bag.yellow + bag.green).choose 2 : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ) +
  1 * ((bag.red.choose 1 * (bag.yellow + bag.green).choose 1) : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ) +
  2 * (bag.red.choose 2 : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ)

/-- The main theorem stating the properties of the bag contents and expected value. -/
theorem bag_contents_theorem (bag : BagContents) :
  bag.red = 4 ∧ 
  probTwoRed bag = 1/6 ∧ 
  probRedYellow bag = 1/3 → 
  bag.yellow - bag.green = 1 ∧ 
  expectedRedBalls bag = 8/9 := by
  sorry


end NUMINAMATH_CALUDE_bag_contents_theorem_l1571_157190


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l1571_157197

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l1571_157197


namespace NUMINAMATH_CALUDE_club_membership_l1571_157109

theorem club_membership (total : Nat) (left_handed : Nat) (jazz_lovers : Nat) (right_handed_jazz_dislikers : Nat) :
  total = 25 →
  left_handed = 12 →
  jazz_lovers = 18 →
  right_handed_jazz_dislikers = 3 →
  (∃ (left_handed_jazz_lovers : Nat),
    left_handed_jazz_lovers +
    (left_handed - left_handed_jazz_lovers) +
    (jazz_lovers - left_handed_jazz_lovers) +
    right_handed_jazz_dislikers = total ∧
    left_handed_jazz_lovers = 8) := by
  sorry

#check club_membership

end NUMINAMATH_CALUDE_club_membership_l1571_157109


namespace NUMINAMATH_CALUDE_max_at_two_implies_a_geq_neg_half_l1571_157139

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * (a + 1) * x - 3

-- State the theorem
theorem max_at_two_implies_a_geq_neg_half (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ f a 2) →
  a ≥ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_at_two_implies_a_geq_neg_half_l1571_157139


namespace NUMINAMATH_CALUDE_f_intersects_iff_m_le_one_l1571_157106

/-- The quadratic function f(x) = mx^2 + (m-3)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 3) * x + 1

/-- The condition that f intersects the x-axis with at least one point to the right of the origin -/
def intersects_positive_x (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ f m x = 0

theorem f_intersects_iff_m_le_one :
  ∀ m : ℝ, intersects_positive_x m ↔ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_intersects_iff_m_le_one_l1571_157106


namespace NUMINAMATH_CALUDE_T_recursive_relation_l1571_157128

/-- The number of binary strings of length n such that any 4 adjacent digits sum to at least 1 -/
def T (n : ℕ) : ℕ :=
  if n < 4 then
    match n with
    | 0 => 1  -- Convention: empty string is valid
    | 1 => 2  -- "0" and "1" are valid
    | 2 => 3  -- "00", "01", "10", "11" are valid except "00"
    | 3 => 6  -- All combinations except "0000"
    | _ => 0  -- Should never reach here
  else
    T (n - 1) + T (n - 2) + T (n - 3) + T (n - 4)

/-- The main theorem stating the recursive relation for T(n) when n ≥ 4 -/
theorem T_recursive_relation (n : ℕ) (h : n ≥ 4) :
  T n = T (n - 1) + T (n - 2) + T (n - 3) + T (n - 4) := by sorry

end NUMINAMATH_CALUDE_T_recursive_relation_l1571_157128


namespace NUMINAMATH_CALUDE_binary_11001_is_25_l1571_157194

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11001_is_25 :
  binary_to_decimal [true, false, false, true, true] = 25 := by
  sorry

end NUMINAMATH_CALUDE_binary_11001_is_25_l1571_157194


namespace NUMINAMATH_CALUDE_rachel_budget_l1571_157179

/-- Given Sara's expenses and Rachel's spending intention, calculate Rachel's budget. -/
theorem rachel_budget (sara_shoes : ℕ) (sara_dress : ℕ) (rachel_multiplier : ℕ) : 
  sara_shoes = 50 → sara_dress = 200 → rachel_multiplier = 2 →
  (rachel_multiplier * sara_shoes + rachel_multiplier * sara_dress) = 500 := by
  sorry

#check rachel_budget

end NUMINAMATH_CALUDE_rachel_budget_l1571_157179


namespace NUMINAMATH_CALUDE_binary_sum_equals_638_l1571_157119

/-- The sum of the binary numbers 111111111₂ and 1111111₂ is equal to 638 in base 10. -/
theorem binary_sum_equals_638 : 
  (2^9 - 1) + (2^7 - 1) = 638 := by
  sorry

#check binary_sum_equals_638

end NUMINAMATH_CALUDE_binary_sum_equals_638_l1571_157119


namespace NUMINAMATH_CALUDE_sector_area_l1571_157111

/-- A sector with perimeter 12 cm and central angle 2 rad has an area of 9 cm² -/
theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (area : ℝ) : 
  perimeter = 12 → central_angle = 2 → area = 9 := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l1571_157111
