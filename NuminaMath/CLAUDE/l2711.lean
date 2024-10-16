import Mathlib

namespace NUMINAMATH_CALUDE_pencil_count_l2711_271144

/-- Proves that given the ratio of pens to pencils is 5:6 and there are 9 more pencils than pens, the number of pencils is 54. -/
theorem pencil_count (pens pencils : ℕ) : 
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 9 →
  pencils = 54 := by
sorry

end NUMINAMATH_CALUDE_pencil_count_l2711_271144


namespace NUMINAMATH_CALUDE_taco_castle_parking_lot_l2711_271164

theorem taco_castle_parking_lot : 
  ∀ (volkswagen ford toyota dodge : ℕ),
    volkswagen = 5 →
    toyota = 2 * volkswagen →
    ford = 2 * toyota →
    3 * ford = dodge →
    dodge = 60 := by
  sorry

end NUMINAMATH_CALUDE_taco_castle_parking_lot_l2711_271164


namespace NUMINAMATH_CALUDE_unique_grid_placement_l2711_271194

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- The sum of adjacent numbers is less than 12 --/
def valid_sum (g : Grid) : Prop :=
  ∀ p1 p2 : Fin 3 × Fin 3, adjacent p1 p2 → g p1.1 p1.2 + g p2.1 p2.2 < 12

/-- The grid contains all numbers from 1 to 9 --/
def contains_all_numbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n.val + 1

/-- The given positions of odd numbers --/
def given_positions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7 ∧ g 0 2 = 9

/-- The theorem to be proved --/
theorem unique_grid_placement :
  ∀ g : Grid,
    valid_sum g →
    contains_all_numbers g →
    given_positions g →
    g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_grid_placement_l2711_271194


namespace NUMINAMATH_CALUDE_pumpkins_left_l2711_271179

theorem pumpkins_left (grown : ℕ) (eaten : ℕ) (h1 : grown = 43) (h2 : eaten = 23) :
  grown - eaten = 20 := by
  sorry

end NUMINAMATH_CALUDE_pumpkins_left_l2711_271179


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2711_271130

/-- Given a geometric sequence {a_n} where a_2 and a_3 are the roots of x^2 - x - 2013 = 0,
    prove that a_1 * a_4 = -2013 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n) →  -- geometric sequence condition
  a 2^2 - a 2 - 2013 = 0 →  -- a_2 is a root
  a 3^2 - a 3 - 2013 = 0 →  -- a_3 is a root
  a 1 * a 4 = -2013 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2711_271130


namespace NUMINAMATH_CALUDE_integer_pair_solution_l2711_271142

theorem integer_pair_solution : 
  ∀ m n : ℕ+, m^2 + 2*n^2 = 3*(m + 2*n) ↔ (m = 3 ∧ n = 3) ∨ (m = 4 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_integer_pair_solution_l2711_271142


namespace NUMINAMATH_CALUDE_root_product_equals_32_l2711_271116

theorem root_product_equals_32 : 
  (256 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_32_l2711_271116


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l2711_271138

/-- Given that the solution set of the inequality (ax-1)(x+1)<0 is (-∞, -1) ∪ (-1/2, +∞),
    prove that a = -2 -/
theorem inequality_solution_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, (a*x - 1)*(x + 1) < 0 ↔ x < -1 ∨ -1/2 < x) → a = -2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l2711_271138


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2711_271169

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed against the current is 16 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 21 2.5 = 16 := by
  sorry

#eval speed_against_current 21 2.5

end NUMINAMATH_CALUDE_mans_speed_against_current_l2711_271169


namespace NUMINAMATH_CALUDE_largest_valid_B_l2711_271137

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 100000
  let d2 := (n / 10000) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 100) % 10
  let d5 := (n / 10) % 10
  let d6 := n % 10
  d1 + d2 + d3 + d4 + d5 + d6

def is_valid_B (B : ℕ) : Prop :=
  B < 10 ∧ 
  is_divisible_by_3 (sum_of_digits (400000 + B * 10000 + 4832)) ∧
  is_divisible_by_4 (last_two_digits (400000 + B * 10000 + 4832))

theorem largest_valid_B :
  ∀ B, is_valid_B B → B ≤ 9 ∧ is_valid_B 9 := by sorry

end NUMINAMATH_CALUDE_largest_valid_B_l2711_271137


namespace NUMINAMATH_CALUDE_subtraction_of_like_terms_l2711_271128

theorem subtraction_of_like_terms (a b : ℝ) : 5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_like_terms_l2711_271128


namespace NUMINAMATH_CALUDE_max_product_of_fractions_l2711_271140

def is_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

theorem max_product_of_fractions (A B C D : ℕ) 
  (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hD : is_digit D)
  (hAB : A ≠ B) (hAC : A ≠ C) (hAD : A ≠ D) (hBC : B ≠ C) (hBD : B ≠ D) (hCD : C ≠ D) :
  (∀ (W X Y Z : ℕ), is_digit W → is_digit X → is_digit Y → is_digit Z →
    W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
    (A : ℚ) / B * (C : ℚ) / D ≥ (W : ℚ) / X * (Y : ℚ) / Z) →
  (A : ℚ) / B * (C : ℚ) / D = 36 := by
sorry

end NUMINAMATH_CALUDE_max_product_of_fractions_l2711_271140


namespace NUMINAMATH_CALUDE_exam_score_proof_l2711_271108

/-- Given an exam with mean score 76, prove that the score 2 standard deviations
    below the mean is 60, knowing that 100 is 3 standard deviations above the mean. -/
theorem exam_score_proof (mean : ℝ) (score_above : ℝ) (std_dev_above : ℝ) (std_dev_below : ℝ) :
  mean = 76 →
  score_above = 100 →
  std_dev_above = 3 →
  std_dev_below = 2 →
  score_above = mean + std_dev_above * ((score_above - mean) / std_dev_above) →
  mean - std_dev_below * ((score_above - mean) / std_dev_above) = 60 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_proof_l2711_271108


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2711_271133

theorem quadratic_inequality_solution (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, ax^2 - (a + 1)*x + 1 < 0 ↔ 
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨ 
     (a > 1 ∧ 1/a < x ∧ x < 1))) ∧
  (a = 1 → ∀ x : ℝ, ¬(x^2 - 2*x + 1 < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2711_271133


namespace NUMINAMATH_CALUDE_apartment_fractions_l2711_271147

theorem apartment_fractions (one_bedroom : Real) (two_bedroom : Real) 
  (h1 : one_bedroom = 0.17)
  (h2 : one_bedroom + two_bedroom = 0.5) :
  two_bedroom = 0.33 := by
sorry

end NUMINAMATH_CALUDE_apartment_fractions_l2711_271147


namespace NUMINAMATH_CALUDE_probability_of_exact_tails_l2711_271115

noncomputable def probability_of_tails : ℚ := 2/3
noncomputable def number_of_flips : ℕ := 10
noncomputable def number_of_tails : ℕ := 4

theorem probability_of_exact_tails :
  (Nat.choose number_of_flips number_of_tails) *
  (probability_of_tails ^ number_of_tails) *
  ((1 - probability_of_tails) ^ (number_of_flips - number_of_tails)) =
  3360/6561 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_exact_tails_l2711_271115


namespace NUMINAMATH_CALUDE_function_min_value_l2711_271134

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem function_min_value 
  (h_max : ∃ (m : ℝ), ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3 ∧ ∃ y ∈ Set.Icc (-2 : ℝ) 2, f y m = 3) :
  ∃ (m : ℝ), ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ -37 ∧ ∃ y ∈ Set.Icc (-2 : ℝ) 2, f y m = -37 :=
sorry

end NUMINAMATH_CALUDE_function_min_value_l2711_271134


namespace NUMINAMATH_CALUDE_kangaroo_jumps_odd_jumps_zero_four_jumps_two_l2711_271188

/-- Represents a regular octagon with vertices labeled from 0 to 7 -/
def Octagon := Fin 8

/-- Defines whether two vertices are adjacent in the octagon -/
def adjacent (v w : Octagon) : Prop :=
  (v.val + 1) % 8 = w.val ∨ (w.val + 1) % 8 = v.val

/-- Defines the number of ways a kangaroo can reach vertex E from A in n jumps -/
def num_ways (n : ℕ) : ℕ :=
  sorry -- Definition to be implemented

/-- Main theorem: Characterizes the number of ways to reach E from A in n jumps -/
theorem kangaroo_jumps (n : ℕ) :
  num_ways n = if n % 2 = 0
    then let m := n / 2
         (((2 : ℝ) + Real.sqrt 2) ^ (m - 1) - ((2 : ℝ) - Real.sqrt 2) ^ (m - 1)) / Real.sqrt 2
    else 0 :=
  sorry

/-- The number of ways to reach E from A in an odd number of jumps is 0 -/
theorem odd_jumps_zero (n : ℕ) (h : n % 2 = 1) :
  num_ways n = 0 :=
  sorry

/-- The number of ways to reach E from A in 4 jumps is 2 -/
theorem four_jumps_two :
  num_ways 4 = 2 :=
  sorry

end NUMINAMATH_CALUDE_kangaroo_jumps_odd_jumps_zero_four_jumps_two_l2711_271188


namespace NUMINAMATH_CALUDE_range_of_a_l2711_271184

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_a (a : ℝ) (h : a ∈ A) : a ∈ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2711_271184


namespace NUMINAMATH_CALUDE_fermat_fourth_power_l2711_271152

theorem fermat_fourth_power (x y z : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^4 + y^4 ≠ z^4 := by
  sorry

end NUMINAMATH_CALUDE_fermat_fourth_power_l2711_271152


namespace NUMINAMATH_CALUDE_a_5_value_l2711_271185

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = r * (a n - a (n - 1))

theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 1 - a 0 = 1 →
  a 5 = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_a_5_value_l2711_271185


namespace NUMINAMATH_CALUDE_max_rectangle_area_l2711_271166

def perimeter : ℝ := 300
def min_length : ℝ := 80
def min_width : ℝ := 40

def rectangle_area (l w : ℝ) : ℝ := l * w

theorem max_rectangle_area :
  ∀ l w : ℝ,
  l ≥ min_length →
  w ≥ min_width →
  2 * l + 2 * w = perimeter →
  rectangle_area l w ≤ 5600 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l2711_271166


namespace NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l2711_271146

theorem three_numbers_in_unit_interval (x y z : ℝ) :
  (0 ≤ x ∧ x < 1) → (0 ≤ y ∧ y < 1) → (0 ≤ z ∧ z < 1) →
  ∃ a b : ℝ, (a = x ∨ a = y ∨ a = z) ∧ (b = x ∨ b = y ∨ b = z) ∧ a ≠ b ∧ |b - a| < (1/2) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l2711_271146


namespace NUMINAMATH_CALUDE_dividend_rate_is_twelve_percent_l2711_271173

/-- Calculates the rate of dividend given investment details and annual income -/
def calculate_dividend_rate (investment : ℚ) (share_face_value : ℚ) (share_quoted_price : ℚ) (annual_income : ℚ) : ℚ :=
  let num_shares : ℚ := investment / share_quoted_price
  annual_income / (num_shares * share_face_value) * 100

/-- Proves that the rate of dividend is 12% given the specified conditions -/
theorem dividend_rate_is_twelve_percent 
  (investment : ℚ) 
  (share_face_value : ℚ) 
  (share_quoted_price : ℚ) 
  (annual_income : ℚ) 
  (h1 : investment = 4455)
  (h2 : share_face_value = 10)
  (h3 : share_quoted_price = 825/100)
  (h4 : annual_income = 648) :
  calculate_dividend_rate investment share_face_value share_quoted_price annual_income = 12 := by
  sorry

end NUMINAMATH_CALUDE_dividend_rate_is_twelve_percent_l2711_271173


namespace NUMINAMATH_CALUDE_symmetric_decreasing_property_l2711_271126

/-- A function f: ℝ → ℝ that is decreasing on (4, +∞) and symmetric about x = 4 -/
def SymmetricDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 4 ∧ y > x → f y < f x) ∧
  (∀ x, f (4 + x) = f (4 - x))

/-- Given a symmetric decreasing function f, prove that f(3) > f(6) -/
theorem symmetric_decreasing_property (f : ℝ → ℝ) 
  (h : SymmetricDecreasingFunction f) : f 3 > f 6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_decreasing_property_l2711_271126


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2711_271111

theorem inequality_solution_set (x : ℝ) : x + 2 > 3 ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2711_271111


namespace NUMINAMATH_CALUDE_all_statements_false_l2711_271136

-- Define the concepts of lines and planes
variable (Line Plane : Type)

-- Define the concept of parallelism between lines
variable (parallel_lines : Line → Line → Prop)

-- Define the concept of parallelism between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the concept of perpendicularity between lines
variable (perpendicular : Line → Line → Prop)

-- Define the concept of a line having no common points with another line
variable (no_common_points : Line → Line → Prop)

-- Define the concept of a line having no common points with countless lines in a plane
variable (no_common_points_with_plane_lines : Line → Plane → Prop)

theorem all_statements_false :
  (∀ (l₁ l₂ : Line) (p : Plane), parallel_line_plane l₁ p → parallel_line_plane l₂ p → parallel_lines l₁ l₂) = False ∧
  (∀ (l₁ l₂ : Line), no_common_points l₁ l₂ → parallel_lines l₁ l₂) = False ∧
  (∀ (l₁ l₂ l₃ : Line), perpendicular l₁ l₃ → perpendicular l₂ l₃ → parallel_lines l₁ l₂) = False ∧
  (∀ (l : Line) (p : Plane), no_common_points_with_plane_lines l p → parallel_line_plane l p) = False :=
sorry

end NUMINAMATH_CALUDE_all_statements_false_l2711_271136


namespace NUMINAMATH_CALUDE_final_pen_count_l2711_271100

def pen_collection (initial : ℕ) (mike_gives : ℕ) (cindy_doubles : Bool) 
  (alex_takes_percent : ℚ) (sharon_gets : ℕ) : ℕ :=
  let after_mike := initial + mike_gives
  let after_cindy := if cindy_doubles then 2 * after_mike else after_mike
  let alex_takes := (alex_takes_percent * after_cindy).ceil.toNat
  let after_alex := after_cindy - alex_takes
  after_alex - sharon_gets

theorem final_pen_count : 
  pen_collection 20 22 true (15 / 100) 19 = 52 := by sorry

end NUMINAMATH_CALUDE_final_pen_count_l2711_271100


namespace NUMINAMATH_CALUDE_imaginary_roots_sum_of_magnitudes_l2711_271117

theorem imaginary_roots_sum_of_magnitudes (m : ℝ) : 
  (∃ α β : ℂ, (3 * α^2 - 6*(m - 1)*α + m^2 + 1 = 0) ∧ 
               (3 * β^2 - 6*(m - 1)*β + m^2 + 1 = 0) ∧ 
               (α.im ≠ 0) ∧ (β.im ≠ 0) ∧
               (Complex.abs α + Complex.abs β = 2)) →
  m = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_roots_sum_of_magnitudes_l2711_271117


namespace NUMINAMATH_CALUDE_unique_element_implies_a_equals_four_l2711_271199

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + a * x + 1 = 0}

-- State the theorem
theorem unique_element_implies_a_equals_four :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ A a) → a = 4 := by sorry

end NUMINAMATH_CALUDE_unique_element_implies_a_equals_four_l2711_271199


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l2711_271186

-- Define what it means for a number to be rational
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality as the negation of rationality
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_two_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l2711_271186


namespace NUMINAMATH_CALUDE_remainder_seven_pow_2023_mod_5_l2711_271163

theorem remainder_seven_pow_2023_mod_5 : 7^2023 % 5 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_seven_pow_2023_mod_5_l2711_271163


namespace NUMINAMATH_CALUDE_robie_cards_l2711_271193

/-- The number of cards in each box -/
def cards_per_box : ℕ := 10

/-- The number of cards not placed in a box -/
def loose_cards : ℕ := 5

/-- The number of boxes Robie gave away -/
def boxes_given_away : ℕ := 2

/-- The number of boxes Robie has left -/
def boxes_left : ℕ := 5

/-- The total number of cards Robie had in the beginning -/
def total_cards : ℕ := (boxes_given_away + boxes_left) * cards_per_box + loose_cards

theorem robie_cards : total_cards = 75 := by
  sorry

end NUMINAMATH_CALUDE_robie_cards_l2711_271193


namespace NUMINAMATH_CALUDE_water_depth_in_cylinder_l2711_271107

/-- Represents the depth of water in a horizontal cylindrical tank. -/
def water_depth (tank_length tank_diameter water_surface_area : ℝ) : Set ℝ :=
  {h : ℝ | ∃ (w : ℝ), 
    tank_length > 0 ∧ 
    tank_diameter > 0 ∧ 
    water_surface_area > 0 ∧
    w > 0 ∧ 
    h > 0 ∧ 
    h < tank_diameter ∧
    w * tank_length = water_surface_area ∧
    w = 2 * Real.sqrt (tank_diameter * h - h^2)}

/-- The main theorem stating the depth of water in the given cylindrical tank. -/
theorem water_depth_in_cylinder : 
  water_depth 12 4 24 = {2 - Real.sqrt 3, 2 + Real.sqrt 3} := by
  sorry


end NUMINAMATH_CALUDE_water_depth_in_cylinder_l2711_271107


namespace NUMINAMATH_CALUDE_triangle_problem_l2711_271162

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  A = π/4 ∧
  b = Real.sqrt 6 ∧
  (1/2) * b * c * Real.sin A = (3 + Real.sqrt 3)/2 →
  c = 1 + Real.sqrt 3 ∧ B = π/3 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2711_271162


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2711_271139

def P : Set ℝ := {x | |x - 1| < 4}
def Q : Set ℝ := {x | ∃ y, y = Real.log (x + 2)}

theorem intersection_P_Q : P ∩ Q = Set.Ioo (-2 : ℝ) 5 := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2711_271139


namespace NUMINAMATH_CALUDE_correct_calculation_l2711_271122

theorem correct_calculation (a b : ℝ) : (a * b)^2 / (-a * b) = -a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2711_271122


namespace NUMINAMATH_CALUDE_boys_in_art_class_l2711_271160

theorem boys_in_art_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) 
  (h1 : total = 35) 
  (h2 : ratio_girls = 4) 
  (h3 : ratio_boys = 3) : 
  (ratio_boys * total) / (ratio_girls + ratio_boys) = 15 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_art_class_l2711_271160


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l2711_271120

/-- A quadratic function passing through (2, 5) with vertex at (1, 3) has a - b + c = 11 -/
theorem quadratic_coefficient_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (x - 1)^2 + 3) →  -- vertex form
  a * 2^2 + b * 2 + c = 5 →                         -- passes through (2, 5)
  a - b + c = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l2711_271120


namespace NUMINAMATH_CALUDE_eighth_equation_sum_l2711_271192

theorem eighth_equation_sum (a t : ℝ) (ha : a > 0) (ht : t > 0) :
  (8 + a / t).sqrt = 8 * (a / t).sqrt → a + t = 71 := by
  sorry

end NUMINAMATH_CALUDE_eighth_equation_sum_l2711_271192


namespace NUMINAMATH_CALUDE_current_rate_l2711_271149

/-- Calculates the rate of the current given a man's rowing speeds -/
theorem current_rate (downstream_speed upstream_speed still_water_speed : ℝ) 
  (h1 : downstream_speed = 32)
  (h2 : upstream_speed = 17)
  (h3 : still_water_speed = 24.5)
  : (downstream_speed - still_water_speed) = 7.5 := by
  sorry

#check current_rate

end NUMINAMATH_CALUDE_current_rate_l2711_271149


namespace NUMINAMATH_CALUDE_odd_numbers_property_l2711_271102

theorem odd_numbers_property (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_odd_numbers_property_l2711_271102


namespace NUMINAMATH_CALUDE_power_multiplication_l2711_271154

theorem power_multiplication (m : ℝ) : m^5 * m = m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2711_271154


namespace NUMINAMATH_CALUDE_telescope_visual_range_l2711_271170

theorem telescope_visual_range (original_range : ℝ) : 
  (original_range + 1.5 * original_range = 150) → original_range = 60 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l2711_271170


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l2711_271105

/-- Given a point A with coordinates (2, 3), its symmetric point with respect to the x-axis has coordinates (2, -3). -/
theorem symmetric_point_wrt_x_axis :
  let A : ℝ × ℝ := (2, 3)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point A = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l2711_271105


namespace NUMINAMATH_CALUDE_smallest_integer_larger_than_sqrt3_minus_sqrt2_to_6th_l2711_271196

theorem smallest_integer_larger_than_sqrt3_minus_sqrt2_to_6th :
  ∃ n : ℤ, (n = 133 ∧ (∀ m : ℤ, m > (Real.sqrt 3 - Real.sqrt 2)^6 → m ≥ n)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_larger_than_sqrt3_minus_sqrt2_to_6th_l2711_271196


namespace NUMINAMATH_CALUDE_factor_divisor_statements_l2711_271172

theorem factor_divisor_statements : 
  (∃ n : ℕ, 45 = 5 * n) ∧ 
  (∃ m : ℕ, 209 = 19 * m) ∧ 
  (¬ ∃ k : ℕ, 63 = 19 * k) ∧ 
  (∃ p : ℕ, 96 = 12 * p) := by
  sorry

end NUMINAMATH_CALUDE_factor_divisor_statements_l2711_271172


namespace NUMINAMATH_CALUDE_scaling_transformation_maps_line_l2711_271135

/-- A scaling transformation in 2D space -/
structure ScalingTransformation where
  x_scale : ℝ
  y_scale : ℝ

/-- The original line equation -/
def original_line (x y : ℝ) : Prop := x + y + 2 = 0

/-- The transformed line equation -/
def transformed_line (x y : ℝ) : Prop := 8*x + y + 8 = 0

/-- Theorem stating that the given scaling transformation maps the original line to the transformed line -/
theorem scaling_transformation_maps_line :
  ∃ (t : ScalingTransformation),
    (∀ (x y : ℝ), original_line x y ↔ transformed_line (t.x_scale * x) (t.y_scale * y)) ∧
    t.x_scale = 1/2 ∧ t.y_scale = 4 := by
  sorry

end NUMINAMATH_CALUDE_scaling_transformation_maps_line_l2711_271135


namespace NUMINAMATH_CALUDE_apples_left_is_ten_l2711_271177

/-- Represents the number of apples picked by Mike -/
def mike_apples : ℕ := 12

/-- Represents the number of apples eaten by Nancy -/
def nancy_apples : ℕ := 7

/-- Represents the number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- Represents the number of pears picked by Keith -/
def keith_pears : ℕ := 4

/-- Represents the number of apples picked by Christine -/
def christine_apples : ℕ := 10

/-- Represents the number of pears picked by Christine -/
def christine_pears : ℕ := 3

/-- Represents the number of bananas picked by Christine -/
def christine_bananas : ℕ := 5

/-- Represents the number of apples eaten by Greg -/
def greg_apples : ℕ := 9

/-- Represents the number of peaches picked by an unknown person -/
def unknown_peaches : ℕ := 14

/-- Represents the number of plums picked by an unknown person -/
def unknown_plums : ℕ := 7

/-- Represents the ratio of pears picked to apples disappeared -/
def pears_per_apple : ℕ := 3

/-- Theorem stating that the number of apples left is 10 -/
theorem apples_left_is_ten : 
  mike_apples + keith_apples + christine_apples - 
  nancy_apples - greg_apples - 
  ((keith_pears + christine_pears) / pears_per_apple) = 10 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_is_ten_l2711_271177


namespace NUMINAMATH_CALUDE_prob_consonant_correct_l2711_271148

/-- The word from which letters are selected -/
def word : String := "barkhint"

/-- The number of letters in the word -/
def word_length : Nat := word.length

/-- The number of vowels in the word -/
def vowel_count : Nat := (word.toList.filter (fun c => c ∈ ['a', 'e', 'i', 'o', 'u'])).length

/-- The probability of selecting at least one consonant when choosing two letters at random -/
def prob_at_least_one_consonant : ℚ := 27 / 28

/-- Theorem stating that the probability of selecting at least one consonant
    when choosing two letters at random from the word "barkhint" is 27/28 -/
theorem prob_consonant_correct :
  prob_at_least_one_consonant = 1 - (vowel_count / word_length) * ((vowel_count - 1) / (word_length - 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_prob_consonant_correct_l2711_271148


namespace NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l2711_271101

theorem least_positive_integer_for_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (528 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 → (528 + m) % 5 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l2711_271101


namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_for_specific_pyramid_l2711_271109

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  base_edge : ℝ
  side_edge : ℝ

/-- Radius of the circumscribed sphere of a regular triangular pyramid -/
def circumscribed_sphere_radius (p : RegularTriangularPyramid) : ℝ :=
  -- Definition to be proved
  sorry

/-- Theorem: The radius of the circumscribed sphere of a regular triangular pyramid
    with base edge 6 and side edge 4 is 4 -/
theorem circumscribed_sphere_radius_for_specific_pyramid :
  let p : RegularTriangularPyramid := ⟨6, 4⟩
  circumscribed_sphere_radius p = 4 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_for_specific_pyramid_l2711_271109


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l2711_271104

theorem average_of_a_and_b (a b c : ℝ) : 
  (b + c) / 2 = 50 → 
  c - a = 10 → 
  (a + b) / 2 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l2711_271104


namespace NUMINAMATH_CALUDE_january_salary_solve_salary_problem_l2711_271125

/-- Represents the monthly salary structure -/
structure MonthlySalary where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ
  may : ℕ

/-- Theorem stating the salary for January given the conditions -/
theorem january_salary (s : MonthlySalary) :
  (s.january + s.february + s.march + s.april) / 4 = 8000 →
  (s.february + s.march + s.april + s.may) / 4 = 8400 →
  s.may = 6500 →
  s.january = 4900 := by
  sorry

/-- Main theorem proving the salary calculation -/
theorem solve_salary_problem :
  ∃ s : MonthlySalary,
    (s.january + s.february + s.march + s.april) / 4 = 8000 ∧
    (s.february + s.march + s.april + s.may) / 4 = 8400 ∧
    s.may = 6500 ∧
    s.january = 4900 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_solve_salary_problem_l2711_271125


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2711_271110

theorem shaded_area_calculation (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 40 →
  triangle_base = 15 →
  triangle_height = 15 →
  square_side * square_side - 2 * (1/2 * triangle_base * triangle_height) = 1375 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2711_271110


namespace NUMINAMATH_CALUDE_simplify_expression_l2711_271182

theorem simplify_expression (x y : ℝ) : x^5 * x^3 * y^2 * y^4 = x^8 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2711_271182


namespace NUMINAMATH_CALUDE_function_inequality_and_sum_product_l2711_271114

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 2|

-- State the theorem
theorem function_inequality_and_sum_product (m M a b : ℝ) :
  (∀ x, f x ≥ |m - 1|) →
  (-2 ≤ m ∧ m ≤ 4) ∧
  (M = 4 →
   a > 0 →
   b > 0 →
   a^2 + b^2 = M/2 →
   a + b ≥ 2*a*b) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_and_sum_product_l2711_271114


namespace NUMINAMATH_CALUDE_ant_path_impossibility_l2711_271151

/-- Represents a vertex of a cube --/
inductive Vertex
| V1 | V2 | V3 | V4 | V5 | V6 | V7 | V8

/-- Represents the label of a vertex (+1 or -1) --/
def vertexLabel (v : Vertex) : Int :=
  match v with
  | Vertex.V1 | Vertex.V3 | Vertex.V6 | Vertex.V8 => 1
  | Vertex.V2 | Vertex.V4 | Vertex.V5 | Vertex.V7 => -1

/-- Represents a path of an ant on the cube --/
def AntPath := List Vertex

/-- Checks if the path is valid (no backtracking) --/
def isValidPath (path : AntPath) : Prop :=
  sorry

/-- Counts the number of visits to each vertex --/
def countVisits (path : AntPath) : Vertex → Nat :=
  sorry

/-- The main theorem to prove --/
theorem ant_path_impossibility :
  ¬ ∃ (path : AntPath),
    isValidPath path ∧
    (∃ (v : Vertex),
      countVisits path v = 25 ∧
      ∀ (w : Vertex), w ≠ v → countVisits path w = 20) :=
sorry

end NUMINAMATH_CALUDE_ant_path_impossibility_l2711_271151


namespace NUMINAMATH_CALUDE_sum_xyz_equals_negative_one_l2711_271174

theorem sum_xyz_equals_negative_one (x y z : ℝ) : 
  (x + 1)^2 + |y - 2| = -(2*x - z)^2 → x + y + z = -1 :=
by sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_negative_one_l2711_271174


namespace NUMINAMATH_CALUDE_chess_game_results_l2711_271175

/-- Represents the outcome of a chess game. -/
inductive GameOutcome
  | Win
  | Loss
  | Draw

/-- Calculates points for a given game outcome. -/
def pointsForOutcome (outcome : GameOutcome) : Int :=
  match outcome with
  | GameOutcome.Win => 3
  | GameOutcome.Loss => -2
  | GameOutcome.Draw => 0

/-- Represents a player's game results. -/
structure PlayerResults :=
  (wins : Nat)
  (losses : Nat)
  (draws : Nat)

/-- Calculates total points for a player given their results. -/
def totalPoints (results : PlayerResults) : Int :=
  (results.wins * pointsForOutcome GameOutcome.Win) +
  (results.losses * pointsForOutcome GameOutcome.Loss) +
  (results.draws * pointsForOutcome GameOutcome.Draw)

theorem chess_game_results : ∃ (petr_losses : Nat),
  let petr := PlayerResults.mk 6 petr_losses 2
  let karel := PlayerResults.mk (petr_losses) 6 2
  totalPoints karel = 9 ∧
  petr.wins + petr.losses + petr.draws = 15 ∧
  totalPoints karel > totalPoints petr :=
by sorry

end NUMINAMATH_CALUDE_chess_game_results_l2711_271175


namespace NUMINAMATH_CALUDE_cubic_sum_l2711_271118

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_l2711_271118


namespace NUMINAMATH_CALUDE_expression_simplification_l2711_271195

theorem expression_simplification (m n x : ℝ) :
  (3 * m^2 + 2 * m * n - 5 * m^2 + 3 * m * n = -2 * m^2 + 5 * m * n) ∧
  ((x^2 + 2 * x) - 2 * (x^2 - x) = -x^2 + 4 * x) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2711_271195


namespace NUMINAMATH_CALUDE_katya_problems_l2711_271187

theorem katya_problems (p_katya : ℝ) (p_pen : ℝ) (total_problems : ℕ) (good_grade : ℝ) 
  (h_katya : p_katya = 4/5)
  (h_pen : p_pen = 1/2)
  (h_total : total_problems = 20)
  (h_good : good_grade = 13) :
  ∃ x : ℝ, x ≥ 10 ∧ 
    x * p_katya + (total_problems - x) * p_pen ≥ good_grade ∧
    ∀ y : ℝ, y < 10 → y * p_katya + (total_problems - y) * p_pen < good_grade := by
  sorry

end NUMINAMATH_CALUDE_katya_problems_l2711_271187


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_10080_l2711_271153

def digit_product (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem largest_five_digit_with_product_10080 :
  ∀ n : Nat, 10000 ≤ n ∧ n < 100000 ∧ digit_product n = 10080 → n ≤ 98754 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_10080_l2711_271153


namespace NUMINAMATH_CALUDE_age_difference_l2711_271124

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 17) : A - C = 17 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2711_271124


namespace NUMINAMATH_CALUDE_tangent_lines_imply_a_range_l2711_271180

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.sqrt x

def has_two_tangent_lines (f g : ℝ → ℝ) : Prop :=
  ∃ (l₁ l₂ : ℝ → ℝ), l₁ ≠ l₂ ∧
    (∃ (x₁ : ℝ), l₁ x₁ = f x₁ ∧ (∀ x, l₁ x ≤ f x)) ∧
    (∃ (x₂ : ℝ), l₂ x₂ = f x₂ ∧ (∀ x, l₂ x ≤ f x)) ∧
    (∃ (y₁ : ℝ), l₁ y₁ = g y₁ ∧ (∀ y, l₁ y ≤ g y)) ∧
    (∃ (y₂ : ℝ), l₂ y₂ = g y₂ ∧ (∀ y, l₂ y ≤ g y))

theorem tangent_lines_imply_a_range (a : ℝ) :
  has_two_tangent_lines f (g a) → 0 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_imply_a_range_l2711_271180


namespace NUMINAMATH_CALUDE_supermarket_promotion_cost_l2711_271197

/-- Represents the cost calculation for a supermarket promotion --/
def supermarket_promotion (x : ℕ) : Prop :=
  let teapot_price : ℕ := 20
  let teacup_price : ℕ := 6
  let num_teapots : ℕ := 5
  x > 5 →
  (num_teapots * teapot_price + (x - num_teapots) * teacup_price = 6 * x + 70) ∧
  ((num_teapots * teapot_price + x * teacup_price) * 9 / 10 = 54 * x / 10 + 90)

theorem supermarket_promotion_cost (x : ℕ) : supermarket_promotion x :=
by sorry

end NUMINAMATH_CALUDE_supermarket_promotion_cost_l2711_271197


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l2711_271191

theorem power_equality_implies_exponent (m : ℝ) : (81 : ℝ) ^ (1/4 : ℝ) = 3^m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l2711_271191


namespace NUMINAMATH_CALUDE_serenity_shoes_pairs_serenity_bought_three_pairs_l2711_271132

theorem serenity_shoes_pairs : ℕ → ℕ → ℕ → Prop :=
  fun total_shoes shoes_per_pair pairs_bought =>
    total_shoes = 6 ∧ shoes_per_pair = 2 →
    pairs_bought = total_shoes / shoes_per_pair ∧
    pairs_bought = 3

-- Proof
theorem serenity_bought_three_pairs : serenity_shoes_pairs 6 2 3 := by
  sorry

end NUMINAMATH_CALUDE_serenity_shoes_pairs_serenity_bought_three_pairs_l2711_271132


namespace NUMINAMATH_CALUDE_emily_new_salary_l2711_271150

def emily_initial_salary : ℕ := 1000000
def employee_salaries : List ℕ := [30000, 30000, 25000, 35000, 20000]
def min_salary : ℕ := 35000
def tax_rate : ℚ := 15 / 100

def calculate_new_salary (initial_salary : ℕ) (employee_salaries : List ℕ) (min_salary : ℕ) (tax_rate : ℚ) : ℕ :=
  sorry

theorem emily_new_salary :
  calculate_new_salary emily_initial_salary employee_salaries min_salary tax_rate = 959750 :=
sorry

end NUMINAMATH_CALUDE_emily_new_salary_l2711_271150


namespace NUMINAMATH_CALUDE_problem_1_l2711_271168

theorem problem_1 : (-8) + 10 - 2 + (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_problem_1_l2711_271168


namespace NUMINAMATH_CALUDE_ratio_equality_l2711_271106

theorem ratio_equality : ∃ x : ℚ, (3/4) / (1/2) = x / (2/6) ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2711_271106


namespace NUMINAMATH_CALUDE_equation_equivalence_l2711_271198

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + y^2) + Real.sqrt ((x + 4)^2 + y^2) = 10

-- Define the simplified ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

-- Theorem stating the equivalence of the two equations
theorem equation_equivalence :
  ∀ x y : ℝ, original_equation x y ↔ ellipse_equation x y :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2711_271198


namespace NUMINAMATH_CALUDE_babysitter_scream_charge_l2711_271178

/-- Calculates the charge per scream for a new babysitter given the following conditions:
  * The current babysitter charges $16 per hour
  * The new babysitter charges $12 per hour plus an extra amount for each scream
  * The babysitting duration is 6 hours
  * The kids usually scream twice per babysitting session
  * The new babysitter will cost $18 less than the current babysitter
-/
theorem babysitter_scream_charge 
  (current_rate : ℝ) 
  (new_base_rate : ℝ) 
  (hours : ℝ) 
  (screams : ℕ) 
  (total_savings : ℝ) 
  (h1 : current_rate = 16) 
  (h2 : new_base_rate = 12) 
  (h3 : hours = 6) 
  (h4 : screams = 2) 
  (h5 : total_savings = 18) : 
  (current_rate * hours - (new_base_rate * hours + total_savings)) / screams = 3 :=
sorry

end NUMINAMATH_CALUDE_babysitter_scream_charge_l2711_271178


namespace NUMINAMATH_CALUDE_total_turnips_l2711_271159

theorem total_turnips (keith_turnips alyssa_turnips : ℕ) 
  (h1 : keith_turnips = 6) 
  (h2 : alyssa_turnips = 9) : 
  keith_turnips + alyssa_turnips = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l2711_271159


namespace NUMINAMATH_CALUDE_sin_780_degrees_l2711_271123

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_780_degrees_l2711_271123


namespace NUMINAMATH_CALUDE_lilies_per_centerpiece_l2711_271165

/-- Proves that the number of lilies per centerpiece is 6 given the specified conditions -/
theorem lilies_per_centerpiece
  (num_centerpieces : ℕ)
  (roses_per_centerpiece : ℕ)
  (orchids_per_centerpiece : ℕ)
  (total_budget : ℚ)
  (flower_cost : ℚ)
  (h1 : num_centerpieces = 6)
  (h2 : roses_per_centerpiece = 8)
  (h3 : orchids_per_centerpiece = 2 * roses_per_centerpiece)
  (h4 : total_budget = 2700)
  (h5 : flower_cost = 15)
  : (total_budget / flower_cost / num_centerpieces : ℚ) - roses_per_centerpiece - orchids_per_centerpiece = 6 :=
sorry

end NUMINAMATH_CALUDE_lilies_per_centerpiece_l2711_271165


namespace NUMINAMATH_CALUDE_five_integer_chords_l2711_271121

/-- A circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Count of integer length chords passing through P -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The specific circle and point from the problem -/
def problem_circle : CircleWithPoint :=
  { radius := 17,
    distance_to_p := 8 }

/-- The theorem stating that there are 5 integer length chords -/
theorem five_integer_chords :
  count_integer_chords problem_circle = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_integer_chords_l2711_271121


namespace NUMINAMATH_CALUDE_third_side_length_l2711_271113

theorem third_side_length (a b c : ℝ) : 
  a = 4 → b = 10 → c = 12 →
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end NUMINAMATH_CALUDE_third_side_length_l2711_271113


namespace NUMINAMATH_CALUDE_first_part_count_l2711_271167

theorem first_part_count (total_count : Nat) (total_avg : Nat) (first_avg : Nat) (last_avg : Nat) (thirteenth_result : Nat) :
  total_count = 25 →
  total_avg = 18 →
  first_avg = 10 →
  last_avg = 20 →
  thirteenth_result = 90 →
  ∃ n : Nat, n = 14 ∧ 
    n * first_avg + thirteenth_result + (total_count - n) * last_avg = total_count * total_avg :=
by sorry

end NUMINAMATH_CALUDE_first_part_count_l2711_271167


namespace NUMINAMATH_CALUDE_expression_equivalence_l2711_271157

theorem expression_equivalence : 
  (4+5)*(4^2+5^2)*(4^4+5^4)*(4^8+5^8)*(4^16+5^16)*(4^32+5^32)*(4^64+5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l2711_271157


namespace NUMINAMATH_CALUDE_nine_chapters_equations_correct_l2711_271189

/-- Represents the scenario of cars and people as described in "The Nine Chapters on the Mathematical Art" problem --/
def nine_chapters_problem (x y : ℤ) : Prop :=
  (y = 2*x + 9) ∧ (y = 3*(x - 2))

/-- Theorem stating that the equations correctly represent the described scenario --/
theorem nine_chapters_equations_correct :
  ∀ x y : ℤ, 
    nine_chapters_problem x y →
    (y = 2*x + 9) ∧ 
    (y = 3*(x - 2)) ∧
    (x > 0) ∧ 
    (y > 0) := by
  sorry

end NUMINAMATH_CALUDE_nine_chapters_equations_correct_l2711_271189


namespace NUMINAMATH_CALUDE_screen_area_difference_l2711_271143

/-- The area difference between two square screens given their diagonal lengths -/
theorem screen_area_difference (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 18) :
  d1^2 - d2^2 = 76 := by
  sorry

#check screen_area_difference

end NUMINAMATH_CALUDE_screen_area_difference_l2711_271143


namespace NUMINAMATH_CALUDE_equation_solution_for_all_y_l2711_271161

theorem equation_solution_for_all_y :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_for_all_y_l2711_271161


namespace NUMINAMATH_CALUDE_g_of_three_l2711_271145

/-- Given a function g such that g(x-1) = 2x + 6 for all x, prove that g(3) = 14 -/
theorem g_of_three (g : ℝ → ℝ) (h : ∀ x, g (x - 1) = 2 * x + 6) : g 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_l2711_271145


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l2711_271119

theorem greatest_integer_difference (x y : ℚ) 
  (hx : 3 < x) (hxy : x < (3/2)^3) (hyz : (3/2)^3 < y) (hy : y < 7) :
  ∃ (n : ℕ), n = 2 ∧ ∀ (m : ℕ), (m : ℚ) ≤ y - x → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l2711_271119


namespace NUMINAMATH_CALUDE_exists_special_function_l2711_271129

theorem exists_special_function :
  ∃ f : ℕ → ℕ,
    (∀ m n : ℕ, m < n → f m < f n) ∧
    f 1 = 2 ∧
    ∀ n : ℕ, f (f n) = f n + n :=
by sorry

end NUMINAMATH_CALUDE_exists_special_function_l2711_271129


namespace NUMINAMATH_CALUDE_area_of_M_figure_l2711_271103

-- Define the set of points M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ α : ℝ, (p.1 - 3 * Real.cos α)^2 + (p.2 - 3 * Real.sin α)^2 = 25}

-- Define the area of the figure formed by all points in M
noncomputable def area_of_figure : ℝ := Real.pi * ((3 + 5)^2 - (5 - 3)^2)

-- Theorem statement
theorem area_of_M_figure : area_of_figure = 60 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_M_figure_l2711_271103


namespace NUMINAMATH_CALUDE_min_value_abc_l2711_271181

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 → (a + b) / (a * b * c) ≤ (x + y) / (x * y * z)) →
  (a + b) / (a * b * c) = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_abc_l2711_271181


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2711_271158

theorem rectangular_prism_diagonal 
  (a b c : ℝ) 
  (h1 : 2 * a * b + 2 * b * c + 2 * c * a = 11) 
  (h2 : 4 * (a + b + c) = 24) : 
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2711_271158


namespace NUMINAMATH_CALUDE_book_price_theorem_l2711_271183

theorem book_price_theorem (suggested_retail_price : ℝ) 
  (h1 : suggested_retail_price > 0) : 
  let marked_price := 0.6 * suggested_retail_price
  let alice_paid := 0.75 * marked_price
  alice_paid / suggested_retail_price = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_book_price_theorem_l2711_271183


namespace NUMINAMATH_CALUDE_square_root_three_squared_l2711_271127

theorem square_root_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_three_squared_l2711_271127


namespace NUMINAMATH_CALUDE_nth_equation_l2711_271176

theorem nth_equation (n : ℕ) (hn : n > 0) :
  1 + 1 / n - 2 / (2 * n + 1) = (2 * n^2 + n + 1) / (n * (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l2711_271176


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l2711_271112

theorem abc_sum_sqrt (a b c : ℝ) 
  (eq1 : b + c = 20) 
  (eq2 : c + a = 22) 
  (eq3 : a + b = 24) : 
  Real.sqrt (a * b * c * (a + b + c)) = 357 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l2711_271112


namespace NUMINAMATH_CALUDE_horse_grazing_area_l2711_271131

/-- The area over which a horse can graze when tethered to one corner of a rectangular field --/
theorem horse_grazing_area (field_length : ℝ) (field_width : ℝ) (rope_length : ℝ) 
    (h1 : field_length = 40)
    (h2 : field_width = 24)
    (h3 : rope_length = 14)
    (h4 : rope_length ≤ field_length / 2)
    (h5 : rope_length ≤ field_width / 2) :
  (1/4 : ℝ) * Real.pi * rope_length^2 = 49 * Real.pi := by
  sorry

#check horse_grazing_area

end NUMINAMATH_CALUDE_horse_grazing_area_l2711_271131


namespace NUMINAMATH_CALUDE_remainder_1743_base12_div_9_l2711_271171

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The base-12 representation of 1743 --/
def num1743Base12 : List Nat := [3, 4, 7, 1]

theorem remainder_1743_base12_div_9 :
  (base12ToBase10 num1743Base12) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1743_base12_div_9_l2711_271171


namespace NUMINAMATH_CALUDE_rock_collecting_contest_l2711_271141

theorem rock_collecting_contest (sydney_initial conner_initial : ℕ)
  (sydney_day1 conner_day1_multiplier : ℕ)
  (sydney_day3_multiplier conner_day3 : ℕ) :
  sydney_initial = 837 →
  conner_initial = 723 →
  sydney_day1 = 4 →
  conner_day1_multiplier = 8 →
  sydney_day3_multiplier = 2 →
  conner_day3 = 27 →
  ∃ (conner_day2 : ℕ),
    sydney_initial + sydney_day1 + sydney_day3_multiplier * (conner_day1_multiplier * sydney_day1) ≤
    conner_initial + (conner_day1_multiplier * sydney_day1) + conner_day2 + conner_day3 ∧
    conner_day2 = 123 :=
by sorry

end NUMINAMATH_CALUDE_rock_collecting_contest_l2711_271141


namespace NUMINAMATH_CALUDE_valid_configurations_count_l2711_271155

/-- Represents a configuration of lit and unlit bulbs -/
def BulbConfiguration := List Bool

/-- Checks if a configuration is valid (no adjacent lit bulbs) -/
def isValidConfiguration (config : BulbConfiguration) : Bool :=
  match config with
  | [] => true
  | [_] => true
  | true :: true :: _ => false
  | _ :: rest => isValidConfiguration rest

/-- Counts the number of lit bulbs in a configuration -/
def countLitBulbs (config : BulbConfiguration) : Nat :=
  config.filter id |>.length

/-- Generates all possible configurations for n bulbs -/
def allConfigurations (n : Nat) : List BulbConfiguration :=
  sorry

/-- Counts valid configurations with at least k lit bulbs out of n total bulbs -/
def countValidConfigurations (n k : Nat) : Nat :=
  (allConfigurations n).filter (fun config => 
    isValidConfiguration config && countLitBulbs config ≥ k
  ) |>.length

theorem valid_configurations_count : 
  countValidConfigurations 7 3 = 11 := by sorry

end NUMINAMATH_CALUDE_valid_configurations_count_l2711_271155


namespace NUMINAMATH_CALUDE_multiplication_simplification_l2711_271190

theorem multiplication_simplification :
  2000 * 2992 * 0.2992 * 20 = 4 * 2992^2 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_simplification_l2711_271190


namespace NUMINAMATH_CALUDE_negation_of_parallelogram_is_rhombus_is_true_l2711_271156

-- Define the property of being a parallelogram
def is_parallelogram (shape : Type) : Prop := sorry

-- Define the property of being a rhombus
def is_rhombus (shape : Type) : Prop := sorry

-- The statement we want to prove
theorem negation_of_parallelogram_is_rhombus_is_true :
  ∃ (shape : Type), is_parallelogram shape ∧ ¬is_rhombus shape := by sorry

end NUMINAMATH_CALUDE_negation_of_parallelogram_is_rhombus_is_true_l2711_271156
