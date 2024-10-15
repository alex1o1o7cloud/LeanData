import Mathlib

namespace NUMINAMATH_CALUDE_genevieve_code_lines_l1322_132260

/-- Represents the number of lines of code per debugging session -/
def lines_per_debug : ℕ := 100

/-- Represents the number of errors found per debugging session -/
def errors_per_debug : ℕ := 3

/-- Represents the total number of errors fixed so far -/
def total_errors_fixed : ℕ := 129

/-- Calculates the number of lines of code written based on the given conditions -/
def lines_of_code : ℕ := (total_errors_fixed / errors_per_debug) * lines_per_debug

/-- Theorem stating that the number of lines of code written is 4300 -/
theorem genevieve_code_lines : lines_of_code = 4300 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_code_lines_l1322_132260


namespace NUMINAMATH_CALUDE_smallest_prime_not_three_l1322_132223

theorem smallest_prime_not_three : ¬(∀ p : ℕ, Prime p → p ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_not_three_l1322_132223


namespace NUMINAMATH_CALUDE_optimal_configuration_prevents_loosening_l1322_132246

/-- A rectangular prism box with trapezoid end faces on a cart -/
structure BoxOnCart where
  d : ℝ  -- Distance between parallel sides of trapezoid
  k : ℝ  -- Width of cart
  b : ℝ  -- Height of trapezoid at one end
  c : ℝ  -- Height of trapezoid at other end
  h_k_gt_d : k > d
  h_b_gt_c : b > c

/-- The optimal configuration of the box on the cart -/
def optimal_configuration (box : BoxOnCart) : Prop :=
  let DC₁ := box.c / (box.b - box.c) * (box.k - box.d)
  let AB₁ := box.b / (box.b + box.c) * (box.k - box.d)
  DC₁ > 0 ∧ AB₁ > 0 ∧ box.k ≤ 2 * box.b * box.d / (box.b - box.c)

/-- The theorem stating the optimal configuration prevents the rope from loosening -/
theorem optimal_configuration_prevents_loosening (box : BoxOnCart) :
  optimal_configuration box →
  ∃ (DC₁ AB₁ : ℝ),
    DC₁ = box.c / (box.b - box.c) * (box.k - box.d) ∧
    AB₁ = box.b / (box.b + box.c) * (box.k - box.d) ∧
    DC₁ > 0 ∧ AB₁ > 0 ∧
    box.k ≤ 2 * box.b * box.d / (box.b - box.c) :=
by sorry

end NUMINAMATH_CALUDE_optimal_configuration_prevents_loosening_l1322_132246


namespace NUMINAMATH_CALUDE_sum_of_squares_l1322_132209

theorem sum_of_squares (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 1) (h5 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5 + 1) :
  a^2 + b^2 + c^2 = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1322_132209


namespace NUMINAMATH_CALUDE_part1_part2_l1322_132250

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| - 1

-- Part 1
theorem part1 (m : ℝ) : 
  (∀ x, f m x ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5) → m = 2 := by sorry

-- Part 2
theorem part2 (t : ℝ) :
  (∀ x, f 2 x + f 2 (x + 5) ≥ t - 2) → t ≤ 5 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1322_132250


namespace NUMINAMATH_CALUDE_path_of_vertex_A_l1322_132208

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  ab : ℝ
  bc : ℝ

/-- Calculates the path traveled by vertex A of a rectangle when rotated 90° around D and translated -/
def pathTraveledByA (rect : Rectangle) (rotationAngle : ℝ) (translation : ℝ) : ℝ :=
  sorry

/-- Theorem stating the path traveled by vertex A of the specific rectangle -/
theorem path_of_vertex_A :
  let rect : Rectangle := { ab := 3, bc := 5 }
  let rotationAngle : ℝ := π / 2  -- 90° in radians
  let translation : ℝ := 3
  pathTraveledByA rect rotationAngle translation = 2.5 * π + 3 := by
  sorry

end NUMINAMATH_CALUDE_path_of_vertex_A_l1322_132208


namespace NUMINAMATH_CALUDE_train_passengers_proof_l1322_132261

/-- Calculates the number of passengers on each return trip given the total number of round trips, 
    passengers on each one-way trip, and total passengers transported. -/
def return_trip_passengers (round_trips : ℕ) (one_way_passengers : ℕ) (total_passengers : ℕ) : ℕ :=
  (total_passengers - round_trips * one_way_passengers) / round_trips

/-- Proves that given the specified conditions, the number of passengers on each return trip is 60. -/
theorem train_passengers_proof :
  let round_trips : ℕ := 4
  let one_way_passengers : ℕ := 100
  let total_passengers : ℕ := 640
  return_trip_passengers round_trips one_way_passengers total_passengers = 60 := by
  sorry

#eval return_trip_passengers 4 100 640

end NUMINAMATH_CALUDE_train_passengers_proof_l1322_132261


namespace NUMINAMATH_CALUDE_product_of_roots_l1322_132207

theorem product_of_roots (x : ℝ) : 
  (2 * x^3 - 24 * x^2 + 96 * x + 56 = 0) → 
  (∃ r₁ r₂ r₃ : ℝ, (x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ * r₂ * r₃ = -28) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l1322_132207


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_2002_l1322_132248

theorem sum_of_last_two_digits_of_9_pow_2002 : ∃ (n : ℕ), 9^2002 = 100 * n + 81 :=
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_2002_l1322_132248


namespace NUMINAMATH_CALUDE_apple_distribution_l1322_132252

theorem apple_distribution (jim jerry : ℕ) (h1 : jim = 20) (h2 : jerry = 40) : 
  ∃ jane : ℕ, 
    (2 * jim = (jim + jerry + jane) / 3) ∧ 
    (jane = 30) := by
sorry

end NUMINAMATH_CALUDE_apple_distribution_l1322_132252


namespace NUMINAMATH_CALUDE_fish_weight_l1322_132228

/-- Given a barrel of fish with the following properties:
    - The initial weight of the barrel with all fish is 54 kg
    - The weight of the barrel with half of the fish removed is 29 kg
    This theorem proves that the total weight of the fish is 50 kg. -/
theorem fish_weight (initial_weight : ℝ) (half_removed_weight : ℝ) 
  (h1 : initial_weight = 54)
  (h2 : half_removed_weight = 29) :
  ∃ (barrel_weight fish_weight : ℝ),
    barrel_weight + fish_weight = initial_weight ∧
    barrel_weight + fish_weight / 2 = half_removed_weight ∧
    fish_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_fish_weight_l1322_132228


namespace NUMINAMATH_CALUDE_unique_solution_l1322_132295

theorem unique_solution (a b c : ℝ) (h1 : a = 6 - b) (h2 : c^2 = a*b - 9) : a = 3 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1322_132295


namespace NUMINAMATH_CALUDE_parabola_focus_l1322_132201

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := y = -1/8 * x^2

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 2)

/-- Theorem: The focus of the parabola y = -1/8 * x^2 is (0, 2) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola_eq x y →
  ∃ (d : ℝ), 
    (x - focus.1)^2 + (y - focus.2)^2 = (y - d)^2 ∧
    (∀ (x' y' : ℝ), parabola_eq x' y' → 
      (x' - focus.1)^2 + (y' - focus.2)^2 = (y' - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1322_132201


namespace NUMINAMATH_CALUDE_even_sum_of_even_l1322_132298

theorem even_sum_of_even (a b : ℤ) : Even a ∧ Even b → Even (a + b) := by sorry

end NUMINAMATH_CALUDE_even_sum_of_even_l1322_132298


namespace NUMINAMATH_CALUDE_mikes_salary_increase_l1322_132257

theorem mikes_salary_increase (freds_salary_then : ℝ) (mikes_salary_now : ℝ) :
  freds_salary_then = 1000 →
  mikes_salary_now = 15400 →
  let mikes_salary_then := 10 * freds_salary_then
  (mikes_salary_now - mikes_salary_then) / mikes_salary_then * 100 = 54 := by
  sorry

end NUMINAMATH_CALUDE_mikes_salary_increase_l1322_132257


namespace NUMINAMATH_CALUDE_ladder_wall_distance_l1322_132272

/-- The distance between two walls problem -/
theorem ladder_wall_distance
  (w : ℝ) -- Distance between walls
  (a : ℝ) -- Length of the ladder
  (k : ℝ) -- Height of point Q
  (h : ℝ) -- Height of point R
  (hw_pos : w > 0)
  (ha_pos : a > 0)
  (hk_pos : k > 0)
  (hh_pos : h > 0)
  (h_45_deg : a = k * Real.sqrt 2) -- Condition for 45° angle
  (h_75_deg : a = h * Real.sqrt (4 - 2 * Real.sqrt 3)) -- Condition for 75° angle
  : w = h :=
by sorry

end NUMINAMATH_CALUDE_ladder_wall_distance_l1322_132272


namespace NUMINAMATH_CALUDE_equal_coefficients_implies_n_seven_l1322_132211

theorem equal_coefficients_implies_n_seven (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_equal_coefficients_implies_n_seven_l1322_132211


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1322_132203

theorem diophantine_equation_solution : ∃ (x y : ℕ), x^2 + y^2 = 61^3 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1322_132203


namespace NUMINAMATH_CALUDE_range_of_a_l1322_132274

open Real

theorem range_of_a (m : ℝ) (hm : m > 0) :
  (∃ x : ℝ, x + a * (2*x + 2*m - 4*Real.exp 1*x) * (log (x + m) - log x) = 0) →
  a ∈ Set.Iic 0 ∪ Set.Ici (1 / (2 * Real.exp 1)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1322_132274


namespace NUMINAMATH_CALUDE_sequence_limit_l1322_132251

def x (n : ℕ) : ℚ := (2 * n - 1) / (3 * n + 5)

theorem sequence_limit : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - 2/3| < ε := by
  sorry

end NUMINAMATH_CALUDE_sequence_limit_l1322_132251


namespace NUMINAMATH_CALUDE_cos_negative_135_degrees_l1322_132270

theorem cos_negative_135_degrees : Real.cos ((-135 : ℝ) * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_135_degrees_l1322_132270


namespace NUMINAMATH_CALUDE_solve_for_y_l1322_132264

theorem solve_for_y (x y : ℝ) 
  (h1 : x = 151) 
  (h2 : x^3 * y - 3 * x^2 * y + 3 * x * y = 3423000) : 
  y = 3423000 / 3375001 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l1322_132264


namespace NUMINAMATH_CALUDE_pen_cost_l1322_132263

theorem pen_cost (x y : ℕ) (h1 : 5 * x + 4 * y = 345) (h2 : 3 * x + 6 * y = 285) : x = 52 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l1322_132263


namespace NUMINAMATH_CALUDE_log_sum_equality_l1322_132293

theorem log_sum_equality : 2 * Real.log 25 / Real.log 5 + 3 * Real.log 64 / Real.log 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1322_132293


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1322_132281

/-- Given a line with slope -8 passing through (4, -3), prove that m + b = 21 in y = mx + b -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -8 → 
  -3 = m * 4 + b → 
  m + b = 21 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1322_132281


namespace NUMINAMATH_CALUDE_common_points_characterization_l1322_132278

-- Define the square S
def S : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the set C_t
def C (t : ℝ) : Set (ℝ × ℝ) :=
  {p ∈ S | p.2 ≥ ((1 - t) / t) * p.1 + (1 - t)}

-- Define the intersection of all C_t
def CommonPoints : Set (ℝ × ℝ) :=
  ⋂ t ∈ {t | 0 < t ∧ t < 1}, C t

-- State the theorem
theorem common_points_characterization :
  ∀ p ∈ S, p ∈ CommonPoints ↔ Real.sqrt p.1 + Real.sqrt p.2 ≥ 1 := by sorry

end NUMINAMATH_CALUDE_common_points_characterization_l1322_132278


namespace NUMINAMATH_CALUDE_even_quadratic_iff_b_eq_zero_l1322_132217

/-- A quadratic function f(x) = ax^2 + bx + c is even if and only if b = 0, given a ≠ 0 -/
theorem even_quadratic_iff_b_eq_zero (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (-x)^2 + b * (-x) + c) ↔ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_iff_b_eq_zero_l1322_132217


namespace NUMINAMATH_CALUDE_inverse_composition_l1322_132282

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the condition
variable (h : ∀ x, f⁻¹ (g x) = 7 * x - 4)

-- State the theorem
theorem inverse_composition :
  g⁻¹ (f (-9)) = -5/7 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_l1322_132282


namespace NUMINAMATH_CALUDE_javier_speech_time_l1322_132200

/-- Represents the time spent on different activities of speech preparation --/
structure SpeechTime where
  outline : ℕ
  write : ℕ
  practice : ℕ

/-- Calculates the total time spent on speech preparation --/
def totalTime (st : SpeechTime) : ℕ :=
  st.outline + st.write + st.practice

/-- Theorem stating the total time Javier spends on his speech --/
theorem javier_speech_time :
  ∃ (st : SpeechTime),
    st.outline = 30 ∧
    st.write = st.outline + 28 ∧
    st.practice = st.write / 2 ∧
    totalTime st = 117 := by
  sorry


end NUMINAMATH_CALUDE_javier_speech_time_l1322_132200


namespace NUMINAMATH_CALUDE_exists_distinct_subaveraging_equal_entries_imply_infinite_equal_pairs_l1322_132279

-- Define a subaveraging sequence
def IsSubaveraging (s : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, s n = (s (n - 1) + s (n + 1)) / 4

-- Part (a): Existence of a subaveraging sequence with all distinct entries
theorem exists_distinct_subaveraging :
  ∃ s : ℤ → ℝ, IsSubaveraging s ∧ (∀ m n : ℤ, m ≠ n → s m ≠ s n) :=
sorry

-- Part (b): If two entries are equal, infinitely many pairs are equal
theorem equal_entries_imply_infinite_equal_pairs
  (s : ℤ → ℝ) (h : IsSubaveraging s) :
  (∃ m n : ℤ, m ≠ n ∧ s m = s n) →
  (∀ k : ℕ, ∃ i j : ℤ, i ≠ j ∧ s i = s j ∧ |i - j| > k) :=
sorry

end NUMINAMATH_CALUDE_exists_distinct_subaveraging_equal_entries_imply_infinite_equal_pairs_l1322_132279


namespace NUMINAMATH_CALUDE_constant_d_value_l1322_132237

theorem constant_d_value (x y d : ℝ) 
  (h1 : x / (2 * y) = d / 2)
  (h2 : (7 * x + 4 * y) / (x - 2 * y) = 25) :
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_constant_d_value_l1322_132237


namespace NUMINAMATH_CALUDE_center_sum_l1322_132219

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 18*y + 9

-- Define the center of the circle
def is_center (h k : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h - 18*k - 9)

-- Theorem statement
theorem center_sum : ∃ h k, is_center h k ∧ h + k = 12 :=
sorry

end NUMINAMATH_CALUDE_center_sum_l1322_132219


namespace NUMINAMATH_CALUDE_expression_evaluation_l1322_132276

theorem expression_evaluation (a b c : ℝ) 
  (ha : a = 3) (hb : b = 2) (hc : c = 1) : 
  (a^2 + b + c)^2 - (a^2 - b - c)^2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1322_132276


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1322_132253

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 2) :
  (1/x + 1/y + 1/z) ≥ 9/2 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 ∧ 1/x + 1/y + 1/z = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1322_132253


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1322_132245

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 - 6*m*x + 2*m = 0) → m = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1322_132245


namespace NUMINAMATH_CALUDE_max_cake_pieces_l1322_132202

def cakeSize : ℕ := 50
def pieceSize1 : ℕ := 4
def pieceSize2 : ℕ := 6
def pieceSize3 : ℕ := 8

theorem max_cake_pieces :
  let maxLargePieces := (cakeSize / pieceSize3) ^ 2
  let remainingWidth := cakeSize - (cakeSize / pieceSize3) * pieceSize3
  let maxSmallPieces := 2 * (cakeSize / pieceSize1)
  maxLargePieces + maxSmallPieces = 60 :=
by sorry

end NUMINAMATH_CALUDE_max_cake_pieces_l1322_132202


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l1322_132238

theorem fraction_sum_simplification : 1 / 462 + 23 / 42 = 127 / 231 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l1322_132238


namespace NUMINAMATH_CALUDE_cone_volume_l1322_132254

theorem cone_volume (s : Real) (c : Real) (h : s = 8) (k : c = 6 * Real.pi) :
  let r := c / (2 * Real.pi)
  let height := Real.sqrt (s^2 - r^2)
  (1/3) * Real.pi * r^2 * height = 3 * Real.sqrt 55 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cone_volume_l1322_132254


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1322_132227

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {2, 5, 8}
def B : Set Nat := {1, 3, 5, 7}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 3, 7} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1322_132227


namespace NUMINAMATH_CALUDE_max_negative_coefficients_no_real_roots_l1322_132277

def polynomial_coefficients (p : ℝ → ℝ) : List ℤ :=
  sorry

def has_no_real_roots (p : ℝ → ℝ) : Prop :=
  sorry

def count_negative_coefficients (coeffs : List ℤ) : ℕ :=
  sorry

theorem max_negative_coefficients_no_real_roots 
  (p : ℝ → ℝ) 
  (h1 : ∃ (coeffs : List ℤ), polynomial_coefficients p = coeffs ∧ coeffs.length = 2011)
  (h2 : has_no_real_roots p) :
  count_negative_coefficients (polynomial_coefficients p) ≤ 1005 :=
sorry

end NUMINAMATH_CALUDE_max_negative_coefficients_no_real_roots_l1322_132277


namespace NUMINAMATH_CALUDE_function_characterization_l1322_132286

theorem function_characterization (a : ℝ) (f : ℝ → ℝ) 
  (h : ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z ≠ 0 → 
    a * f (x / y) + a * f (x / z) - f x * f ((y + z) / 2) ≥ a^2) :
  ∀ (x : ℝ), x ≠ 0 → f x = a := by
sorry

end NUMINAMATH_CALUDE_function_characterization_l1322_132286


namespace NUMINAMATH_CALUDE_book_prices_l1322_132258

def total_cost : ℕ := 104

def is_valid_price (price : ℕ) : Prop :=
  ∃ (n : ℕ), 10 < n ∧ n < 60 ∧ n * price = total_cost

theorem book_prices :
  {p : ℕ | is_valid_price p} = {2, 4, 8} :=
by sorry

end NUMINAMATH_CALUDE_book_prices_l1322_132258


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_zero_l1322_132242

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_zero :
  (lg 2)^2 + (lg 2) * (lg 5) + lg 5 - (Real.sqrt 2 - 1)^0 = 0 :=
by
  -- Assume the given condition
  have h : lg 2 + lg 5 = 1 := by sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_zero_l1322_132242


namespace NUMINAMATH_CALUDE_negation_of_proposition_is_true_l1322_132273

theorem negation_of_proposition_is_true : 
  ∃ a : ℝ, (a > 2 ∧ a^2 ≥ 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_is_true_l1322_132273


namespace NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l1322_132232

/-- The volume of a cone whose lateral surface unfolds into a semicircle with radius 2 -/
theorem cone_volume_from_lateral_surface (r : Real) (h : Real) : 
  (r = 1) → (h = Real.sqrt 3) → (2 * π * r = 2 * π) → 
  (1 / 3 : Real) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l1322_132232


namespace NUMINAMATH_CALUDE_triangle_similarity_problem_l1322_132218

theorem triangle_similarity_problem (DC CB : ℝ) (AB ED AD : ℝ) (FC : ℝ) :
  DC = 9 →
  CB = 9 →
  AB = (1 / 3) * AD →
  ED = (2 / 3) * AD →
  -- Assuming triangle similarity
  (∃ (k₁ k₂ : ℝ), k₁ > 0 ∧ k₂ > 0 ∧
    AB / AD = k₁ ∧
    FC / (CB + AB) = k₁ ∧
    ED / AD = k₂ ∧
    FC / (CB + AB) = k₂) →
  FC = 12 :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_problem_l1322_132218


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_bound_l1322_132284

theorem inequality_solution_implies_a_bound (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2*x^2 - 8*x - 4 - a > 0) → a < -4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_bound_l1322_132284


namespace NUMINAMATH_CALUDE_fatimas_number_probability_l1322_132283

def first_three_options : List Nat := [296, 299, 295]
def last_five_digits : List Nat := [0, 1, 6, 7, 8]

def total_possibilities : Nat :=
  (first_three_options.length) * (Nat.factorial last_five_digits.length)

theorem fatimas_number_probability :
  (1 : ℚ) / total_possibilities = (1 : ℚ) / 360 := by
  sorry

end NUMINAMATH_CALUDE_fatimas_number_probability_l1322_132283


namespace NUMINAMATH_CALUDE_draw_with_even_ball_l1322_132247

/-- The number of balls in the bin -/
def total_balls : ℕ := 15

/-- The number of balls to be drawn -/
def drawn_balls : ℕ := 4

/-- The number of odd-numbered balls -/
def odd_balls : ℕ := 8

/-- Calculate the number of ways to draw n balls from m balls in order -/
def ways_to_draw (m n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * (m - i)) 1

/-- The main theorem: number of ways to draw 4 balls with at least one even-numbered ball -/
theorem draw_with_even_ball :
  ways_to_draw total_balls drawn_balls - ways_to_draw odd_balls drawn_balls = 31080 := by
  sorry

end NUMINAMATH_CALUDE_draw_with_even_ball_l1322_132247


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1322_132275

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1322_132275


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1322_132213

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ)) ∧
  (x : ℕ) + (y : ℕ) = 64 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1322_132213


namespace NUMINAMATH_CALUDE_six_planes_max_parts_l1322_132249

/-- The maximum number of parts that n planes can divide space into -/
def max_parts (n : ℕ) : ℕ := (n^3 + 5*n + 6) / 6

/-- Theorem: 6 planes can divide space into at most 42 parts -/
theorem six_planes_max_parts : max_parts 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_six_planes_max_parts_l1322_132249


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l1322_132244

def hyperbola (a b h k : ℝ) (x y : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

def asymptote1 (x y : ℝ) : Prop := y = 3 * x + 6
def asymptote2 (x y : ℝ) : Prop := y = -3 * x + 2

theorem hyperbola_theorem (a b h k : ℝ) :
  a > 0 → b > 0 →
  (∀ x y, asymptote1 x y ∨ asymptote2 x y → hyperbola a b h k x y) →
  hyperbola a b h k 1 5 →
  a + h = 6 * Real.sqrt 2 - 2/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l1322_132244


namespace NUMINAMATH_CALUDE_dave_tickets_l1322_132259

def tickets_problem (initial_tickets spent_tickets later_tickets : ℕ) : Prop :=
  initial_tickets - spent_tickets + later_tickets = 16

theorem dave_tickets : tickets_problem 11 5 10 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l1322_132259


namespace NUMINAMATH_CALUDE_probability_less_than_one_third_l1322_132230

/-- The probability of selecting a number less than 1/3 from the interval (0, 1/2) is 2/3 -/
theorem probability_less_than_one_third : 
  let total_interval : ℝ := 1/2 - 0
  let desired_interval : ℝ := 1/3 - 0
  desired_interval / total_interval = 2/3 := by
sorry

end NUMINAMATH_CALUDE_probability_less_than_one_third_l1322_132230


namespace NUMINAMATH_CALUDE_train_passing_time_l1322_132243

/-- Proves that a train of given length and speed takes a specific time to pass a stationary object. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 180 →
  train_speed_kmh = 36 →
  passing_time = 18 →
  passing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1322_132243


namespace NUMINAMATH_CALUDE_lost_card_number_l1322_132214

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ∈ Set.range (λ i => i : Fin n → ℕ)) : 
  ∃ (k : Fin n), k.val + 1 = 4 ∧ (n * (n + 1)) / 2 - (k.val + 1) = 101 :=
by sorry

end NUMINAMATH_CALUDE_lost_card_number_l1322_132214


namespace NUMINAMATH_CALUDE_crop_planting_arrangement_l1322_132240

theorem crop_planting_arrangement (n : ℕ) (h : n = 10) : 
  (Finset.sum (Finset.range (n - 6)) (λ i => n - i - 6)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_crop_planting_arrangement_l1322_132240


namespace NUMINAMATH_CALUDE_laptop_price_l1322_132269

theorem laptop_price (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) : 
  upfront_percentage = 0.20 → 
  upfront_payment = 200 → 
  upfront_percentage * total_price = upfront_payment → 
  total_price = 1000 :=
by sorry

end NUMINAMATH_CALUDE_laptop_price_l1322_132269


namespace NUMINAMATH_CALUDE_alcohol_concentration_in_second_vessel_l1322_132271

/-- 
Given two vessels with different capacities and alcohol concentrations, 
prove that when mixed and diluted to a certain concentration, 
the alcohol percentage in the second vessel can be determined.
-/
theorem alcohol_concentration_in_second_vessel 
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (total_capacity : ℝ)
  (final_mixture_capacity : ℝ)
  (final_mixture_percentage : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 30)
  (h3 : vessel2_capacity = 6)
  (h4 : total_capacity = vessel1_capacity + vessel2_capacity)
  (h5 : final_mixture_capacity = 10)
  (h6 : final_mixture_percentage = 30) :
  ∃ vessel2_alcohol_percentage : ℝ, 
    vessel2_alcohol_percentage = 30 ∧
    vessel1_capacity * (vessel1_alcohol_percentage / 100) + 
    vessel2_capacity * (vessel2_alcohol_percentage / 100) = 
    total_capacity * (final_mixture_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_concentration_in_second_vessel_l1322_132271


namespace NUMINAMATH_CALUDE_range_of_m_l1322_132206

-- Define the sets M and N
def M (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def N : Set ℝ := {x | x^2 - 2*x - 8 < 0}

-- Define the universe U as the real numbers
def U : Set ℝ := Set.univ

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∃ x, x ∈ (U \ M m) ∩ N) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1322_132206


namespace NUMINAMATH_CALUDE_square_of_difference_l1322_132233

theorem square_of_difference (x : ℝ) : (x - 3)^2 = x^2 - 6*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l1322_132233


namespace NUMINAMATH_CALUDE_stamps_problem_l1322_132204

theorem stamps_problem (A B C D : ℕ) : 
  A + B + C + D = 251 →
  A = 2 * B + 2 →
  A = 3 * C + 6 →
  A = 4 * D - 16 →
  D = 32 := by
sorry

end NUMINAMATH_CALUDE_stamps_problem_l1322_132204


namespace NUMINAMATH_CALUDE_probability_two_even_toys_l1322_132210

def total_toys : ℕ := 21
def even_toys : ℕ := 10

theorem probability_two_even_toys :
  let p1 := even_toys / total_toys
  let p2 := (even_toys - 1) / (total_toys - 1)
  p1 * p2 = 3 / 14 := by sorry

end NUMINAMATH_CALUDE_probability_two_even_toys_l1322_132210


namespace NUMINAMATH_CALUDE_unknown_number_value_l1322_132285

theorem unknown_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 35 * 63) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l1322_132285


namespace NUMINAMATH_CALUDE_smallest_divisible_by_hundred_million_l1322_132212

/-- The smallest positive integer n such that the nth term of a geometric sequence
    with first term 5/6 and second term 25 is divisible by 100 million. -/
theorem smallest_divisible_by_hundred_million : ℕ :=
  let a₁ : ℚ := 5 / 6  -- First term
  let a₂ : ℚ := 25     -- Second term
  let r : ℚ := a₂ / a₁ -- Common ratio
  let aₙ : ℕ → ℚ := λ n => r ^ (n - 1) * a₁  -- nth term
  9  -- The smallest n (to be proved)

#check smallest_divisible_by_hundred_million

end NUMINAMATH_CALUDE_smallest_divisible_by_hundred_million_l1322_132212


namespace NUMINAMATH_CALUDE_candy_left_after_eating_l1322_132225

/-- The number of candy pieces left after two people eat some from a total collection --/
def candy_left (total : ℕ) (people : ℕ) (eaten_per_person : ℕ) : ℕ :=
  total - (people * eaten_per_person)

/-- Theorem stating that 60 pieces of candy are left when 2 people each eat 4 pieces from a total of 68 --/
theorem candy_left_after_eating : 
  candy_left 68 2 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_candy_left_after_eating_l1322_132225


namespace NUMINAMATH_CALUDE_tobias_played_one_week_l1322_132235

/-- Calculates the number of weeks Tobias played given the conditions of the problem -/
def tobias_weeks (nathan_hours_per_day : ℕ) (nathan_weeks : ℕ) (tobias_hours_per_day : ℕ) (total_hours : ℕ) : ℕ :=
  let nathan_total_hours := nathan_hours_per_day * 7 * nathan_weeks
  let tobias_total_hours := total_hours - nathan_total_hours
  tobias_total_hours / (tobias_hours_per_day * 7)

/-- Theorem stating that Tobias played for 1 week given the problem conditions -/
theorem tobias_played_one_week :
  tobias_weeks 3 2 5 77 = 1 := by
  sorry

#eval tobias_weeks 3 2 5 77

end NUMINAMATH_CALUDE_tobias_played_one_week_l1322_132235


namespace NUMINAMATH_CALUDE_zachary_needs_money_l1322_132289

/-- The amount of additional money Zachary needs to buy football equipment -/
def additional_money_needed (football_cost shorts_cost shoes_cost socks_cost bottle_cost : ℝ)
  (shorts_count socks_count : ℕ) (current_money : ℝ) : ℝ :=
  let total_cost := football_cost + shorts_count * shorts_cost + shoes_cost +
                    socks_count * socks_cost + bottle_cost
  total_cost - current_money

/-- Theorem stating the additional money Zachary needs -/
theorem zachary_needs_money :
  additional_money_needed 3.756 2.498 11.856 1.329 7.834 2 4 24.042 = 9.716 := by
  sorry

end NUMINAMATH_CALUDE_zachary_needs_money_l1322_132289


namespace NUMINAMATH_CALUDE_initial_egg_count_l1322_132266

theorem initial_egg_count (total : ℕ) (taken : ℕ) (left : ℕ) : 
  taken = 5 → left = 42 → total = taken + left → total = 47 := by
  sorry

end NUMINAMATH_CALUDE_initial_egg_count_l1322_132266


namespace NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l1322_132288

theorem sum_reciprocals_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 1 / y ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l1322_132288


namespace NUMINAMATH_CALUDE_bull_count_l1322_132241

theorem bull_count (total_cattle : ℕ) (cow_ratio bull_ratio : ℕ) 
  (h1 : total_cattle = 555)
  (h2 : cow_ratio = 10)
  (h3 : bull_ratio = 27) : 
  (bull_ratio : ℚ) / (cow_ratio + bull_ratio : ℚ) * total_cattle = 405 :=
by sorry

end NUMINAMATH_CALUDE_bull_count_l1322_132241


namespace NUMINAMATH_CALUDE_expected_girls_left_of_boys_l1322_132256

theorem expected_girls_left_of_boys (num_boys num_girls : ℕ) 
  (h1 : num_boys = 10) (h2 : num_girls = 7) :
  let total := num_boys + num_girls
  let expected_value := (num_girls : ℚ) / (total + 1 : ℚ)
  expected_value = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expected_girls_left_of_boys_l1322_132256


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l1322_132224

theorem quadratic_solution_property (a b : ℝ) : 
  (a * 1^2 + b * 1 + 1 = 0) → (2022 - a - b = 2023) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l1322_132224


namespace NUMINAMATH_CALUDE_problem_statement_l1322_132239

theorem problem_statement (a b c : ℝ) (h1 : b < c) (h2 : 1 < a) (h3 : a < b + c) (h4 : b + c < a + 1) :
  b < a := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1322_132239


namespace NUMINAMATH_CALUDE_circle_center_distance_l1322_132231

/-- The distance between the center of the circle defined by x^2 + y^2 = 8x - 2y + 16 and the point (-3, 4) is √74. -/
theorem circle_center_distance : 
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 - 8*x + 2*y - 16 = 0
  let center := (fun (x y : ℝ) => circle_eq x y ∧ 
                 ∀ (x' y' : ℝ), circle_eq x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2)
  let distance := fun (x₁ y₁ x₂ y₂ : ℝ) => Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  ∃ (cx cy : ℝ), center cx cy ∧ distance cx cy (-3) 4 = Real.sqrt 74 := by sorry

end NUMINAMATH_CALUDE_circle_center_distance_l1322_132231


namespace NUMINAMATH_CALUDE_gcd_problem_l1322_132265

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 3883) :
  Int.gcd (4 * b^2 + 35 * b + 56) (3 * b + 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1322_132265


namespace NUMINAMATH_CALUDE_resulting_polygon_has_24_sides_l1322_132205

/-- Calculates the number of sides in the resulting polygon formed by sequentially 
    attaching regular polygons from triangle to octagon. -/
def resulting_polygon_sides : ℕ :=
  let initial_triangle := 3
  let square_addition := 4 - 2
  let pentagon_addition := 5 - 2
  let hexagon_addition := 6 - 2
  let heptagon_addition := 7 - 2
  let octagon_addition := 8 - 1
  initial_triangle + square_addition + pentagon_addition + 
  hexagon_addition + heptagon_addition + octagon_addition

/-- The resulting polygon has 24 sides. -/
theorem resulting_polygon_has_24_sides : resulting_polygon_sides = 24 := by
  sorry

end NUMINAMATH_CALUDE_resulting_polygon_has_24_sides_l1322_132205


namespace NUMINAMATH_CALUDE_circle_properties_l1322_132280

/-- The equation of a circle in the form ax² + by² + cx + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

def circle_equation : CircleEquation :=
  { a := 1, b := 1, c := -2, d := 4, e := 3 }

theorem circle_properties :
  ∃ (props : CircleProperties),
    props.center = (1, -2) ∧ 
    props.radius = Real.sqrt 2 ∧
    ∀ (x y : ℝ),
      (circle_equation.a * x^2 + circle_equation.b * y^2 + 
       circle_equation.c * x + circle_equation.d * y + 
       circle_equation.e = 0) ↔
      ((x - props.center.1)^2 + (y - props.center.2)^2 = props.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1322_132280


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_B_union_A_complementB_l1322_132216

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Define the complement of B
def complementB : Set ℝ := {x | ¬ (x ∈ B)}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < -1} := by sorry

theorem complement_B : complementB = {x | -1 ≤ x ∧ x ≤ 4} := by sorry

theorem union_A_complementB : A ∪ complementB = {x | -2 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_B_union_A_complementB_l1322_132216


namespace NUMINAMATH_CALUDE_set_A_properties_l1322_132215

def A : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2^k}

theorem set_A_properties :
  (∀ a ∈ A, ∀ b : ℕ, b ≠ 0 → b < 2*a - 1 → ¬(2*a ∣ b*(b+1))) ∧
  (∀ a ∉ A, a ≠ 1 → ∃ b : ℕ, b ≠ 0 ∧ b < 2*a - 1 ∧ (2*a ∣ b*(b+1))) :=
by sorry

end NUMINAMATH_CALUDE_set_A_properties_l1322_132215


namespace NUMINAMATH_CALUDE_remainder_problem_l1322_132220

theorem remainder_problem : 7 * 12^24 + 3^24 ≡ 0 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1322_132220


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l1322_132297

/-- Proves that the cost of gas per gallon is $4 given the specified conditions --/
theorem gas_cost_per_gallon 
  (pay_rate : ℝ) 
  (truck_efficiency : ℝ) 
  (profit : ℝ) 
  (trip_distance : ℝ) 
  (h1 : pay_rate = 0.50)
  (h2 : truck_efficiency = 20)
  (h3 : profit = 180)
  (h4 : trip_distance = 600) :
  (trip_distance * pay_rate - profit) / (trip_distance / truck_efficiency) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l1322_132297


namespace NUMINAMATH_CALUDE_grains_in_gray_parts_grains_in_gray_parts_specific_l1322_132255

/-- Given two circles with the same number of grains in their white parts,
    and their respective total grains, calculate the sum of grains in both gray parts. -/
theorem grains_in_gray_parts
  (white_grains : ℕ)
  (total_grains_1 total_grains_2 : ℕ)
  (h1 : total_grains_1 ≥ white_grains)
  (h2 : total_grains_2 ≥ white_grains) :
  (total_grains_1 - white_grains) + (total_grains_2 - white_grains) = 61 :=
by
  sorry

/-- Specific instance of the theorem with given values -/
theorem grains_in_gray_parts_specific :
  (87 - 68) + (110 - 68) = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_grains_in_gray_parts_grains_in_gray_parts_specific_l1322_132255


namespace NUMINAMATH_CALUDE_prize_location_questions_l1322_132268

/-- Represents the three doors in the game show. -/
inductive Door
| left
| center
| right

/-- Represents the host's response to a question. -/
inductive Response
| yes
| no

/-- The maximum number of lies the host can tell. -/
def max_lies : ℕ := 10

/-- The function that determines the minimum number of questions needed to locate the prize. -/
def min_questions_to_locate_prize (doors : List Door) (max_lies : ℕ) : ℕ :=
  sorry

/-- The theorem stating that 32 questions are needed to locate the prize with certainty. -/
theorem prize_location_questions (doors : List Door) (h1 : doors.length = 3) :
  min_questions_to_locate_prize doors max_lies = 32 :=
sorry

end NUMINAMATH_CALUDE_prize_location_questions_l1322_132268


namespace NUMINAMATH_CALUDE_probability_union_mutually_exclusive_l1322_132291

theorem probability_union_mutually_exclusive (A B : Set Ω) (P : Set Ω → ℝ) 
  (h_mutex : A ∩ B = ∅) (h_prob_A : P A = 0.25) (h_prob_B : P B = 0.18) :
  P (A ∪ B) = 0.43 := by
  sorry

end NUMINAMATH_CALUDE_probability_union_mutually_exclusive_l1322_132291


namespace NUMINAMATH_CALUDE_fourth_week_sugar_l1322_132292

def sugar_amount (week : ℕ) : ℚ :=
  24 / (2 ^ (week - 1))

theorem fourth_week_sugar : sugar_amount 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_week_sugar_l1322_132292


namespace NUMINAMATH_CALUDE_tangent_slope_circle_l1322_132267

/-- The slope of the line tangent to a circle at the point (8, 3) is -1, 
    given that the center of the circle is at (1, -4). -/
theorem tangent_slope_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (1, -4) → point = (8, 3) → 
  (point.1 - center.1) * (point.2 - center.2) = -1 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_circle_l1322_132267


namespace NUMINAMATH_CALUDE_max_a_value_exists_max_a_l1322_132221

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| ≥ a*x) → 
  a ≤ 10 :=
by sorry

theorem exists_max_a : 
  ∃ a : ℝ, a = 10 ∧ 
  (∀ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| ≥ a*x) ∧
  ∀ b > a, ∃ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| < b*x :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_exists_max_a_l1322_132221


namespace NUMINAMATH_CALUDE_sum_of_specific_S_l1322_132262

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -n / 2 else (n + 1) / 2

theorem sum_of_specific_S : S 15 + S 28 + S 39 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_l1322_132262


namespace NUMINAMATH_CALUDE_average_of_multiples_of_seven_l1322_132294

def is_between (a b x : ℝ) : Prop := a < x ∧ x < b

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = y * k

theorem average_of_multiples_of_seven (numbers : List ℕ) : 
  (∀ n ∈ numbers, is_between 6 36 n ∧ divisible_by n 7) →
  numbers.length > 0 →
  (numbers.sum / numbers.length : ℝ) = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_seven_l1322_132294


namespace NUMINAMATH_CALUDE_complex_number_equality_l1322_132222

theorem complex_number_equality (b : ℝ) : 
  let z := (3 - b * Complex.I) / (2 + Complex.I)
  z.re = z.im → b = -9 := by sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1322_132222


namespace NUMINAMATH_CALUDE_large_sphere_radius_l1322_132290

theorem large_sphere_radius (n : ℕ) (r : ℝ) (R : ℝ) : 
  n = 12 → r = 2 → (4 / 3 * Real.pi * R^3) = n * (4 / 3 * Real.pi * r^3) → R = (96 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_large_sphere_radius_l1322_132290


namespace NUMINAMATH_CALUDE_blue_tetrahedron_volume_l1322_132234

/-- Represents a cube with alternately colored vertices -/
structure ColoredCube where
  sideLength : ℝ
  vertexColors : Fin 8 → Bool  -- True for blue, False for red

/-- The volume of a tetrahedron formed by four vertices of a cube -/
def tetrahedronVolume (c : ColoredCube) (v1 v2 v3 v4 : Fin 8) : ℝ :=
  sorry

/-- Theorem: The volume of the tetrahedron formed by blue vertices in a cube with side length 8 is 170⅔ -/
theorem blue_tetrahedron_volume (c : ColoredCube) 
  (h1 : c.sideLength = 8)
  (h2 : ∀ i j : Fin 8, i ≠ j → c.vertexColors i ≠ c.vertexColors j) :
  ∃ v1 v2 v3 v4 : Fin 8, 
    (c.vertexColors v1 ∧ c.vertexColors v2 ∧ c.vertexColors v3 ∧ c.vertexColors v4) ∧
    tetrahedronVolume c v1 v2 v3 v4 = 170 + 2/3 :=
  sorry

end NUMINAMATH_CALUDE_blue_tetrahedron_volume_l1322_132234


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l1322_132287

/-- The first element of the nth set -/
def first_element (n : ℕ) : ℕ := 1 + (n - 1) * n / 2

/-- The last element of the nth set -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem: The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l1322_132287


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l1322_132236

theorem power_zero_eq_one (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l1322_132236


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1322_132299

theorem p_sufficient_not_necessary_for_q :
  ∃ (p q : Prop),
    (p ↔ (∃ x : ℝ, x = 2)) ∧
    (q ↔ (∃ x : ℝ, (x - 2) * (x + 3) = 0)) ∧
    (p → q) ∧
    ¬(q → p) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1322_132299


namespace NUMINAMATH_CALUDE_total_cost_with_tip_l1322_132226

def hair_cost : ℝ := 50
def nail_cost : ℝ := 30
def tip_percentage : ℝ := 0.20

theorem total_cost_with_tip : 
  (hair_cost + nail_cost) * (1 + tip_percentage) = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_with_tip_l1322_132226


namespace NUMINAMATH_CALUDE_harmonic_sum_identity_l1322_132296

def h (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_identity (n : ℕ) (h_n : n ≥ 2) :
  (n : ℚ) + (Finset.range (n - 1)).sum h = n * h n :=
by sorry

end NUMINAMATH_CALUDE_harmonic_sum_identity_l1322_132296


namespace NUMINAMATH_CALUDE_planes_perpendicular_l1322_132229

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersects : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular
  (α β : Plane) (a b c : Line)
  (h1 : contains α a)
  (h2 : contains α b)
  (h3 : intersects a b)
  (h4 : perpendicular c a)
  (h5 : perpendicular c b)
  (h6 : parallel c β) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l1322_132229
