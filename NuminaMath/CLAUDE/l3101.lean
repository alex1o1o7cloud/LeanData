import Mathlib

namespace NUMINAMATH_CALUDE_three_digit_congruence_solutions_l3101_310114

theorem three_digit_congruence_solutions : 
  (Finset.filter (fun y : ℕ => 100 ≤ y ∧ y ≤ 999 ∧ (1945 * y + 243) % 17 = 605 % 17) 
    (Finset.range 1000)).card = 53 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_congruence_solutions_l3101_310114


namespace NUMINAMATH_CALUDE_original_average_age_l3101_310129

/-- Proves that the original average age of a class is 40 years given the specified conditions. -/
theorem original_average_age (original_strength : ℕ) (new_students : ℕ) (new_average_age : ℕ) (average_decrease : ℕ) :
  original_strength = 17 →
  new_students = 17 →
  new_average_age = 32 →
  average_decrease = 4 →
  ∃ (original_average : ℕ), 
    original_average * original_strength + new_students * new_average_age = 
    (original_strength + new_students) * (original_average - average_decrease) ∧
    original_average = 40 :=
by sorry

end NUMINAMATH_CALUDE_original_average_age_l3101_310129


namespace NUMINAMATH_CALUDE_factorization_of_x4_plus_81_l3101_310106

theorem factorization_of_x4_plus_81 (x : ℝ) : 
  x^4 + 81 = (x^2 + 3*x + 4.5) * (x^2 - 3*x + 4.5) := by sorry

end NUMINAMATH_CALUDE_factorization_of_x4_plus_81_l3101_310106


namespace NUMINAMATH_CALUDE_tan_value_from_double_angle_formula_l3101_310115

theorem tan_value_from_double_angle_formula (θ : Real) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : Real.sin (2 * θ) = 2 - 2 * Real.cos (2 * θ)) : 
  Real.tan θ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_double_angle_formula_l3101_310115


namespace NUMINAMATH_CALUDE_perfect_square_iff_divisibility_l3101_310193

theorem perfect_square_iff_divisibility (A : ℕ+) :
  (∃ d : ℕ+, A = d^2) ↔
  (∀ n : ℕ+, ∃ j : ℕ+, j ≤ n ∧ n ∣ ((A + j)^2 - A)) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_iff_divisibility_l3101_310193


namespace NUMINAMATH_CALUDE_largest_smallest_valid_numbers_l3101_310157

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000000 ∧ n < 10000000000) ∧  -- 10-digit number
  (n % 11 = 0) ∧                        -- divisible by 11
  (∀ i j, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10))  -- no repeated digits

theorem largest_smallest_valid_numbers :
  (∀ n : ℕ, is_valid_number n → n ≤ 9876524130) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ 1024375869) ∧
  is_valid_number 9876524130 ∧
  is_valid_number 1024375869 :=
sorry

end NUMINAMATH_CALUDE_largest_smallest_valid_numbers_l3101_310157


namespace NUMINAMATH_CALUDE_track_circumference_track_circumference_is_720_l3101_310145

/-- The circumference of a circular track given specific meeting conditions of two joggers --/
theorem track_circumference : ℝ → ℝ → ℝ → Prop :=
  fun first_meet second_meet circumference =>
    let half_circumference := circumference / 2
    first_meet = 150 ∧
    second_meet = circumference - 90 ∧
    first_meet / (half_circumference - first_meet) = (half_circumference + 90) / (circumference - 90) →
    circumference = 720

/-- The main theorem stating that the track circumference is 720 yards --/
theorem track_circumference_is_720 : ∃ (first_meet second_meet : ℝ),
  track_circumference first_meet second_meet 720 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_track_circumference_is_720_l3101_310145


namespace NUMINAMATH_CALUDE_M_union_N_equals_R_l3101_310103

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set N
def N : Set ℝ := {x | |x| < Real.sqrt 5}

-- Theorem statement
theorem M_union_N_equals_R : M ∪ N = Set.univ := by sorry

end NUMINAMATH_CALUDE_M_union_N_equals_R_l3101_310103


namespace NUMINAMATH_CALUDE_soda_weight_proof_l3101_310195

/-- Calculates the amount of soda in each can given the total weight, number of cans, and weight of empty cans. -/
def soda_per_can (total_weight : ℕ) (soda_cans : ℕ) (empty_cans : ℕ) (empty_can_weight : ℕ) : ℕ :=
  (total_weight - (soda_cans + empty_cans) * empty_can_weight) / soda_cans

/-- Proves that the amount of soda in each can is 12 ounces given the problem conditions. -/
theorem soda_weight_proof (total_weight : ℕ) (soda_cans : ℕ) (empty_cans : ℕ) (empty_can_weight : ℕ)
  (h1 : total_weight = 88)
  (h2 : soda_cans = 6)
  (h3 : empty_cans = 2)
  (h4 : empty_can_weight = 2) :
  soda_per_can total_weight soda_cans empty_cans empty_can_weight = 12 := by
  sorry

end NUMINAMATH_CALUDE_soda_weight_proof_l3101_310195


namespace NUMINAMATH_CALUDE_root_of_polynomial_l3101_310170

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 16*x^2 + 4

-- State the theorem
theorem root_of_polynomial :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 16*x^2 + 4) ∧
  -- The polynomial has degree 4
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- √5 + √3 is a root of the polynomial
  p (Real.sqrt 5 + Real.sqrt 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l3101_310170


namespace NUMINAMATH_CALUDE_four_number_sequence_l3101_310169

theorem four_number_sequence (x y z t : ℝ) : 
  (y - x = z - y) →  -- arithmetic sequence condition
  (z^2 = y * t) →    -- geometric sequence condition
  (x + t = 37) → 
  (y + z = 36) → 
  (x = 12 ∧ y = 16 ∧ z = 20 ∧ t = 25) := by
sorry

end NUMINAMATH_CALUDE_four_number_sequence_l3101_310169


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3101_310153

theorem inequality_solution_set (k : ℝ) :
  let S := {x : ℝ | k * x^2 - (k + 2) * x + 2 < 0}
  (k = 0 → S = {x : ℝ | x < 1}) ∧
  (0 < k ∧ k < 2 → S = {x : ℝ | x < 1 ∨ x > 2/k}) ∧
  (k = 2 → S = {x : ℝ | x ≠ 1}) ∧
  (k > 2 → S = {x : ℝ | x < 2/k ∨ x > 1}) ∧
  (k < 0 → S = {x : ℝ | 2/k < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3101_310153


namespace NUMINAMATH_CALUDE_min_moves_to_single_color_l3101_310142

/-- Represents a move on the chessboard -/
structure Move where
  m : Nat
  n : Nat

/-- Represents the chessboard -/
def Chessboard := Fin 7 → Fin 7 → Bool

/-- Applies a move to the chessboard -/
def applyMove (board : Chessboard) (move : Move) : Chessboard :=
  sorry

/-- Checks if the board is of a single color -/
def isSingleColor (board : Chessboard) : Bool :=
  sorry

/-- Initial chessboard with alternating colors -/
def initialBoard : Chessboard :=
  sorry

/-- Theorem stating the minimum number of moves required -/
theorem min_moves_to_single_color :
  ∃ (moves : List Move),
    moves.length = 6 ∧
    isSingleColor (moves.foldl applyMove initialBoard) ∧
    ∀ (otherMoves : List Move),
      isSingleColor (otherMoves.foldl applyMove initialBoard) →
      otherMoves.length ≥ 6 :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_single_color_l3101_310142


namespace NUMINAMATH_CALUDE_average_speed_bicycle_and_walk_l3101_310131

/-- Proves that the average speed of a pedestrian who rode a bicycle for 40 minutes at 5 m/s
    and then walked for 2 hours at 5 km/h is 8.25 km/h. -/
theorem average_speed_bicycle_and_walk (
  bicycle_time : Real) (bicycle_speed : Real) (walk_time : Real) (walk_speed : Real)
  (h1 : bicycle_time = 40 / 60) -- 40 minutes in hours
  (h2 : bicycle_speed = 5 * 3.6) -- 5 m/s converted to km/h
  (h3 : walk_time = 2) -- 2 hours
  (h4 : walk_speed = 5) -- 5 km/h
  : (bicycle_time * bicycle_speed + walk_time * walk_speed) / (bicycle_time + walk_time) = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_bicycle_and_walk_l3101_310131


namespace NUMINAMATH_CALUDE_logarithm_equation_l3101_310125

theorem logarithm_equation (x : ℝ) :
  1 - Real.log 5 = (1/3) * (Real.log (1/2) + Real.log x + (1/3) * Real.log 5) →
  x = 16 / Real.rpow 5 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equation_l3101_310125


namespace NUMINAMATH_CALUDE_smallest_whole_number_solution_l3101_310149

theorem smallest_whole_number_solution : 
  (∀ n : ℕ, n < 6 → (2 : ℚ) / 5 + (n : ℚ) / 9 ≤ 1) ∧ 
  ((2 : ℚ) / 5 + (6 : ℚ) / 9 > 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_solution_l3101_310149


namespace NUMINAMATH_CALUDE_min_sum_dimensions_2310_l3101_310194

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the volume of a box given its dimensions -/
def volume (d : BoxDimensions) : ℕ :=
  d.length.val * d.width.val * d.height.val

/-- Calculates the sum of dimensions of a box -/
def sumDimensions (d : BoxDimensions) : ℕ :=
  d.length.val + d.width.val + d.height.val

/-- Theorem: The minimum sum of dimensions for a box with volume 2310 is 42 -/
theorem min_sum_dimensions_2310 :
  (∃ d : BoxDimensions, volume d = 2310) →
  (∀ d : BoxDimensions, volume d = 2310 → sumDimensions d ≥ 42) ∧
  (∃ d : BoxDimensions, volume d = 2310 ∧ sumDimensions d = 42) :=
sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_2310_l3101_310194


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l3101_310133

theorem mean_equality_implies_z_value : 
  (∃ z : ℝ, (8 + 10 + 24) / 3 = (16 + z) / 2) → 
  (∃ z : ℝ, (8 + 10 + 24) / 3 = (16 + z) / 2 ∧ z = 12) :=
by sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l3101_310133


namespace NUMINAMATH_CALUDE_eric_egg_collection_days_l3101_310175

/-- Proves that Eric waited 3 days to collect 36 eggs from 4 chickens laying 3 eggs each per day -/
theorem eric_egg_collection_days (num_chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (total_eggs : ℕ) : 
  num_chickens = 4 → 
  eggs_per_chicken_per_day = 3 → 
  total_eggs = 36 → 
  (total_eggs / (num_chickens * eggs_per_chicken_per_day) : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_eric_egg_collection_days_l3101_310175


namespace NUMINAMATH_CALUDE_henry_trays_problem_l3101_310117

theorem henry_trays_problem (trays_per_trip : ℕ) (trips : ℕ) (trays_second_table : ℕ) :
  trays_per_trip = 9 →
  trips = 9 →
  trays_second_table = 52 →
  trays_per_trip * trips - trays_second_table = 29 :=
by sorry

end NUMINAMATH_CALUDE_henry_trays_problem_l3101_310117


namespace NUMINAMATH_CALUDE_average_of_w_and_x_l3101_310138

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 7 / w + 7 / x = 7 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = y / 2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_w_and_x_l3101_310138


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3101_310182

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    prove that if a_3 = 2S_2 + 1 and a_4 = 2S_3 + 1, then q = 3. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- a_n is a geometric sequence with common ratio q
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- S_n is the sum of first n terms
  a 3 = 2 * S 2 + 1 →  -- a_3 = 2S_2 + 1
  a 4 = 2 * S 3 + 1 →  -- a_4 = 2S_3 + 1
  q = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3101_310182


namespace NUMINAMATH_CALUDE_monochromatic_four_clique_exists_l3101_310100

/-- A two-color edge coloring of a complete graph. -/
def TwoColorEdgeColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- The existence of a monochromatic 4-clique in a two-color edge coloring of K_18. -/
theorem monochromatic_four_clique_exists :
  ∀ (coloring : TwoColorEdgeColoring 18),
  ∃ (vertices : Fin 4 → Fin 18),
    (∀ (i j : Fin 4), i ≠ j →
      coloring (vertices i) (vertices j) = coloring (vertices 0) (vertices 1)) :=
by sorry

end NUMINAMATH_CALUDE_monochromatic_four_clique_exists_l3101_310100


namespace NUMINAMATH_CALUDE_f_decreasing_range_l3101_310141

/-- A piecewise function f(x) defined by a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

/-- The theorem stating the range of a for which f is strictly decreasing -/
theorem f_decreasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) ↔ 
  (0 < a ∧ a ≤ 1/4) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_range_l3101_310141


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3101_310124

/-- Two variables are inversely proportional if their product is constant -/
def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  inversely_proportional x y →
  x + y = 40 →
  x - y = 8 →
  x = 7 →
  y = 54 + 6/7 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3101_310124


namespace NUMINAMATH_CALUDE_domestic_needs_fraction_l3101_310181

def total_income : ℚ := 200
def provident_fund_rate : ℚ := 1/16
def insurance_premium_rate : ℚ := 1/15
def bank_deposit : ℚ := 50

def remaining_after_provident_fund : ℚ := total_income * (1 - provident_fund_rate)
def remaining_after_insurance : ℚ := remaining_after_provident_fund * (1 - insurance_premium_rate)

theorem domestic_needs_fraction :
  (remaining_after_insurance - bank_deposit) / remaining_after_insurance = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_domestic_needs_fraction_l3101_310181


namespace NUMINAMATH_CALUDE_final_position_theorem_supplement_angle_beta_theorem_l3101_310105

-- Define the initial position of point A
def initial_position : Int := -5

-- Define the movement of point A
def move_right : Int := 4
def move_left : Int := 1

-- Define the angle α
def angle_alpha : Int := 40

-- Theorem for the final position of point A
theorem final_position_theorem :
  initial_position + move_right - move_left = -2 := by sorry

-- Theorem for the supplement of angle β
theorem supplement_angle_beta_theorem :
  180 - (90 - angle_alpha) = 130 := by sorry

end NUMINAMATH_CALUDE_final_position_theorem_supplement_angle_beta_theorem_l3101_310105


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3101_310176

/-- The hyperbola and parabola intersect at two points A and B -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The common focus of the hyperbola and parabola -/
def CommonFocus : ℝ × ℝ := (1, 2)

/-- The hyperbola equation -/
def isOnHyperbola (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (p.1^2 / a^2) - (p.2^2 / b^2) = 1

/-- The parabola equation -/
def isOnParabola (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

/-- Line AB passes through the common focus -/
def lineABThroughFocus (points : IntersectionPoints) : Prop :=
  ∃ (t : ℝ), (1 - t) * points.A.1 + t * points.B.1 = CommonFocus.1 ∧
             (1 - t) * points.A.2 + t * points.B.2 = CommonFocus.2

/-- Theorem: The length of the real axis of the hyperbola is 2√2 - 2 -/
theorem hyperbola_real_axis_length (a b : ℝ) (points : IntersectionPoints) :
  isOnHyperbola a b CommonFocus →
  isOnParabola CommonFocus →
  lineABThroughFocus points →
  2 * a = 2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3101_310176


namespace NUMINAMATH_CALUDE_angle_side_inequality_l3101_310191

-- Define a structure for triangles
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the property that larger angles are opposite longer sides
axiom larger_angle_longer_side {t : Triangle} : 
  ∀ (x y : Real), (x = t.A ∧ y = t.a) ∨ (x = t.B ∧ y = t.b) ∨ (x = t.C ∧ y = t.c) →
  ∀ (p q : Real), (p = t.A ∧ q = t.a) ∨ (p = t.B ∧ q = t.b) ∨ (p = t.C ∧ q = t.c) →
  x > p → y > q

-- Theorem statement
theorem angle_side_inequality (t : Triangle) : t.A < t.B → t.a < t.b := by
  sorry

end NUMINAMATH_CALUDE_angle_side_inequality_l3101_310191


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l3101_310116

def A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -4; -3, 2]

theorem matrix_inverse_proof :
  ∃ (B : Matrix (Fin 2) (Fin 2) ℝ),
    B = !![1, 2; 1.5, 3.5] ∧ A * B = 1 ∧ B * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l3101_310116


namespace NUMINAMATH_CALUDE_max_sales_revenue_l3101_310121

/-- Sales price as a function of time -/
def sales_price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

/-- Daily sales volume as a function of time -/
def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

/-- Daily sales revenue as a function of time -/
def sales_revenue (t : ℕ) : ℝ :=
  sales_price t * sales_volume t

/-- The maximum daily sales revenue and the day it occurs -/
theorem max_sales_revenue :
  (∃ (t : ℕ), 0 < t ∧ t ≤ 30 ∧ 
    sales_revenue t = 1125 ∧
    ∀ (s : ℕ), 0 < s ∧ s ≤ 30 → sales_revenue s ≤ sales_revenue t) ∧
  (∀ (s : ℕ), 0 < s ∧ s ≤ 30 ∧ sales_revenue s = 1125 → s = 25) :=
by sorry

end NUMINAMATH_CALUDE_max_sales_revenue_l3101_310121


namespace NUMINAMATH_CALUDE_shaded_area_of_square_with_circles_l3101_310118

/-- Given a square with side length 24 inches and three circles, each tangent to two sides of the square
    and one adjacent circle, the shaded area (area not covered by the circles) is 576 - 108π square inches. -/
theorem shaded_area_of_square_with_circles (side : ℝ) (circles : ℕ) : 
  side = 24 → circles = 3 → (side^2 - circles * (side/4)^2 * Real.pi) = 576 - 108 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_with_circles_l3101_310118


namespace NUMINAMATH_CALUDE_circle_radius_zero_l3101_310140

theorem circle_radius_zero (x y : ℝ) : 
  4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0 → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l3101_310140


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3101_310165

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3101_310165


namespace NUMINAMATH_CALUDE_smallest_money_sum_l3101_310113

/-- Represents a sum of money in pounds, shillings, pence, and farthings -/
structure Money where
  pounds : ℕ
  shillings : ℕ
  pence : ℕ
  farthings : ℕ
  shillings_valid : shillings < 20
  pence_valid : pence < 12
  farthings_valid : farthings < 4

/-- Checks if a list of digits contains each of 1 to 9 exactly once -/
def valid_digits (digits : List ℕ) : Prop :=
  digits.length = 9 ∧ (∀ d, d ∈ digits → d ≥ 1 ∧ d ≤ 9) ∧ digits.Nodup

/-- Converts a Money value to its total value in farthings -/
def to_farthings (m : Money) : ℕ :=
  m.pounds * 960 + m.shillings * 48 + m.pence * 4 + m.farthings

/-- The theorem to be proved -/
theorem smallest_money_sum :
  ∃ (m : Money) (digits : List ℕ),
    valid_digits digits ∧
    to_farthings m = to_farthings ⟨2567, 18, 9, 3, by sorry, by sorry, by sorry⟩ ∧
    (∀ (m' : Money) (digits' : List ℕ),
      valid_digits digits' →
      to_farthings m' ≥ to_farthings m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_money_sum_l3101_310113


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l3101_310101

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (75/23, -64/23)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 8 * x - 5 * y = 40

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 14

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations : 
  line1 intersection_point.1 intersection_point.2 ∧ 
  line2 intersection_point.1 intersection_point.2 := by
  sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique (x y : ℚ) : 
  line1 x y ∧ line2 x y → (x, y) = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l3101_310101


namespace NUMINAMATH_CALUDE_factorial_inequality_l3101_310197

theorem factorial_inequality (k : ℕ) (h : k ≥ 2) :
  ((k + 1) / 2 : ℝ) ^ k > k! :=
by sorry

end NUMINAMATH_CALUDE_factorial_inequality_l3101_310197


namespace NUMINAMATH_CALUDE_rectangle_diagonal_pi_irrational_l3101_310152

theorem rectangle_diagonal_pi_irrational 
  (m n p q : ℤ) 
  (hn : n ≠ 0) 
  (hq : q ≠ 0) :
  let l : ℚ := m / n
  let w : ℚ := p / q
  let d : ℝ := Real.sqrt ((l * l + w * w : ℚ) : ℝ)
  Irrational (π * d) := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_pi_irrational_l3101_310152


namespace NUMINAMATH_CALUDE_money_conditions_l3101_310168

theorem money_conditions (a b : ℝ) 
  (h1 : b - 4*a < 78)
  (h2 : 6*a - b = 36)
  : a < 57 ∧ b > -36 := by
  sorry

end NUMINAMATH_CALUDE_money_conditions_l3101_310168


namespace NUMINAMATH_CALUDE_problem_triangle_integer_lengths_l3101_310136

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths of line segments
    that can be drawn from a vertex to the opposite side in a right triangle -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle in the problem -/
def problemTriangle : RightTriangle :=
  { de := 15, ef := 36 }

/-- Theorem stating that the number of distinct integer lengths
    in the problem triangle is 24 -/
theorem problem_triangle_integer_lengths :
  countIntegerLengths problemTriangle = 24 := by
  sorry

end NUMINAMATH_CALUDE_problem_triangle_integer_lengths_l3101_310136


namespace NUMINAMATH_CALUDE_ingredient_problem_l3101_310107

/-- Represents the quantities and prices of ingredients A and B -/
structure Ingredients where
  total_quantity : ℕ
  price_a : ℕ
  price_b_base : ℕ
  price_b_decrease : ℚ
  quantity_b : ℕ

/-- The total cost function for the ingredients -/
def total_cost (i : Ingredients) : ℚ :=
  if i.quantity_b ≤ 300 then
    (i.total_quantity - i.quantity_b) * i.price_a + i.quantity_b * i.price_b_base
  else
    (i.total_quantity - i.quantity_b) * i.price_a + 
    i.quantity_b * (i.price_b_base - (i.quantity_b - 300) / 10 * i.price_b_decrease)

/-- The main theorem encompassing all parts of the problem -/
theorem ingredient_problem (i : Ingredients) 
  (h_total : i.total_quantity = 600)
  (h_price_a : i.price_a = 5)
  (h_price_b_base : i.price_b_base = 9)
  (h_price_b_decrease : i.price_b_decrease = 0.1)
  (h_quantity_b_multiple : i.quantity_b % 10 = 0) :
  (∃ (x : ℕ), x < 300 ∧ i.quantity_b = x ∧ total_cost i = 3800 → 
    i.total_quantity - x = 400 ∧ x = 200) ∧
  (∃ (x : ℕ), x > 300 ∧ i.quantity_b = x ∧ 2 * (i.total_quantity - x) ≥ x → 
    ∃ (min_cost : ℚ), min_cost = 4200 ∧ 
    ∀ (y : ℕ), y > 300 ∧ 2 * (i.total_quantity - y) ≥ y → 
      total_cost { i with quantity_b := y } ≥ min_cost) ∧
  (∃ (m : ℕ), m < 250 ∧ 
    (∀ (x : ℕ), x > 300 ∧ i.total_quantity - x ≤ m → 
      total_cost { i with quantity_b := x } ≤ 4000) ∧
    (∃ (x : ℕ), x > 300 ∧ i.total_quantity - x ≤ m ∧ 
      total_cost { i with quantity_b := x } = 4000) →
    m = 100) := by
  sorry

end NUMINAMATH_CALUDE_ingredient_problem_l3101_310107


namespace NUMINAMATH_CALUDE_emily_orange_ratio_l3101_310159

/-- Given the following conditions about oranges:
  * Emily has some times as many oranges as Sandra
  * Sandra has 3 times as many oranges as Betty
  * Betty has 12 oranges
  * Emily has 252 oranges
Prove that Emily has 7 times more oranges than Sandra. -/
theorem emily_orange_ratio (betty_oranges sandra_oranges emily_oranges : ℕ) 
  (h1 : sandra_oranges = 3 * betty_oranges)
  (h2 : betty_oranges = 12)
  (h3 : emily_oranges = 252) :
  emily_oranges / sandra_oranges = 7 := by
  sorry

end NUMINAMATH_CALUDE_emily_orange_ratio_l3101_310159


namespace NUMINAMATH_CALUDE_two_digit_divisible_by_digit_product_l3101_310160

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def ones_digit (n : ℕ) : ℕ := n % 10

def divisible_by_digit_product (n : ℕ) : Prop :=
  is_two_digit n ∧ n % (tens_digit n * ones_digit n) = 0

theorem two_digit_divisible_by_digit_product :
  {n : ℕ | divisible_by_digit_product n} = {11, 12, 24, 36, 15} :=
by sorry

end NUMINAMATH_CALUDE_two_digit_divisible_by_digit_product_l3101_310160


namespace NUMINAMATH_CALUDE_max_volume_cutout_length_l3101_310135

/-- The side length of the original square sheet of iron in centimeters -/
def original_side_length : ℝ := 36

/-- The volume of the box as a function of the side length of the cut-out square -/
def volume (x : ℝ) : ℝ := x * (original_side_length - 2*x)^2

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := 12 * (18 - x) * (6 - x)

theorem max_volume_cutout_length :
  ∃ (x : ℝ), 0 < x ∧ x < original_side_length / 2 ∧
  volume_derivative x = 0 ∧
  (∀ y, 0 < y → y < original_side_length / 2 → volume y ≤ volume x) ∧
  x = 6 := by sorry

end NUMINAMATH_CALUDE_max_volume_cutout_length_l3101_310135


namespace NUMINAMATH_CALUDE_rent_comparison_l3101_310132

theorem rent_comparison (last_year_earnings : ℝ) : 
  let last_year_rent := 0.25 * last_year_earnings
  let this_year_earnings := last_year_earnings * 1.35
  let this_year_rent := 0.40 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 216 := by
sorry

end NUMINAMATH_CALUDE_rent_comparison_l3101_310132


namespace NUMINAMATH_CALUDE_cube_squared_equals_sixth_power_l3101_310123

theorem cube_squared_equals_sixth_power (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_squared_equals_sixth_power_l3101_310123


namespace NUMINAMATH_CALUDE_first_term_is_0_375_l3101_310192

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the first 40 terms is 600 -/
  sum_first_40 : (40 : ℝ) / 2 * (2 * a + 39 * d) = 600
  /-- The sum of the next 40 terms (terms 41 to 80) is 1800 -/
  sum_next_40 : (40 : ℝ) / 2 * (2 * (a + 40 * d) + 39 * d) = 1800

/-- The first term of the arithmetic sequence with the given properties is 0.375 -/
theorem first_term_is_0_375 (seq : ArithmeticSequence) : seq.a = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_0_375_l3101_310192


namespace NUMINAMATH_CALUDE_hotel_room_charges_l3101_310178

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * 0.8)  -- P is 20% less than R
  (h2 : P = G * 0.9)  -- P is 10% less than G
  : R = G * 1.125 :=  -- R is 12.5% greater than G
by sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l3101_310178


namespace NUMINAMATH_CALUDE_diane_age_when_condition_met_l3101_310119

/-- Represents the ages of Diane, Alex, and Allison at the time when the condition is met -/
structure Ages where
  diane : ℕ
  alex : ℕ
  allison : ℕ

/-- Checks if the given ages satisfy the condition -/
def satisfiesCondition (ages : Ages) : Prop :=
  ages.diane = ages.alex / 2 ∧ ages.diane = 2 * ages.allison

/-- Represents the current ages of Diane, Alex, and Allison -/
structure CurrentAges where
  diane : ℕ
  alexPlusAllison : ℕ

/-- Theorem stating that Diane will be 78 when the condition is met -/
theorem diane_age_when_condition_met (current : CurrentAges)
    (h1 : current.diane = 16)
    (h2 : current.alexPlusAllison = 47) :
    ∃ (ages : Ages), satisfiesCondition ages ∧ ages.diane = 78 :=
  sorry

end NUMINAMATH_CALUDE_diane_age_when_condition_met_l3101_310119


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l3101_310196

/-- Represents a square tile pattern -/
structure TilePattern :=
  (side : ℕ)
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- The initial square pattern -/
def initial_pattern : TilePattern :=
  { side := 5
  , black_tiles := 8
  , white_tiles := 17 }

/-- Extends a tile pattern by adding a black border -/
def extend_pattern (p : TilePattern) : TilePattern :=
  { side := p.side + 2
  , black_tiles := p.black_tiles + 2 * p.side + 2 * (p.side + 2)
  , white_tiles := p.white_tiles }

/-- The theorem to be proved -/
theorem extended_pattern_ratio (p : TilePattern) : 
  p = initial_pattern → 
  (extend_pattern p).black_tiles = 32 ∧ (extend_pattern p).white_tiles = 17 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l3101_310196


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3101_310144

theorem nested_fraction_evaluation :
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3101_310144


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3101_310122

theorem polynomial_factorization (x : ℝ) : 
  x^6 + 6*x^5 + 15*x^4 + 20*x^3 + 15*x^2 + 6*x + 1 = (x+1)^6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3101_310122


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3101_310112

theorem geometric_series_ratio (a : ℝ) (r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4 / (1 - r))) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3101_310112


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l3101_310162

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation 
  (rate : ℝ) (interest : ℝ) (time : ℝ) :
  rate = 4.783950617283951 →
  interest = 155 →
  time = 4 →
  ∃ (principal : ℝ), 
    (principal * rate * time) / 100 = interest ∧ 
    abs (principal - 810.13) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l3101_310162


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3101_310166

theorem consecutive_integers_sum (m : ℤ) :
  let sequence := [m, m+1, m+2, m+3, m+4, m+5, m+6]
  (sequence.sum - (sequence.take 3).sum) = 4*m + 18 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3101_310166


namespace NUMINAMATH_CALUDE_binomial_sum_l3101_310110

theorem binomial_sum : Nat.choose 18 4 + Nat.choose 5 2 = 3070 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l3101_310110


namespace NUMINAMATH_CALUDE_difference_of_squares_and_perfect_squares_l3101_310190

theorem difference_of_squares_and_perfect_squares : 
  (102^2 - 98^2 = 800) ∧ 
  (¬ ∃ n : ℕ, n^2 = 102) ∧ 
  (¬ ∃ m : ℕ, m^2 = 98) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_and_perfect_squares_l3101_310190


namespace NUMINAMATH_CALUDE_snyder_cookies_l3101_310109

/-- Given that Mrs. Snyder made a total of 86 cookies, with only red and pink colors,
    and 50 of them are pink, prove that she made 36 red cookies. -/
theorem snyder_cookies (total : ℕ) (pink : ℕ) (red : ℕ) : 
  total = 86 → pink = 50 → total = pink + red → red = 36 := by
  sorry

end NUMINAMATH_CALUDE_snyder_cookies_l3101_310109


namespace NUMINAMATH_CALUDE_large_circle_radius_l3101_310177

/-- Configuration of circles -/
structure CircleConfiguration where
  small_radius : ℝ
  chord_length : ℝ
  small_circle_count : ℕ

/-- Theorem: If five identical circles are placed in a line inside a larger circle,
    and the chord connecting the endpoints of the line of circles has length 16,
    then the radius of the large circle is 8. -/
theorem large_circle_radius
  (config : CircleConfiguration)
  (h1 : config.small_circle_count = 5)
  (h2 : config.chord_length = 16) :
  4 * config.small_radius = 8 := by
  sorry

#check large_circle_radius

end NUMINAMATH_CALUDE_large_circle_radius_l3101_310177


namespace NUMINAMATH_CALUDE_expression_equality_l3101_310120

theorem expression_equality : 
  (2 / 3 * Real.sqrt 15 - Real.sqrt 20) / (1 / 3 * Real.sqrt 5) = 2 * Real.sqrt 3 - 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3101_310120


namespace NUMINAMATH_CALUDE_optimal_discount_savings_l3101_310151

def initial_order : ℝ := 15000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def apply_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

def discount_set1 : List ℝ := [0.25, 0.15, 0.10]
def discount_set2 : List ℝ := [0.30, 0.10, 0.05]

theorem optimal_discount_savings :
  apply_discounts initial_order discount_set2 - apply_discounts initial_order discount_set1 = 371.25 := by
  sorry

end NUMINAMATH_CALUDE_optimal_discount_savings_l3101_310151


namespace NUMINAMATH_CALUDE_doughnuts_left_l3101_310158

theorem doughnuts_left (total_doughnuts : ℕ) (staff_count : ℕ) (doughnuts_per_staff : ℕ) :
  total_doughnuts = 50 →
  staff_count = 19 →
  doughnuts_per_staff = 2 →
  total_doughnuts - (staff_count * doughnuts_per_staff) = 12 := by
sorry

end NUMINAMATH_CALUDE_doughnuts_left_l3101_310158


namespace NUMINAMATH_CALUDE_cannot_form_square_l3101_310198

/-- Represents the number of sticks of each length --/
structure Sticks :=
  (length1 : ℕ)
  (length2 : ℕ)
  (length3 : ℕ)
  (length4 : ℕ)

/-- Calculates the total length of all sticks --/
def totalLength (s : Sticks) : ℕ :=
  s.length1 * 1 + s.length2 * 2 + s.length3 * 3 + s.length4 * 4

/-- Represents the given set of sticks --/
def givenSticks : Sticks :=
  { length1 := 6
  , length2 := 3
  , length3 := 6
  , length4 := 5 }

/-- Theorem stating that it's impossible to form a square with the given sticks --/
theorem cannot_form_square (s : Sticks) (h : s = givenSticks) :
  ¬ ∃ (side : ℕ), side > 0 ∧ 4 * side = totalLength s :=
by sorry


end NUMINAMATH_CALUDE_cannot_form_square_l3101_310198


namespace NUMINAMATH_CALUDE_percentage_increase_l3101_310155

theorem percentage_increase (x : ℝ) (h : x = 105.6) :
  (x - 88) / 88 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3101_310155


namespace NUMINAMATH_CALUDE_simplify_polynomial_product_l3101_310172

theorem simplify_polynomial_product (a : ℝ) : 
  (1 : ℝ) * (2 * a) * (3 * a^2) * (4 * a^3) * (5 * a^4) * (6 * a^5) = 720 * a^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_product_l3101_310172


namespace NUMINAMATH_CALUDE_davids_english_marks_l3101_310171

/-- Given David's marks in various subjects and his average, prove his marks in English --/
theorem davids_english_marks :
  let math_marks : ℕ := 95
  let physics_marks : ℕ := 82
  let chemistry_marks : ℕ := 97
  let biology_marks : ℕ := 95
  let average_marks : ℕ := 93
  let total_subjects : ℕ := 5
  let total_marks : ℕ := average_marks * total_subjects
  let known_marks_sum : ℕ := math_marks + physics_marks + chemistry_marks + biology_marks
  let english_marks : ℕ := total_marks - known_marks_sum
  english_marks = 96 := by
sorry

end NUMINAMATH_CALUDE_davids_english_marks_l3101_310171


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3101_310126

/-- The sum of a geometric series with first term 3, common ratio -2, and last term 768 is 513. -/
theorem geometric_series_sum : 
  ∀ (n : ℕ) (a : ℝ) (r : ℝ) (S : ℝ),
  a = 3 →
  r = -2 →
  a * r^(n-1) = 768 →
  S = (a * (1 - r^n)) / (1 - r) →
  S = 513 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3101_310126


namespace NUMINAMATH_CALUDE_lucy_snack_bar_total_cost_l3101_310108

/-- The cost of a single sandwich at Lucy's Snack Bar -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda at Lucy's Snack Bar -/
def soda_cost : ℕ := 3

/-- The number of sandwiches Lucy wants to buy -/
def num_sandwiches : ℕ := 7

/-- The number of sodas Lucy wants to buy -/
def num_sodas : ℕ := 8

/-- The theorem stating that the total cost of Lucy's purchase is $52 -/
theorem lucy_snack_bar_total_cost : 
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = 52 := by
  sorry

end NUMINAMATH_CALUDE_lucy_snack_bar_total_cost_l3101_310108


namespace NUMINAMATH_CALUDE_expression_evaluation_l3101_310189

theorem expression_evaluation :
  let x : ℚ := -1
  let y : ℚ := 2
  (2*x + y) * (2*x - y) - (8*x^3*y - 2*x*y^3 - x^2*y^2) / (2*x*y) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3101_310189


namespace NUMINAMATH_CALUDE_equilateral_triangle_count_l3101_310130

/-- Represents a line in the 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (x y : ℝ) → a * x + b * y + c = 0

/-- Generates the set of lines given by y = k, y = x + 3k, and y = -x + 3k for k from -6 to 6 --/
def generateLines : Set Line := sorry

/-- Checks if three lines form an equilateral triangle of side 1 --/
def formEquilateralTriangle (l1 l2 l3 : Line) : Prop := sorry

/-- Counts the number of equilateral triangles formed by the intersection of lines --/
def countEquilateralTriangles (lines : Set Line) : ℕ := sorry

/-- The main theorem stating that the number of equilateral triangles is 444 --/
theorem equilateral_triangle_count :
  ∃ (lines : Set Line), 
    lines = generateLines ∧ 
    countEquilateralTriangles lines = 444 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_count_l3101_310130


namespace NUMINAMATH_CALUDE_log_product_equality_l3101_310199

open Real

theorem log_product_equality (A m n p : ℝ) (hA : A > 0) (hm : m > 0) (hn : n > 0) (hp : p > 0) :
  (log A / log m) * (log A / log n) + (log A / log n) * (log A / log p) + (log A / log p) * (log A / log m) =
  (log (m * n * p) / log A) * (log A / log p) * (log A / log n) * (log A / log m) :=
by sorry

#check log_product_equality

end NUMINAMATH_CALUDE_log_product_equality_l3101_310199


namespace NUMINAMATH_CALUDE_statement_a_statement_d_l3101_310127

-- Statement A
theorem statement_a (a b c : ℝ) (h1 : c ≠ 0) (h2 : a * c^2 > b * c^2) : a > b := by
  sorry

-- Statement D
theorem statement_d (a b : ℝ) (h : a > b ∧ b > 0) : a + 1/b > b + 1/a := by
  sorry

end NUMINAMATH_CALUDE_statement_a_statement_d_l3101_310127


namespace NUMINAMATH_CALUDE_three_digit_numbers_count_l3101_310111

def Digits : Finset Nat := {1, 2, 3, 4}

theorem three_digit_numbers_count : 
  Finset.card (Finset.product (Finset.product Digits Digits) Digits) = 64 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_count_l3101_310111


namespace NUMINAMATH_CALUDE_cubic_term_simplification_l3101_310143

theorem cubic_term_simplification (a : ℝ) : a^3 + 7*a^3 - 5*a^3 = 3*a^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_term_simplification_l3101_310143


namespace NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l3101_310187

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - (2 * Q.quadrilateral_faces)

/-- Theorem: A convex polyhedron Q with 30 vertices, 70 edges, 42 faces
    (30 triangular and 12 quadrilateral) has 341 space diagonals -/
theorem space_diagonals_of_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 42,
    triangular_faces := 30,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 341 := by sorry

end NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l3101_310187


namespace NUMINAMATH_CALUDE_mica_sandwich_options_l3101_310154

-- Define the types of sandwich components
def BreadTypes : ℕ := 6
def MeatTypes : ℕ := 7
def CheeseTypes : ℕ := 6

-- Define the restricted combinations
def TurkeySwissCombinations : ℕ := BreadTypes
def SourdoughChickenCombinations : ℕ := CheeseTypes
def SalamiRyeCombinations : ℕ := CheeseTypes

-- Define the total number of restricted combinations
def TotalRestrictedCombinations : ℕ :=
  TurkeySwissCombinations + SourdoughChickenCombinations + SalamiRyeCombinations

-- Define the total number of possible sandwich combinations
def TotalPossibleCombinations : ℕ := BreadTypes * MeatTypes * CheeseTypes

-- Define the number of sandwiches Mica could order
def MicaSandwichOptions : ℕ := TotalPossibleCombinations - TotalRestrictedCombinations

-- Theorem statement
theorem mica_sandwich_options :
  MicaSandwichOptions = 234 := by sorry

end NUMINAMATH_CALUDE_mica_sandwich_options_l3101_310154


namespace NUMINAMATH_CALUDE_melanie_dimes_value_l3101_310164

def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4
def dime_value : ℚ := 0.1

def total_dimes : ℕ := initial_dimes + dimes_from_dad + dimes_from_mom

theorem melanie_dimes_value :
  (total_dimes : ℚ) * dime_value = 1.9 := by sorry

end NUMINAMATH_CALUDE_melanie_dimes_value_l3101_310164


namespace NUMINAMATH_CALUDE_sum_difference_theorem_l3101_310161

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (n : ℕ) : ℕ :=
  let m := n % 5
  if m < 3 then n - m else n + (5 - m)

def emma_sum (n : ℕ) : ℕ :=
  List.range n |> List.map round_to_nearest_five |> List.sum

theorem sum_difference_theorem :
  sum_to_n 100 - emma_sum 100 = 4750 := by sorry

end NUMINAMATH_CALUDE_sum_difference_theorem_l3101_310161


namespace NUMINAMATH_CALUDE_cake_box_height_l3101_310148

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of items that can fit along a dimension -/
def maxItemsAlongDimension (containerSize itemSize : ℕ) : ℕ :=
  containerSize / itemSize

/-- Represents the problem of determining the height of cake boxes in a carton -/
def cakeBoxProblem (cartonDims : Dimensions) (cakeBoxBase : Dimensions) (maxBoxes : ℕ) : Prop :=
  let boxesAlongLength := maxItemsAlongDimension cartonDims.length cakeBoxBase.length
  let boxesAlongWidth := maxItemsAlongDimension cartonDims.width cakeBoxBase.width
  let boxesPerLayer := boxesAlongLength * boxesAlongWidth
  let numLayers := maxBoxes / boxesPerLayer
  let cakeBoxHeight := cartonDims.height / numLayers
  cakeBoxHeight = 5

/-- The main theorem stating that the height of a cake box is 5 inches -/
theorem cake_box_height :
  cakeBoxProblem
    (Dimensions.mk 25 42 60)  -- Carton dimensions
    (Dimensions.mk 8 7 0)     -- Cake box base dimensions (height is unknown)
    210                       -- Maximum number of boxes
  := by sorry

end NUMINAMATH_CALUDE_cake_box_height_l3101_310148


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3101_310179

theorem decimal_to_fraction (x : ℚ) (h : x = 3.68) : 
  ∃ (n d : ℕ), d ≠ 0 ∧ x = n / d ∧ n = 92 ∧ d = 25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3101_310179


namespace NUMINAMATH_CALUDE_ratio_problem_l3101_310104

theorem ratio_problem (a b x m : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.75 * a) 
  (h5 : m = b - 0.80 * b) : 
  m / x = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3101_310104


namespace NUMINAMATH_CALUDE_exists_double_application_square_l3101_310188

theorem exists_double_application_square :
  ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n^2 := by sorry

end NUMINAMATH_CALUDE_exists_double_application_square_l3101_310188


namespace NUMINAMATH_CALUDE_lunch_percentage_boys_l3101_310180

theorem lunch_percentage_boys (C B G : ℝ) (P_b : ℝ) :
  B / G = 3 / 2 →
  C = B + G →
  0.52 * C = (P_b / 100) * B + (40 / 100) * G →
  P_b = 60 := by
  sorry

end NUMINAMATH_CALUDE_lunch_percentage_boys_l3101_310180


namespace NUMINAMATH_CALUDE_sine_cosine_roots_l3101_310137

theorem sine_cosine_roots (α : Real) (m : Real) : 
  α ∈ Set.Ioo 0 (2 * Real.pi) →
  (∃ (x y : Real), x = Real.sin α ∧ y = Real.cos α ∧
    2 * x^2 - (Real.sqrt 3 + 1) * x + m / 3 = 0 ∧
    2 * y^2 - (Real.sqrt 3 + 1) * y + m / 3 = 0) →
  m = 3 * Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_roots_l3101_310137


namespace NUMINAMATH_CALUDE_total_oranges_approx_45_l3101_310163

/-- The number of bags of oranges -/
def num_bags : ℝ := 1.956521739

/-- The number of pounds of oranges per bag -/
def pounds_per_bag : ℝ := 23.0

/-- The total pounds of oranges -/
def total_pounds : ℝ := num_bags * pounds_per_bag

/-- Theorem stating that the total pounds of oranges is approximately 45.00 pounds -/
theorem total_oranges_approx_45 :
  ∃ ε > 0, |total_pounds - 45.00| < ε :=
sorry

end NUMINAMATH_CALUDE_total_oranges_approx_45_l3101_310163


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3101_310134

def cabinet_price : ℝ := 1200
def cabinet_discount : ℝ := 0.15
def dining_table_price : ℝ := 1800
def dining_table_discount : ℝ := 0.20
def sofa_price : ℝ := 2500
def sofa_discount : ℝ := 0.10
def sales_tax : ℝ := 0.08

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_discounted_price : ℝ :=
  discounted_price cabinet_price cabinet_discount +
  discounted_price dining_table_price dining_table_discount +
  discounted_price sofa_price sofa_discount

def total_cost : ℝ :=
  total_discounted_price * (1 + sales_tax)

theorem total_cost_calculation :
  total_cost = 5086.80 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3101_310134


namespace NUMINAMATH_CALUDE_edward_book_spending_l3101_310128

/-- Given Edward's initial amount, amount spent on pens, and remaining amount,
    prove that the amount spent on books is $6. -/
theorem edward_book_spending (initial : ℕ) (spent_on_pens : ℕ) (remaining : ℕ) 
    (h1 : initial = 41)
    (h2 : spent_on_pens = 16)
    (h3 : remaining = 19) :
    initial - remaining - spent_on_pens = 6 := by
  sorry

end NUMINAMATH_CALUDE_edward_book_spending_l3101_310128


namespace NUMINAMATH_CALUDE_min_value_sum_l3101_310167

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 1 / Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l3101_310167


namespace NUMINAMATH_CALUDE_banana_apple_equivalence_l3101_310174

-- Define the worth of bananas in terms of apples
def banana_worth (b : ℚ) : ℚ := 
  (12 : ℚ) / ((3 / 4) * 16)

-- Theorem statement
theorem banana_apple_equivalence : 
  banana_worth ((1 / 3) * 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_banana_apple_equivalence_l3101_310174


namespace NUMINAMATH_CALUDE_inequality_solution_set_not_equal_function_always_negative_implies_k_range_negation_of_inequality_solution_set_is_true_l3101_310156

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) * Real.sqrt (x^2 - 3*x + 2) ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ 2}

-- Define the function y
def y (k x : ℝ) : ℝ := k * x^2 - k * x - 1

theorem inequality_solution_set_not_equal : 
  {x : ℝ | inequality x} ≠ solution_set := by sorry

theorem function_always_negative_implies_k_range (k : ℝ) :
  (∀ x, y k x < 0) → -4 < k ∧ k ≤ 0 := by sorry

theorem negation_of_inequality_solution_set_is_true :
  ¬({x : ℝ | inequality x} = solution_set) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_not_equal_function_always_negative_implies_k_range_negation_of_inequality_solution_set_is_true_l3101_310156


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l3101_310185

theorem quadratic_inequality_solution_condition (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ |a| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l3101_310185


namespace NUMINAMATH_CALUDE_right_triangle_side_lengths_l3101_310173

theorem right_triangle_side_lengths (x : ℝ) :
  (((2*x + 2)^2 = (x + 4)^2 + (x + 2)^2 ∨ (x + 4)^2 = (2*x + 2)^2 + (x + 2)^2) ∧ 
   x > 0 ∧ 2*x + 2 > 0 ∧ x + 4 > 0 ∧ x + 2 > 0) ↔ 
  (x = 4 ∨ x = 1) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_side_lengths_l3101_310173


namespace NUMINAMATH_CALUDE_no_matrix_transformation_l3101_310183

theorem no_matrix_transformation (a b c d : ℝ) : 
  ¬ ∃ (N : Matrix (Fin 2) (Fin 2) ℝ), 
    N • !![a, b; c, d] = !![d, c; b, a] := by
  sorry

end NUMINAMATH_CALUDE_no_matrix_transformation_l3101_310183


namespace NUMINAMATH_CALUDE_octahedron_theorem_l3101_310186

/-- A point in 3D space -/
structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Checks if a point lies on any plane defined by x ± y ± z = n for integer n -/
def liesOnPlane (p : Point3D) : Prop :=
  ∃ n : ℤ, (p.x + p.y + p.z = n) ∨ (p.x + p.y - p.z = n) ∨
           (p.x - p.y + p.z = n) ∨ (p.x - p.y - p.z = n) ∨
           (-p.x + p.y + p.z = n) ∨ (-p.x + p.y - p.z = n) ∨
           (-p.x - p.y + p.z = n) ∨ (-p.x - p.y - p.z = n)

/-- Checks if a point lies strictly inside an octahedron -/
def insideOctahedron (p : Point3D) : Prop :=
  ∃ n : ℤ, (n < p.x + p.y + p.z) ∧ (p.x + p.y + p.z < n + 1) ∧
           (n < p.x + p.y - p.z) ∧ (p.x + p.y - p.z < n + 1) ∧
           (n < p.x - p.y + p.z) ∧ (p.x - p.y + p.z < n + 1) ∧
           (n < -p.x + p.y + p.z) ∧ (-p.x + p.y + p.z < n + 1)

theorem octahedron_theorem (p : Point3D) (h : ¬ liesOnPlane p) :
  ∃ k : ℕ, insideOctahedron ⟨k * p.x, k * p.y, k * p.z⟩ := by
  sorry

end NUMINAMATH_CALUDE_octahedron_theorem_l3101_310186


namespace NUMINAMATH_CALUDE_function_inequality_l3101_310139

-- Define a function f that is differentiable on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the condition that 2f(x) - f'(x) > 0 for all x in ℝ
variable (h : ∀ x : ℝ, 2 * f x - deriv f x > 0)

-- State the theorem
theorem function_inequality : f 1 > f 2 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3101_310139


namespace NUMINAMATH_CALUDE_decimal_representation_of_three_fortieths_l3101_310102

theorem decimal_representation_of_three_fortieths : (3 : ℚ) / 40 = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_of_three_fortieths_l3101_310102


namespace NUMINAMATH_CALUDE_new_house_cost_l3101_310147

def first_house_cost : ℝ := 100000

def value_increase_percentage : ℝ := 0.25

def new_house_down_payment_percentage : ℝ := 0.25

theorem new_house_cost (old_house_value : ℝ) (new_house_cost : ℝ) : 
  old_house_value = first_house_cost * (1 + value_increase_percentage) ∧
  old_house_value = new_house_cost * new_house_down_payment_percentage →
  new_house_cost = 500000 := by
  sorry

end NUMINAMATH_CALUDE_new_house_cost_l3101_310147


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l3101_310146

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 > 4}

def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem intersection_M_complement_N :
  M ∩ (U \ N) = {x : ℝ | x > 3 ∨ x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l3101_310146


namespace NUMINAMATH_CALUDE_loss_per_metre_calculation_l3101_310150

/-- Calculates the loss per metre of cloth given the total metres sold, total selling price, and cost price per metre. -/
def loss_per_metre (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) : ℕ :=
  (total_metres * cost_price_per_metre - total_selling_price) / total_metres

/-- Theorem stating that given 200 metres of cloth sold for Rs. 18000 with a cost price of Rs. 95 per metre, the loss per metre is Rs. 5. -/
theorem loss_per_metre_calculation :
  loss_per_metre 200 18000 95 = 5 := by
  sorry

end NUMINAMATH_CALUDE_loss_per_metre_calculation_l3101_310150


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l3101_310184

theorem quadratic_equation_range :
  {a : ℝ | ∃ x : ℝ, x^2 - 4*x + a = 0} = Set.Iic 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l3101_310184
