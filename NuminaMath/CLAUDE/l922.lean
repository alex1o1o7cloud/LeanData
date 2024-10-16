import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l922_92266

theorem simplify_expression (x y : ℝ) : 
  (15 * x + 45 * y) + (20 * x + 58 * y) - (18 * x + 75 * y) = 17 * x + 28 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l922_92266


namespace NUMINAMATH_CALUDE_inequality_theorem_l922_92208

theorem inequality_theorem (p q : ℝ) :
  q > 0 ∧ p ≥ 0 →
  ((4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔
  (0 ≤ p ∧ p < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l922_92208


namespace NUMINAMATH_CALUDE_matrix_inverse_zero_if_singular_l922_92227

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 8; -2, -4]

theorem matrix_inverse_zero_if_singular :
  Matrix.det A = 0 → A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_zero_if_singular_l922_92227


namespace NUMINAMATH_CALUDE_completing_square_proof_l922_92228

theorem completing_square_proof (x : ℝ) : 
  (x^2 - 8*x + 5 = 0) ↔ ((x - 4)^2 = 11) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_proof_l922_92228


namespace NUMINAMATH_CALUDE_leaves_blown_away_l922_92207

theorem leaves_blown_away (initial_leaves left_leaves : ℕ) 
  (h1 : initial_leaves = 5678)
  (h2 : left_leaves = 1432) :
  initial_leaves - left_leaves = 4246 := by
  sorry

end NUMINAMATH_CALUDE_leaves_blown_away_l922_92207


namespace NUMINAMATH_CALUDE_systematic_sampling_removal_l922_92229

/-- The number of individuals to be removed from a population to make it divisible by a given sample size -/
def individualsToRemove (populationSize sampleSize : ℕ) : ℕ :=
  populationSize - sampleSize * (populationSize / sampleSize)

/-- Theorem stating that 4 individuals should be removed from a population of 3,204 to make it divisible by 80 -/
theorem systematic_sampling_removal :
  individualsToRemove 3204 80 = 4 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_removal_l922_92229


namespace NUMINAMATH_CALUDE_parallel_lines_m_opposite_sides_m_range_l922_92254

-- Define the lines and points
def l1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := (x + 2) * (y - 4) = (x - m) * (y - m)
def point_A (m : ℝ) := (-2, m)
def point_B (m : ℝ) := (m, 4)

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ x y, l1 x y → l2 m x y

-- Define points on opposite sides of a line
def opposite_sides (m : ℝ) : Prop :=
  (2 * (-2) + m - 1) * (2 * m + 4 - 1) < 0

-- Theorem statements
theorem parallel_lines_m (m : ℝ) : parallel m → m = -8 := by sorry

theorem opposite_sides_m_range (m : ℝ) : opposite_sides m → -3/2 < m ∧ m < 5 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_opposite_sides_m_range_l922_92254


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l922_92289

theorem complex_fraction_simplification :
  (1 + 2 * Complex.I) / (1 - 2 * Complex.I) = -(3/5 : ℂ) + (4/5 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l922_92289


namespace NUMINAMATH_CALUDE_hcf_of_three_numbers_l922_92212

theorem hcf_of_three_numbers (a b c : ℕ+) : 
  (a + b + c : ℝ) = 60 →
  Nat.lcm (a : ℕ) (Nat.lcm b c) = 180 →
  (1 / (a : ℝ) + 1 / (b : ℝ) + 1 / (c : ℝ)) = 11 / 120 →
  (a * b * c : ℕ) = 900 →
  Nat.gcd (a : ℕ) (Nat.gcd b c) = 5 := by
sorry


end NUMINAMATH_CALUDE_hcf_of_three_numbers_l922_92212


namespace NUMINAMATH_CALUDE_conjugate_complex_modulus_l922_92226

theorem conjugate_complex_modulus (α β : ℂ) :
  (∃ (x y : ℝ), α = x + y * I ∧ β = x - y * I) →  -- α and β are conjugates
  (α^2 / β).im = 0 →                              -- α²/β is real
  Complex.abs (α - β) = 4 →                       -- |α - β| = 4
  Complex.abs α = 4 * Real.sqrt 3 / 3 :=          -- |α| = 4√3/3
by sorry

end NUMINAMATH_CALUDE_conjugate_complex_modulus_l922_92226


namespace NUMINAMATH_CALUDE_error_clock_correct_fraction_l922_92224

/-- Represents a 12-hour digital clock with display errors -/
structure ErrorClock where
  /-- The clock displays '1' as '9' -/
  one_as_nine : Bool
  /-- The clock displays '2' as '5' -/
  two_as_five : Bool

/-- Calculates the fraction of the day when the clock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : ℚ :=
  sorry

/-- Theorem stating that for a clock with both display errors, 
    the fraction of correct time is 49/144 -/
theorem error_clock_correct_fraction :
  ∀ (clock : ErrorClock), 
  clock.one_as_nine ∧ clock.two_as_five → 
  correct_time_fraction clock = 49 / 144 :=
sorry

end NUMINAMATH_CALUDE_error_clock_correct_fraction_l922_92224


namespace NUMINAMATH_CALUDE_work_completion_time_l922_92200

/-- The number of days it takes to complete a task when two people work together -/
def combined_work_time (john_rate : ℚ) (rose_rate : ℚ) : ℚ :=
  1 / (john_rate + rose_rate)

/-- Theorem: John and Rose complete the work in 8 days when working together -/
theorem work_completion_time :
  let john_rate : ℚ := 1 / 10
  let rose_rate : ℚ := 1 / 40
  combined_work_time john_rate rose_rate = 8 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l922_92200


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_11_with_reverse_divisible_by_63_l922_92247

def reverse_number (n : ℕ) : ℕ :=
  let digits := Nat.digits 10 n
  Nat.ofDigits 10 (List.reverse digits)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_11_with_reverse_divisible_by_63 :
  ∃ (p : ℕ),
    is_four_digit p ∧
    p % 63 = 0 ∧
    (reverse_number p) % 63 = 0 ∧
    p % 11 = 0 ∧
    ∀ (q : ℕ),
      is_four_digit q ∧
      q % 63 = 0 ∧
      (reverse_number q) % 63 = 0 ∧
      q % 11 = 0 →
      q ≤ p ∧
    p = 7623 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_11_with_reverse_divisible_by_63_l922_92247


namespace NUMINAMATH_CALUDE_count_eight_digit_integers_l922_92240

/-- The number of different 8-digit positive integers where the first digit is not 0
    and the last digit is neither 0 nor 1 -/
def eight_digit_integers : ℕ :=
  9 * (10 ^ 6) * 8

theorem count_eight_digit_integers :
  eight_digit_integers = 72000000 := by
  sorry

end NUMINAMATH_CALUDE_count_eight_digit_integers_l922_92240


namespace NUMINAMATH_CALUDE_both_locks_stall_time_l922_92283

/-- The time (in minutes) the first lock stalls the raccoons -/
def first_lock_time : ℕ := 5

/-- The time (in minutes) the second lock stalls the raccoons -/
def second_lock_time : ℕ := 3 * first_lock_time - 3

/-- The time (in minutes) both locks together stall the raccoons -/
def both_locks_time : ℕ := 5 * second_lock_time

/-- Theorem stating that both locks together stall the raccoons for 60 minutes -/
theorem both_locks_stall_time : both_locks_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_both_locks_stall_time_l922_92283


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l922_92271

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y + Real.sqrt 3 = 0

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem intersection_distance_squared :
  ∀ (A B M : ℝ × ℝ),
    curve_C A.1 A.2 →
    curve_C B.1 B.2 →
    line_l A.1 A.2 →
    line_l B.1 B.2 →
    line_l M.1 M.2 →
    y_axis M.1 →
    (Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2) + Real.sqrt ((B.1 - M.1)^2 + (B.2 - M.2)^2))^2 = 16 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l922_92271


namespace NUMINAMATH_CALUDE_integer_congruence_problem_l922_92225

theorem integer_congruence_problem :
  ∀ n : ℤ, 5 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [ZMOD 6] → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_integer_congruence_problem_l922_92225


namespace NUMINAMATH_CALUDE_ramola_rank_from_last_l922_92252

theorem ramola_rank_from_last (total_students : ℕ) (rank_from_start : ℕ) :
  total_students = 26 →
  rank_from_start = 14 →
  total_students - rank_from_start + 1 = 14 :=
by sorry

end NUMINAMATH_CALUDE_ramola_rank_from_last_l922_92252


namespace NUMINAMATH_CALUDE_expand_expression_l922_92209

theorem expand_expression (x : ℝ) : 3 * (x - 6) * (x - 7) = 3 * x^2 - 39 * x + 126 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l922_92209


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l922_92265

theorem algebraic_expression_value (a b : ℝ) (h : 2 * a - b = 5) :
  4 * a - 2 * b + 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l922_92265


namespace NUMINAMATH_CALUDE_diamond_two_four_l922_92279

-- Define the Diamond operation
def Diamond (a b : ℤ) : ℤ := a * b^3 - b + 2

-- Theorem statement
theorem diamond_two_four : Diamond 2 4 = 126 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_four_l922_92279


namespace NUMINAMATH_CALUDE_product_remainder_l922_92230

theorem product_remainder (a b m : ℕ) (ha : a = 1492) (hb : b = 1999) (hm : m = 500) :
  (a * b) % m = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l922_92230


namespace NUMINAMATH_CALUDE_simplest_fraction_sum_l922_92249

theorem simplest_fraction_sum (a b : ℕ+) : 
  (a : ℚ) / b = 0.4375 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / d = 0.4375 → a ≤ c ∧ b ≤ d →
  a + b = 23 := by
sorry

end NUMINAMATH_CALUDE_simplest_fraction_sum_l922_92249


namespace NUMINAMATH_CALUDE_lincoln_county_housing_boom_l922_92237

/-- The number of houses built during a housing boom in Lincoln County. -/
def houses_built (original current : ℕ) : ℕ := current - original

/-- Theorem stating the number of houses built during the housing boom in Lincoln County. -/
theorem lincoln_county_housing_boom :
  houses_built 20817 118558 = 97741 := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_housing_boom_l922_92237


namespace NUMINAMATH_CALUDE_cubic_decimeter_to_meter_seconds_to_minutes_minutes_to_hours_square_centimeter_to_decimeter_milliliters_to_liters_l922_92267

-- Define conversion rates
def cubic_meter_to_decimeter : ℚ := 1000
def minute_to_second : ℚ := 60
def hour_to_minute : ℚ := 60
def square_decimeter_to_centimeter : ℚ := 100
def liter_to_milliliter : ℚ := 1000

-- Theorem statements
theorem cubic_decimeter_to_meter : (35 : ℚ) / cubic_meter_to_decimeter = 7 / 200 := by sorry

theorem seconds_to_minutes : (53 : ℚ) / minute_to_second = 53 / 60 := by sorry

theorem minutes_to_hours : (5 : ℚ) / hour_to_minute = 1 / 12 := by sorry

theorem square_centimeter_to_decimeter : (1 : ℚ) / square_decimeter_to_centimeter = 1 / 100 := by sorry

theorem milliliters_to_liters : (450 : ℚ) / liter_to_milliliter = 9 / 20 := by sorry

end NUMINAMATH_CALUDE_cubic_decimeter_to_meter_seconds_to_minutes_minutes_to_hours_square_centimeter_to_decimeter_milliliters_to_liters_l922_92267


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l922_92235

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}

theorem P_sufficient_not_necessary_for_Q :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l922_92235


namespace NUMINAMATH_CALUDE_inequality_theorem_l922_92214

theorem inequality_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  2 * Real.log m - n > 2 * Real.log n - m := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l922_92214


namespace NUMINAMATH_CALUDE_right_triangle_from_angle_condition_l922_92276

theorem right_triangle_from_angle_condition (A B C : Real) :
  -- Triangle condition
  A + B + C = 180 →
  -- Given angle condition
  A = B ∧ A = (1/2) * C →
  -- Conclusion: C is a right angle
  C = 90 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_from_angle_condition_l922_92276


namespace NUMINAMATH_CALUDE_total_chocolate_bars_l922_92297

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 16

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 25

/-- The total number of chocolate bars in the large box -/
def total_bars : ℕ := num_small_boxes * bars_per_small_box

theorem total_chocolate_bars : total_bars = 400 := by
  sorry

end NUMINAMATH_CALUDE_total_chocolate_bars_l922_92297


namespace NUMINAMATH_CALUDE_circumcircle_point_values_l922_92217

-- Define the points A, B, and C
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 3)
def C : ℝ × ℝ := (3, -1)

-- Define the equation of a circle
def circle_equation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

-- Define the circumcircle of triangle ABC
def circumcircle (x y : ℝ) : Prop :=
  ∃ D E F : ℝ,
    circle_equation x y D E F ∧
    circle_equation A.1 A.2 D E F ∧
    circle_equation B.1 B.2 D E F ∧
    circle_equation C.1 C.2 D E F

-- Theorem statement
theorem circumcircle_point_values :
  ∀ a : ℝ, circumcircle a 2 → a = 2 ∨ a = 6 :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_point_values_l922_92217


namespace NUMINAMATH_CALUDE_path_count_l922_92284

/-- The number of paths between two points -/
def num_paths (start finish : Point) : ℕ := sorry

/-- The set of points in the problem -/
inductive Point
| A
| B
| C
| D

/-- The total number of paths from A to C -/
def total_paths : ℕ := sorry

theorem path_count :
  (num_paths Point.A Point.B = 2) →
  (num_paths Point.B Point.D = 2) →
  (num_paths Point.D Point.C = 2) →
  (num_paths Point.A Point.C = 1 + num_paths Point.A Point.B * num_paths Point.B Point.D * num_paths Point.D Point.C) →
  (total_paths = 9) :=
by sorry

end NUMINAMATH_CALUDE_path_count_l922_92284


namespace NUMINAMATH_CALUDE_parabola_directrix_l922_92251

/-- Given a fixed point A(2,1) and a parabola y^2 = 2px (p > 0) whose focus lies on the perpendicular 
    bisector of OA, prove that the directrix of the parabola has the equation x = -5/4 -/
theorem parabola_directrix (p : ℝ) (h1 : p > 0) : 
  let A : ℝ × ℝ := (2, 1)
  let O : ℝ × ℝ := (0, 0)
  let focus : ℝ × ℝ := (p/2, 0)
  let perp_bisector (x y : ℝ) := 4*x + 2*y - 5 = 0
  let parabola (x y : ℝ) := y^2 = 2*p*x
  let directrix (x : ℝ) := x = -5/4
  (perp_bisector (focus.1) (focus.2)) →
  (∀ x y, parabola x y → (x = -p/2 ↔ directrix x)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l922_92251


namespace NUMINAMATH_CALUDE_unique_solution_system_l922_92259

theorem unique_solution_system : 
  ∃! (x y : ℝ), x + y = 3 ∧ x^4 - y^4 = 8*x - y ∧ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l922_92259


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_three_l922_92233

theorem subset_implies_a_equals_three (A B : Set ℕ) (a : ℕ) 
  (h1 : A = {1, 3})
  (h2 : B = {1, 2, a})
  (h3 : A ⊆ B) : 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_three_l922_92233


namespace NUMINAMATH_CALUDE_no_valid_m_l922_92253

theorem no_valid_m : ¬ ∃ (m : ℕ+), (∃ (a b : ℕ+), (1806 : ℤ) = a * (m.val ^ 2 - 2) ∧ (1806 : ℤ) = b * (m.val ^ 2 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_m_l922_92253


namespace NUMINAMATH_CALUDE_even_function_iff_a_eq_zero_l922_92244

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

theorem even_function_iff_a_eq_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_iff_a_eq_zero_l922_92244


namespace NUMINAMATH_CALUDE_triangle_height_l922_92215

theorem triangle_height (x y : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) : 
  let area := (x^3 * y)^2
  let side := (2 * x * y)^2
  let height := (1/2) * x^4
  area = (1/2) * side * height := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l922_92215


namespace NUMINAMATH_CALUDE_decimal_3_is_binary_11_binary_11_is_decimal_3_l922_92232

/-- Converts a natural number to its binary representation as a list of bits (0s and 1s) --/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 2) ((m % 2) :: acc)
    go n []

/-- Converts a list of bits (0s and 1s) to its decimal representation --/
def fromBinary (bits : List ℕ) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + b) 0

theorem decimal_3_is_binary_11 : toBinary 3 = [1, 1] := by
  sorry

theorem binary_11_is_decimal_3 : fromBinary [1, 1] = 3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_3_is_binary_11_binary_11_is_decimal_3_l922_92232


namespace NUMINAMATH_CALUDE_vending_machine_failure_rate_l922_92234

/-- Calculates the failure rate of a vending machine. -/
theorem vending_machine_failure_rate 
  (total_users : ℕ) 
  (snacks_dropped : ℕ) 
  (extra_snack_rate : ℚ) : 
  total_users = 30 → 
  snacks_dropped = 28 → 
  extra_snack_rate = 1/10 → 
  (total_users : ℚ) - snacks_dropped = 
    total_users * (1 - extra_snack_rate) * (1/6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_vending_machine_failure_rate_l922_92234


namespace NUMINAMATH_CALUDE_min_points_theorem_min_points_is_minimal_l922_92221

/-- Represents a point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the angle APB given three points A, P, and B -/
def angle (A P B : Point) : ℝ := sorry

/-- The minimum number of points satisfying the given condition -/
def min_points : ℕ := 1993

theorem min_points_theorem (A B : Point) :
  ∀ (points : Finset Point),
    points.card ≥ min_points →
    ∃ (Pi Pj : Point), Pi ∈ points ∧ Pj ∈ points ∧ Pi ≠ Pj ∧
      |Real.sin (angle A Pi B) - Real.sin (angle A Pj B)| ≤ 1 / 1992 :=
by sorry

theorem min_points_is_minimal :
  ∀ k : ℕ, k < min_points →
    ∃ (A B : Point) (points : Finset Point),
      points.card = k ∧
      ∀ (Pi Pj : Point), Pi ∈ points → Pj ∈ points → Pi ≠ Pj →
        |Real.sin (angle A Pi B) - Real.sin (angle A Pj B)| > 1 / 1992 :=
by sorry

end NUMINAMATH_CALUDE_min_points_theorem_min_points_is_minimal_l922_92221


namespace NUMINAMATH_CALUDE_small_tile_position_l922_92298

/-- Represents a tile in the square --/
inductive Tile
| Large : Tile  -- 1×3 tile
| Small : Tile  -- 1×1 tile

/-- Represents a position in the 7×7 square --/
structure Position :=
(row : Fin 7)
(col : Fin 7)

/-- Defines if a position is in the center or adjacent to the border --/
def is_center_or_border (p : Position) : Prop :=
  (p.row = 3 ∧ p.col = 3) ∨ 
  p.row = 0 ∨ p.row = 6 ∨ p.col = 0 ∨ p.col = 6

/-- Represents the arrangement of tiles in the square --/
def Arrangement := Position → Tile

/-- The theorem to be proved --/
theorem small_tile_position 
  (arr : Arrangement) 
  (h1 : ∃! p, arr p = Tile.Small) 
  (h2 : ∀ p, arr p = Tile.Large → 
       ∃ p1 p2, p1 ≠ p ∧ p2 ≠ p ∧ p1 ≠ p2 ∧ 
       arr p1 = Tile.Large ∧ arr p2 = Tile.Large) 
  (h3 : ∀ p, arr p = Tile.Large ∨ arr p = Tile.Small) :
  ∃ p, arr p = Tile.Small ∧ is_center_or_border p :=
sorry

end NUMINAMATH_CALUDE_small_tile_position_l922_92298


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l922_92216

theorem gcd_lcm_sum : Nat.gcd 42 70 + Nat.lcm 8 32 = 46 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l922_92216


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l922_92274

theorem absolute_value_inequality_solution_set :
  ∀ x : ℝ, |x - 2| < 1 ↔ x ∈ Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l922_92274


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l922_92260

theorem symmetric_complex_product (z₁ z₂ : ℂ) : 
  (z₁.re = 1 ∧ z₁.im = 1) → 
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) → 
  z₁ * z₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l922_92260


namespace NUMINAMATH_CALUDE_max_area_rectangle_d_l922_92285

/-- Given a rectangle divided into four smaller rectangles A, B, C, and D,
    where the perimeters of A, B, and C are known, 
    prove that the maximum possible area of rectangle D is 16 cm². -/
theorem max_area_rectangle_d (perim_A perim_B perim_C : ℝ) 
  (h_perim_A : perim_A = 10)
  (h_perim_B : perim_B = 12)
  (h_perim_C : perim_C = 14) :
  ∃ (area_D : ℝ), area_D ≤ 16 ∧ 
  ∀ (other_area : ℝ), (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*(a+b) = perim_B + perim_C - perim_A ∧ other_area = a*b) 
  → other_area ≤ area_D := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_d_l922_92285


namespace NUMINAMATH_CALUDE_extended_parallelepiped_volume_2_5_6_l922_92269

/-- The volume of points within or exactly one unit from a rectangular parallelepiped -/
def extended_parallelepiped_volume (length width height : ℝ) : ℝ :=
  (length + 2) * (width + 2) * (height + 2) - length * width * height

/-- The volume of the set of points within or exactly one unit from a 2x5x6 parallelepiped -/
theorem extended_parallelepiped_volume_2_5_6 :
  extended_parallelepiped_volume 2 5 6 = (1008 + 44 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_extended_parallelepiped_volume_2_5_6_l922_92269


namespace NUMINAMATH_CALUDE_initial_birds_on_fence_l922_92281

theorem initial_birds_on_fence (initial_birds additional_birds total_birds : ℕ) :
  initial_birds + additional_birds = total_birds →
  additional_birds = 4 →
  total_birds = 6 →
  initial_birds = 2 := by
sorry

end NUMINAMATH_CALUDE_initial_birds_on_fence_l922_92281


namespace NUMINAMATH_CALUDE_jacket_price_after_discounts_l922_92202

theorem jacket_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 20 →
  discount1 = 0.25 →
  discount2 = 0.40 →
  original_price * (1 - discount1) * (1 - discount2) = 9 := by
sorry

end NUMINAMATH_CALUDE_jacket_price_after_discounts_l922_92202


namespace NUMINAMATH_CALUDE_old_soldiers_participation_l922_92286

/-- Represents the distribution of soldiers in different age groups for a parade. -/
structure ParadeDistribution where
  total_soldiers : ℕ
  young_soldiers : ℕ
  middle_soldiers : ℕ
  old_soldiers : ℕ
  parade_spots : ℕ
  young_soldiers_le : young_soldiers ≤ total_soldiers
  middle_soldiers_le : middle_soldiers ≤ total_soldiers
  old_soldiers_le : old_soldiers ≤ total_soldiers
  total_sum : young_soldiers + middle_soldiers + old_soldiers = total_soldiers

/-- The number of soldiers aged over 23 participating in the parade. -/
def old_soldiers_in_parade (d : ParadeDistribution) : ℕ :=
  min d.old_soldiers (d.parade_spots - (d.parade_spots / 3 * 2))

/-- Theorem stating that for the given distribution, 2 soldiers aged over 23 will participate. -/
theorem old_soldiers_participation (d : ParadeDistribution) 
  (h1 : d.total_soldiers = 45)
  (h2 : d.young_soldiers = 15)
  (h3 : d.middle_soldiers = 20)
  (h4 : d.old_soldiers = 10)
  (h5 : d.parade_spots = 9) :
  old_soldiers_in_parade d = 2 := by
  sorry


end NUMINAMATH_CALUDE_old_soldiers_participation_l922_92286


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l922_92213

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (5, 0)

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := 3 * y = 4 * x

-- Theorem statement
theorem distance_focus_to_asymptote :
  let F := right_focus
  ∃ (d : ℝ), d = 4 ∧
  ∀ (x y : ℝ), asymptote x y →
    (F.1 - x)^2 + (F.2 - y)^2 = d^2 :=
sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l922_92213


namespace NUMINAMATH_CALUDE_factorization_sum_l922_92248

theorem factorization_sum (a b c d e f g h j k : ℤ) :
  (∀ x y : ℝ, 64 * x^6 - 729 * y^6 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2)) →
  a + b + c + d + e + f + g + h + j + k = 30 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l922_92248


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l922_92287

-- Define the hyperbola C
def C (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal distance
def focal_distance (c : ℝ) : Prop := c = 2

-- Define a point on the hyperbola
def point_on_hyperbola (a b : ℝ) : Prop :=
  C a b 2 3

-- Theorem statement
theorem hyperbola_eccentricity (a b c : ℝ) :
  C a b 2 3 → focal_distance c → c / a = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l922_92287


namespace NUMINAMATH_CALUDE_egg_ratio_is_two_to_one_l922_92222

def egg_laying_problem (day1 day2 day3 day4 total : ℕ) : Prop :=
  day1 = 50 ∧
  day2 = 2 * day1 ∧
  day3 = day2 + 20 ∧
  total = 810 ∧
  day4 = total - (day1 + day2 + day3)

theorem egg_ratio_is_two_to_one :
  ∀ day1 day2 day3 day4 total : ℕ,
    egg_laying_problem day1 day2 day3 day4 total →
    day4 * (day1 + day2 + day3) = 2 * (day1 + day2 + day3) * (day1 + day2 + day3) :=
by
  sorry

end NUMINAMATH_CALUDE_egg_ratio_is_two_to_one_l922_92222


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l922_92263

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {-1, 0, 1, 3}
def N : Set Int := {-2, 0, 2, 3}

theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l922_92263


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l922_92296

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given quadratic equation x^2 - 6x + 2m - 1 = 0 with two real roots -/
def givenEquation (m : ℝ) : QuadraticEquation :=
  { a := 1, b := -6, c := 2 * m - 1 }

theorem quadratic_roots_theorem (m : ℝ) :
  let eq := givenEquation m
  let x₁ : ℝ := 1
  let x₂ : ℝ := 5
  (∃ (x₁ x₂ : ℝ), x₁^2 - 6*x₁ + 2*m - 1 = 0 ∧ x₂^2 - 6*x₂ + 2*m - 1 = 0) →
  x₁ = 1 →
  (x₂ = 5 ∧ m = 3) ∧
  (∃ m' : ℝ, (x₁ - 1) * (x₂ - 1) = 6 / (m' - 5) ∧ m' = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_theorem_l922_92296


namespace NUMINAMATH_CALUDE_parallelogram_area_theorem_l922_92295

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : Real) : Real :=
  base * height

/-- The inclination of the parallelogram -/
def inclination : Real := 6

theorem parallelogram_area_theorem (base height : Real) 
  (h_base : base = 20) 
  (h_height : height = 4) :
  parallelogram_area base height = 80 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_theorem_l922_92295


namespace NUMINAMATH_CALUDE_l_shaped_field_area_l922_92261

theorem l_shaped_field_area :
  let field_length : ℕ := 10
  let field_width : ℕ := 7
  let removed_length_diff : ℕ := 3
  let removed_width_diff : ℕ := 2
  let removed_length : ℕ := field_length - removed_length_diff
  let removed_width : ℕ := field_width - removed_width_diff
  let total_area : ℕ := field_length * field_width
  let removed_area : ℕ := removed_length * removed_width
  let l_shaped_area : ℕ := total_area - removed_area
  l_shaped_area = 35 := by sorry

end NUMINAMATH_CALUDE_l_shaped_field_area_l922_92261


namespace NUMINAMATH_CALUDE_two_solutions_characterization_l922_92204

def has_two_solutions (a : ℕ) : Prop :=
  ∃ x y : ℕ, x < 2007 ∧ y < 2007 ∧ x ≠ y ∧
  x^2 + a ≡ 0 [ZMOD 2007] ∧ y^2 + a ≡ 0 [ZMOD 2007] ∧
  ∀ z : ℕ, z < 2007 → z^2 + a ≡ 0 [ZMOD 2007] → (z = x ∨ z = y)

theorem two_solutions_characterization :
  {a : ℕ | a < 2007 ∧ has_two_solutions a} = {446, 1115, 1784} := by sorry

end NUMINAMATH_CALUDE_two_solutions_characterization_l922_92204


namespace NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l922_92206

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem cubic_inequality_negation : 
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l922_92206


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_polynomial_values_l922_92262

/-- A polynomial with integer coefficients -/
def IntPolynomial := List ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ :=
  (p.length - 1).max 0

/-- Evaluate a polynomial at a given integer -/
def eval (p : IntPolynomial) (x : ℤ) : ℤ :=
  p.enum.foldl (fun acc (i, a) => acc + a * x ^ i) 0

theorem infinite_primes_dividing_polynomial_values (p : IntPolynomial)
  (h : degree p ≥ 1) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    (∀ q ∈ S, Prime q ∧ ∃ n : ℕ, (eval p n) % q = 0) :=
  sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_polynomial_values_l922_92262


namespace NUMINAMATH_CALUDE_parabola_opens_downwards_l922_92275

/-- A parabola opens downwards if its quadratic coefficient is negative -/
def opens_downwards (a b c : ℝ) : Prop :=
  a < 0

/-- The theorem states that for a = -3, the parabola y = ax^2 + bx + c opens downwards -/
theorem parabola_opens_downwards :
  let a : ℝ := -3
  opens_downwards a b c := by sorry

end NUMINAMATH_CALUDE_parabola_opens_downwards_l922_92275


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l922_92277

theorem sales_tax_percentage (total_allowed : ℝ) (food_cost : ℝ) (tip_percentage : ℝ) :
  total_allowed = 75 →
  food_cost = 61.48 →
  tip_percentage = 15 →
  ∃ (sales_tax_percentage : ℝ),
    sales_tax_percentage ≤ 6.95 ∧
    food_cost * (1 + sales_tax_percentage / 100 + tip_percentage / 100) ≤ total_allowed :=
by sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l922_92277


namespace NUMINAMATH_CALUDE_parabola_chord_through_focus_l922_92292

/-- Given a parabola y² = 2px with p > 0, if a chord AB passes through the focus F
    such that |AF| = 2 and |BF| = 3, then p = 12/5 -/
theorem parabola_chord_through_focus (p : ℝ) (A B F : ℝ × ℝ) :
  p > 0 →
  (∀ x y, y^2 = 2*p*x) →
  F.1 = p/2 ∧ F.2 = 0 →
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = 4 →
  (B.1 - F.1)^2 + (B.2 - F.2)^2 = 9 →
  p = 12/5 := by
sorry

end NUMINAMATH_CALUDE_parabola_chord_through_focus_l922_92292


namespace NUMINAMATH_CALUDE_square_roots_problem_l922_92270

theorem square_roots_problem (x a : ℝ) (hx : x > 0) :
  ((-a + 2)^2 = x ∧ (2*a - 1)^2 = x) → (a = -1 ∧ x = 9) := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l922_92270


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l922_92258

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), ∀ x, (x + 3) * (x - 3) = 2 * x → a * x^2 + b * x + c = 0 ∧ a = 1 ∧ b = -2 ∧ c = -9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l922_92258


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l922_92239

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a (-2) →
  a 1 + a 4 + a 7 = 50 →
  a 6 + a 9 + a 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l922_92239


namespace NUMINAMATH_CALUDE_expression_evaluation_l922_92242

theorem expression_evaluation :
  let a : ℝ := Real.sqrt 2 - 1
  (1 - 1 / (a + 1)) * ((a^2 + 2*a + 1) / a) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l922_92242


namespace NUMINAMATH_CALUDE_clothing_price_proof_l922_92203

/-- Given the following conditions:
    - Total spent on 7 pieces of clothing is $610
    - One piece costs $49
    - Another piece costs $81
    - The remaining pieces all cost the same
    - The price of the remaining pieces is a multiple of 5
    Prove that each of the remaining pieces costs $96 -/
theorem clothing_price_proof (total_spent : ℕ) (total_pieces : ℕ) (price1 : ℕ) (price2 : ℕ) (price_other : ℕ) :
  total_spent = 610 →
  total_pieces = 7 →
  price1 = 49 →
  price2 = 81 →
  (total_spent - price1 - price2) % (total_pieces - 2) = 0 →
  price_other % 5 = 0 →
  price_other * (total_pieces - 2) + price1 + price2 = total_spent →
  price_other = 96 := by
  sorry

#eval 96 * 5 + 49 + 81  -- Should output 610

end NUMINAMATH_CALUDE_clothing_price_proof_l922_92203


namespace NUMINAMATH_CALUDE_sum_sqrt_inequality_l922_92245

theorem sum_sqrt_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  Real.sqrt (x*y/(z+x*y)) + Real.sqrt (y*z/(x+y*z)) + Real.sqrt (z*x/(y+z*x)) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_inequality_l922_92245


namespace NUMINAMATH_CALUDE_quadratic_function_value_l922_92210

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_value (a b c : ℝ) :
  f a b c 1 = 3 → f a b c 2 = 12 → f a b c 3 = 27 → f a b c 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l922_92210


namespace NUMINAMATH_CALUDE_multiple_optimal_solutions_l922_92293

/-- The feasible region defined by the given linear constraints -/
def FeasibleRegion (x y : ℝ) : Prop :=
  2 * x - y + 2 ≥ 0 ∧ x - 3 * y + 1 ≤ 0 ∧ x + y - 2 ≤ 0

/-- The objective function z -/
def ObjectiveFunction (a x y : ℝ) : ℝ := a * x - y

/-- The theorem stating that a = 1/3 results in multiple optimal solutions -/
theorem multiple_optimal_solutions :
  ∃ (a : ℝ), a > 0 ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    FeasibleRegion x₁ y₁ ∧ FeasibleRegion x₂ y₂ ∧
    ObjectiveFunction a x₁ y₁ = ObjectiveFunction a x₂ y₂ ∧
    (∀ (x y : ℝ), FeasibleRegion x y → ObjectiveFunction a x y ≤ ObjectiveFunction a x₁ y₁)) ∧
  a = 1/3 :=
sorry

end NUMINAMATH_CALUDE_multiple_optimal_solutions_l922_92293


namespace NUMINAMATH_CALUDE_detergent_water_ratio_change_l922_92294

-- Define the original ratio
def original_ratio : Fin 3 → ℚ
  | 0 => 2  -- bleach
  | 1 => 40 -- detergent
  | 2 => 100 -- water

-- Define the altered ratio
def altered_ratio : Fin 3 → ℚ
  | 0 => 6  -- bleach (tripled)
  | 1 => 40 -- detergent
  | 2 => 200 -- water

-- Theorem to prove
theorem detergent_water_ratio_change :
  (altered_ratio 1 / altered_ratio 2) / (original_ratio 1 / original_ratio 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_detergent_water_ratio_change_l922_92294


namespace NUMINAMATH_CALUDE_problem_solution_l922_92250

theorem problem_solution (a b : ℤ) 
  (eq1 : 1010 * a + 1014 * b = 1018)
  (eq2 : 1012 * a + 1016 * b = 1020) : 
  a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l922_92250


namespace NUMINAMATH_CALUDE_parallelogram_split_slope_l922_92288

/-- A parallelogram in a 2D plane --/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- A line in a 2D plane represented by its slope --/
structure Line where
  slope : ℝ

/-- Predicate to check if a line passes through the origin and splits a parallelogram into two congruent parts --/
def splits_congruently (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- The main theorem --/
theorem parallelogram_split_slope :
  let p := Parallelogram.mk (8, 50) (8, 120) (30, 160) (30, 90)
  let l := Line.mk (265 / 38)
  splits_congruently p l := by sorry

end NUMINAMATH_CALUDE_parallelogram_split_slope_l922_92288


namespace NUMINAMATH_CALUDE_sum_of_150_consecutive_integers_l922_92236

def is_sum_of_consecutive_integers (n : ℕ) (k : ℕ) : Prop :=
  ∃ a : ℕ, n = (k * (2 * a + k - 1)) / 2

theorem sum_of_150_consecutive_integers :
  is_sum_of_consecutive_integers 4692583675 150 ∧
  ¬ is_sum_of_consecutive_integers 1627386425 150 ∧
  ¬ is_sum_of_consecutive_integers 2345680925 150 ∧
  ¬ is_sum_of_consecutive_integers 3579113450 150 ∧
  ¬ is_sum_of_consecutive_integers 5815939525 150 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_150_consecutive_integers_l922_92236


namespace NUMINAMATH_CALUDE_total_sales_proof_l922_92282

def robyn_sales : ℕ := 55
def lucy_sales : ℕ := 43

theorem total_sales_proof : robyn_sales + lucy_sales = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_proof_l922_92282


namespace NUMINAMATH_CALUDE_seating_arrangement_l922_92264

/-- Given a seating arrangement where each row seats either 6 or 9 people,
    and 57 people are to be seated with all seats occupied,
    prove that there is exactly 1 row seating 9 people. -/
theorem seating_arrangement (x y : ℕ) : 
  9 * x + 6 * y = 57 → 
  x + y > 0 →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_seating_arrangement_l922_92264


namespace NUMINAMATH_CALUDE_deepak_current_age_l922_92211

/-- Represents the ages of Rahul and Deepak -/
structure Ages where
  rahul : ℕ
  deepak : ℕ

/-- The ratio between Rahul and Deepak's ages -/
def age_ratio (ages : Ages) : ℚ :=
  ages.rahul / ages.deepak

/-- Rahul's age after 6 years -/
def rahul_future_age (ages : Ages) : ℕ :=
  ages.rahul + 6

theorem deepak_current_age (ages : Ages) :
  age_ratio ages = 4/3 →
  rahul_future_age ages = 42 →
  ages.deepak = 27 := by
sorry

end NUMINAMATH_CALUDE_deepak_current_age_l922_92211


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l922_92272

/-- Given sets A, B, and C, prove that if C is a subset of A ∩ B, then the range of a is [-1, 1] -/
theorem subset_implies_a_range (a : ℝ) :
  let A : Set ℝ := {x | (2 - x) / (3 + x) ≥ 0}
  let B : Set ℝ := {x | x^2 - 2*x - 3 < 0}
  let C : Set ℝ := {x | x^2 - (2*a + 1)*x + a*(a + 1) < 0}
  C ⊆ (A ∩ B) → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

#check subset_implies_a_range

end NUMINAMATH_CALUDE_subset_implies_a_range_l922_92272


namespace NUMINAMATH_CALUDE_cubes_not_touching_foil_count_l922_92280

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of cubes not touching tin foil in a prism -/
def cubes_not_touching_foil (outer_width : ℕ) (inner : PrismDimensions) : ℕ :=
  inner.length * inner.width * inner.height

/-- Theorem stating the number of cubes not touching tin foil -/
theorem cubes_not_touching_foil_count 
  (outer_width : ℕ) 
  (inner : PrismDimensions) 
  (h1 : outer_width = 10)
  (h2 : inner.width = 2 * inner.length)
  (h3 : inner.width = 2 * inner.height)
  (h4 : inner.width = outer_width - 4) :
  cubes_not_touching_foil outer_width inner = 54 := by
  sorry

#eval cubes_not_touching_foil 10 ⟨6, 3, 3⟩

end NUMINAMATH_CALUDE_cubes_not_touching_foil_count_l922_92280


namespace NUMINAMATH_CALUDE_field_height_rise_l922_92299

/-- Calculates the rise in height of a field after digging a pit and spreading the removed earth --/
theorem field_height_rise (field_length field_width pit_length pit_width pit_depth : ℝ) 
  (h_field_length : field_length = 20)
  (h_field_width : field_width = 10)
  (h_pit_length : pit_length = 8)
  (h_pit_width : pit_width = 5)
  (h_pit_depth : pit_depth = 2) :
  let total_area := field_length * field_width
  let pit_area := pit_length * pit_width
  let remaining_area := total_area - pit_area
  let pit_volume := pit_length * pit_width * pit_depth
  pit_volume / remaining_area = 0.5 := by sorry

end NUMINAMATH_CALUDE_field_height_rise_l922_92299


namespace NUMINAMATH_CALUDE_inverse_proportion_x_relationship_l922_92257

/-- 
Given three points A(x₁, -2), B(x₂, 1), and C(x₃, 2) on the graph of the inverse proportion function y = -2/x,
prove that x₂ < x₃ < x₁.
-/
theorem inverse_proportion_x_relationship (x₁ x₂ x₃ : ℝ) : 
  (-2 = -2 / x₁) → (1 = -2 / x₂) → (2 = -2 / x₃) → x₂ < x₃ ∧ x₃ < x₁ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_x_relationship_l922_92257


namespace NUMINAMATH_CALUDE_students_on_bleachers_l922_92223

/-- Given a total of 26 students and a ratio of 11:13 for students on the floor to total students,
    prove that the number of students on the bleachers is 4. -/
theorem students_on_bleachers :
  ∀ (floor bleachers : ℕ),
    floor + bleachers = 26 →
    floor / (floor + bleachers : ℚ) = 11 / 13 →
    bleachers = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_on_bleachers_l922_92223


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l922_92256

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l922_92256


namespace NUMINAMATH_CALUDE_division_problem_l922_92273

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 3086 →
  divisor = 85 →
  remainder = 26 →
  dividend = divisor * quotient + remainder →
  quotient = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l922_92273


namespace NUMINAMATH_CALUDE_class_ratio_l922_92231

theorem class_ratio (total : ℕ) (difference : ℕ) (girls boys : ℕ) : 
  total = 32 →
  difference = 6 →
  girls = boys + difference →
  girls + boys = total →
  (girls : ℚ) / (boys : ℚ) = 19 / 13 := by
sorry

end NUMINAMATH_CALUDE_class_ratio_l922_92231


namespace NUMINAMATH_CALUDE_diagonal_planes_increment_l922_92220

/-- The number of diagonal planes in a prism with k edges -/
def f (k : ℕ) : ℕ := k * (k - 3) / 2

/-- Theorem: The number of diagonal planes in a prism with k+1 edges
    is equal to the number of diagonal planes in a prism with k edges plus k-1 -/
theorem diagonal_planes_increment (k : ℕ) :
  f (k + 1) = f k + k - 1 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_planes_increment_l922_92220


namespace NUMINAMATH_CALUDE_second_shipment_weight_l922_92238

/-- The weight of the second shipment of couscous -/
def second_shipment : ℝ := sorry

/-- The weight of the first shipment of couscous -/
def first_shipment : ℝ := 7

/-- The weight of the third shipment of couscous -/
def third_shipment : ℝ := 45

/-- The number of dishes made -/
def num_dishes : ℕ := 13

/-- The amount of couscous used per dish -/
def couscous_per_dish : ℝ := 5

theorem second_shipment_weight :
  second_shipment = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_second_shipment_weight_l922_92238


namespace NUMINAMATH_CALUDE_base_b_proof_l922_92255

theorem base_b_proof (b : ℕ) (h : b > 1) :
  (7 * b^2 + 8 * b + 4 = (2 * b + 8)^2) → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_b_proof_l922_92255


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l922_92246

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![2, 2]

-- State the theorem
theorem vector_sum_magnitude :
  ‖(a + b)‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l922_92246


namespace NUMINAMATH_CALUDE_remaining_concert_time_l922_92243

def concert_duration : ℕ := 165 -- 2 hours and 45 minutes in minutes
def intermission1 : ℕ := 12
def intermission2 : ℕ := 10
def intermission3 : ℕ := 8
def regular_song_duration : ℕ := 4
def ballad_duration : ℕ := 7
def medley_duration : ℕ := 15
def num_regular_songs : ℕ := 15
def num_ballads : ℕ := 6

theorem remaining_concert_time : 
  concert_duration - 
  (intermission1 + intermission2 + intermission3 + 
   num_regular_songs * regular_song_duration + 
   num_ballads * ballad_duration + 
   medley_duration) = 18 := by sorry

end NUMINAMATH_CALUDE_remaining_concert_time_l922_92243


namespace NUMINAMATH_CALUDE_roots_product_zero_l922_92201

theorem roots_product_zero (a b c d : ℝ) : 
  (a^2 + 57*a + 1 = 0) →
  (b^2 + 57*b + 1 = 0) →
  (c^2 - 57*c + 1 = 0) →
  (d^2 - 57*d + 1 = 0) →
  (a + c) * (b + c) * (a - d) * (b - d) = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_product_zero_l922_92201


namespace NUMINAMATH_CALUDE_a_investment_l922_92278

/-- A partnership business with three partners A, B, and C. -/
structure Partnership where
  investA : ℕ
  investB : ℕ
  investC : ℕ
  totalProfit : ℕ
  cShareProfit : ℕ

/-- The partnership satisfies the given conditions -/
def validPartnership (p : Partnership) : Prop :=
  p.investB = 8000 ∧
  p.investC = 9000 ∧
  p.totalProfit = 88000 ∧
  p.cShareProfit = 36000

/-- The theorem stating A's investment amount -/
theorem a_investment (p : Partnership) (h : validPartnership p) : 
  p.investA = 5000 := by
  sorry

end NUMINAMATH_CALUDE_a_investment_l922_92278


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l922_92218

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2013 = i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l922_92218


namespace NUMINAMATH_CALUDE_octal_to_decimal_l922_92205

theorem octal_to_decimal (octal_num : ℕ) : octal_num = 362 → 
  (3 * 8^2 + 6 * 8^1 + 2 * 8^0) = 242 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_l922_92205


namespace NUMINAMATH_CALUDE_commute_speed_ratio_l922_92290

/-- Proves that the ratio of speeds for a commuter is 2:1 given specific conditions -/
theorem commute_speed_ratio 
  (distance : ℝ) 
  (total_time : ℝ) 
  (return_speed : ℝ) 
  (h1 : distance = 28) 
  (h2 : total_time = 6) 
  (h3 : return_speed = 14) : 
  return_speed / ((2 * distance) / total_time - return_speed) = 2 := by
  sorry

#check commute_speed_ratio

end NUMINAMATH_CALUDE_commute_speed_ratio_l922_92290


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l922_92268

/-- The function f(x) = a^(x-2) + 1 has a fixed point at (2, 2) for all a > 0 and a ≠ 1 -/
theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∀ x : ℝ, (x = 2 ∧ a^(x-2) + 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l922_92268


namespace NUMINAMATH_CALUDE_fraction_simplification_l922_92219

theorem fraction_simplification (x : ℝ) : (2*x - 3)/4 + (5 - 4*x)/3 = (-10*x + 11)/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l922_92219


namespace NUMINAMATH_CALUDE_cubic_intersection_line_l922_92291

theorem cubic_intersection_line (a b c M : ℝ) : 
  a < b ∧ b < c ∧ 
  2 * (b - a) = c - b ∧
  a^3 - 84*a = M ∧
  b^3 - 84*b = M ∧
  c^3 - 84*c = M →
  M = 160 := by
sorry

end NUMINAMATH_CALUDE_cubic_intersection_line_l922_92291


namespace NUMINAMATH_CALUDE_hash_seven_two_l922_92241

-- Define the # operation
def hash (a b : ℤ) : ℤ := 4 * a - 2 * b

-- State the theorem
theorem hash_seven_two : hash 7 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_hash_seven_two_l922_92241
