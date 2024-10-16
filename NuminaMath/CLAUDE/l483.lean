import Mathlib

namespace NUMINAMATH_CALUDE_abs_and_reciprocal_l483_48303

theorem abs_and_reciprocal :
  (abs (-9 : ℝ) = 9) ∧ (((-3 : ℝ)⁻¹) = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_abs_and_reciprocal_l483_48303


namespace NUMINAMATH_CALUDE_lottery_expected_correct_guesses_l483_48371

/-- Represents the number of matches in the lottery -/
def num_matches : ℕ := 12

/-- Represents the number of possible outcomes for each match -/
def num_outcomes : ℕ := 3

/-- Probability of guessing correctly for a single match -/
def p_correct : ℚ := 1 / num_outcomes

/-- Probability of guessing incorrectly for a single match -/
def p_incorrect : ℚ := 1 - p_correct

/-- Expected number of correct guesses in the lottery -/
def expected_correct_guesses : ℚ := num_matches * p_correct

theorem lottery_expected_correct_guesses :
  expected_correct_guesses = 4 := by sorry

end NUMINAMATH_CALUDE_lottery_expected_correct_guesses_l483_48371


namespace NUMINAMATH_CALUDE_smallest_positive_root_floor_l483_48307

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - Real.cos x + 2 * Real.tan x

theorem smallest_positive_root_floor :
  ∃ s : ℝ, s > 0 ∧ g s = 0 ∧ (∀ t, t > 0 ∧ g t = 0 → s ≤ t) ∧ 4 ≤ s ∧ s < 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_root_floor_l483_48307


namespace NUMINAMATH_CALUDE_problem_solution_l483_48311

theorem problem_solution (x y : ℝ) : x / y = 12 / 3 → y = 27 → x = 108 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l483_48311


namespace NUMINAMATH_CALUDE_jinx_hak_not_flog_l483_48313

-- Define the sets
variable (U : Type) -- Universe set
variable (Flog Grep Hak Jinx : Set U)

-- Define the given conditions
variable (h1 : Flog ⊆ Grep)
variable (h2 : Hak ⊆ Grep)
variable (h3 : Hak ⊆ Jinx)
variable (h4 : Flog ∩ Jinx = ∅)

-- Theorem to prove
theorem jinx_hak_not_flog : 
  Jinx ⊆ Hak ∧ ∃ x, x ∈ Jinx ∧ x ∉ Flog :=
sorry

end NUMINAMATH_CALUDE_jinx_hak_not_flog_l483_48313


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l483_48385

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (45 * x + 13) % 17 = 5 % 17 ∧
  ∀ (y : ℕ), y > 0 ∧ (45 * y + 13) % 17 = 5 % 17 → x ≤ y :=
by
  use 11
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l483_48385


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l483_48337

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l483_48337


namespace NUMINAMATH_CALUDE_wx_length_is_25_l483_48309

/-- A quadrilateral with two right angles and specific side lengths -/
structure RightQuadrilateral where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  right_angle_X : (X.1 - W.1) * (Y.1 - X.1) + (X.2 - W.2) * (Y.2 - X.2) = 0
  right_angle_Y : (Y.1 - X.1) * (Z.1 - Y.1) + (Y.2 - X.2) * (Z.2 - Y.2) = 0
  wz_length : Real.sqrt ((W.1 - Z.1)^2 + (W.2 - Z.2)^2) = 7
  xy_length : Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 14
  yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 24

/-- The length of WX in the given quadrilateral is 25 -/
theorem wx_length_is_25 (q : RightQuadrilateral) :
  Real.sqrt ((q.W.1 - q.X.1)^2 + (q.W.2 - q.X.2)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_wx_length_is_25_l483_48309


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l483_48367

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x - 4| > 6 ↔ x < 0 ∨ x > 12 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l483_48367


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_three_l483_48366

theorem sqrt_sum_eq_three (a : ℝ) (h : a + 1/a = 7) : 
  Real.sqrt a + 1 / Real.sqrt a = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_three_l483_48366


namespace NUMINAMATH_CALUDE_complex_modulus_one_minus_i_l483_48383

theorem complex_modulus_one_minus_i :
  let z : ℂ := 1 - I
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_one_minus_i_l483_48383


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l483_48377

theorem divisibility_equivalence (a m x n : ℕ) :
  m ∣ n ↔ (x^m - a^m) ∣ (x^n - a^n) := by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l483_48377


namespace NUMINAMATH_CALUDE_science_club_enrollment_l483_48347

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_math : math = 95)
  (h_physics : physics = 70)
  (h_both : both = 25) :
  total - (math + physics - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l483_48347


namespace NUMINAMATH_CALUDE_rectangle_area_l483_48394

theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l = 3 * w ∧ x^2 = l^2 + w^2 ∧ l * w = (3 * x^2) / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l483_48394


namespace NUMINAMATH_CALUDE_extended_pattern_black_tiles_l483_48333

/-- Represents a square pattern of tiles -/
structure SquarePattern :=
  (size : Nat)
  (blackTiles : Nat)

/-- Extends a square pattern by adding a border of black tiles -/
def extendPattern (pattern : SquarePattern) : SquarePattern :=
  { size := pattern.size + 2,
    blackTiles := pattern.blackTiles + (pattern.size + 2) * 4 - 4 }

theorem extended_pattern_black_tiles :
  let originalPattern : SquarePattern := { size := 5, blackTiles := 10 }
  let extendedPattern := extendPattern originalPattern
  extendedPattern.blackTiles = 34 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_black_tiles_l483_48333


namespace NUMINAMATH_CALUDE_triangle_medians_inequalities_l483_48346

-- Define a structure for a triangle with medians and circumradius
structure Triangle where
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  R : ℝ
  h_positive : m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧ R > 0

-- Theorem statement
theorem triangle_medians_inequalities (t : Triangle) : 
  t.m_a^2 + t.m_b^2 + t.m_c^2 ≤ (27 * t.R^2) / 4 ∧ 
  t.m_a + t.m_b + t.m_c ≤ (9 * t.R) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_medians_inequalities_l483_48346


namespace NUMINAMATH_CALUDE_two_fifths_of_number_l483_48312

theorem two_fifths_of_number (x : ℚ) : (2 / 9 : ℚ) * x = 10 → (2 / 5 : ℚ) * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_of_number_l483_48312


namespace NUMINAMATH_CALUDE_tinas_career_difference_l483_48301

/-- Represents the career of a boxer --/
structure BoxerCareer where
  initial_wins : ℕ
  additional_wins_before_first_loss : ℕ
  losses : ℕ

/-- Calculates the total wins for a boxer's career --/
def total_wins (career : BoxerCareer) : ℕ :=
  career.initial_wins + career.additional_wins_before_first_loss + 
  (career.initial_wins + career.additional_wins_before_first_loss)

/-- Theorem stating the difference between wins and losses for Tina's career --/
theorem tinas_career_difference : 
  ∀ (career : BoxerCareer), 
  career.initial_wins = 10 → 
  career.additional_wins_before_first_loss = 5 → 
  career.losses = 2 → 
  total_wins career - career.losses = 43 :=
by sorry

end NUMINAMATH_CALUDE_tinas_career_difference_l483_48301


namespace NUMINAMATH_CALUDE_polynomial_root_product_l483_48324

theorem polynomial_root_product (a b c d : ℝ) : 
  let Q : ℝ → ℝ := λ x => x^4 + a*x^3 + b*x^2 + c*x + d
  (Q (Real.cos (2*π/9)) = 0) ∧ 
  (Q (Real.cos (4*π/9)) = 0) ∧ 
  (Q (Real.cos (6*π/9)) = 0) ∧ 
  (Q (Real.cos (8*π/9)) = 0) →
  a * b * c * d = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l483_48324


namespace NUMINAMATH_CALUDE_repeating_decimal_56_eq_fraction_l483_48357

/-- The decimal representation of a number with infinitely repeating digits 56 after the decimal point -/
def repeating_decimal_56 : ℚ :=
  56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_56_eq_fraction : repeating_decimal_56 = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_56_eq_fraction_l483_48357


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l483_48317

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 5) * (Real.sqrt 8 / Real.sqrt 9) * (Real.sqrt 3 / Real.sqrt 7) = 
  (4 * Real.sqrt 105) / 105 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l483_48317


namespace NUMINAMATH_CALUDE_projectiles_meeting_time_l483_48359

/-- Theorem: Time for two projectiles to meet --/
theorem projectiles_meeting_time
  (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : distance = 2520)
  (h2 : speed1 = 432)
  (h3 : speed2 = 576) :
  (distance / (speed1 + speed2)) * 60 = 150 :=
by sorry

end NUMINAMATH_CALUDE_projectiles_meeting_time_l483_48359


namespace NUMINAMATH_CALUDE_prism_volume_l483_48315

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h1 : side_area = 18)
  (h2 : front_area = 12)
  (h3 : bottom_area = 8) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l483_48315


namespace NUMINAMATH_CALUDE_possible_m_values_l483_48326

def A : Set ℝ := {x | x^2 - 9*x - 10 = 0}

def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem possible_m_values : 
  ∀ m : ℝ, (A ∪ B m = A) ↔ m ∈ ({0, 1, -(1/10)} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_possible_m_values_l483_48326


namespace NUMINAMATH_CALUDE_no_integer_solutions_to_3x2_plus_7y2_eq_z4_l483_48348

theorem no_integer_solutions_to_3x2_plus_7y2_eq_z4 :
  ¬ ∃ (x y z : ℤ), 3 * x^2 + 7 * y^2 = z^4 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_to_3x2_plus_7y2_eq_z4_l483_48348


namespace NUMINAMATH_CALUDE_pie_order_cost_l483_48382

/-- Represents the cost of fruit for Michael's pie order -/
def total_cost (peach_pies apple_pies blueberry_pies : ℕ) 
  (fruit_per_pie : ℕ) 
  (peach_price apple_price blueberry_price : ℚ) : ℚ :=
  (peach_pies * fruit_per_pie : ℚ) * peach_price +
  (apple_pies * fruit_per_pie : ℚ) * apple_price +
  (blueberry_pies * fruit_per_pie : ℚ) * blueberry_price

/-- Theorem stating that the total cost of fruit for Michael's pie order is $51.00 -/
theorem pie_order_cost : 
  total_cost 5 4 3 3 2 1 1 = 51 := by
  sorry

end NUMINAMATH_CALUDE_pie_order_cost_l483_48382


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l483_48370

-- Define the ☆ operation
def star (m n : ℝ) : ℝ := m^2 - m*n + n

-- Theorem statements
theorem problem_1 : star 3 4 = 1 := by sorry

theorem problem_2 : star (-1) (star 2 (-3)) = 15 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l483_48370


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l483_48399

theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∃ x : ℕ+, (3 * x^2 + a * x + 26) / (x + 1) ≤ 2) →
  a ≤ -15 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l483_48399


namespace NUMINAMATH_CALUDE_log_10_7_in_terms_of_r_s_l483_48310

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_10_7_in_terms_of_r_s (r s : ℝ) 
  (h1 : log 4 2 = r) 
  (h2 : log 2 7 = s) : 
  log 10 7 = s / (1 + s) := by
  sorry

end NUMINAMATH_CALUDE_log_10_7_in_terms_of_r_s_l483_48310


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_factorial_series_l483_48308

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- Function to get the last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- The series in question -/
def series : List ℕ := [1, 2, 5, 13, 34]

/-- Theorem stating that the sum of the last two digits of the factorial series is 23 -/
theorem sum_of_last_two_digits_of_factorial_series : 
  (series.map (λ n => lastTwoDigits (factorial n))).sum = 23 := by sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_factorial_series_l483_48308


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l483_48319

/-- A geometric sequence with common ratio 4 and sum of first three terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 4 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The general term formula for the geometric sequence -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ n : ℕ, a n = 4^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l483_48319


namespace NUMINAMATH_CALUDE_car_mileage_l483_48392

/-- Given a car that travels 200 kilometers using 5 gallons of gasoline, its mileage is 40 kilometers per gallon. -/
theorem car_mileage (distance : ℝ) (gasoline : ℝ) (h1 : distance = 200) (h2 : gasoline = 5) :
  distance / gasoline = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_mileage_l483_48392


namespace NUMINAMATH_CALUDE_tony_bread_slices_left_l483_48320

/-- The number of slices of bread Tony uses in a week -/
def bread_used (weekday_slices : ℕ) (saturday_slices : ℕ) (sunday_slices : ℕ) : ℕ :=
  5 * weekday_slices + saturday_slices + sunday_slices

/-- The number of slices left from a loaf -/
def slices_left (total_slices : ℕ) (used_slices : ℕ) : ℕ :=
  total_slices - used_slices

/-- Theorem stating the number of slices left from Tony's bread usage -/
theorem tony_bread_slices_left :
  let weekday_slices := 2
  let saturday_slices := 5
  let sunday_slices := 1
  let total_slices := 22
  let used_slices := bread_used weekday_slices saturday_slices sunday_slices
  slices_left total_slices used_slices = 6 := by
  sorry

end NUMINAMATH_CALUDE_tony_bread_slices_left_l483_48320


namespace NUMINAMATH_CALUDE_book_price_increase_percentage_l483_48376

theorem book_price_increase_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 300)
  (h2 : new_price = 480) :
  (new_price - original_price) / original_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_price_increase_percentage_l483_48376


namespace NUMINAMATH_CALUDE_f_max_min_values_l483_48340

noncomputable def f (x : ℝ) : ℝ := 5 * (Real.cos x)^2 - 6 * Real.sin (2 * x) + 20 * Real.sin x - 30 * Real.cos x + 7

theorem f_max_min_values :
  (∀ x, f x ≤ 16 + 10 * Real.sqrt 13) ∧
  (∀ x, f x ≥ 16 - 10 * Real.sqrt 13) ∧
  (∃ x, f x = 16 + 10 * Real.sqrt 13) ∧
  (∃ x, f x = 16 - 10 * Real.sqrt 13) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_values_l483_48340


namespace NUMINAMATH_CALUDE_smallest_number_of_blocks_l483_48318

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  height : ℕ
  length : ℕ

/-- Represents the dimensions of the wall -/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to build the wall -/
def blocksNeeded (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  let oddRowBlocks := wall.length / 2
  let evenRowBlocks := oddRowBlocks + 1
  let numRows := wall.height / block.height
  let oddRows := numRows / 2
  let evenRows := numRows - oddRows
  oddRows * oddRowBlocks + evenRows * evenRowBlocks

/-- The theorem stating the smallest number of blocks needed -/
theorem smallest_number_of_blocks 
  (wall : WallDimensions)
  (block : BlockDimensions)
  (h1 : wall.length = 120)
  (h2 : wall.height = 8)
  (h3 : block.height = 1)
  (h4 : block.length = 2 ∨ block.length = 1)
  (h5 : wall.length % 2 = 0) -- Ensures wall is even on the ends
  : blocksNeeded wall block = 484 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_blocks_l483_48318


namespace NUMINAMATH_CALUDE_line_translation_upwards_l483_48339

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically --/
def translateLine (l : Line) (c : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + c }

/-- The equation of a line in slope-intercept form --/
def lineEquation (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem line_translation_upwards 
  (original : Line) 
  (c : ℝ) 
  (h : c > 0) : 
  ∀ x y : ℝ, lineEquation original x y ↔ lineEquation (translateLine original c) x (y + c) :=
by sorry

end NUMINAMATH_CALUDE_line_translation_upwards_l483_48339


namespace NUMINAMATH_CALUDE_place_value_sum_place_value_sum_holds_l483_48304

theorem place_value_sum : Real → Prop :=
  fun x => 
    let ten_thousands : Real := 4
    let thousands : Real := 3
    let hundreds : Real := 7
    let tens : Real := 5
    let ones : Real := 2
    let tenths : Real := 8
    let hundredths : Real := 4
    x = ten_thousands * 10000 + thousands * 1000 + hundreds * 100 + 
        tens * 10 + ones + tenths / 10 + hundredths / 100 ∧ 
    x = 43752.84

theorem place_value_sum_holds : ∃ x, place_value_sum x := by
  sorry

end NUMINAMATH_CALUDE_place_value_sum_place_value_sum_holds_l483_48304


namespace NUMINAMATH_CALUDE_quadratic_max_value_l483_48314

theorem quadratic_max_value (m : ℝ) (h_m : m ≠ 0) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, m * x^2 - 2 * m * x + 2 ≤ 4) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, m * x^2 - 2 * m * x + 2 = 4) →
  m = 2/3 ∨ m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l483_48314


namespace NUMINAMATH_CALUDE_license_plate_count_l483_48345

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of consonants in the alphabet (including Y) -/
def num_consonants : ℕ := 21

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_consonants * num_consonants * num_vowels * num_vowels * num_digits

theorem license_plate_count :
  total_license_plates = 110250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l483_48345


namespace NUMINAMATH_CALUDE_negation_of_positive_quadratic_l483_48387

theorem negation_of_positive_quadratic (x : ℝ) :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_positive_quadratic_l483_48387


namespace NUMINAMATH_CALUDE_octagon_area_l483_48355

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
theorem octagon_area (r : ℝ) (h : r = 3) : 
  let octagon_area := 8 * (1/2 * r^2 * Real.sin (π/4))
  octagon_area = 18 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l483_48355


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l483_48306

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := fun x ↦ (x + 2) * (x - 3)
  {x : ℝ | f x < 0} = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l483_48306


namespace NUMINAMATH_CALUDE_expression_simplification_l483_48334

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (1 + 1 / (x - 2)) / ((x^2 - 2*x + 1) / (x - 2)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l483_48334


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l483_48349

theorem angle_sum_theorem (c d : Real) (h1 : 0 < c ∧ c < π/2) (h2 : 0 < d ∧ d < π/2)
  (eq1 : 4 * (Real.cos c)^2 + 3 * (Real.sin d)^2 = 1)
  (eq2 : 4 * Real.sin (2*c) - 3 * Real.cos (2*d) = 0) :
  2*c + 3*d = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l483_48349


namespace NUMINAMATH_CALUDE_fifth_element_row_20_l483_48390

-- Define Pascal's triangle function
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem statement
theorem fifth_element_row_20 : pascal 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_l483_48390


namespace NUMINAMATH_CALUDE_triangle_area_l483_48386

theorem triangle_area (A B C : Real) (a b c : Real) :
  (b = c * (2 * Real.sin A + Real.cos A)) →
  (a = Real.sqrt 2) →
  (B = 3 * Real.pi / 4) →
  (∃ (S : Real), S = (1 / 2) * a * c * Real.sin B ∧ S = 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l483_48386


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l483_48302

/-- Given two positive integers with specific HCF and LCM properties, prove the other factor of their LCM -/
theorem lcm_factor_proof (A B : ℕ) (hA : A = 460) (hHCF : Nat.gcd A B = 20) 
  (hLCM_factor : ∃ (k : ℕ), Nat.lcm A B = 20 * 21 * k) : 
  ∃ (k : ℕ), Nat.lcm A B = 20 * 21 * 23 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l483_48302


namespace NUMINAMATH_CALUDE_prob_at_least_one_event_l483_48352

theorem prob_at_least_one_event (P₁ P₂ : ℝ) 
  (h₁ : 0 ≤ P₁ ∧ P₁ ≤ 1) 
  (h₂ : 0 ≤ P₂ ∧ P₂ ≤ 1) :
  P₁ + P₂ - P₁ * P₂ = 1 - (1 - P₁) * (1 - P₂) :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_event_l483_48352


namespace NUMINAMATH_CALUDE_tony_winnings_l483_48335

/-- Calculates the total winnings for lottery tickets with identical numbers -/
def totalWinnings (numTickets : ℕ) (winningNumbersPerTicket : ℕ) (valuePerWinningNumber : ℕ) : ℕ :=
  numTickets * winningNumbersPerTicket * valuePerWinningNumber

/-- Theorem: Tony's total winnings are $300 -/
theorem tony_winnings :
  totalWinnings 3 5 20 = 300 := by
  sorry

end NUMINAMATH_CALUDE_tony_winnings_l483_48335


namespace NUMINAMATH_CALUDE_alyssas_soccer_games_l483_48397

theorem alyssas_soccer_games (games_this_year games_next_year total_games : ℕ) 
  (h1 : games_this_year = 11)
  (h2 : games_next_year = 15)
  (h3 : total_games = 39) :
  total_games - games_this_year - games_next_year = 13 := by
  sorry

end NUMINAMATH_CALUDE_alyssas_soccer_games_l483_48397


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l483_48343

theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 60 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  cost_per_meter * perimeter = total_cost →
  length = 80 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l483_48343


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l483_48350

/-- The quadratic inequality -/
def quadratic_inequality (k x : ℝ) : Prop := k * x^2 - 2 * x + 6 * k < 0

/-- The solution set for the first case -/
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2

/-- The solution set for the second case -/
def solution_set_2 : Set ℝ := Set.univ

theorem quadratic_inequality_theorem :
  (∀ k : ℝ, k ≠ 0 →
    (∀ x : ℝ, quadratic_inequality k x ↔ solution_set_1 x) →
    k = -2/5) ∧
  (∀ k : ℝ, k ≠ 0 →
    (∀ x : ℝ, quadratic_inequality k x) →
    k < -Real.sqrt 6 / 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l483_48350


namespace NUMINAMATH_CALUDE_a_plus_2b_equals_one_l483_48327

theorem a_plus_2b_equals_one (a b : ℝ) 
  (ha : a^3 - 21*a^2 + 140*a - 120 = 0)
  (hb : 4*b^3 - 12*b^2 - 32*b + 448 = 0) :
  a + 2*b = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_2b_equals_one_l483_48327


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l483_48344

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def B : Set ℝ := {x | x^2 - x - 2 < 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l483_48344


namespace NUMINAMATH_CALUDE_complex_equation_solution_l483_48389

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution (a : ℂ) :
  a / (1 - i) = (1 + i) / i → a = -2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l483_48389


namespace NUMINAMATH_CALUDE_nancy_carrot_nv_l483_48322

/-- Calculates the total nutritional value of carrots based on given conditions -/
def total_nutritional_value (initial_carrots : ℕ) (kept_carrots : ℕ) (new_seeds : ℕ) 
  (growth_factor : ℕ) (base_nv : ℝ) (nv_per_cm : ℝ) (growth_cm : ℝ) : ℝ :=
  let new_carrots := new_seeds * growth_factor
  let total_carrots := initial_carrots - kept_carrots + new_carrots
  let good_carrots := total_carrots - (total_carrots / 3)
  let new_carrot_nv := new_carrots * (base_nv + nv_per_cm * growth_cm)
  let kept_carrot_nv := kept_carrots * base_nv
  new_carrot_nv + kept_carrot_nv

/-- Theorem stating that the total nutritional value of Nancy's carrots is 92 -/
theorem nancy_carrot_nv : 
  total_nutritional_value 12 2 5 3 1 0.5 12 = 92 := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrot_nv_l483_48322


namespace NUMINAMATH_CALUDE_jenny_recycling_l483_48358

theorem jenny_recycling (total_weight : ℕ) (can_weight : ℕ) (num_cans : ℕ)
  (bottle_price : ℕ) (can_price : ℕ) (total_earnings : ℕ) :
  total_weight = 100 →
  can_weight = 2 →
  num_cans = 20 →
  bottle_price = 10 →
  can_price = 3 →
  total_earnings = 160 →
  ∃ (bottle_weight : ℕ), 
    bottle_weight = 6 ∧
    bottle_weight * ((total_weight - (can_weight * num_cans)) / bottle_weight) = 
      total_weight - (can_weight * num_cans) ∧
    bottle_price * ((total_weight - (can_weight * num_cans)) / bottle_weight) + 
      can_price * num_cans = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_jenny_recycling_l483_48358


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l483_48354

-- Problem 1
theorem problem_1 : -1^2 - |(-2)| + (1/3 - 3/4) * 12 = -8 := by sorry

-- Problem 2
theorem problem_2 :
  ∃ (x y : ℚ), (x / 2 - (y + 1) / 3 = 1) ∧ (3 * x + 2 * y = 10) ∧ (x = 3) ∧ (y = 1/2) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l483_48354


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l483_48381

/-- The set of possible slopes for a line with y-intercept (0, -3) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (2/110) ∨ m ≥ Real.sqrt (2/110)}

/-- Theorem stating the possible slopes of the line -/
theorem line_ellipse_intersection_slopes :
  ∀ (m : ℝ), (∃ (x y : ℝ), 4*x^2 + 25*y^2 = 100 ∧ y = m*x - 3) ↔ m ∈ possible_slopes := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l483_48381


namespace NUMINAMATH_CALUDE_distance_AB_is_250_l483_48361

/-- The distance between two points A and B, where two people walk towards each other and meet under specific conditions. -/
def distance_AB : ℝ :=
  let first_meeting_distance := 100 -- meters from B
  let second_meeting_distance := 50 -- meters from A
  let total_distance := first_meeting_distance + second_meeting_distance + 100
  total_distance

/-- Theorem stating that the distance between points A and B is 250 meters. -/
theorem distance_AB_is_250 : distance_AB = 250 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_250_l483_48361


namespace NUMINAMATH_CALUDE_power_series_identity_l483_48321

/-- Given that (1 - hx)⁻¹ (1 - kx)⁻¹ = ∑(i≥0) aᵢ xⁱ, 
    prove that (1 + hkx)(1 - hkx)⁻¹ (1 - h²x)⁻¹ (1 - k²x)⁻¹ = ∑(i≥0) aᵢ² xⁱ -/
theorem power_series_identity 
  (h k : ℝ) (x : ℝ) (a : ℕ → ℝ) :
  (∀ x, (1 - h*x)⁻¹ * (1 - k*x)⁻¹ = ∑' i, a i * x^i) →
  (1 + h*k*x) * (1 - h*k*x)⁻¹ * (1 - h^2*x)⁻¹ * (1 - k^2*x)⁻¹ = ∑' i, (a i)^2 * x^i :=
by
  sorry

end NUMINAMATH_CALUDE_power_series_identity_l483_48321


namespace NUMINAMATH_CALUDE_max_water_bottles_proof_l483_48323

/-- Given a total number of water bottles and athletes, with each athlete receiving at least one water bottle,
    calculate the maximum number of water bottles one athlete could have received. -/
def max_water_bottles (total_bottles : ℕ) (total_athletes : ℕ) : ℕ :=
  total_bottles - (total_athletes - 1)

/-- Prove that given 40 water bottles distributed among 25 athletes, with each athlete receiving at least one water bottle,
    the maximum number of water bottles one athlete could have received is 16. -/
theorem max_water_bottles_proof :
  max_water_bottles 40 25 = 16 := by
  sorry

#eval max_water_bottles 40 25

end NUMINAMATH_CALUDE_max_water_bottles_proof_l483_48323


namespace NUMINAMATH_CALUDE_toms_reading_speed_l483_48331

/-- Tom's reading speed problem -/
theorem toms_reading_speed (normal_speed : ℕ) : 
  (2 * (3 * normal_speed) = 72) → normal_speed = 12 := by
  sorry

#check toms_reading_speed

end NUMINAMATH_CALUDE_toms_reading_speed_l483_48331


namespace NUMINAMATH_CALUDE_no_equal_division_of_scalene_triangle_l483_48365

/-- A triangle represented by its three vertices in ℝ² -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- A triangle is scalene if all its sides have different lengths -/
def isScalene (t : Triangle) : Prop := sorry

/-- A point D that divides the triangle into two equal parts -/
def dividingPoint (t : Triangle) (D : ℝ × ℝ) : Prop :=
  triangleArea ⟨t.A, t.B, D⟩ = triangleArea ⟨t.A, t.C, D⟩

/-- Theorem: A scalene triangle cannot be divided into two equal triangles -/
theorem no_equal_division_of_scalene_triangle (t : Triangle) :
  isScalene t → ¬∃ D : ℝ × ℝ, dividingPoint t D := by
  sorry

end NUMINAMATH_CALUDE_no_equal_division_of_scalene_triangle_l483_48365


namespace NUMINAMATH_CALUDE_inequality_proof_l483_48353

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b)
  (h2 : b ≥ c)
  (h3 : c > 0)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) :
  a > 1 ∧ 1 > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l483_48353


namespace NUMINAMATH_CALUDE_sum_of_squares_l483_48325

theorem sum_of_squares : 
  1000^2 + 1001^2 + 1002^2 + 1003^2 + 1004^2 = 5020030 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l483_48325


namespace NUMINAMATH_CALUDE_star_equation_solution_l483_48332

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ :=
  a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- State the theorem
theorem star_equation_solution :
  ∃ y : ℝ, star 3 y = 18 ∧ y = 30 := by
sorry

end NUMINAMATH_CALUDE_star_equation_solution_l483_48332


namespace NUMINAMATH_CALUDE_segments_in_proportion_l483_48336

-- Define the set of line segments
def segments : List ℝ := [2, 3, 4, 6]

-- Define what it means for a list of four numbers to be in proportion
def isInProportion (l : List ℝ) : Prop :=
  l.length = 4 ∧ l[0]! * l[3]! = l[1]! * l[2]!

-- Theorem statement
theorem segments_in_proportion : isInProportion segments := by
  sorry

end NUMINAMATH_CALUDE_segments_in_proportion_l483_48336


namespace NUMINAMATH_CALUDE_mixed_selection_probability_l483_48396

/-- Represents the number of volunteers from each grade -/
structure Volunteers where
  first_grade : ℕ
  second_grade : ℕ

/-- Represents the number of temporary leaders selected from each grade -/
structure Leaders where
  first_grade : ℕ
  second_grade : ℕ

/-- Calculates the number of leaders proportionally selected from each grade -/
def selectLeaders (v : Volunteers) : Leaders :=
  { first_grade := (5 * v.first_grade) / (v.first_grade + v.second_grade),
    second_grade := (5 * v.second_grade) / (v.first_grade + v.second_grade) }

/-- Calculates the probability of selecting one leader from each grade -/
def probabilityOfMixedSelection (l : Leaders) : ℚ :=
  (l.first_grade * l.second_grade : ℚ) / ((l.first_grade + l.second_grade) * (l.first_grade + l.second_grade - 1) / 2 : ℚ)

theorem mixed_selection_probability 
  (v : Volunteers) 
  (h1 : v.first_grade = 150) 
  (h2 : v.second_grade = 100) : 
  probabilityOfMixedSelection (selectLeaders v) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_mixed_selection_probability_l483_48396


namespace NUMINAMATH_CALUDE_mode_and_median_of_game_scores_l483_48364

def game_scores : List Int := [20, 18, 23, 17, 20, 20, 18]

def mode (l : List Int) : Int := sorry

def median (l : List Int) : Int := sorry

theorem mode_and_median_of_game_scores :
  mode game_scores = 20 ∧ median game_scores = 20 := by sorry

end NUMINAMATH_CALUDE_mode_and_median_of_game_scores_l483_48364


namespace NUMINAMATH_CALUDE_min_colors_is_thirteen_l483_48305

/-- A coloring function for a 25x25 chessboard. -/
def Coloring := Fin 25 → Fin 25 → ℕ

/-- Predicate to check if a coloring satisfies the given condition. -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ i j s t, 1 ≤ i ∧ i < j ∧ j ≤ 25 ∧ 1 ≤ s ∧ s < t ∧ t ≤ 25 →
    (c i s ≠ c j s ∨ c i s ≠ c j t ∨ c j s ≠ c j t)

/-- The minimum number of colors needed for a valid coloring. -/
def MinColors : ℕ := 13

/-- Theorem stating that 13 is the smallest number of colors needed for a valid coloring. -/
theorem min_colors_is_thirteen :
  (∃ c : Coloring, ValidColoring c ∧ (∀ i j, c i j < MinColors)) ∧
  (∀ n : ℕ, n < MinColors →
    ¬∃ c : Coloring, ValidColoring c ∧ (∀ i j, c i j < n)) := by
  sorry

end NUMINAMATH_CALUDE_min_colors_is_thirteen_l483_48305


namespace NUMINAMATH_CALUDE_chris_pears_equal_lily_apples_l483_48388

/-- Represents the number of fruits in the box -/
structure FruitBox where
  apples : ℕ
  pears : ℕ
  apples_twice_pears : apples = 2 * pears

/-- Represents the distribution of fruits between Chris and Lily -/
structure FruitDistribution where
  box : FruitBox
  chris_apples : ℕ
  chris_pears : ℕ
  lily_apples : ℕ
  lily_pears : ℕ
  total_distributed : chris_apples + chris_pears + lily_apples + lily_pears = box.apples + box.pears
  chris_twice_lily : chris_apples + chris_pears = 2 * (lily_apples + lily_pears)

/-- Theorem stating that Chris took as many pears as Lily took apples -/
theorem chris_pears_equal_lily_apples (dist : FruitDistribution) : 
  dist.chris_pears = dist.lily_apples := by sorry

end NUMINAMATH_CALUDE_chris_pears_equal_lily_apples_l483_48388


namespace NUMINAMATH_CALUDE_plane_equation_correct_l483_48338

/-- The equation of a plane given the foot of the perpendicular from the origin -/
def plane_equation (foot : ℝ × ℝ × ℝ) : ℤ × ℤ × ℤ × ℤ := sorry

/-- Check if the given coefficients satisfy the required conditions -/
def valid_coefficients (coeffs : ℤ × ℤ × ℤ × ℤ) : Prop :=
  let (A, B, C, D) := coeffs
  A > 0 ∧ Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

theorem plane_equation_correct (foot : ℝ × ℝ × ℝ) :
  foot = (10, -2, 1) →
  plane_equation foot = (10, -2, 1, -105) ∧
  valid_coefficients (plane_equation foot) := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l483_48338


namespace NUMINAMATH_CALUDE_original_number_problem_l483_48362

theorem original_number_problem (x : ℝ) : ((x + 5 - 2) / 4 = 7) → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l483_48362


namespace NUMINAMATH_CALUDE_jill_trips_to_fill_tank_l483_48372

/-- Represents the water fetching problem with Jack and Jill -/
def WaterFetchingProblem (tank_capacity : ℕ) (bucket_capacity : ℕ) (jack_buckets : ℕ) 
  (jill_buckets : ℕ) (jack_trips : ℕ) (jill_trips : ℕ) (leak_rate : ℕ) : Prop :=
  ∃ (jill_total_trips : ℕ),
    -- The tank capacity is 600 gallons
    tank_capacity = 600 ∧
    -- Each bucket holds 5 gallons
    bucket_capacity = 5 ∧
    -- Jack carries 2 buckets per trip
    jack_buckets = 2 ∧
    -- Jill carries 1 bucket per trip
    jill_buckets = 1 ∧
    -- Jack makes 3 trips for every 2 trips Jill makes
    jack_trips = 3 ∧
    jill_trips = 2 ∧
    -- The tank leaks 2 gallons every time both return
    leak_rate = 2 ∧
    -- The number of trips Jill makes is 20
    jill_total_trips = 20 ∧
    -- The tank is filled after Jill's trips
    jill_total_trips * jill_trips * (jack_buckets * bucket_capacity * jack_trips + 
      jill_buckets * bucket_capacity * jill_trips - leak_rate) / (jack_trips + jill_trips) ≥ tank_capacity

/-- Theorem stating that given the conditions, Jill will make 20 trips before the tank is filled -/
theorem jill_trips_to_fill_tank : 
  WaterFetchingProblem 600 5 2 1 3 2 2 := by sorry

end NUMINAMATH_CALUDE_jill_trips_to_fill_tank_l483_48372


namespace NUMINAMATH_CALUDE_train_length_l483_48384

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) : 
  speed_kmph = 72 → crossing_time = 7 → 
  (speed_kmph * 1000 / 3600) * crossing_time = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l483_48384


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_76_l483_48380

-- Define the vertices of the quadrilateral
def v1 : ℝ × ℝ := (4, -3)
def v2 : ℝ × ℝ := (4, 7)
def v3 : ℝ × ℝ := (12, 2)
def v4 : ℝ × ℝ := (12, -7)

-- Define the function to calculate the area of the quadrilateral
def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area_is_76 : 
  quadrilateralArea v1 v2 v3 v4 = 76 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_76_l483_48380


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l483_48398

def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic a)
  (h_sum : a 4 + a 9 = 24)
  (h_a6 : a 6 = 11) :
  a 7 = 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l483_48398


namespace NUMINAMATH_CALUDE_temperature_conversion_l483_48342

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 122 → t = 50 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l483_48342


namespace NUMINAMATH_CALUDE_seating_arrangements_l483_48375

structure Table :=
  (chairs : ℕ)
  (couples : ℕ)

def valid_seating (t : Table) (arrangements : ℕ) : Prop :=
  t.chairs = 12 ∧
  t.couples = 6 ∧
  arrangements = 43200

theorem seating_arrangements (t : Table) :
  valid_seating t 43200 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l483_48375


namespace NUMINAMATH_CALUDE_missing_number_proof_l483_48300

theorem missing_number_proof : ∃ x : ℝ, 0.72 * x + 0.12 * 0.34 = 0.3504 :=
  by
  use 0.43
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l483_48300


namespace NUMINAMATH_CALUDE_maya_car_arrangement_l483_48328

theorem maya_car_arrangement (current_cars : ℕ) (cars_per_row : ℕ) (additional_cars : ℕ) : 
  current_cars = 29 →
  cars_per_row = 7 →
  (current_cars + additional_cars) % cars_per_row = 0 →
  ∀ n : ℕ, n < additional_cars → (current_cars + n) % cars_per_row ≠ 0 →
  additional_cars = 6 := by
sorry

end NUMINAMATH_CALUDE_maya_car_arrangement_l483_48328


namespace NUMINAMATH_CALUDE_exponent_rules_l483_48351

theorem exponent_rules (a : ℝ) : 
  (a^2 * a^4 ≠ a^8) ∧ 
  ((-2*a^2)^3 ≠ -6*a^6) ∧ 
  (a^4 / a = a^3) ∧ 
  (2*a + 3*a ≠ 5*a^2) := by
sorry

end NUMINAMATH_CALUDE_exponent_rules_l483_48351


namespace NUMINAMATH_CALUDE_circle_radius_l483_48373

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y + 1 = 0) → 
  ∃ (h k r : ℝ), r = 2 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l483_48373


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l483_48360

/-- Given an arithmetic sequence with first term a₁ = -3, last term aₙ = 45, 
    and common difference d = 3, prove that the number of terms n is 17. -/
theorem arithmetic_sequence_count : 
  ∀ (n : ℕ) (a : ℕ → ℤ), 
    a 1 = -3 ∧ 
    (∀ k, a (k + 1) = a k + 3) ∧ 
    a n = 45 → 
    n = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l483_48360


namespace NUMINAMATH_CALUDE_fraction_problem_l483_48330

theorem fraction_problem : ∃ F : ℚ, F = 11/77 ∧ F * 1925 - (1/11) * 1925 = 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l483_48330


namespace NUMINAMATH_CALUDE_car_speed_from_tire_rotation_l483_48374

/-- Given a tire rotating at a certain rate with a specific circumference,
    calculate the speed of the car in km/h. -/
theorem car_speed_from_tire_rotation 
  (revolutions_per_minute : ℝ) 
  (tire_circumference : ℝ) 
  (h1 : revolutions_per_minute = 400) 
  (h2 : tire_circumference = 5) : 
  (revolutions_per_minute * tire_circumference * 60) / 1000 = 120 := by
  sorry

#check car_speed_from_tire_rotation

end NUMINAMATH_CALUDE_car_speed_from_tire_rotation_l483_48374


namespace NUMINAMATH_CALUDE_function_maximum_implies_a_range_l483_48316

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a then -(x + 1) * Real.exp x else -2 * x - 1

theorem function_maximum_implies_a_range (a : ℝ) :
  (∃ (M : ℝ), ∀ (x : ℝ), f a x ≤ M) →
  a ≥ -(1/2) - 1/(2 * Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_function_maximum_implies_a_range_l483_48316


namespace NUMINAMATH_CALUDE_cube_sum_in_interval_l483_48393

theorem cube_sum_in_interval (n : ℕ) : ∃ k x y : ℕ,
  (n : ℝ) - 4 * Real.sqrt (n : ℝ) ≤ k ∧
  k ≤ (n : ℝ) + 4 * Real.sqrt (n : ℝ) ∧
  k = x^3 + y^3 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_in_interval_l483_48393


namespace NUMINAMATH_CALUDE_largest_base3_3digit_in_base10_l483_48363

/-- The largest three-digit number in base 3 -/
def largest_base3_3digit : ℕ := 2 * 3^2 + 2 * 3^1 + 2 * 3^0

/-- Theorem: The largest three-digit number in base 3, when converted to base 10, equals 26 -/
theorem largest_base3_3digit_in_base10 : largest_base3_3digit = 26 := by
  sorry

end NUMINAMATH_CALUDE_largest_base3_3digit_in_base10_l483_48363


namespace NUMINAMATH_CALUDE_school_commute_theorem_l483_48378

/-- Represents the number of students in different commute categories -/
structure SchoolCommute where
  localStudents : ℕ
  publicTransport : ℕ
  privateTransport : ℕ
  train : ℕ
  bus : ℕ
  cycle : ℕ
  drivenByParents : ℕ

/-- Given the commute ratios and public transport users, proves the number of train commuters
    minus parent-driven students and the total number of students -/
theorem school_commute_theorem (sc : SchoolCommute) 
  (h1 : sc.localStudents = 3 * (sc.publicTransport + sc.privateTransport))
  (h2 : 3 * sc.privateTransport = 2 * sc.publicTransport)
  (h3 : 7 * sc.bus = 5 * sc.train)
  (h4 : 5 * sc.drivenByParents = 3 * sc.cycle)
  (h5 : sc.publicTransport = 24)
  (h6 : sc.publicTransport = sc.train + sc.bus)
  (h7 : sc.privateTransport = sc.cycle + sc.drivenByParents) :
  sc.train - sc.drivenByParents = 8 ∧ 
  sc.localStudents + sc.publicTransport + sc.privateTransport = 160 := by
  sorry


end NUMINAMATH_CALUDE_school_commute_theorem_l483_48378


namespace NUMINAMATH_CALUDE_min_value_fraction_l483_48379

theorem min_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∃ (m : ℝ), m = 3 ∧ ∀ z, z = (x * y) / x → m ≤ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l483_48379


namespace NUMINAMATH_CALUDE_fullTimeAndYearCount_l483_48395

/-- Represents a company with employees. -/
structure Company where
  total : ℕ
  fullTime : ℕ
  atLeastYear : ℕ
  neitherFullTimeNorYear : ℕ

/-- The number of full-time employees who have worked at least a year. -/
def fullTimeAndYear (c : Company) : ℕ :=
  c.fullTime + c.atLeastYear - c.total + c.neitherFullTimeNorYear

/-- Theorem stating the number of full-time employees who have worked at least a year. -/
theorem fullTimeAndYearCount (c : Company) 
    (h1 : c.total = 130)
    (h2 : c.fullTime = 80)
    (h3 : c.atLeastYear = 100)
    (h4 : c.neitherFullTimeNorYear = 20) :
    fullTimeAndYear c = 70 := by
  sorry

end NUMINAMATH_CALUDE_fullTimeAndYearCount_l483_48395


namespace NUMINAMATH_CALUDE_unique_common_tangent_common_tangent_segments_bisect_l483_48391

-- Define the parabolas
def C₁ (x : ℝ) : ℝ := x^2 + 2*x
def C₂ (a x : ℝ) : ℝ := -x^2 + a

-- Define the common tangent line
def commonTangent (k b : ℝ) (x : ℝ) : ℝ := k*x + b

-- Define the tangency points
structure TangencyPoint where
  x : ℝ
  y : ℝ

-- Theorem for part 1
theorem unique_common_tangent (a : ℝ) :
  a = -1/2 →
  ∃! (k b : ℝ), ∀ (x : ℝ),
    (C₁ x = commonTangent k b x ∧ C₂ a x = commonTangent k b x) →
    k = 1 ∧ b = -1/4 :=
sorry

-- Theorem for part 2
theorem common_tangent_segments_bisect (a : ℝ) :
  a ≠ -1/2 →
  ∃ (A B C D : TangencyPoint),
    (C₁ A.x = commonTangent k₁ b₁ A.x ∧ C₂ a A.x = commonTangent k₁ b₁ A.x) ∧
    (C₁ B.x = commonTangent k₁ b₁ B.x ∧ C₂ a B.x = commonTangent k₁ b₁ B.x) ∧
    (C₁ C.x = commonTangent k₂ b₂ C.x ∧ C₂ a C.x = commonTangent k₂ b₂ C.x) ∧
    (C₁ D.x = commonTangent k₂ b₂ D.x ∧ C₂ a D.x = commonTangent k₂ b₂ D.x) →
    (A.x + C.x) / 2 = -1/2 ∧ (A.y + C.y) / 2 = (a - 1) / 2 ∧
    (B.x + D.x) / 2 = -1/2 ∧ (B.y + D.y) / 2 = (a - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_unique_common_tangent_common_tangent_segments_bisect_l483_48391


namespace NUMINAMATH_CALUDE_unique_n_l483_48356

theorem unique_n : ∃! n : ℤ, 
  50 ≤ n ∧ n ≤ 150 ∧ 
  7 ∣ n ∧ 
  n % 9 = 3 ∧ 
  n % 4 = 3 ∧
  n = 147 := by
sorry

end NUMINAMATH_CALUDE_unique_n_l483_48356


namespace NUMINAMATH_CALUDE_old_man_silver_dollars_l483_48329

theorem old_man_silver_dollars :
  ∃ (n : ℕ), n > 0 ∧
  (∃ (a b c d r : ℕ),
    n = a + b + c + d + r + 5 ∧
    a = (n - 1) / 4 ∧
    b = ((3 * (n - 1) / 4) - 1) / 4 ∧
    c = ((3 * ((3 * (n - 1) / 4) - 1) / 4) - 1) / 4 ∧
    d = ((3 * ((3 * ((3 * (n - 1) / 4) - 1) / 4) - 1) / 4) - 1) / 4 ∧
    r = ((3 * ((3 * ((3 * ((3 * (n - 1) / 4) - 1) / 4) - 1) / 4) - 1) / 4) - 1) / 4 ∧
    r % 4 = 0) ∧
  (∀ (m : ℕ), m < n →
    ¬(∃ (a b c d r : ℕ),
      m = a + b + c + d + r + 5 ∧
      a = (m - 1) / 4 ∧
      b = ((3 * (m - 1) / 4) - 1) / 4 ∧
      c = ((3 * ((3 * (m - 1) / 4) - 1) / 4) - 1) / 4 ∧
      d = ((3 * ((3 * ((3 * (m - 1) / 4) - 1) / 4) - 1) / 4) - 1) / 4 ∧
      r = ((3 * ((3 * ((3 * ((3 * (m - 1) / 4) - 1) / 4) - 1) / 4) - 1) / 4) - 1) / 4 ∧
      r % 4 = 0)) ∧
  n = 1021 := by
sorry

end NUMINAMATH_CALUDE_old_man_silver_dollars_l483_48329


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l483_48368

def a (n : ℕ) : ℚ := (3 - n^2) / (4 + 2*n^2)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - (-1/2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l483_48368


namespace NUMINAMATH_CALUDE_approval_ratio_rounded_l483_48369

/-- The ratio of regions needed for approval to total regions -/
def approval_ratio : ℚ := 8 / 15

/-- Rounding a rational number to the nearest tenth -/
def round_to_tenth (q : ℚ) : ℚ := 
  ⌊q * 10 + 1/2⌋ / 10

theorem approval_ratio_rounded : round_to_tenth approval_ratio = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_approval_ratio_rounded_l483_48369


namespace NUMINAMATH_CALUDE_prob_no_absolute_winner_is_correct_l483_48341

/-- Represents a player in the mini-tournament -/
inductive Player : Type
| Alyosha : Player
| Borya : Player
| Vasya : Player

/-- Represents the result of a match between two players -/
def MatchResult (p1 p2 : Player) : Type := Bool

/-- The probability that Alyosha wins against Borya -/
def prob_Alyosha_wins_Borya : ℝ := 0.6

/-- The probability that Borya wins against Vasya -/
def prob_Borya_wins_Vasya : ℝ := 0.4

/-- The score of a player in the mini-tournament -/
def score (p : Player) (results : Π p1 p2 : Player, MatchResult p1 p2) : ℕ :=
  sorry

/-- There is an absolute winner if one player has a score of 2 -/
def has_absolute_winner (results : Π p1 p2 : Player, MatchResult p1 p2) : Prop :=
  ∃ p : Player, score p results = 2

/-- The probability of no absolute winner in the mini-tournament -/
def prob_no_absolute_winner : ℝ :=
  sorry

theorem prob_no_absolute_winner_is_correct :
  prob_no_absolute_winner = 0.24 :=
sorry

end NUMINAMATH_CALUDE_prob_no_absolute_winner_is_correct_l483_48341
