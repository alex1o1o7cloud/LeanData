import Mathlib

namespace NUMINAMATH_CALUDE_gcd_1729_867_l1161_116122

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_867_l1161_116122


namespace NUMINAMATH_CALUDE_last_two_digits_product_l1161_116186

/-- Given an integer n, returns its last two digits as a pair -/
def lastTwoDigits (n : ℤ) : ℤ × ℤ :=
  let tens := (n / 10) % 10
  let ones := n % 10
  (tens, ones)

/-- Given an integer n, returns true if it's divisible by 8 -/
def divisibleBy8 (n : ℤ) : Prop :=
  n % 8 = 0

theorem last_two_digits_product (n : ℤ) :
  divisibleBy8 n ∧ (let (a, b) := lastTwoDigits n; a + b = 14) →
  (let (a, b) := lastTwoDigits n; a * b = 48) :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l1161_116186


namespace NUMINAMATH_CALUDE_four_variable_inequality_l1161_116185

theorem four_variable_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d)^2 ≤ 4 * (a^2 + b^2 + c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_four_variable_inequality_l1161_116185


namespace NUMINAMATH_CALUDE_hank_aaron_home_runs_l1161_116113

/-- The number of home runs hit by Dave Winfield -/
def dave_winfield_hr : ℕ := 465

/-- The number of home runs hit by Hank Aaron -/
def hank_aaron_hr : ℕ := 2 * dave_winfield_hr - 175

/-- Theorem stating that Hank Aaron hit 755 home runs -/
theorem hank_aaron_home_runs : hank_aaron_hr = 755 := by sorry

end NUMINAMATH_CALUDE_hank_aaron_home_runs_l1161_116113


namespace NUMINAMATH_CALUDE_base_10_to_base_12_250_l1161_116157

def base_12_digit (n : ℕ) : Char :=
  if n < 10 then Char.ofNat (n + 48)
  else if n = 10 then 'A'
  else 'B'

def to_base_12 (n : ℕ) : List Char :=
  if n < 12 then [base_12_digit n]
  else (to_base_12 (n / 12)) ++ [base_12_digit (n % 12)]

theorem base_10_to_base_12_250 :
  to_base_12 250 = ['1', 'A'] :=
sorry

end NUMINAMATH_CALUDE_base_10_to_base_12_250_l1161_116157


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_projection_trajectory_l1161_116149

/-- The line equation as a function of x, y, and m -/
def line_equation (x y m : ℝ) : Prop := 2 * x + (1 + m) * y + 2 * m = 0

/-- The fixed point through which all lines pass -/
def fixed_point : ℝ × ℝ := (1, -2)

/-- Point P coordinates -/
def point_p : ℝ × ℝ := (-1, 0)

/-- Trajectory equation of point M -/
def trajectory_equation (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 2

theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation (fixed_point.1) (fixed_point.2) m := by sorry

theorem projection_trajectory :
  ∀ x y : ℝ, (∃ m : ℝ, line_equation x y m ∧ 
    (x - point_p.1)^2 + (y - point_p.2)^2 = 
    ((x - point_p.1) * (fixed_point.1 - point_p.1) + (y - point_p.2) * (fixed_point.2 - point_p.2))^2 / 
    ((fixed_point.1 - point_p.1)^2 + (fixed_point.2 - point_p.2)^2)) →
  trajectory_equation x y := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_projection_trajectory_l1161_116149


namespace NUMINAMATH_CALUDE_original_price_calculation_l1161_116174

/-- Given an article sold for $115 with a 15% gain, prove that the original price was $100 --/
theorem original_price_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 115 ∧ gain_percent = 15 → 
  ∃ (original_price : ℝ), 
    original_price = 100 ∧ 
    selling_price = original_price * (1 + gain_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1161_116174


namespace NUMINAMATH_CALUDE_count_six_digit_numbers_with_at_least_two_zeros_l1161_116107

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers with no zeros -/
def numbers_with_no_zeros : ℕ := 531441

/-- The number of 6-digit numbers with exactly one zero -/
def numbers_with_one_zero : ℕ := 295245

/-- The number of 6-digit numbers with at least two zeros -/
def numbers_with_at_least_two_zeros : ℕ := 
  total_six_digit_numbers - (numbers_with_no_zeros + numbers_with_one_zero)

theorem count_six_digit_numbers_with_at_least_two_zeros : 
  numbers_with_at_least_two_zeros = 73314 := by
  sorry

end NUMINAMATH_CALUDE_count_six_digit_numbers_with_at_least_two_zeros_l1161_116107


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1161_116121

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

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1161_116121


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l1161_116165

/-- Given a person who jogs and walks, this theorem proves their walking speed. -/
theorem walking_speed_calculation 
  (jog_speed : ℝ) 
  (jog_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : jog_speed = 2) 
  (h2 : jog_distance = 3) 
  (h3 : total_time = 3) : 
  jog_distance / (total_time - jog_distance / jog_speed) = 2 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l1161_116165


namespace NUMINAMATH_CALUDE_line_increase_l1161_116163

/-- Given an initial number of lines and an increased number of lines with a specific percentage increase, 
    prove that the increase in the number of lines is 110. -/
theorem line_increase (L : ℝ) : 
  let L' : ℝ := 240
  let percent_increase : ℝ := 84.61538461538461
  (L' - L) / L * 100 = percent_increase →
  L' - L = 110 := by
sorry

end NUMINAMATH_CALUDE_line_increase_l1161_116163


namespace NUMINAMATH_CALUDE_max_value_of_x_minus_x_squared_l1161_116198

theorem max_value_of_x_minus_x_squared (x : ℝ) :
  0 < x → x < 1 → ∃ (y : ℝ), y = 1/2 ∧ ∀ z, 0 < z → z < 1 → x * (1 - x) ≤ y * (1 - y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_minus_x_squared_l1161_116198


namespace NUMINAMATH_CALUDE_product_of_roots_l1161_116110

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 18 → ∃ y : ℝ, (x + 3) * (x - 4) = 18 ∧ (y + 3) * (y - 4) = 18 ∧ x * y = -30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1161_116110


namespace NUMINAMATH_CALUDE_smallest_square_area_l1161_116137

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum side length of a square that can contain two rectangles,
    one of which is rotated 90 degrees -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r1.height) (max r2.width r2.height)

/-- Theorem stating the smallest possible area of the square -/
theorem smallest_square_area (r1 r2 : Rectangle)
  (h1 : r1 = ⟨4, 2⟩ ∨ r1 = ⟨2, 4⟩)
  (h2 : r2 = ⟨5, 3⟩ ∨ r2 = ⟨3, 5⟩) :
  (minSquareSide r1 r2) ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l1161_116137


namespace NUMINAMATH_CALUDE_correct_product_l1161_116129

/- Define a function to reverse digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/- Main theorem -/
theorem correct_product (a b : ℕ) :
  a ≥ 10 ∧ a ≤ 99 ∧  -- a is a two-digit number
  0 < b ∧  -- b is positive
  (reverse_digits a) * b = 187 →
  a * b = 187 :=
by
  sorry


end NUMINAMATH_CALUDE_correct_product_l1161_116129


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1161_116173

def M : Set ℝ := {x | x^2 = 1}
def N : Set ℝ := {1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1161_116173


namespace NUMINAMATH_CALUDE_import_tax_threshold_l1161_116120

/-- Proves that the amount in excess of which import tax was applied is $1000 -/
theorem import_tax_threshold (total_value : ℝ) (tax_rate : ℝ) (tax_paid : ℝ) (threshold : ℝ) : 
  total_value = 2570 →
  tax_rate = 0.07 →
  tax_paid = 109.90 →
  tax_rate * (total_value - threshold) = tax_paid →
  threshold = 1000 := by
sorry

end NUMINAMATH_CALUDE_import_tax_threshold_l1161_116120


namespace NUMINAMATH_CALUDE_ginos_popsicle_sticks_l1161_116124

/-- Gino's popsicle stick problem -/
theorem ginos_popsicle_sticks (initial : Real) (given : Real) (remaining : Real) :
  initial = 63.0 →
  given = 50.0 →
  remaining = initial - given →
  remaining = 13.0 := by sorry

end NUMINAMATH_CALUDE_ginos_popsicle_sticks_l1161_116124


namespace NUMINAMATH_CALUDE_babysitting_time_calculation_l1161_116125

/-- Calculates the time spent babysitting given the hourly rate and total earnings -/
def time_spent (hourly_rate : ℚ) (total_earnings : ℚ) : ℚ :=
  (total_earnings / hourly_rate) * 60

/-- Proves that given an hourly rate of $12 and total earnings of $10, the time spent babysitting is 50 minutes -/
theorem babysitting_time_calculation (hourly_rate : ℚ) (total_earnings : ℚ) 
  (h1 : hourly_rate = 12)
  (h2 : total_earnings = 10) :
  time_spent hourly_rate total_earnings = 50 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_time_calculation_l1161_116125


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1161_116101

theorem inequality_solution_set (x : ℝ) : 
  (∃ y, y > 1 ∧ y < x) ↔ (x^2 - x) * (Real.exp x - 1) > 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1161_116101


namespace NUMINAMATH_CALUDE_corners_count_is_even_l1161_116109

/-- A corner is a shape on a grid paper -/
structure Corner where
  position : ℤ × ℤ

/-- A rectangle is a 1x4 shape on a grid paper -/
structure Rectangle where
  position : ℤ × ℤ

/-- A centrally symmetric figure on a grid paper -/
structure CentrallySymmetricFigure where
  corners : List Corner
  rectangles : List Rectangle
  is_centrally_symmetric : Bool

/-- The theorem states that in a centrally symmetric figure composed of corners and 1x4 rectangles, 
    the number of corners must be even -/
theorem corners_count_is_even (figure : CentrallySymmetricFigure) 
  (h : figure.is_centrally_symmetric = true) : 
  Even (figure.corners.length) := by
  sorry

end NUMINAMATH_CALUDE_corners_count_is_even_l1161_116109


namespace NUMINAMATH_CALUDE_test_questions_count_l1161_116177

theorem test_questions_count (total_points : ℕ) (two_point_questions : ℕ) :
  total_points = 100 →
  two_point_questions = 30 →
  ∃ (four_point_questions : ℕ),
    total_points = 2 * two_point_questions + 4 * four_point_questions ∧
    two_point_questions + four_point_questions = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l1161_116177


namespace NUMINAMATH_CALUDE_shaded_area_in_square_with_triangles_l1161_116164

/-- The area of the shaded region in a square with two right-angle triangles -/
theorem shaded_area_in_square_with_triangles (square_side : ℝ) (triangle_leg : ℝ)
  (h_square : square_side = 40)
  (h_triangle : triangle_leg = 25) :
  square_side ^ 2 - 2 * (triangle_leg ^ 2 / 2) = 975 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_in_square_with_triangles_l1161_116164


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1161_116171

theorem polynomial_division_remainder : ∀ (x : ℝ), ∃ (q : ℝ), 2*x^2 - 17*x + 47 = (x - 5) * q + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1161_116171


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l1161_116175

def total_players : ℕ := 15
def predetermined_players : ℕ := 3
def players_to_choose : ℕ := 2

theorem starting_lineup_combinations :
  Nat.choose (total_players - predetermined_players) players_to_choose = 66 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l1161_116175


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1161_116155

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (a^2 - 2 * Real.sqrt 3 * a + 2 = 0) →
  (b^2 - 2 * Real.sqrt 3 * b + 2 = 0) →
  (2 * Real.cos (A + B) = -1) →
  -- Definitions of triangle
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  -- Law of cosines
  (Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) →
  -- Conclusions
  (C = π / 3) ∧ 
  (c = Real.sqrt 6) ∧ 
  (1/2 * a * b * Real.sin C = Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1161_116155


namespace NUMINAMATH_CALUDE_inverse_function_graph_point_l1161_116140

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse function of f
variable (h_inv : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f)

-- Given condition: f(3) = 0
variable (h_f_3 : f 3 = 0)

-- Theorem statement
theorem inverse_function_graph_point :
  (f_inv ((-1) + 1) = 3) ∧ (f_inv ∘ (fun x ↦ x + 1)) (-1) = 3 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_graph_point_l1161_116140


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l1161_116144

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_point_one :
  f 1 = 3 ∧ f' 1 = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l1161_116144


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_ratio_l1161_116153

/-- Given vectors a and b are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_imply_ratio (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, -1)
  let b : ℝ × ℝ := (Real.cos x, 2)
  are_parallel a b →
  (Real.cos x - Real.sin x) / (Real.cos x + Real.sin x) = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_ratio_l1161_116153


namespace NUMINAMATH_CALUDE_rate_of_interest_l1161_116135

/-- Simple interest formula -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

/-- Given conditions -/
def principal : ℝ := 400
def interest : ℝ := 160
def time : ℝ := 2

/-- Theorem: The rate of interest is 0.2 given the conditions -/
theorem rate_of_interest :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_rate_of_interest_l1161_116135


namespace NUMINAMATH_CALUDE_triple_composition_even_l1161_116115

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem triple_composition_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by sorry

end NUMINAMATH_CALUDE_triple_composition_even_l1161_116115


namespace NUMINAMATH_CALUDE_hawks_victory_margin_l1161_116182

/-- Calculates the total score for a team given their scoring details --/
def team_score (touchdowns extra_points two_point_conversions field_goals safeties : ℕ) : ℕ :=
  touchdowns * 6 + extra_points + two_point_conversions * 2 + field_goals * 3 + safeties * 2

/-- Represents the scoring details of the Hawks --/
def hawks_score : ℕ :=
  team_score 4 2 1 2 1

/-- Represents the scoring details of the Eagles --/
def eagles_score : ℕ :=
  team_score 3 3 1 3 1

/-- Theorem stating that the Hawks won by a margin of 2 points --/
theorem hawks_victory_margin :
  hawks_score - eagles_score = 2 :=
sorry

end NUMINAMATH_CALUDE_hawks_victory_margin_l1161_116182


namespace NUMINAMATH_CALUDE_rectangle_area_l1161_116123

theorem rectangle_area (a b : ℕ) : 
  a ≠ b →                  -- rectangle is not a square
  a % 2 = 0 →              -- one side is even
  a * b = 3 * (2 * a + 2 * b) →  -- area is three times perimeter
  a * b = 162              -- area is 162
:= by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1161_116123


namespace NUMINAMATH_CALUDE_volleyball_committee_combinations_l1161_116189

/-- The number of teams in the volleyball league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members sent by the host team to the committee -/
def host_committee_size : ℕ := 4

/-- The number of members sent by each non-host team to the committee -/
def non_host_committee_size : ℕ := 3

/-- The total size of the tournament committee -/
def total_committee_size : ℕ := 16

/-- The number of different committees that can be formed -/
def num_committees : ℕ := 3442073600

theorem volleyball_committee_combinations :
  num_committees = num_teams * (Nat.choose team_size host_committee_size) * 
    (Nat.choose team_size non_host_committee_size)^(num_teams - 1) :=
by sorry

end NUMINAMATH_CALUDE_volleyball_committee_combinations_l1161_116189


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1161_116112

def p (x : ℝ) : ℝ := x^3 + 2*x + 3
def q (x : ℝ) : ℝ := 2*x^4 + x^2 + 7

theorem constant_term_expansion : 
  (p 0) * (q 0) = 21 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1161_116112


namespace NUMINAMATH_CALUDE_carpet_cost_specific_carpet_cost_l1161_116126

/-- The total cost of carpet squares needed to cover a rectangular floor and an irregular section -/
theorem carpet_cost (rectangular_length : ℝ) (rectangular_width : ℝ) (irregular_area : ℝ)
  (carpet_side : ℝ) (carpet_cost : ℝ) : ℝ :=
  let rectangular_area := rectangular_length * rectangular_width
  let carpet_area := carpet_side * carpet_side
  let rectangular_squares := rectangular_area / carpet_area
  let irregular_squares := irregular_area / carpet_area
  let total_squares := rectangular_squares + irregular_squares + 1 -- Adding 1 for potential waste
  total_squares * carpet_cost

/-- The specific problem statement -/
theorem specific_carpet_cost : carpet_cost 24 64 128 8 24 = 648 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_specific_carpet_cost_l1161_116126


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_lines_l1161_116119

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x + y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 3*y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-4, 6)

-- Define a function to check if a point lies on a line
def point_on_line (x y : ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line x y

-- Define a function to check if a line passes through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  point_on_line x₁ y₁ line ∧ point_on_line x₂ y₂ line

-- Define a function to check if two lines are perpendicular
def perpendicular (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∃ m₁ m₂ : ℝ, m₁ * m₂ = -1 ∧
  (∀ x y : ℝ, line1 x y → y = m₁ * x + (P.2 - m₁ * P.1)) ∧
  (∀ x y : ℝ, line2 x y → y = m₂ * x + (P.2 - m₂ * P.1))

-- Theorem statement
theorem intersection_and_perpendicular_lines :
  (point_on_line P.1 P.2 l₁ ∧ point_on_line P.1 P.2 l₂) →
  (∃ line1 : ℝ → ℝ → Prop, line_through_points P.1 P.2 0 0 line1 ∧
    ∀ x y : ℝ, line1 x y ↔ 3*x + 2*y = 0) ∧
  (∃ line2 : ℝ → ℝ → Prop, line_through_points P.1 P.2 P.1 P.2 line2 ∧
    perpendicular line2 l₃ ∧
    ∀ x y : ℝ, line2 x y ↔ 3*x + y + 6 = 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_lines_l1161_116119


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_of_squares_l1161_116161

theorem consecutive_even_numbers_sum_of_squares (n : ℤ) : 
  (∀ k : ℕ, k < 6 → 2 ∣ (n + 2 * k)) → 
  (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) = 90) →
  (n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 + (n + 8)^2 + (n + 10)^2 = 1420) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_of_squares_l1161_116161


namespace NUMINAMATH_CALUDE_b_investment_is_10000_l1161_116143

/-- Represents the capital and profit distribution in a business partnership --/
structure BusinessPartnership where
  capitalA : ℝ
  capitalB : ℝ
  capitalC : ℝ
  profitShareB : ℝ
  profitShareDiffAC : ℝ

/-- Theorem stating that under given conditions, B's investment is 10000 --/
theorem b_investment_is_10000 (bp : BusinessPartnership)
  (h1 : bp.capitalA = 8000)
  (h2 : bp.capitalC = 12000)
  (h3 : bp.profitShareB = 1900)
  (h4 : bp.profitShareDiffAC = 760) :
  bp.capitalB = 10000 := by
  sorry

#check b_investment_is_10000

end NUMINAMATH_CALUDE_b_investment_is_10000_l1161_116143


namespace NUMINAMATH_CALUDE_a_2017_is_one_sixty_fifth_l1161_116191

/-- Represents a proper fraction -/
structure ProperFraction where
  numerator : Nat
  denominator : Nat
  is_proper : numerator < denominator

/-- The sequence of proper fractions -/
def fraction_sequence : Nat → ProperFraction := sorry

/-- The 2017th term of the sequence -/
def a_2017 : ProperFraction := fraction_sequence 2017

/-- Theorem stating that the 2017th term is 1/65 -/
theorem a_2017_is_one_sixty_fifth : 
  a_2017.numerator = 1 ∧ a_2017.denominator = 65 := by sorry

end NUMINAMATH_CALUDE_a_2017_is_one_sixty_fifth_l1161_116191


namespace NUMINAMATH_CALUDE_chocolate_cake_eggs_l1161_116180

/-- The number of eggs needed for each cheesecake -/
def eggs_per_cheesecake : ℕ := 8

/-- The number of additional eggs needed for 9 cheesecakes compared to 5 chocolate cakes -/
def additional_eggs : ℕ := 57

/-- The number of cheesecakes in the comparison -/
def num_cheesecakes : ℕ := 9

/-- The number of chocolate cakes in the comparison -/
def num_chocolate_cakes : ℕ := 5

/-- The number of eggs needed for each chocolate cake -/
def eggs_per_chocolate_cake : ℕ := 3

theorem chocolate_cake_eggs :
  eggs_per_chocolate_cake * num_chocolate_cakes = 
  eggs_per_cheesecake * num_cheesecakes - additional_eggs :=
by sorry

end NUMINAMATH_CALUDE_chocolate_cake_eggs_l1161_116180


namespace NUMINAMATH_CALUDE_min_value_on_line_l1161_116162

theorem min_value_on_line (x y : ℝ) (h : x + 2 * y = 3) :
  ∃ (m : ℝ), m = 4 * Real.sqrt 2 ∧ ∀ (a b : ℝ), a + 2 * b = 3 → 2^a + 4^b ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_line_l1161_116162


namespace NUMINAMATH_CALUDE_x_range_for_negative_f_l1161_116169

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + a + 1

-- Define the theorem
theorem x_range_for_negative_f :
  (∀ x : ℝ, ∀ a ∈ Set.Icc (-1 : ℝ) 1, f a x < 0) →
  (∀ x : ℝ, f (-1) x < 0 ∧ f 1 x < 0) →
  {x : ℝ | 1 < x ∧ x < 2} = {x : ℝ | ∀ a ∈ Set.Icc (-1 : ℝ) 1, f a x < 0} :=
by sorry


end NUMINAMATH_CALUDE_x_range_for_negative_f_l1161_116169


namespace NUMINAMATH_CALUDE_cafeteria_pie_count_l1161_116148

/-- Given a cafeteria with apples and pie-making scenario, calculate the number of pies that can be made -/
theorem cafeteria_pie_count (total_apples handed_out apples_per_pie : ℕ) 
  (h1 : total_apples = 96)
  (h2 : handed_out = 42)
  (h3 : apples_per_pie = 6) :
  (total_apples - handed_out) / apples_per_pie = 9 :=
by
  sorry

#check cafeteria_pie_count

end NUMINAMATH_CALUDE_cafeteria_pie_count_l1161_116148


namespace NUMINAMATH_CALUDE_triangle_bisector_inequality_l1161_116116

/-- In any triangle ABC, the product of the lengths of its internal angle bisectors 
    is less than or equal to (3√3 / 8) times the product of its side lengths. -/
theorem triangle_bisector_inequality (a b c t_a t_b t_c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < t_a ∧ 0 < t_b ∧ 0 < t_c →  -- Positive bisector lengths
  t_a + t_b > c ∧ t_b + t_c > a ∧ t_c + t_a > b →  -- Triangle inequality for bisectors
  t_a * t_b * t_c ≤ (3 * Real.sqrt 3 / 8) * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_bisector_inequality_l1161_116116


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1161_116134

/-- An isosceles right triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- Length of the leg of the isosceles right triangle -/
  a : ℝ
  /-- Side length of the inscribed square -/
  s : ℝ
  /-- The triangle is isosceles and right-angled -/
  isIsoscelesRight : True
  /-- The square is inscribed with one vertex on the hypotenuse -/
  squareOnHypotenuse : True
  /-- The square has one vertex at the right angle of the triangle -/
  squareAtRightAngle : True
  /-- The square has two vertices on the legs of the triangle -/
  squareOnLegs : True
  /-- The leg length is positive -/
  a_pos : 0 < a

/-- The side length of the inscribed square is half the leg length of the triangle -/
theorem inscribed_square_side_length 
  (triangle : IsoscelesRightTriangleWithSquare) : 
  triangle.s = triangle.a / 2 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_square_side_length_l1161_116134


namespace NUMINAMATH_CALUDE_rulers_remaining_l1161_116158

theorem rulers_remaining (initial_rulers : ℕ) (rulers_taken : ℕ) : 
  initial_rulers = 46 → rulers_taken = 25 → initial_rulers - rulers_taken = 21 :=
by sorry

end NUMINAMATH_CALUDE_rulers_remaining_l1161_116158


namespace NUMINAMATH_CALUDE_angle_on_bisector_l1161_116196

-- Define the set of integers
variable (k : ℤ)

-- Define the angle in degrees
def angle (k : ℤ) : ℝ := k * 180 + 135

-- Define the property of being on the bisector of the second or fourth quadrant
def on_bisector_2nd_or_4th (θ : ℝ) : Prop :=
  ∃ n : ℤ, θ = 135 + n * 360 ∨ θ = 315 + n * 360

-- Theorem statement
theorem angle_on_bisector :
  ∀ θ : ℝ, on_bisector_2nd_or_4th θ ↔ ∃ k : ℤ, θ = angle k :=
sorry

end NUMINAMATH_CALUDE_angle_on_bisector_l1161_116196


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_l1161_116103

-- Define the type for people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person
| Fiona : Person

-- Define a seating arrangement as a function from position to person
def SeatingArrangement := Fin 6 → Person

-- Define the conditions for a valid seating arrangement
def IsValidArrangement (arrangement : SeatingArrangement) : Prop :=
  -- Alice is not next to Bob or Carla
  (∀ i : Fin 5, arrangement i = Person.Alice → 
    arrangement (i + 1) ≠ Person.Bob ∧ arrangement (i + 1) ≠ Person.Carla) ∧
  (∀ i : Fin 5, arrangement (i + 1) = Person.Alice → 
    arrangement i ≠ Person.Bob ∧ arrangement i ≠ Person.Carla) ∧
  -- Derek is not next to Eric
  (∀ i : Fin 5, arrangement i = Person.Derek → arrangement (i + 1) ≠ Person.Eric) ∧
  (∀ i : Fin 5, arrangement (i + 1) = Person.Derek → arrangement i ≠ Person.Eric) ∧
  -- Fiona is not at either end
  (arrangement 0 ≠ Person.Fiona) ∧ (arrangement 5 ≠ Person.Fiona) ∧
  -- All people are seated and each seat has exactly one person
  (∀ p : Person, ∃! i : Fin 6, arrangement i = p)

-- The theorem to be proved
theorem valid_seating_arrangements :
  (∃ (arrangements : Finset SeatingArrangement), 
    (∀ arr ∈ arrangements, IsValidArrangement arr) ∧ 
    arrangements.card = 16) :=
sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_l1161_116103


namespace NUMINAMATH_CALUDE_x_greater_abs_y_sufficient_not_necessary_l1161_116181

theorem x_greater_abs_y_sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x > |y| → x > y) ∧
  (∃ x y : ℝ, x > y ∧ ¬(x > |y|)) :=
sorry

end NUMINAMATH_CALUDE_x_greater_abs_y_sufficient_not_necessary_l1161_116181


namespace NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l1161_116176

/-- The curve y = x^3 + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

/-- Point P₀ -/
def P₀ : ℝ × ℝ := (-1, -4)

/-- The slope of the line parallel to the tangent at P₀ -/
def m : ℝ := 4

/-- The equation of the line perpendicular to the tangent at P₀ -/
def l (x y : ℝ) : Prop := x + 4*y + 17 = 0

theorem tangent_point_and_perpendicular_line :
  (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ f' x = m) →  -- P₀ exists in third quadrant with slope m
  (P₀.1 = -1 ∧ P₀.2 = -4) ∧  -- P₀ has coordinates (-1, -4)
  (∀ (x y : ℝ), l x y ↔ y - P₀.2 = -(1/m) * (x - P₀.1)) :=  -- l is perpendicular to tangent at P₀
by sorry

end NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l1161_116176


namespace NUMINAMATH_CALUDE_fraction_inequality_l1161_116150

theorem fraction_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  c / a > d / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1161_116150


namespace NUMINAMATH_CALUDE_expression_simplification_l1161_116194

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^2 - 2*a*b + b^2) / (2*a*b) - (2*a*b - b^2) / (3*a*b - 3*a^2) = (a - b)^2 / (2*a*b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1161_116194


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1161_116166

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    P = 21/4 ∧ Q = 15 ∧ R = -11/2 ∧
    ∀ (x : ℚ), x ≠ 2 → x ≠ 4 →
      5*x + 1 = (x - 4)*(x - 2)^2 * (P/(x - 4) + Q/(x - 2) + R/(x - 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1161_116166


namespace NUMINAMATH_CALUDE_tetrahedron_volume_from_midpoint_distances_l1161_116195

/-- The volume of a regular tetrahedron, given specific midpoint distances -/
theorem tetrahedron_volume_from_midpoint_distances :
  ∀ (midpoint_to_face midpoint_to_edge : ℝ),
    midpoint_to_face = 2 →
    midpoint_to_edge = Real.sqrt 5 →
    ∃ (volume : ℝ), volume = 27 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_from_midpoint_distances_l1161_116195


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l1161_116133

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_450 :
  let M := sum_of_divisors 450
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ M ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ M → q ≤ p ∧ p = 31 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l1161_116133


namespace NUMINAMATH_CALUDE_calculate_expression_l1161_116132

theorem calculate_expression : 3000 * (3000^2999) * 2 = 2 * 3000^3000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1161_116132


namespace NUMINAMATH_CALUDE_square_side_estimate_l1161_116131

theorem square_side_estimate (A : ℝ) (h : A = 30) :
  ∃ s : ℝ, s^2 = A ∧ 5 < s ∧ s < 6 := by
  sorry

end NUMINAMATH_CALUDE_square_side_estimate_l1161_116131


namespace NUMINAMATH_CALUDE_z_modulus_l1161_116111

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition z(i+1) = i
def condition (z : ℂ) : Prop := z * (i + 1) = i

-- State the theorem
theorem z_modulus (z : ℂ) (h : condition z) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_z_modulus_l1161_116111


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1161_116108

theorem unique_solution_quadratic (c : ℝ) (h : c ≠ 0) :
  (∃! b : ℝ, b > 0 ∧ (∃! x : ℝ, x^2 + 3 * (b + 1/b) * x + c = 0)) ↔ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1161_116108


namespace NUMINAMATH_CALUDE_fraction_count_l1161_116102

/-- A fraction is an expression of the form A/B where A and B are polynomials and B contains letters -/
def IsFraction (expr : String) : Prop := sorry

/-- The set of given expressions -/
def ExpressionSet : Set String := {"1/m", "b/3", "(x-1)/π", "2/(x+y)", "a+1/a"}

/-- Counts the number of fractions in a set of expressions -/
def CountFractions (s : Set String) : ℕ := sorry

theorem fraction_count : CountFractions ExpressionSet = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_count_l1161_116102


namespace NUMINAMATH_CALUDE_fifth_term_is_half_l1161_116106

/-- A geometric sequence with special properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  sum_property : a 1 + a 2 + a 3 = a 2 + 5 * a 1
  a7_value : a 7 = 2

/-- The fifth term of the geometric sequence is 1/2 -/
theorem fifth_term_is_half (seq : GeometricSequence) : seq.a 5 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_half_l1161_116106


namespace NUMINAMATH_CALUDE_average_spring_headcount_equals_10700_l1161_116178

def spring_headcount_02_03 : ℕ := 10900
def spring_headcount_03_04 : ℕ := 10500
def spring_headcount_04_05 : ℕ := 10700

def average_spring_headcount : ℚ :=
  (spring_headcount_02_03 + spring_headcount_03_04 + spring_headcount_04_05) / 3

theorem average_spring_headcount_equals_10700 :
  round average_spring_headcount = 10700 := by
  sorry

end NUMINAMATH_CALUDE_average_spring_headcount_equals_10700_l1161_116178


namespace NUMINAMATH_CALUDE_parallelogram_count_parallelogram_count_equals_function_l1161_116105

/-- Given a triangle ABC where each side is divided into n equal segments and lines are drawn
    parallel to each side through each division point, the total number of parallelograms formed
    is 3 × (n+2 choose 4) + 2. -/
theorem parallelogram_count (n : ℕ) : ℕ :=
  3 * Nat.choose (n + 2) 4 + 2

/-- The function that calculates the number of parallelograms -/
def count_parallelograms (n : ℕ) : ℕ :=
  3 * Nat.choose (n + 2) 4 + 2

theorem parallelogram_count_equals_function (n : ℕ) :
  parallelogram_count n = count_parallelograms n := by sorry

end NUMINAMATH_CALUDE_parallelogram_count_parallelogram_count_equals_function_l1161_116105


namespace NUMINAMATH_CALUDE_yoonseok_handshakes_l1161_116172

/-- Represents a group of people arranged in a dodecagon -/
structure DodecagonGroup :=
  (size : Nat)
  (handshake_rule : Nat → Nat)

/-- The number of handshakes for a person in the DodecagonGroup -/
def handshakes_count (g : DodecagonGroup) : Nat :=
  g.handshake_rule g.size

theorem yoonseok_handshakes (g : DodecagonGroup) :
  g.size = 12 →
  (∀ n : Nat, n ≤ g.size → g.handshake_rule n = n - 3) →
  handshakes_count g = 9 := by
  sorry

#check yoonseok_handshakes

end NUMINAMATH_CALUDE_yoonseok_handshakes_l1161_116172


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1161_116188

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I : ℂ) * 2 + 1 * a + (Complex.I : ℂ) * (2 * a) + b = (Complex.I : ℂ) * 2 → 
  a = 1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1161_116188


namespace NUMINAMATH_CALUDE_certain_number_problem_l1161_116156

theorem certain_number_problem :
  ∃! x : ℝ, ((7 * (x + 10)) / 5) - 5 = 88 / 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1161_116156


namespace NUMINAMATH_CALUDE_total_value_is_18_60_l1161_116100

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a half-dollar coin in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The value of a dollar coin in dollars -/
def dollar_coin_value : ℚ := 1.00

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The number of quarters Tom found -/
def num_quarters : ℕ := 25

/-- The number of dimes Tom found -/
def num_dimes : ℕ := 15

/-- The number of nickels Tom found -/
def num_nickels : ℕ := 12

/-- The number of half-dollar coins Tom found -/
def num_half_dollars : ℕ := 7

/-- The number of dollar coins Tom found -/
def num_dollar_coins : ℕ := 3

/-- The number of pennies Tom found -/
def num_pennies : ℕ := 375

/-- The total value of the coins Tom found -/
def total_value : ℚ :=
  num_quarters * quarter_value +
  num_dimes * dime_value +
  num_nickels * nickel_value +
  num_half_dollars * half_dollar_value +
  num_dollar_coins * dollar_coin_value +
  num_pennies * penny_value

theorem total_value_is_18_60 : total_value = 18.60 := by
  sorry

end NUMINAMATH_CALUDE_total_value_is_18_60_l1161_116100


namespace NUMINAMATH_CALUDE_beef_to_steaks_l1161_116184

/-- Given 15 pounds of beef cut into 12-ounce steaks, prove that the number of steaks obtained is 20. -/
theorem beef_to_steaks :
  let pounds_of_beef : ℕ := 15
  let ounces_per_pound : ℕ := 16
  let ounces_per_steak : ℕ := 12
  let total_ounces : ℕ := pounds_of_beef * ounces_per_pound
  let number_of_steaks : ℕ := total_ounces / ounces_per_steak
  number_of_steaks = 20 :=
by sorry

end NUMINAMATH_CALUDE_beef_to_steaks_l1161_116184


namespace NUMINAMATH_CALUDE_inscribed_squares_circles_area_difference_l1161_116130

/-- The difference between the sum of areas of squares and circles in an infinite inscribed sequence -/
theorem inscribed_squares_circles_area_difference :
  let square_areas : ℕ → ℝ := λ n => (1 / 2 : ℝ) ^ n
  let circle_areas : ℕ → ℝ := λ n => π / 4 * (1 / 2 : ℝ) ^ n
  (∑' n, square_areas n) - (∑' n, circle_areas n) = 2 - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_circles_area_difference_l1161_116130


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1161_116117

theorem absolute_value_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : |a| > -b := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1161_116117


namespace NUMINAMATH_CALUDE_rectangular_plot_difference_l1161_116179

/-- Proves that for a rectangular plot with breadth 8 meters and area 18 times its breadth,
    the difference between the length and the breadth is 10 meters. -/
theorem rectangular_plot_difference (length breadth : ℝ) : 
  breadth = 8 →
  length * breadth = 18 * breadth →
  length - breadth = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_difference_l1161_116179


namespace NUMINAMATH_CALUDE_particle_probabilities_l1161_116147

/-- A particle moves on a line with marked points 0, ±1, ±2, ±3, ... 
    Starting at point 0, it moves to n+1 or n-1 with equal probabilities 1/2 -/
def Particle := ℤ

/-- The probability that the particle will be at point 1 at some time -/
def prob_at_one (p : Particle) : ℝ := sorry

/-- The probability that the particle will be at point -1 at some time -/
def prob_at_neg_one (p : Particle) : ℝ := sorry

/-- The probability that the particle will return to point 0 at some time 
    other than the initial starting point -/
def prob_return_to_zero (p : Particle) : ℝ := sorry

/-- The theorem stating that all three probabilities are equal to 1 -/
theorem particle_probabilities (p : Particle) : 
  prob_at_one p = 1 ∧ prob_at_neg_one p = 1 ∧ prob_return_to_zero p = 1 :=
by sorry

end NUMINAMATH_CALUDE_particle_probabilities_l1161_116147


namespace NUMINAMATH_CALUDE_sadaf_height_l1161_116159

theorem sadaf_height (lily_height : ℝ) (anika_height : ℝ) (sadaf_height : ℝ) 
  (h1 : lily_height = 90)
  (h2 : anika_height = 4/3 * lily_height)
  (h3 : sadaf_height = 5/4 * anika_height) :
  sadaf_height = 150 := by
  sorry

end NUMINAMATH_CALUDE_sadaf_height_l1161_116159


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1161_116167

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1161_116167


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1161_116138

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate for a point being on the curve y = 4x^2 -/
def OnCurve (p : Point) : Prop :=
  p.y = 4 * p.x^2

/-- Predicate for a point satisfying the equation √y = 2x -/
def SatisfiesEquation (p : Point) : Prop :=
  Real.sqrt p.y = 2 * p.x

theorem necessary_not_sufficient :
  (∀ p : Point, OnCurve p → SatisfiesEquation p) ∧
  (∃ p : Point, SatisfiesEquation p ∧ ¬OnCurve p) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1161_116138


namespace NUMINAMATH_CALUDE_exists_same_dimensions_l1161_116139

/-- Represents a rectangle with width and height as powers of two -/
structure Rectangle where
  width : Nat
  height : Nat
  width_pow_two : ∃ k : Nat, width = 2^k
  height_pow_two : ∃ k : Nat, height = 2^k

/-- Represents a tiling of a square -/
structure Tiling where
  n : Nat
  rectangles : List Rectangle
  at_least_two : rectangles.length ≥ 2
  covers_square : ∀ (x y : Nat), x < 2^n ∧ y < 2^n → 
    ∃ (r : Rectangle), r ∈ rectangles ∧ x < r.width ∧ y < r.height
  non_overlapping : ∀ (r1 r2 : Rectangle), r1 ∈ rectangles ∧ r2 ∈ rectangles ∧ r1 ≠ r2 →
    ∀ (x y : Nat), ¬(x < r1.width ∧ y < r1.height ∧ x < r2.width ∧ y < r2.height)

/-- Main theorem: There exist at least two rectangles with the same dimensions in any valid tiling -/
theorem exists_same_dimensions (t : Tiling) : 
  ∃ (r1 r2 : Rectangle), r1 ∈ t.rectangles ∧ r2 ∈ t.rectangles ∧ r1 ≠ r2 ∧ 
    r1.width = r2.width ∧ r1.height = r2.height :=
by sorry

end NUMINAMATH_CALUDE_exists_same_dimensions_l1161_116139


namespace NUMINAMATH_CALUDE_john_total_calories_l1161_116183

/-- The number of potato chips John eats -/
def num_chips : ℕ := 10

/-- The total calories of the potato chips -/
def total_chip_calories : ℕ := 60

/-- The number of cheezits John eats -/
def num_cheezits : ℕ := 6

/-- The calories of one potato chip -/
def calories_per_chip : ℚ := total_chip_calories / num_chips

/-- The calories of one cheezit -/
def calories_per_cheezit : ℚ := calories_per_chip * (1 + 1/3)

/-- The total calories John ate -/
def total_calories : ℚ := num_chips * calories_per_chip + num_cheezits * calories_per_cheezit

theorem john_total_calories : total_calories = 108 := by
  sorry

end NUMINAMATH_CALUDE_john_total_calories_l1161_116183


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1161_116168

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1161_116168


namespace NUMINAMATH_CALUDE_expected_male_athletes_expected_male_athletes_eq_twelve_l1161_116145

/-- Given a team of athletes with a specific male-to-total ratio,
    calculate the expected number of male athletes in a stratified sample. -/
theorem expected_male_athletes 
  (total_athletes : ℕ) 
  (male_ratio : ℚ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = 84) 
  (h2 : male_ratio = 4/7) 
  (h3 : sample_size = 21) : 
  ℕ := by
  sorry

#check expected_male_athletes

theorem expected_male_athletes_eq_twelve 
  (total_athletes : ℕ) 
  (male_ratio : ℚ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = 84) 
  (h2 : male_ratio = 4/7) 
  (h3 : sample_size = 21) : 
  expected_male_athletes total_athletes male_ratio sample_size h1 h2 h3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expected_male_athletes_expected_male_athletes_eq_twelve_l1161_116145


namespace NUMINAMATH_CALUDE_f_value_at_2_l1161_116136

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem f_value_at_2 : 
  (∀ x : ℝ, f (2 * x + 1) = x^2 - 2*x) → f 2 = -3/4 := by sorry

end NUMINAMATH_CALUDE_f_value_at_2_l1161_116136


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1161_116199

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (y : ℝ), (Complex.I : ℂ) * y = (1 - a * Complex.I) / (1 + Complex.I)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1161_116199


namespace NUMINAMATH_CALUDE_a_greater_than_b_greater_than_one_l1161_116192

theorem a_greater_than_b_greater_than_one
  (n : ℕ) (hn : n ≥ 2)
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (eq1 : a^n = a + 1)
  (eq2 : b^(2*n) = b + 3*a) :
  a > b ∧ b > 1 := by
sorry

end NUMINAMATH_CALUDE_a_greater_than_b_greater_than_one_l1161_116192


namespace NUMINAMATH_CALUDE_suv_highway_mileage_l1161_116160

/-- Given an SUV with specified city mileage and maximum distance on a fixed amount of gasoline,
    calculate its highway mileage. -/
theorem suv_highway_mileage 
  (city_mpg : ℝ) 
  (max_distance : ℝ) 
  (gas_amount : ℝ) 
  (h_city_mpg : city_mpg = 7.6)
  (h_max_distance : max_distance = 292.8)
  (h_gas_amount : gas_amount = 24) :
  max_distance / gas_amount = 12.2 := by
  sorry

end NUMINAMATH_CALUDE_suv_highway_mileage_l1161_116160


namespace NUMINAMATH_CALUDE_howard_last_week_money_l1161_116193

/-- Howard's money situation --/
def howard_money (current_money washing_money last_week_money : ℕ) : Prop :=
  current_money = washing_money + last_week_money

/-- Theorem: Howard had 26 dollars last week --/
theorem howard_last_week_money :
  ∃ (last_week_money : ℕ),
    howard_money 52 26 last_week_money ∧ last_week_money = 26 :=
sorry

end NUMINAMATH_CALUDE_howard_last_week_money_l1161_116193


namespace NUMINAMATH_CALUDE_five_digit_sum_contains_zero_l1161_116151

-- Define a five-digit number type
def FiveDigitNumber := { n : ℕ // n ≥ 10000 ∧ n < 100000 }

-- Define a function to check if a number contains 0
def containsZero (n : FiveDigitNumber) : Prop :=
  ∃ (a b c d : ℕ), n.val = 10000 * a + 1000 * b + 100 * c + 10 * d ∨
                    n.val = 10000 * a + 1000 * b + 100 * c + d ∨
                    n.val = 10000 * a + 1000 * b + 10 * c + d ∨
                    n.val = 10000 * a + 100 * b + 10 * c + d ∨
                    n.val = 1000 * a + 100 * b + 10 * c + d

-- Define a function to check if two numbers differ by switching two digits
def differByTwoDigits (n m : FiveDigitNumber) : Prop :=
  ∃ (a b c d e f : ℕ),
    (n.val = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
     m.val = 10000 * a + 1000 * b + 100 * f + 10 * d + e) ∨
    (n.val = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
     m.val = 10000 * a + 1000 * f + 100 * c + 10 * d + e) ∨
    (n.val = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
     m.val = 10000 * f + 1000 * b + 100 * c + 10 * d + e)

theorem five_digit_sum_contains_zero (n m : FiveDigitNumber)
  (h1 : differByTwoDigits n m)
  (h2 : n.val + m.val = 111111) :
  containsZero n ∨ containsZero m :=
sorry

end NUMINAMATH_CALUDE_five_digit_sum_contains_zero_l1161_116151


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1161_116197

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) : 
  total_students = 42 →
  boy_ratio = 3 →
  girl_ratio = 4 →
  ∃ (boys girls : ℕ), 
    boys * girl_ratio = girls * boy_ratio ∧
    boys + girls = total_students ∧
    girls - boys = 6 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1161_116197


namespace NUMINAMATH_CALUDE_abc_together_probability_l1161_116118

-- Define the number of people and available positions
def num_people : ℕ := 8
def available_positions : ℕ := 6

-- Define the probability function
def probability_abc_together : ℚ := 1 / 84

-- Theorem statement
theorem abc_together_probability :
  probability_abc_together = 1 / 84 :=
by sorry

end NUMINAMATH_CALUDE_abc_together_probability_l1161_116118


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l1161_116142

/-- Given Isabella's initial and final hair lengths, prove the amount of hair growth. -/
theorem isabellas_hair_growth 
  (initial_length : ℝ) 
  (final_length : ℝ) 
  (h1 : initial_length = 18) 
  (h2 : final_length = 24) : 
  final_length - initial_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l1161_116142


namespace NUMINAMATH_CALUDE_happy_number_iff_multiple_of_eight_l1161_116152

/-- A number is "happy" if it is equal to the square difference of two consecutive odd numbers. -/
def is_happy_number (n : ℤ) : Prop :=
  ∃ k : ℤ, n = (2*k + 1)^2 - (2*k - 1)^2

/-- The theorem states that a number is a "happy number" if and only if it is a multiple of 8. -/
theorem happy_number_iff_multiple_of_eight (n : ℤ) :
  is_happy_number n ↔ ∃ m : ℤ, n = 8 * m :=
by sorry

end NUMINAMATH_CALUDE_happy_number_iff_multiple_of_eight_l1161_116152


namespace NUMINAMATH_CALUDE_expression_value_l1161_116127

theorem expression_value (x : ℝ) (h : Real.tan (Real.pi - x) = -2) :
  4 * Real.sin x ^ 2 - 3 * Real.sin x * Real.cos x - 5 * Real.cos x ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1161_116127


namespace NUMINAMATH_CALUDE_min_value_sum_l1161_116141

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 / x) + (8 / y) = 1) : x + y ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l1161_116141


namespace NUMINAMATH_CALUDE_max_quarters_and_dimes_l1161_116146

theorem max_quarters_and_dimes (total : ℚ) (h_total : total = 425/100) :
  ∃ (quarters dimes pennies : ℕ),
    quarters = dimes ∧
    quarters * (25 : ℚ)/100 + dimes * (10 : ℚ)/100 + pennies * (1 : ℚ)/100 = total ∧
    ∀ q d p : ℕ, q = d →
      q * (25 : ℚ)/100 + d * (10 : ℚ)/100 + p * (1 : ℚ)/100 = total →
      q ≤ quarters :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_and_dimes_l1161_116146


namespace NUMINAMATH_CALUDE_absolute_value_complex_power_l1161_116170

theorem absolute_value_complex_power : 
  Complex.abs ((5 : ℂ) + (Complex.I * Real.sqrt 11)) ^ 4 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_complex_power_l1161_116170


namespace NUMINAMATH_CALUDE_floor_sum_for_specific_x_l1161_116190

theorem floor_sum_for_specific_x : 
  let x : ℝ := 9.42
  ⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋ = 55 := by sorry

end NUMINAMATH_CALUDE_floor_sum_for_specific_x_l1161_116190


namespace NUMINAMATH_CALUDE_william_has_more_money_l1161_116187

def oliver_initial_usd : ℝ := 10 * 20 + 3 * 5
def oliver_uk_pounds : ℝ := 200
def oliver_japan_yen : ℝ := 7000
def pound_to_usd : ℝ := 1.38
def yen_to_usd : ℝ := 0.0091
def oliver_expense_usd : ℝ := 75
def oliver_expense_pounds : ℝ := 55
def oliver_expense_yen : ℝ := 3000

def william_initial_usd : ℝ := 15 * 10 + 4 * 5
def william_europe_euro : ℝ := 250
def william_canada_cad : ℝ := 350
def euro_to_usd : ℝ := 1.18
def cad_to_usd : ℝ := 0.78
def william_expense_usd : ℝ := 20
def william_expense_euro : ℝ := 105
def william_expense_cad : ℝ := 150

theorem william_has_more_money :
  let oliver_remaining := (oliver_initial_usd - oliver_expense_usd) +
                          (oliver_uk_pounds * pound_to_usd - oliver_expense_pounds * pound_to_usd) +
                          (oliver_japan_yen * yen_to_usd - oliver_expense_yen * yen_to_usd)
  let william_remaining := (william_initial_usd - william_expense_usd) +
                           (william_europe_euro * euro_to_usd - william_expense_euro * euro_to_usd) +
                           (william_canada_cad * cad_to_usd - william_expense_cad * cad_to_usd)
  william_remaining - oliver_remaining = 100.6 := by sorry

end NUMINAMATH_CALUDE_william_has_more_money_l1161_116187


namespace NUMINAMATH_CALUDE_average_and_square_multiple_l1161_116104

theorem average_and_square_multiple (n : ℝ) (m : ℝ) (h1 : n ≠ 0) (h2 : n = 9) 
  (h3 : (n + n^2) / 2 = m * n) : m = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_and_square_multiple_l1161_116104


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l1161_116114

theorem square_sum_reciprocal (w : ℝ) (hw : w > 0) (heq : w - 1/w = 5) :
  (w + 1/w)^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l1161_116114


namespace NUMINAMATH_CALUDE_custom_mul_theorem_l1161_116128

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 2 * a - b^3

/-- Theorem stating that if a * 3 = 15 under the custom multiplication, then a = 21 -/
theorem custom_mul_theorem (a : ℝ) (h : custom_mul a 3 = 15) : a = 21 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_theorem_l1161_116128


namespace NUMINAMATH_CALUDE_max_m_value_l1161_116154

theorem max_m_value (m : ℕ+) 
  (h : ∃ (k : ℕ), (m.val ^ 4 + 16 * m.val + 8 : ℕ) = (k * (k + 1))) : 
  m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_max_m_value_l1161_116154
