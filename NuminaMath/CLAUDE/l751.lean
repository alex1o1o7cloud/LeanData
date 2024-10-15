import Mathlib

namespace NUMINAMATH_CALUDE_min_operations_rectangle_l751_75129

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Measures the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Checks if two numbers are equal -/
def compare (a b : ℝ) : Bool :=
  sorry

/-- Checks if a quadrilateral is a rectangle -/
def isRectangle (q : Quadrilateral) : Bool :=
  sorry

/-- Counts the number of operations needed to determine if a quadrilateral is a rectangle -/
def countOperations (q : Quadrilateral) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of operations to determine if a quadrilateral is a rectangle is 9 -/
theorem min_operations_rectangle (q : Quadrilateral) : 
  (countOperations q = 9) ∧ (∀ n : ℕ, n < 9 → ¬(∀ q' : Quadrilateral, isRectangle q' ↔ countOperations q' ≤ n)) :=
  sorry

end NUMINAMATH_CALUDE_min_operations_rectangle_l751_75129


namespace NUMINAMATH_CALUDE_minimize_reciprocal_sum_l751_75146

theorem minimize_reciprocal_sum (a b : ℕ+) (h : 4 * a.val + b.val = 30) :
  (1 : ℚ) / a.val + (1 : ℚ) / b.val ≥ (1 : ℚ) / 5 + (1 : ℚ) / 10 :=
sorry

end NUMINAMATH_CALUDE_minimize_reciprocal_sum_l751_75146


namespace NUMINAMATH_CALUDE_exists_number_with_nine_nines_squared_l751_75162

theorem exists_number_with_nine_nines_squared : ∃ n : ℕ, 
  ∃ k : ℕ, n^2 = 999999999 * 10^k + m ∧ m < 10^k :=
sorry

end NUMINAMATH_CALUDE_exists_number_with_nine_nines_squared_l751_75162


namespace NUMINAMATH_CALUDE_disjunction_not_implies_both_true_l751_75151

theorem disjunction_not_implies_both_true :
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by sorry

end NUMINAMATH_CALUDE_disjunction_not_implies_both_true_l751_75151


namespace NUMINAMATH_CALUDE_log21_not_calculable_l751_75134

-- Define the given logarithm values
def log5 : ℝ := 0.6990
def log7 : ℝ := 0.8451

-- Define a function to represent the ability to calculate a logarithm
def can_calculate (x : ℝ) : Prop := ∃ (a b : ℝ), x = a * log5 + b * log7

-- Theorem stating that log 21 cannot be calculated directly
theorem log21_not_calculable : ¬(can_calculate (Real.log 21)) :=
sorry

end NUMINAMATH_CALUDE_log21_not_calculable_l751_75134


namespace NUMINAMATH_CALUDE_number_equation_solution_l751_75196

theorem number_equation_solution :
  ∃ x : ℝ, x - (1002 / 20.04) = 1295 ∧ x = 1345 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l751_75196


namespace NUMINAMATH_CALUDE_brothers_ratio_l751_75177

theorem brothers_ratio (aaron_brothers bennett_brothers : ℕ) 
  (h1 : aaron_brothers = 4) 
  (h2 : bennett_brothers = 6) : 
  (bennett_brothers : ℚ) / aaron_brothers = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ratio_l751_75177


namespace NUMINAMATH_CALUDE_sugar_cups_in_lemonade_l751_75156

theorem sugar_cups_in_lemonade (total_cups : ℕ) (sugar_ratio water_ratio : ℕ) : 
  total_cups = 84 → sugar_ratio = 1 → water_ratio = 2 → 
  (sugar_ratio * total_cups) / (sugar_ratio + water_ratio) = 28 := by
sorry

end NUMINAMATH_CALUDE_sugar_cups_in_lemonade_l751_75156


namespace NUMINAMATH_CALUDE_susan_spending_l751_75178

theorem susan_spending (initial_amount : ℝ) (h1 : initial_amount = 600) : 
  let after_clothes := initial_amount / 2
  let after_books := after_clothes / 2
  after_books = 150 := by
sorry

end NUMINAMATH_CALUDE_susan_spending_l751_75178


namespace NUMINAMATH_CALUDE_stream_speed_l751_75145

/-- Given a boat traveling downstream, this theorem proves the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 24 →
  downstream_distance = 84 →
  downstream_time = 3 →
  ∃ stream_speed : ℝ, stream_speed = 4 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l751_75145


namespace NUMINAMATH_CALUDE_matrix_det_minus_two_l751_75128

theorem matrix_det_minus_two (A : Matrix (Fin 2) (Fin 2) ℤ) :
  A = ![![9, 5], ![-3, 4]] →
  Matrix.det A - 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_matrix_det_minus_two_l751_75128


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l751_75126

theorem sqrt_sum_inequality (a b : ℝ) 
  (h1 : a + b = 1) 
  (h2 : (a + 1/2) * (b + 1/2) ≥ 0) : 
  Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l751_75126


namespace NUMINAMATH_CALUDE_election_votes_theorem_l751_75166

theorem election_votes_theorem (total_votes : ℕ) (winner_votes second_votes third_votes : ℕ) :
  (winner_votes : ℚ) = 45 / 100 * total_votes ∧
  (second_votes : ℚ) = 35 / 100 * total_votes ∧
  winner_votes = second_votes + 150 ∧
  winner_votes + second_votes + third_votes = total_votes →
  total_votes = 1500 ∧ winner_votes = 675 ∧ second_votes = 525 ∧ third_votes = 300 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l751_75166


namespace NUMINAMATH_CALUDE_banana_permutations_l751_75133

-- Define the word BANANA
def word : String := "BANANA"

-- Define the total number of letters
def total_letters : Nat := word.length

-- Define the number of As
def num_A : Nat := 3

-- Define the number of Ns
def num_N : Nat := 2

-- Theorem statement
theorem banana_permutations : 
  (Nat.factorial total_letters) / (Nat.factorial num_A * Nat.factorial num_N) = 60 :=
by sorry

end NUMINAMATH_CALUDE_banana_permutations_l751_75133


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l751_75179

theorem geometric_mean_of_4_and_16 (x : ℝ) :
  x ^ 2 = 4 * 16 → x = 8 ∨ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l751_75179


namespace NUMINAMATH_CALUDE_middle_school_students_l751_75168

theorem middle_school_students (elementary : ℕ) (middle : ℕ) : 
  elementary = 4 * middle - 3 →
  elementary + middle = 247 →
  middle = 50 := by
sorry

end NUMINAMATH_CALUDE_middle_school_students_l751_75168


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l751_75118

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 * x + 9) = 11 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l751_75118


namespace NUMINAMATH_CALUDE_fraction_simplification_l751_75123

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l751_75123


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l751_75171

/-- Given two squares ABCD and DEFG where CE = 14 and AG = 2, prove that the sum of their areas is 100 -/
theorem sum_of_square_areas (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 14 → a - b = 2 → a^2 + b^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l751_75171


namespace NUMINAMATH_CALUDE_roots_difference_abs_l751_75103

theorem roots_difference_abs (r₁ r₂ : ℝ) : 
  r₁^2 - 7*r₁ + 12 = 0 → 
  r₂^2 - 7*r₂ + 12 = 0 → 
  |r₁ - r₂| = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_difference_abs_l751_75103


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l751_75111

theorem min_value_of_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 3033 ≥ 3032 :=
sorry

theorem equality_achieved : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 3033 = 3032 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l751_75111


namespace NUMINAMATH_CALUDE_polynomial_identity_l751_75164

theorem polynomial_identity (g : ℝ → ℝ) (h : ∀ x, g (x^2 + 2) = x^4 + 6*x^2 + 4) :
  ∀ x, g (x^2 - 2) = x^4 - 2*x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l751_75164


namespace NUMINAMATH_CALUDE_function_properties_l751_75104

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := 3 * (a * x^3 + b * x^2)

/-- The derivative of f(x) with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 9 * a * x^2 + 6 * b * x

theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x ≤ f a b 1) ∧
    (f a b 1 = 3) ∧
    (f_derivative a b 1 = 0) ∧
    (a = -2) ∧
    (b = 3) ∧
    (∀ x ∈ Set.Icc (-1) 3, f a b x ≤ 15) ∧
    (∃ x ∈ Set.Icc (-1) 3, f a b x = 15) ∧
    (∀ x ∈ Set.Icc (-1) 3, f a b x ≥ -81) ∧
    (∃ x ∈ Set.Icc (-1) 3, f a b x = -81) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l751_75104


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l751_75114

theorem arithmetic_sequence_problem (n : ℕ) : 
  let a₁ : ℤ := 1
  let d : ℤ := 3
  let aₙ : ℤ := a₁ + (n - 1) * d
  aₙ = 298 → n = 100 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l751_75114


namespace NUMINAMATH_CALUDE_sqrt_rational_sum_l751_75112

theorem sqrt_rational_sum (a b r : ℚ) (h : Real.sqrt a + Real.sqrt b = r) :
  ∃ (c d : ℚ), Real.sqrt a = c ∧ Real.sqrt b = d := by
  sorry

end NUMINAMATH_CALUDE_sqrt_rational_sum_l751_75112


namespace NUMINAMATH_CALUDE_change_received_l751_75136

def skirt_price : ℝ := 13
def blouse_price : ℝ := 6
def shoes_price : ℝ := 25
def handbag_price : ℝ := 35
def handbag_discount_rate : ℝ := 0.1
def coupon_discount : ℝ := 5
def amount_paid : ℝ := 150

def total_cost : ℝ := 2 * skirt_price + 3 * blouse_price + shoes_price + handbag_price

def discounted_handbag_price : ℝ := handbag_price * (1 - handbag_discount_rate)

def total_cost_after_discounts : ℝ := 
  2 * skirt_price + 3 * blouse_price + shoes_price + discounted_handbag_price - coupon_discount

theorem change_received : 
  amount_paid - total_cost_after_discounts = 54.5 := by sorry

end NUMINAMATH_CALUDE_change_received_l751_75136


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l751_75132

theorem isosceles_triangle_perimeter : ∀ (a b : ℝ),
  a^2 - 6*a + 5 = 0 →
  b^2 - 6*b + 5 = 0 →
  a ≠ b →
  (a + a + b = 11 ∨ b + b + a = 11) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l751_75132


namespace NUMINAMATH_CALUDE_semicircle_area_l751_75174

theorem semicircle_area (diameter : ℝ) (h : diameter = 3) : 
  let radius : ℝ := diameter / 2
  let semicircle_area : ℝ := (π * radius^2) / 2
  semicircle_area = 9 * π / 8 := by
sorry

end NUMINAMATH_CALUDE_semicircle_area_l751_75174


namespace NUMINAMATH_CALUDE_spiral_grid_third_row_sum_l751_75135

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the spiral grid -/
def SpiralGrid := Position → ℕ

/-- Creates a spiral grid with the given dimensions -/
def create_spiral_grid (n : ℕ) : SpiralGrid :=
  sorry

/-- Returns the numbers in a given row of the grid -/
def numbers_in_row (grid : SpiralGrid) (row : ℕ) : List ℕ :=
  sorry

theorem spiral_grid_third_row_sum :
  let grid := create_spiral_grid 17
  let third_row_numbers := numbers_in_row grid 3
  let min_number := third_row_numbers.minimum?
  let max_number := third_row_numbers.maximum?
  ∀ min max, min_number = some min → max_number = some max →
    min + max = 577 := by
  sorry

end NUMINAMATH_CALUDE_spiral_grid_third_row_sum_l751_75135


namespace NUMINAMATH_CALUDE_stamps_collection_theorem_l751_75165

def kylie_stamps : ℕ := 34
def nelly_stamps_difference : ℕ := 44

def total_stamps : ℕ := kylie_stamps + (kylie_stamps + nelly_stamps_difference)

theorem stamps_collection_theorem : total_stamps = 112 := by
  sorry

end NUMINAMATH_CALUDE_stamps_collection_theorem_l751_75165


namespace NUMINAMATH_CALUDE_exponential_function_property_l751_75185

theorem exponential_function_property (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → (fun x => a^x) (x + y) = (fun x => a^x) x * (fun x => a^x) y :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l751_75185


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l751_75139

theorem rectangular_field_dimensions (m : ℝ) : 
  (2 * m + 9) * (m - 4) = 88 → m = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l751_75139


namespace NUMINAMATH_CALUDE_f_properties_l751_75122

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.tan x

theorem f_properties :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (x + 2 * Real.pi) = f x) ∧
  (∀ x, f (Real.pi - x) = f (Real.pi + x)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l751_75122


namespace NUMINAMATH_CALUDE_length_MN_is_six_l751_75119

-- Define the points
variable (A B C D M N : ℝ)

-- Define the conditions
axiom on_segment : A < C ∧ C < D ∧ D < B
axiom midpoint_M : M = (A + C) / 2
axiom midpoint_N : N = (D + B) / 2
axiom length_AB : B - A = 10
axiom length_CD : D - C = 2

-- Theorem statement
theorem length_MN_is_six : N - M = 6 := by sorry

end NUMINAMATH_CALUDE_length_MN_is_six_l751_75119


namespace NUMINAMATH_CALUDE_sum_of_greater_is_greater_l751_75186

theorem sum_of_greater_is_greater (a b c d : ℝ) : a > b → c > d → a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_of_greater_is_greater_l751_75186


namespace NUMINAMATH_CALUDE_tin_addition_theorem_l751_75188

/-- Proves that adding 1.5 kg of pure tin to a 12 kg alloy containing 45% copper 
    will result in a new alloy containing 40% copper. -/
theorem tin_addition_theorem (initial_mass : ℝ) (initial_copper_percentage : ℝ) 
    (final_copper_percentage : ℝ) (tin_added : ℝ) : 
    initial_mass = 12 →
    initial_copper_percentage = 0.45 →
    final_copper_percentage = 0.4 →
    tin_added = 1.5 →
    initial_mass * initial_copper_percentage = 
    final_copper_percentage * (initial_mass + tin_added) := by
  sorry

#check tin_addition_theorem

end NUMINAMATH_CALUDE_tin_addition_theorem_l751_75188


namespace NUMINAMATH_CALUDE_sin_theta_value_l751_75113

theorem sin_theta_value (θ : ℝ) (h : Real.cos (π / 4 - θ / 2) = 2 / 3) : 
  Real.sin θ = -1 / 9 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_value_l751_75113


namespace NUMINAMATH_CALUDE_arrangements_with_A_or_B_at_ends_eq_84_l751_75167

/-- The number of ways to arrange n distinct objects --/
def permutations (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange 5 people in a row with at least one of A or B at the ends --/
def arrangements_with_A_or_B_at_ends : ℕ :=
  permutations 5 - (3 * 2 * permutations 3)

theorem arrangements_with_A_or_B_at_ends_eq_84 :
  arrangements_with_A_or_B_at_ends = 84 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_A_or_B_at_ends_eq_84_l751_75167


namespace NUMINAMATH_CALUDE_max_attempts_l751_75195

/-- The number of unique arrangements of a four-digit number containing one 2, one 9, and two 6s -/
def password_arrangements : ℕ := sorry

/-- The maximum number of attempts needed to find the correct password -/
theorem max_attempts : password_arrangements = 12 := by sorry

end NUMINAMATH_CALUDE_max_attempts_l751_75195


namespace NUMINAMATH_CALUDE_chord_length_circle_line_l751_75147

/-- The chord length cut by a circle from a line -/
theorem chord_length_circle_line (r : ℝ) (a b c : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | ∃ t, x = a + b*t ∧ y = c + t}
  let chord_length := 2 * Real.sqrt (r^2 - (a^2 + b^2 - 2*a*c + c^2) / (b^2 + 1))
  r = 3 ∧ a = 1 ∧ b = 2 ∧ c = 2 → chord_length = 12 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_chord_length_circle_line_l751_75147


namespace NUMINAMATH_CALUDE_triangle_area_l751_75182

/-- The area of a right triangle with vertices at (0, 0), (0, 10), and (-10, 0) is 50 square units,
    given that the points (-3, 7) and (-7, 3) lie on its hypotenuse. -/
theorem triangle_area : 
  let p1 : ℝ × ℝ := (-3, 7)
  let p2 : ℝ × ℝ := (-7, 3)
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (0, 10)
  let v3 : ℝ × ℝ := (-10, 0)
  (p1.1 - p2.1) / (p1.2 - p2.2) = 1 →  -- Slope of the line through p1 and p2 is 1
  (∃ t : ℝ, v2 = p1 + t • (1, 1)) →  -- v2 lies on the line through p1 with slope 1
  (∃ t : ℝ, v3 = p2 + t • (1, 1)) →  -- v3 lies on the line through p2 with slope 1
  (1/2) * (v2.2 - v1.2) * (v1.1 - v3.1) = 50 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_l751_75182


namespace NUMINAMATH_CALUDE_common_point_linear_functions_l751_75102

/-- Three linear functions with a common point -/
theorem common_point_linear_functions
  (a b c d : ℝ)
  (h1 : a ≠ b)
  (h2 : ∃ (x y : ℝ), (y = a * x + a) ∧ (y = b * x + b) ∧ (y = c * x + d)) :
  c = d :=
sorry

end NUMINAMATH_CALUDE_common_point_linear_functions_l751_75102


namespace NUMINAMATH_CALUDE_triangle_yz_length_l751_75160

/-- Given a triangle XYZ where cos(2X-Z) + sin(X+Z) = 2 and XY = 6, prove that YZ = 3 -/
theorem triangle_yz_length (X Y Z : ℝ) (h1 : 0 < X ∧ 0 < Y ∧ 0 < Z)
  (h2 : X + Y + Z = π) (h3 : Real.cos (2*X - Z) + Real.sin (X + Z) = 2) (h4 : 6 = 6) : 
  3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_yz_length_l751_75160


namespace NUMINAMATH_CALUDE_tank_capacity_l751_75138

theorem tank_capacity (C : ℝ) 
  (h1 : (3/4 : ℝ) * C + 8 = (7/8 : ℝ) * C) : C = 64 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l751_75138


namespace NUMINAMATH_CALUDE_smallest_square_enclosing_circle_l751_75190

theorem smallest_square_enclosing_circle (r : ℝ) (h : r = 5) : 
  (2 * r) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_enclosing_circle_l751_75190


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l751_75127

theorem difference_of_squares_special_case (m : ℝ) : 
  (2 * m + 1/2) * (2 * m - 1/2) = 4 * m^2 - 1/4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l751_75127


namespace NUMINAMATH_CALUDE_probability_sum_10_l751_75157

def die_faces : Nat := 6

def total_outcomes : Nat := die_faces * die_faces

def favorable_outcomes : Nat := 3 * 2 - 1

theorem probability_sum_10 : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_sum_10_l751_75157


namespace NUMINAMATH_CALUDE_chess_pieces_count_l751_75172

theorem chess_pieces_count (black_pieces : ℕ) (prob_black : ℚ) (white_pieces : ℕ) : 
  black_pieces = 6 → 
  prob_black = 1 / 5 → 
  (black_pieces : ℚ) / ((black_pieces : ℚ) + (white_pieces : ℚ)) = prob_black →
  white_pieces = 24 := by
sorry

end NUMINAMATH_CALUDE_chess_pieces_count_l751_75172


namespace NUMINAMATH_CALUDE_divisibility_criterion_l751_75197

theorem divisibility_criterion (A m k : ℕ) (h_pos : A > 0) (h_m_pos : m > 0) (h_k_pos : k > 0) :
  let g := k * m + 1
  let remainders : List ℕ := sorry
  let sum_remainders := remainders.sum
  (A % m = 0) ↔ (sum_remainders % m = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l751_75197


namespace NUMINAMATH_CALUDE_possible_x_values_l751_75125

def M (x : ℝ) : Set ℝ := {3, 9, 3*x}
def N (x : ℝ) : Set ℝ := {3, x^2}

theorem possible_x_values :
  ∀ x : ℝ, N x ⊆ M x → x = -3 ∨ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_possible_x_values_l751_75125


namespace NUMINAMATH_CALUDE_remainder_set_different_l751_75184

theorem remainder_set_different (a b c : ℤ) 
  (ha : 0 < a ∧ a < c - 1) 
  (hb : 1 < b ∧ b < c) : 
  let r : ℤ → ℤ := λ k => (k * b) % c
  (∀ k, 0 ≤ k ∧ k ≤ a → 0 ≤ r k ∧ r k < c) →
  {k : ℤ | 0 ≤ k ∧ k ≤ a}.image r ≠ {k : ℤ | 0 ≤ k ∧ k ≤ a} := by
  sorry

end NUMINAMATH_CALUDE_remainder_set_different_l751_75184


namespace NUMINAMATH_CALUDE_M_definition_sum_of_digits_M_l751_75170

def M : ℕ := sorry

-- M is the smallest positive integer divisible by every positive integer less than 8
theorem M_definition : 
  M > 0 ∧ 
  (∀ k : ℕ, k > 0 → k < 8 → M % k = 0) ∧
  (∀ n : ℕ, n > 0 → (∀ k : ℕ, k > 0 → k < 8 → n % k = 0) → n ≥ M) :=
sorry

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Theorem stating that the sum of digits of M is 6
theorem sum_of_digits_M : sum_of_digits M = 6 :=
sorry

end NUMINAMATH_CALUDE_M_definition_sum_of_digits_M_l751_75170


namespace NUMINAMATH_CALUDE_equation_roots_l751_75187

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 - (x - 3)
  (f 3 = 0 ∧ f 4 = 0) ∧ ∀ x : ℝ, f x = 0 → (x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l751_75187


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l751_75192

/-- A geometric sequence with its sum function -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1  -- Geometric sequence property
  sum_property : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))  -- Sum formula

/-- Theorem: For a geometric sequence with S_5 = 3 and S_10 = 9, S_15 = 21 -/
theorem geometric_sequence_sum (seq : GeometricSequence) 
  (h1 : seq.S 5 = 3) 
  (h2 : seq.S 10 = 9) : 
  seq.S 15 = 21 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l751_75192


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l751_75105

/-- Given two lines that are perpendicular, prove that the value of m is 1/2 -/
theorem perpendicular_lines_m_value (m : ℝ) : 
  (∀ x y : ℝ, x - m * y + 2 * m = 0 ∨ x + 2 * y - m = 0) → 
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ - m * y₁ + 2 * m = 0 → 
    x₂ + 2 * y₂ - m = 0 → 
    (x₁ - x₂) * (y₁ - y₂) = 0) →
  m = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l751_75105


namespace NUMINAMATH_CALUDE_nursery_paintable_area_l751_75152

/-- Calculates the total paintable wall area for three identical rooms -/
def totalPaintableArea (length width height : ℝ) (unpaintableArea : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let paintableAreaPerRoom := wallArea - unpaintableArea
  3 * paintableAreaPerRoom

/-- Theorem stating that the total paintable area for three rooms with given dimensions is 1200 sq ft -/
theorem nursery_paintable_area :
  totalPaintableArea 14 11 9 50 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_nursery_paintable_area_l751_75152


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l751_75143

/-- The x-coordinate of the intersection point of two linear functions -/
theorem intersection_point_x_coordinate (k b : ℝ) (h : k ≠ b) : 
  ∃ x : ℝ, k * x + b = b * x + k ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l751_75143


namespace NUMINAMATH_CALUDE_number_of_boys_l751_75193

theorem number_of_boys (total_students : ℕ) (boys_fraction : ℚ) (boys_count : ℕ) : 
  total_students = 12 →
  boys_fraction = 2/3 →
  boys_count = (total_students : ℚ) * boys_fraction →
  boys_count = 8 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l751_75193


namespace NUMINAMATH_CALUDE_tan_100_degrees_l751_75101

theorem tan_100_degrees (k : ℝ) (h : Real.sin (-(80 * π / 180)) = k) :
  Real.tan ((100 * π) / 180) = k / Real.sqrt (1 - k^2) := by sorry

end NUMINAMATH_CALUDE_tan_100_degrees_l751_75101


namespace NUMINAMATH_CALUDE_field_ratio_proof_l751_75198

/-- Proves that for a rectangular field with length 24 meters and width 13.5 meters,
    the ratio of twice the width to the length is 9:8. -/
theorem field_ratio_proof (length width : ℝ) : 
  length = 24 → width = 13.5 → (2 * width) / length = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_proof_l751_75198


namespace NUMINAMATH_CALUDE_prob_two_of_three_suits_l751_75124

/-- The probability of drawing a specific suit from a standard 52-card deck -/
def prob_suit : ℚ := 1/4

/-- The number of cards drawn -/
def num_draws : ℕ := 6

/-- The number of desired cards for each suit (hearts, diamonds, clubs) -/
def num_each_suit : ℕ := 2

/-- The probability of drawing exactly two hearts, two diamonds, and two clubs
    when drawing six cards with replacement from a standard 52-card deck -/
theorem prob_two_of_three_suits : 
  (num_draws.choose num_each_suit * num_draws.choose num_each_suit * num_draws.choose num_each_suit) *
  (prob_suit ^ num_each_suit * prob_suit ^ num_each_suit * prob_suit ^ num_each_suit) = 90/4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_of_three_suits_l751_75124


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l751_75130

theorem complex_number_quadrant (z : ℂ) (h : (2 - Complex.I) * z = Complex.I) :
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l751_75130


namespace NUMINAMATH_CALUDE_election_votes_l751_75173

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (52 : ℚ) / 100 * total_votes - (48 : ℚ) / 100 * total_votes = 288) : 
  ((52 : ℚ) / 100 * total_votes : ℚ).floor = 3744 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l751_75173


namespace NUMINAMATH_CALUDE_inheritance_calculation_l751_75154

/-- The original inheritance amount -/
def inheritance : ℝ := sorry

/-- The state tax rate -/
def state_tax_rate : ℝ := 0.15

/-- The federal tax rate -/
def federal_tax_rate : ℝ := 0.25

/-- The total tax paid -/
def total_tax_paid : ℝ := 18000

theorem inheritance_calculation : 
  state_tax_rate * inheritance + 
  federal_tax_rate * (1 - state_tax_rate) * inheritance = 
  total_tax_paid := by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l751_75154


namespace NUMINAMATH_CALUDE_smaller_area_with_center_l751_75149

/-- Represents a circular sector with a central angle of 60 degrees -/
structure Sector60 where
  radius : ℝ
  center : ℝ × ℝ

/-- Represents the line that cuts the sector -/
structure CuttingLine where
  slope : ℝ
  intercept : ℝ

/-- Represents the two parts after cutting the sector -/
structure SectorParts where
  part_with_center : Set (ℝ × ℝ)
  other_part : Set (ℝ × ℝ)

/-- Function to cut the sector -/
def cut_sector (s : Sector60) (l : CuttingLine) : SectorParts := sorry

/-- Function to calculate perimeter of a part -/
def perimeter (part : Set (ℝ × ℝ)) : ℝ := sorry

/-- Function to calculate area of a part -/
def area (part : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem smaller_area_with_center (s : Sector60) :
  ∃ (l : CuttingLine),
    let parts := cut_sector s l
    perimeter parts.part_with_center = perimeter parts.other_part →
    area parts.part_with_center < area parts.other_part :=
  sorry

end NUMINAMATH_CALUDE_smaller_area_with_center_l751_75149


namespace NUMINAMATH_CALUDE_knitting_productivity_l751_75163

theorem knitting_productivity (girl1_work_time girl1_break_time girl2_work_time girl2_break_time : ℕ) 
  (h1 : girl1_work_time = 5)
  (h2 : girl1_break_time = 1)
  (h3 : girl2_work_time = 7)
  (h4 : girl2_break_time = 1)
  : (girl1_work_time * (girl1_work_time + girl1_break_time)) / 
    (girl2_work_time * (girl2_work_time + girl2_break_time)) = 20 / 21 :=
by sorry

end NUMINAMATH_CALUDE_knitting_productivity_l751_75163


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l751_75189

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x ≥ a^2 - a} = {a : ℝ | -1 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l751_75189


namespace NUMINAMATH_CALUDE_problem_solution_l751_75159

theorem problem_solution : ((-4)^2) * (((-1)^2023) + (3/4) + ((-1/2)^3)) = -6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l751_75159


namespace NUMINAMATH_CALUDE_specific_figure_area_l751_75161

/-- A fifteen-sided figure drawn on a 1 cm × 1 cm graph paper -/
structure FifteenSidedFigure where
  /-- The number of full unit squares within the figure -/
  full_squares : ℕ
  /-- The number of rectangles within the figure -/
  rectangles : ℕ
  /-- The width of each rectangle in cm -/
  rectangle_width : ℝ
  /-- The height of each rectangle in cm -/
  rectangle_height : ℝ
  /-- The figure has fifteen sides -/
  sides : ℕ
  sides_eq : sides = 15

/-- The area of the fifteen-sided figure in cm² -/
def figure_area (f : FifteenSidedFigure) : ℝ :=
  f.full_squares + f.rectangles * f.rectangle_width * f.rectangle_height

/-- Theorem stating that the area of the specific fifteen-sided figure is 15 cm² -/
theorem specific_figure_area :
  ∃ f : FifteenSidedFigure, figure_area f = 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_figure_area_l751_75161


namespace NUMINAMATH_CALUDE_solve_equation_l751_75100

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (n : ℝ) : Prop := (2 : ℂ) / (1 - i) = 1 + n * i

-- State the theorem
theorem solve_equation : ∃ (n : ℝ), equation n ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l751_75100


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l751_75109

/-- Given a geometric sequence {a_n} where a_1+1, a_3+4, a_5+7 form an arithmetic sequence,
    the common difference of this arithmetic sequence is 3. -/
theorem geometric_arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d) :
  ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l751_75109


namespace NUMINAMATH_CALUDE_jenny_activities_alignment_l751_75194

def dance_interval : ℕ := 6
def karate_interval : ℕ := 12
def swimming_interval : ℕ := 15
def painting_interval : ℕ := 20
def library_interval : ℕ := 18
def sick_days : ℕ := 7

def next_alignment_day : ℕ := 187

theorem jenny_activities_alignment :
  let intervals := [dance_interval, karate_interval, swimming_interval, painting_interval, library_interval]
  let lcm_intervals := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm dance_interval karate_interval) swimming_interval) painting_interval) library_interval
  next_alignment_day = lcm_intervals + sick_days := by
  sorry

end NUMINAMATH_CALUDE_jenny_activities_alignment_l751_75194


namespace NUMINAMATH_CALUDE_new_average_after_removal_l751_75175

theorem new_average_after_removal (numbers : List ℝ) : 
  numbers.length = 12 → 
  numbers.sum / numbers.length = 90 → 
  80 ∈ numbers → 
  84 ∈ numbers → 
  let remaining := numbers.filter (λ x => x ≠ 80 ∧ x ≠ 84)
  remaining.sum / remaining.length = 91.6 := by
sorry

end NUMINAMATH_CALUDE_new_average_after_removal_l751_75175


namespace NUMINAMATH_CALUDE_tomato_pick_ratio_l751_75150

/-- Represents the number of tomatoes picked in each week and the remaining tomatoes -/
structure TomatoPicks where
  initial : ℕ
  first_week : ℕ
  second_week : ℕ
  third_week : ℕ
  remaining : ℕ

/-- Calculates the ratio of tomatoes picked in the third week to the second week -/
def pick_ratio (picks : TomatoPicks) : ℚ :=
  picks.third_week / picks.second_week

/-- Theorem stating the ratio of tomatoes picked in the third week to the second week -/
theorem tomato_pick_ratio : 
  ∀ (picks : TomatoPicks), 
  picks.initial = 100 ∧ 
  picks.first_week = picks.initial / 4 ∧
  picks.second_week = 20 ∧
  picks.remaining = 15 ∧
  picks.initial = picks.first_week + picks.second_week + picks.third_week + picks.remaining
  → pick_ratio picks = 2 := by
  sorry

#check tomato_pick_ratio

end NUMINAMATH_CALUDE_tomato_pick_ratio_l751_75150


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l751_75199

theorem complex_fraction_simplification :
  (7 : ℂ) + 18 * I / (3 - 4 * I) = -(51 : ℚ) / 25 + (82 : ℚ) / 25 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l751_75199


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l751_75106

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem rectangular_solid_surface_area 
  (l w h : ℕ) 
  (prime_l : is_prime l) 
  (prime_w : is_prime w) 
  (prime_h : is_prime h) 
  (volume_eq : l * w * h = 1001) : 
  2 * (l * w + w * h + h * l) = 622 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l751_75106


namespace NUMINAMATH_CALUDE_china_population_in_scientific_notation_l751_75110

/-- Definition of scientific notation -/
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10

/-- The population of China according to the sixth national census -/
def china_population : ℝ := 1370540000

/-- The scientific notation representation of China's population -/
def china_population_scientific : ℝ := 1.37054 * (10 : ℝ) ^ 9

theorem china_population_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n china_population ∧
  china_population_scientific = a * (10 : ℝ) ^ n :=
sorry

end NUMINAMATH_CALUDE_china_population_in_scientific_notation_l751_75110


namespace NUMINAMATH_CALUDE_parabola_coefficient_l751_75180

/-- Given a parabola y = ax^2 + bx + c with vertex (q/2, q/2) and y-intercept (0, -2q),
    where q ≠ 0, prove that b = 10 -/
theorem parabola_coefficient (a b c q : ℝ) (h_q : q ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (q/2, q/2) = (-(b / (2 * a)), a * (-(b / (2 * a)))^2 + b * (-(b / (2 * a))) + c) →
  c = -2 * q →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l751_75180


namespace NUMINAMATH_CALUDE_zhang_hong_weight_estimate_l751_75115

/-- Regression equation for weight based on height -/
def weight_estimate (height : ℝ) : ℝ := 0.72 * height - 58.2

/-- Age range for which the regression equation is valid -/
def valid_age_range : Set ℝ := Set.Icc 18 38

theorem zhang_hong_weight_estimate :
  20 ∈ valid_age_range →
  weight_estimate 178 = 69.96 := by
  sorry

end NUMINAMATH_CALUDE_zhang_hong_weight_estimate_l751_75115


namespace NUMINAMATH_CALUDE_meals_for_adults_l751_75183

/-- The number of meals initially available for adults -/
def A : ℕ := 18

/-- The number of children that can be fed with all the meals -/
def C : ℕ := 90

/-- Theorem stating that A is the correct number of meals initially available for adults -/
theorem meals_for_adults : 
  (∀ x : ℕ, x * (C / A) = 72 → x = 14) ∧ 
  (A : ℚ) = C / (72 / 14) :=
sorry

end NUMINAMATH_CALUDE_meals_for_adults_l751_75183


namespace NUMINAMATH_CALUDE_marikas_fathers_age_l751_75158

/-- Given that Marika was 10 years old in 2006 and her father's age was five times her age,
    prove that the year when Marika's father's age will be twice her age is 2036. -/
theorem marikas_fathers_age (marika_birth_year : ℕ) (father_birth_year : ℕ) : 
  marika_birth_year = 1996 →
  father_birth_year = 1956 →
  ∃ (year : ℕ), year = 2036 ∧ 
    (year - father_birth_year) = 2 * (year - marika_birth_year) :=
by sorry

end NUMINAMATH_CALUDE_marikas_fathers_age_l751_75158


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l751_75144

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q) (h_pos : a 1 > 0) :
  (increasing_sequence a → q > 0) ∧
  ¬(q > 0 → increasing_sequence a) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l751_75144


namespace NUMINAMATH_CALUDE_arccos_sin_three_l751_75181

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_three_l751_75181


namespace NUMINAMATH_CALUDE_tan_one_implies_sin_2a_minus_cos_sq_a_eq_half_l751_75153

theorem tan_one_implies_sin_2a_minus_cos_sq_a_eq_half (α : Real) 
  (h : Real.tan α = 1) : Real.sin (2 * α) - Real.cos α ^ 2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_one_implies_sin_2a_minus_cos_sq_a_eq_half_l751_75153


namespace NUMINAMATH_CALUDE_remainder_equivalence_l751_75191

theorem remainder_equivalence (x : ℤ) : x % 5 = 4 → x % 61 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_equivalence_l751_75191


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l751_75169

theorem polynomial_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + a₅*(1-x)^5) →
  (a₃ = -10 ∧ a₁ + a₃ + a₅ = -16) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l751_75169


namespace NUMINAMATH_CALUDE_dealer_gross_profit_l751_75107

-- Define the parameters
def purchase_price : ℝ := 150
def markup_rate : ℝ := 0.25
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05

-- Define the initial selling price
noncomputable def initial_selling_price : ℝ :=
  purchase_price / (1 - markup_rate)

-- Define the discounted price
noncomputable def discounted_price : ℝ :=
  initial_selling_price * (1 - discount_rate)

-- Define the final selling price (including tax)
noncomputable def final_selling_price : ℝ :=
  discounted_price * (1 + tax_rate)

-- Define the gross profit
noncomputable def gross_profit : ℝ :=
  final_selling_price - purchase_price

-- Theorem statement
theorem dealer_gross_profit :
  gross_profit = 19 := by sorry

end NUMINAMATH_CALUDE_dealer_gross_profit_l751_75107


namespace NUMINAMATH_CALUDE_range_of_S_l751_75117

theorem range_of_S (a b : ℝ) (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) :
  -2 ≤ (a + 1) * (b + 1) ∧ (a + 1) * (b + 1) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_S_l751_75117


namespace NUMINAMATH_CALUDE_remainder_theorem_l751_75108

theorem remainder_theorem (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l751_75108


namespace NUMINAMATH_CALUDE_total_students_correct_l751_75176

/-- The number of students who tried out for the school's trivia teams. -/
def total_students : ℕ := 64

/-- The number of students who didn't get picked for the team. -/
def not_picked : ℕ := 36

/-- The number of groups the picked students were divided into. -/
def num_groups : ℕ := 4

/-- The number of students in each group of picked students. -/
def students_per_group : ℕ := 7

/-- Theorem stating that the total number of students who tried out is correct. -/
theorem total_students_correct :
  total_students = not_picked + num_groups * students_per_group :=
by sorry

end NUMINAMATH_CALUDE_total_students_correct_l751_75176


namespace NUMINAMATH_CALUDE_baker_cakes_remaining_l751_75120

/-- Given the initial number of cakes, additional cakes made, and cakes sold,
    prove that the number of cakes remaining is equal to 67. -/
theorem baker_cakes_remaining 
  (initial_cakes : ℕ) 
  (additional_cakes : ℕ) 
  (sold_cakes : ℕ) 
  (h1 : initial_cakes = 62)
  (h2 : additional_cakes = 149)
  (h3 : sold_cakes = 144) :
  initial_cakes + additional_cakes - sold_cakes = 67 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_remaining_l751_75120


namespace NUMINAMATH_CALUDE_drawer_probability_l751_75137

theorem drawer_probability (shirts : ℕ) (shorts : ℕ) (socks : ℕ) :
  shirts = 6 →
  shorts = 7 →
  socks = 8 →
  let total := shirts + shorts + socks
  let favorable := Nat.choose shirts 2 * Nat.choose shorts 1 * Nat.choose socks 1
  let total_outcomes := Nat.choose total 4
  (favorable : ℚ) / total_outcomes = 56 / 399 := by
  sorry

end NUMINAMATH_CALUDE_drawer_probability_l751_75137


namespace NUMINAMATH_CALUDE_unique_divisible_by_101_l751_75141

theorem unique_divisible_by_101 : ∃! n : ℕ, 
  201300 ≤ n ∧ n < 201400 ∧ n % 101 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_101_l751_75141


namespace NUMINAMATH_CALUDE_parabola_directrix_l751_75142

/-- Given a parabola with equation x² = 4y, its directrix has equation y = -1 -/
theorem parabola_directrix (x y : ℝ) : x^2 = 4*y → (∃ (k : ℝ), k = -1 ∧ y = k) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l751_75142


namespace NUMINAMATH_CALUDE_shaded_area_regular_octagon_l751_75155

/-- The area of the shaded region in a regular octagon with side length 8 cm,
    formed by connecting the midpoints of consecutive sides. -/
theorem shaded_area_regular_octagon (s : ℝ) (h : s = 8) :
  let outer_area := 2 * (1 + Real.sqrt 2) * s^2
  let inner_side := s * (1 - Real.sqrt 2 / 2)
  let inner_area := 2 * (1 + Real.sqrt 2) * inner_side^2
  outer_area - inner_area = 128 * (1 + Real.sqrt 2) - 2 * (1 + Real.sqrt 2) * (8 - 4 * Real.sqrt 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_regular_octagon_l751_75155


namespace NUMINAMATH_CALUDE_paul_total_crayons_l751_75121

/-- The number of crayons Paul received for his birthday -/
def birthday_crayons : ℝ := 479.0

/-- The number of crayons Paul received at the end of the school year -/
def school_year_crayons : ℝ := 134.0

/-- The total number of crayons Paul has now -/
def total_crayons : ℝ := birthday_crayons + school_year_crayons

/-- Theorem stating that Paul's total number of crayons is 613.0 -/
theorem paul_total_crayons : total_crayons = 613.0 := by
  sorry

end NUMINAMATH_CALUDE_paul_total_crayons_l751_75121


namespace NUMINAMATH_CALUDE_largest_possible_z_value_l751_75131

theorem largest_possible_z_value (a b c z : ℂ) 
  (h1 : Complex.abs a = 2 * Complex.abs b)
  (h2 : Complex.abs a = 2 * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ 1 := by sorry

end NUMINAMATH_CALUDE_largest_possible_z_value_l751_75131


namespace NUMINAMATH_CALUDE_even_function_sum_a_b_l751_75140

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

-- Define the property of being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Main theorem
theorem even_function_sum_a_b :
  ∀ a b : ℝ,
  (∀ x, x ∈ Set.Icc (2 * a - 3) (4 - a) → f a b x = f a b (-x)) →
  a + b = 2 :=
by sorry

end NUMINAMATH_CALUDE_even_function_sum_a_b_l751_75140


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l751_75116

/-- The perimeter of a semicircle with radius 9 is approximately 46.27 units. -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |((9 * Real.pi + 18) : ℝ) - 46.27| < ε := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l751_75116


namespace NUMINAMATH_CALUDE_x_amount_proof_l751_75148

def total_amount : ℝ := 5000
def ratio_x : ℝ := 2
def ratio_y : ℝ := 8

theorem x_amount_proof :
  let total_ratio := ratio_x + ratio_y
  let amount_per_part := total_amount / total_ratio
  let x_amount := amount_per_part * ratio_x
  x_amount = 1000 := by sorry

end NUMINAMATH_CALUDE_x_amount_proof_l751_75148
