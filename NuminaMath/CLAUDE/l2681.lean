import Mathlib

namespace external_diagonals_condition_l2681_268170

/-- Represents the lengths of external diagonals of a right regular prism -/
structure ExternalDiagonals where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if the given lengths could be valid external diagonals of a right regular prism -/
def isValidExternalDiagonals (d : ExternalDiagonals) : Prop :=
  d.x^2 + d.y^2 > d.z^2 ∧ d.y^2 + d.z^2 > d.x^2 ∧ d.x^2 + d.z^2 > d.y^2

theorem external_diagonals_condition (d : ExternalDiagonals) :
  d.x > 0 ∧ d.y > 0 ∧ d.z > 0 → isValidExternalDiagonals d :=
by sorry

end external_diagonals_condition_l2681_268170


namespace largest_in_set_l2681_268102

def a : ℝ := -4

def S : Set ℝ := {-3 * a, 4 * a, 24 / a, a^2, 2 * a + 1, 1}

theorem largest_in_set : ∀ x ∈ S, x ≤ a^2 := by sorry

end largest_in_set_l2681_268102


namespace equation_is_linear_l2681_268194

/-- A linear equation in two variables is of the form Ax + By = C, where A, B, and C are constants, and A and B are not both zero. -/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (A B C : ℝ), (A ≠ 0 ∨ B ≠ 0) ∧ ∀ x y, f x y ↔ A * x + B * y = C

/-- The equation 3x - 1 = 2 - 5y is a linear equation in two variables. -/
theorem equation_is_linear : IsLinearEquationInTwoVariables (fun x y ↦ 3 * x - 1 = 2 - 5 * y) := by
  sorry

#check equation_is_linear

end equation_is_linear_l2681_268194


namespace sphere_radius_equals_cone_lateral_area_l2681_268181

theorem sphere_radius_equals_cone_lateral_area 
  (cone_height : ℝ) 
  (cone_base_radius : ℝ) 
  (sphere_radius : ℝ) :
  cone_height = 3 →
  cone_base_radius = 4 →
  (4 * Real.pi * sphere_radius^2) = (Real.pi * cone_base_radius * (cone_height^2 + cone_base_radius^2).sqrt) →
  sphere_radius = Real.sqrt 5 := by
  sorry

end sphere_radius_equals_cone_lateral_area_l2681_268181


namespace rectangles_in_5x4_grid_l2681_268158

/-- The number of rectangles in a row of length n -/
def rectangles_in_row (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of rectangles in an m x n grid -/
def total_rectangles (m n : ℕ) : ℕ :=
  m * rectangles_in_row n + n * rectangles_in_row m - m * n

theorem rectangles_in_5x4_grid :
  total_rectangles 5 4 = 24 := by
  sorry

end rectangles_in_5x4_grid_l2681_268158


namespace pies_cost_l2681_268199

theorem pies_cost (a b c d : ℕ) : 
  c = 2 * a →                           -- cherry pie costs the same as two apple pies
  b = 2 * d →                           -- blueberry pie costs the same as two damson pies
  c + 2 * d = a + 2 * b →               -- cherry pie and two damson pies cost the same as an apple pie and two blueberry pies
  a + b + c + d = 18                    -- total cost is £18
  := by sorry

end pies_cost_l2681_268199


namespace existence_of_inverse_solvable_problems_l2681_268140

/-- A mathematical problem that can be solved by first considering its inverse -/
structure InverseSolvableProblem where
  problem : Type
  inverse_problem : Type
  solve : inverse_problem → problem

/-- Theorem stating that there exist problems solvable by first solving their inverse -/
theorem existence_of_inverse_solvable_problems :
  ∃ (P : InverseSolvableProblem), True :=
  sorry

end existence_of_inverse_solvable_problems_l2681_268140


namespace not_perfect_square_floor_theorem_l2681_268171

theorem not_perfect_square_floor_theorem (A : ℕ) (h : ¬ ∃ k : ℕ, A = k ^ 2) :
  ∃ n : ℕ, A = ⌊(n : ℝ) + Real.sqrt n + 1/2⌋ := by
  sorry

end not_perfect_square_floor_theorem_l2681_268171


namespace log_equation_solution_l2681_268187

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 7 →
  x = 3 ^ (14 / 3) := by
sorry

end log_equation_solution_l2681_268187


namespace min_t_value_l2681_268197

theorem min_t_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) :
  (∀ t : ℝ, 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1/2) →
  t ≥ Real.sqrt 2 / 2 :=
by sorry

end min_t_value_l2681_268197


namespace birds_in_tree_l2681_268126

theorem birds_in_tree (initial_birds final_birds : ℕ) (h1 : initial_birds = 179) (h2 : final_birds = 217) :
  final_birds - initial_birds = 38 := by
sorry

end birds_in_tree_l2681_268126


namespace quadratic_roots_problem_l2681_268185

theorem quadratic_roots_problem (α β k : ℝ) : 
  (α^2 - α + k - 1 = 0) →
  (β^2 - β + k - 1 = 0) →
  (α^2 - 2*α - β = 4) →
  (k = -4) := by
sorry

end quadratic_roots_problem_l2681_268185


namespace odd_not_divides_power_plus_one_l2681_268141

theorem odd_not_divides_power_plus_one (n m : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ¬(n ∣ (m^(n-1) + 1)) := by
  sorry

end odd_not_divides_power_plus_one_l2681_268141


namespace price_per_dozen_eggs_l2681_268112

/-- Calculates the price per dozen eggs given the number of chickens, eggs per chicken per week,
    eggs per dozen, total revenue, and number of weeks. -/
theorem price_per_dozen_eggs 
  (num_chickens : ℕ) 
  (eggs_per_chicken_per_week : ℕ) 
  (eggs_per_dozen : ℕ) 
  (total_revenue : ℚ) 
  (num_weeks : ℕ) 
  (h1 : num_chickens = 46)
  (h2 : eggs_per_chicken_per_week = 6)
  (h3 : eggs_per_dozen = 12)
  (h4 : total_revenue = 552)
  (h5 : num_weeks = 8) :
  total_revenue / (num_chickens * eggs_per_chicken_per_week * num_weeks / eggs_per_dozen) = 3 := by
  sorry

end price_per_dozen_eggs_l2681_268112


namespace xe_exp_increasing_l2681_268162

/-- The function f(x) = xe^x is increasing for all x > 0 -/
theorem xe_exp_increasing (x : ℝ) (h : x > 0) :
  Monotone (fun x => x * Real.exp x) := by sorry

end xe_exp_increasing_l2681_268162


namespace quadratic_real_roots_l2681_268191

theorem quadratic_real_roots (k d : ℝ) (h : k ≠ 0) :
  (∃ x : ℝ, x^2 + k*x + k^2 + d = 0) ↔ d ≤ -3/4 * k^2 := by
sorry

end quadratic_real_roots_l2681_268191


namespace betty_boxes_l2681_268147

def total_oranges : ℕ := 24
def oranges_per_box : ℕ := 8

theorem betty_boxes : 
  total_oranges / oranges_per_box = 3 := by sorry

end betty_boxes_l2681_268147


namespace greatest_two_digit_product_12_l2681_268154

/-- A function that returns true if a number is a two-digit whole number --/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- A function that returns the product of digits of a two-digit number --/
def digitProduct (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- Theorem stating that 62 is the greatest two-digit number whose digits have a product of 12 --/
theorem greatest_two_digit_product_12 :
  ∀ n : ℕ, isTwoDigit n → digitProduct n = 12 → n ≤ 62 :=
sorry

end greatest_two_digit_product_12_l2681_268154


namespace kylie_picked_220_apples_l2681_268122

/-- The number of apples Kylie picked in the first hour -/
def first_hour_apples : ℕ := 66

/-- The number of apples Kylie picked in the second hour -/
def second_hour_apples : ℕ := 2 * first_hour_apples

/-- The number of apples Kylie picked in the third hour -/
def third_hour_apples : ℕ := first_hour_apples / 3

/-- The total number of apples Kylie picked -/
def total_apples : ℕ := first_hour_apples + second_hour_apples + third_hour_apples

/-- Theorem stating that the total number of apples Kylie picked is 220 -/
theorem kylie_picked_220_apples : total_apples = 220 := by
  sorry

end kylie_picked_220_apples_l2681_268122


namespace digit_difference_in_base_d_l2681_268123

/-- Given two digits A and B in base d > 7, if AB + BA = 202 in base d, then A - B = 2 in base d -/
theorem digit_difference_in_base_d (d : ℕ) (A B : ℕ) : 
  d > 7 →
  A < d →
  B < d →
  (A * d + B) + (B * d + A) = 2 * d^2 + 2 →
  A - B = 2 := by
  sorry

end digit_difference_in_base_d_l2681_268123


namespace banana_arrangements_l2681_268125

/-- The number of ways to arrange letters in a word -/
def arrange_letters (total : ℕ) (freq1 freq2 freq3 : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial freq1 * Nat.factorial freq2 * Nat.factorial freq3)

/-- Theorem: The number of arrangements of BANANA is 60 -/
theorem banana_arrangements :
  arrange_letters 6 1 3 2 = 60 := by
  sorry

end banana_arrangements_l2681_268125


namespace biology_score_calculation_l2681_268107

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 67
def average_score : ℕ := 75
def total_subjects : ℕ := 5

theorem biology_score_calculation :
  let known_scores_sum := math_score + science_score + social_studies_score + english_score
  let total_score := average_score * total_subjects
  total_score - known_scores_sum = 85 := by
  sorry

#check biology_score_calculation

end biology_score_calculation_l2681_268107


namespace expression_simplification_l2681_268151

theorem expression_simplification (x : ℝ) : 3*x + 4*x^2 + 2 - (9 - 3*x - 4*x^2) + Real.sin x = 8*x^2 + 6*x - 7 + Real.sin x := by
  sorry

end expression_simplification_l2681_268151


namespace equilateral_triangle_on_three_lines_l2681_268198

/-- A line in a plane --/
structure Line where
  -- Add necessary fields to represent a line

/-- An equilateral triangle --/
structure EquilateralTriangle where
  -- Add necessary fields to represent an equilateral triangle

/-- A point in a plane --/
structure Point where
  -- Add necessary fields to represent a point

/-- Checks if a point lies on a given line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  sorry

/-- Checks if a triangle is equilateral --/
def isEquilateralTriangle (t : EquilateralTriangle) : Prop :=
  sorry

/-- The main theorem --/
theorem equilateral_triangle_on_three_lines 
  (d₁ d₂ d₃ : Line) : 
  ∃ (t : EquilateralTriangle), 
    isEquilateralTriangle t ∧ 
    (∃ (p₁ p₂ p₃ : Point), 
      pointOnLine p₁ d₁ ∧ 
      pointOnLine p₂ d₂ ∧ 
      pointOnLine p₃ d₃ ∧ 
      -- Add conditions to relate p₁, p₂, p₃ to the vertices of t
      sorry) :=
by
  sorry


end equilateral_triangle_on_three_lines_l2681_268198


namespace decimal_sum_to_fraction_l2681_268157

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 := by
  sorry

end decimal_sum_to_fraction_l2681_268157


namespace least_number_for_divisibility_l2681_268134

theorem least_number_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((5918273 + y) % (41 * 71 * 139) = 0)) ∧ 
  ((5918273 + x) % (41 * 71 * 139) = 0) := by
  sorry

end least_number_for_divisibility_l2681_268134


namespace roots_of_polynomial_l2681_268120

def p (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 4

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 4) ∧
  (∀ x : ℝ, (x = -1 ∨ x = 1 ∨ x = 4) → (deriv p) x ≠ 0) :=
sorry

end roots_of_polynomial_l2681_268120


namespace blood_cell_count_l2681_268127

/-- Given two blood samples with a total of 7341 blood cells, where the first sample
    contains 4221 blood cells, prove that the second sample contains 3120 blood cells. -/
theorem blood_cell_count (total : ℕ) (first_sample : ℕ) (second_sample : ℕ) 
    (h1 : total = 7341)
    (h2 : first_sample = 4221)
    (h3 : total = first_sample + second_sample) : 
  second_sample = 3120 := by
  sorry

end blood_cell_count_l2681_268127


namespace sochi_puzzle_solution_l2681_268195

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Digit
  hundreds : Digit
  tens : Digit
  ones : Digit

/-- Convert a FourDigitNumber to a natural number -/
def FourDigitNumber.toNat (n : FourDigitNumber) : Nat :=
  1000 * n.thousands.val + 100 * n.hundreds.val + 10 * n.tens.val + n.ones.val

/-- Check if all digits in a FourDigitNumber are unique -/
def FourDigitNumber.uniqueDigits (n : FourDigitNumber) : Prop :=
  n.thousands ≠ n.hundreds ∧ n.thousands ≠ n.tens ∧ n.thousands ≠ n.ones ∧
  n.hundreds ≠ n.tens ∧ n.hundreds ≠ n.ones ∧
  n.tens ≠ n.ones

theorem sochi_puzzle_solution :
  ∃ (year sochi : FourDigitNumber),
    year.uniqueDigits ∧
    sochi.uniqueDigits ∧
    2014 + year.toNat = sochi.toNat :=
  sorry

end sochi_puzzle_solution_l2681_268195


namespace monotonic_intervals_and_comparison_l2681_268119

noncomputable def f (x : ℝ) : ℝ := 3 * Real.exp x + x^2
noncomputable def g (x : ℝ) : ℝ := 9*x - 1
noncomputable def φ (x : ℝ) : ℝ := x * Real.exp x + 4*x - f x

theorem monotonic_intervals_and_comparison :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < Real.log 2 → φ x₁ < φ x₂) ∧
  (∀ x₁ x₂, Real.log 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → φ x₁ > φ x₂) ∧
  (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → φ x₁ < φ x₂) ∧
  (∀ x, f x > g x) :=
by sorry

end monotonic_intervals_and_comparison_l2681_268119


namespace mikes_tire_spending_l2681_268193

/-- The problem of calculating Mike's spending on new tires -/
theorem mikes_tire_spending (total_spent : ℚ) (speaker_cost : ℚ) (tire_cost : ℚ) :
  total_spent = 224.87 →
  speaker_cost = 118.54 →
  tire_cost = total_spent - speaker_cost →
  tire_cost = 106.33 := by
  sorry

end mikes_tire_spending_l2681_268193


namespace library_visitors_proof_l2681_268108

/-- The total number of visitors to a library in a week -/
def total_visitors (monday : ℕ) (tuesday_multiplier : ℕ) (remaining_days : ℕ) (avg_remaining : ℕ) : ℕ :=
  monday + (tuesday_multiplier * monday) + (remaining_days * avg_remaining)

/-- Theorem stating that the total number of visitors to the library in a week is 250 -/
theorem library_visitors_proof : 
  total_visitors 50 2 5 20 = 250 := by
  sorry

end library_visitors_proof_l2681_268108


namespace parabola_intersection_theorem_l2681_268131

/-- Parabola type representing y^2 = ax -/
structure Parabola where
  a : ℝ
  hpos : a > 0

/-- Point type representing (x, y) coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing a line with slope k -/
structure Line where
  k : ℝ

def intersect_parabola_line (p : Parabola) (l : Line) : Point × Point := sorry

def extend_line (p1 p2 : Point) : Line := sorry

def slope_of_line (p1 p2 : Point) : ℝ := sorry

theorem parabola_intersection_theorem (p : Parabola) (m : Point) 
  (h_m : m.x = 4 ∧ m.y = 0) (l : Line) (k2 : ℝ) 
  (h_k : l.k = Real.sqrt 2 * k2) :
  let f := Point.mk (p.a / 4) 0
  let (a, b) := intersect_parabola_line p l
  let c := intersect_parabola_line p (extend_line a m)
  let d := intersect_parabola_line p (extend_line b m)
  slope_of_line c.1 d.1 = k2 → p.a = 8 * Real.sqrt 2 := by
  sorry

end parabola_intersection_theorem_l2681_268131


namespace not_perfect_square_l2681_268173

theorem not_perfect_square (a : ℤ) : a ≠ 0 → ¬∃ x : ℤ, a^2 + 4 = x^2 := by sorry

end not_perfect_square_l2681_268173


namespace cubic_inequality_solution_l2681_268186

theorem cubic_inequality_solution (x : ℝ) : x^3 - 9*x^2 > -27*x ↔ (0 < x ∧ x < 3) ∨ (x > 6) := by
  sorry

end cubic_inequality_solution_l2681_268186


namespace sin_B_value_max_perimeter_l2681_268172

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition (2a-c)cosB = bcosC -/
def triangle_condition (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C

/-- Theorem 1: If (2a-c)cosB = bcosC, then sinB = √3/2 -/
theorem sin_B_value (t : Triangle) (h : triangle_condition t) : 
  Real.sin t.B = Real.sqrt 3 / 2 := by sorry

/-- Theorem 2: If b = √7, then the maximum perimeter is 3√7 -/
theorem max_perimeter (t : Triangle) (h : t.b = Real.sqrt 7) :
  t.a + t.b + t.c ≤ 3 * Real.sqrt 7 := by sorry

end sin_B_value_max_perimeter_l2681_268172


namespace sum_of_integers_l2681_268178

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : a - b + 2*c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 5)
  (eq4 : d - a + b = 4) : 
  a + b + c + d = 20 := by
  sorry

end sum_of_integers_l2681_268178


namespace latin_essay_scores_l2681_268111

/-- The maximum score for the Latin essay --/
def max_score : ℕ := 20

/-- Michel's score --/
def michel_score : ℕ := 14

/-- Claude's score --/
def claude_score : ℕ := 6

/-- The average score --/
def average_score : ℚ := (michel_score + claude_score) / 2

theorem latin_essay_scores :
  michel_score > 0 ∧
  michel_score ≤ max_score ∧
  claude_score > 0 ∧
  claude_score ≤ max_score ∧
  michel_score > average_score ∧
  claude_score < average_score ∧
  michel_score - michel_score / 3 = 3 * (claude_score - claude_score / 3) :=
by sorry

end latin_essay_scores_l2681_268111


namespace wedge_volume_of_sphere_l2681_268180

/-- The volume of a wedge of a sphere -/
theorem wedge_volume_of_sphere (circumference : ℝ) (num_wedges : ℕ) : 
  circumference = 18 * Real.pi → 
  num_wedges = 6 → 
  (1 / num_wedges : ℝ) * (4 / 3 : ℝ) * Real.pi * (circumference / (2 * Real.pi))^3 = 162 * Real.pi := by
  sorry

end wedge_volume_of_sphere_l2681_268180


namespace gear_system_rotation_l2681_268168

/-- Represents the rotation direction of a gear -/
inductive Direction
| Clockwise
| Counterclockwise

/-- Represents a system of gears -/
structure GearSystem :=
  (n : ℕ)  -- number of gears

/-- Returns the direction of the i-th gear in the system -/
def gear_direction (sys : GearSystem) (i : ℕ) : Direction :=
  if i % 2 = 0 then Direction.Counterclockwise else Direction.Clockwise

/-- Checks if the gear system can rotate -/
def can_rotate (sys : GearSystem) : Prop :=
  sys.n % 2 = 0

theorem gear_system_rotation (sys : GearSystem) :
  can_rotate sys ↔ sys.n % 2 = 0 :=
sorry

end gear_system_rotation_l2681_268168


namespace cubic_minus_linear_factorization_l2681_268184

theorem cubic_minus_linear_factorization (x : ℝ) : x^3 - x = x*(x+1)*(x-1) := by
  sorry

end cubic_minus_linear_factorization_l2681_268184


namespace stone_length_calculation_l2681_268114

/-- Calculates the length of stones used to pave a hall -/
theorem stone_length_calculation (hall_length hall_width : ℕ) 
  (stone_width num_stones : ℕ) (stone_length : ℚ) : 
  hall_length = 36 ∧ 
  hall_width = 15 ∧ 
  stone_width = 5 ∧ 
  num_stones = 5400 ∧
  (hall_length * 10 * hall_width * 10 : ℚ) = stone_length * stone_width * num_stones →
  stone_length = 2 := by
sorry

end stone_length_calculation_l2681_268114


namespace ellipse_dot_product_range_l2681_268174

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

/-- Definition of point M -/
def M : ℝ × ℝ := (0, 2)

/-- Definition of the dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- Statement of the theorem -/
theorem ellipse_dot_product_range :
  ∀ (P Q : ℝ × ℝ),
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  ∃ (k : ℝ), P.2 - M.2 = k * (P.1 - M.1) ∧ Q.2 - M.2 = k * (Q.1 - M.1) →
  -20 ≤ dot_product P Q + dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2) ∧
  dot_product P Q + dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2) ≤ -52/3 :=
sorry

end ellipse_dot_product_range_l2681_268174


namespace anthony_total_pencils_l2681_268190

/-- The total number of pencils Anthony has after receiving more from Kathryn -/
theorem anthony_total_pencils (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 9 → received = 56 → total = initial + received → total = 65 := by
  sorry

end anthony_total_pencils_l2681_268190


namespace min_value_sum_l2681_268150

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a + b) / (a * b) = 1) :
  a + 2*b ≥ 3 + 2*Real.sqrt 2 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ (a₀ + b₀) / (a₀ * b₀) = 1 ∧ a₀ + 2*b₀ = 3 + 2*Real.sqrt 2 :=
sorry

end min_value_sum_l2681_268150


namespace chicken_problem_l2681_268166

/-- The number of chickens Colten has -/
def colten : ℕ := 37

/-- The number of chickens Skylar has -/
def skylar : ℕ := 3 * colten - 4

/-- The number of chickens Quentin has -/
def quentin : ℕ := 2 * skylar + 25

/-- The total number of chickens -/
def total : ℕ := 383

theorem chicken_problem :
  colten + skylar + quentin = total :=
sorry

end chicken_problem_l2681_268166


namespace basketball_free_throw_probability_l2681_268118

theorem basketball_free_throw_probability (player_A_prob player_B_prob : ℝ) 
  (h1 : player_A_prob = 0.7)
  (h2 : player_B_prob = 0.6)
  (h3 : 0 ≤ player_A_prob ∧ player_A_prob ≤ 1)
  (h4 : 0 ≤ player_B_prob ∧ player_B_prob ≤ 1) :
  1 - (1 - player_A_prob) * (1 - player_B_prob) = 0.88 := by
  sorry


end basketball_free_throw_probability_l2681_268118


namespace inequality_contradiction_l2681_268106

theorem inequality_contradiction (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) := by
  sorry

end inequality_contradiction_l2681_268106


namespace food_waste_scientific_notation_l2681_268155

theorem food_waste_scientific_notation :
  (500 : ℝ) * 1000000000 = 5 * (10 : ℝ)^10 := by sorry

end food_waste_scientific_notation_l2681_268155


namespace y_derivative_l2681_268192

noncomputable def y (x : ℝ) : ℝ := 
  Real.sqrt ((Real.tan x + Real.sqrt (2 * Real.tan x) + 1) / (Real.tan x - Real.sqrt (2 * Real.tan x) + 1))

theorem y_derivative (x : ℝ) : 
  deriv y x = 0 :=
sorry

end y_derivative_l2681_268192


namespace abc_value_l2681_268117

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 15 * Real.sqrt 3)
  (hbc : b * c = 21 * Real.sqrt 3)
  (hac : a * c = 10 * Real.sqrt 3) :
  a * b * c = 15 * Real.sqrt 42 := by
sorry

end abc_value_l2681_268117


namespace smallest_sum_of_reciprocal_sum_l2681_268101

theorem smallest_sum_of_reciprocal_sum (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → (a : ℤ) + b ≥ 45) ∧
  ∃ p q : ℕ+, p ≠ q ∧ (1 : ℚ) / p + (1 : ℚ) / q = (1 : ℚ) / 10 ∧ (p : ℤ) + q = 45 :=
sorry

end smallest_sum_of_reciprocal_sum_l2681_268101


namespace gcd_2024_2048_l2681_268144

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end gcd_2024_2048_l2681_268144


namespace fold_reflection_sum_l2681_268167

/-- The fold line passing through the midpoint of (0,3) and (5,0) -/
def fold_line (x y : ℝ) : Prop := y = (5/3) * x - 1

/-- The property that (m,n) is the reflection of (8,4) across the fold line -/
def reflection_property (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    fold_line x y ∧ 
    (x = (8 + m) / 2 ∧ y = (4 + n) / 2) ∧
    (n - 4) / (m - 8) = -3/5

theorem fold_reflection_sum (m n : ℝ) 
  (h1 : fold_line 0 3)
  (h2 : fold_line 5 0)
  (h3 : reflection_property m n) :
  m + n = 9.75 := by sorry

end fold_reflection_sum_l2681_268167


namespace equipment_percentage_transportation_degrees_l2681_268142

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  salaries : ℝ
  research_development : ℝ
  utilities : ℝ
  supplies : ℝ
  transportation : ℝ
  equipment : ℝ

/-- The theorem stating the correct percentage for equipment in the budget -/
theorem equipment_percentage (b : BudgetAllocation) : b.equipment = 4 :=
  by
    have h1 : b.salaries = 60 := by sorry
    have h2 : b.research_development = 9 := by sorry
    have h3 : b.utilities = 5 := by sorry
    have h4 : b.supplies = 2 := by sorry
    have h5 : b.transportation = 20 := by sorry
    have h6 : b.salaries + b.research_development + b.utilities + b.supplies + b.transportation + b.equipment = 100 := by sorry
    sorry

/-- The function to calculate the degrees in a circle graph for a given percentage -/
def percentToDegrees (percent : ℝ) : ℝ := 3.6 * percent

/-- The theorem stating that 72 degrees represent the transportation budget -/
theorem transportation_degrees (b : BudgetAllocation) : percentToDegrees b.transportation = 72 :=
  by sorry

end equipment_percentage_transportation_degrees_l2681_268142


namespace john_received_120_l2681_268116

/-- The amount of money John received from his grandpa -/
def grandpa_amount : ℕ := 30

/-- The amount of money John received from his grandma -/
def grandma_amount : ℕ := 3 * grandpa_amount

/-- The total amount of money John received from both grandparents -/
def total_amount : ℕ := grandpa_amount + grandma_amount

theorem john_received_120 : total_amount = 120 := by
  sorry

end john_received_120_l2681_268116


namespace avg_people_moving_rounded_l2681_268130

/-- The number of people moving to Texas in two days -/
def people_moving : ℕ := 1500

/-- The number of days -/
def days : ℕ := 2

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculate the average number of people moving to Texas per minute -/
def avg_people_per_minute : ℚ :=
  people_moving / (days * hours_per_day * minutes_per_hour)

/-- Round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem avg_people_moving_rounded :
  round_to_nearest avg_people_per_minute = 1 := by sorry

end avg_people_moving_rounded_l2681_268130


namespace trapezium_other_side_length_l2681_268121

/-- Given a trapezium with the following properties:
  - One parallel side is 20 cm long
  - The distance between parallel sides is 17 cm
  - The area is 323 square centimeters
  Prove that the length of the other parallel side is 18 cm -/
theorem trapezium_other_side_length (a b h : ℝ) (area : ℝ) : 
  a = 20 → h = 17 → area = 323 → area = (a + b) * h / 2 → b = 18 := by
  sorry

end trapezium_other_side_length_l2681_268121


namespace committee_problem_l2681_268128

/-- The number of ways to form a committee with the given constraints -/
def committee_formations (n m k r : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose (n - m) k

theorem committee_problem :
  let total_members : ℕ := 30
  let founding_members : ℕ := 10
  let committee_size : ℕ := 5
  committee_formations total_members founding_members committee_size = 126992 := by
  sorry

end committee_problem_l2681_268128


namespace race_length_l2681_268188

/-- Represents the race scenario -/
structure Race where
  length : ℝ
  samTime : ℝ
  johnTime : ℝ
  headStart : ℝ

/-- The race satisfies the given conditions -/
def validRace (r : Race) : Prop :=
  r.samTime = 17 ∧
  r.johnTime = r.samTime + 5 ∧
  r.headStart = 15 ∧
  r.length / r.samTime = (r.length - r.headStart) / r.johnTime

/-- The theorem to be proved -/
theorem race_length (r : Race) (h : validRace r) : r.length = 66 := by
  sorry

end race_length_l2681_268188


namespace lighthouse_distance_l2681_268133

theorem lighthouse_distance (a : ℝ) (h : a > 0) :
  let A : ℝ × ℝ := (a * Real.cos (20 * π / 180), a * Real.sin (20 * π / 180))
  let B : ℝ × ℝ := (a * Real.cos (220 * π / 180), a * Real.sin (220 * π / 180))
  let C : ℝ × ℝ := (0, 0)
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 3 * a^2 := by
  sorry

end lighthouse_distance_l2681_268133


namespace toys_sold_l2681_268115

theorem toys_sold (selling_price : ℕ) (cost_price : ℕ) (gain : ℕ) :
  selling_price = 16800 →
  gain = 3 * cost_price →
  cost_price = 800 →
  (selling_price - gain) / cost_price = 18 :=
by
  sorry

end toys_sold_l2681_268115


namespace quadratic_inequality_transformation_l2681_268146

-- Define the quadratic function and its solution set
def quadratic_inequality (a b : ℝ) := {x : ℝ | x^2 + a*x + b > 0}

-- Define the given solution set
def given_solution_set := {x : ℝ | x < -3 ∨ x > 1}

-- Define the transformed quadratic inequality
def transformed_inequality (a b : ℝ) := {x : ℝ | a*x^2 + b*x - 2 < 0}

-- Define the expected solution set
def expected_solution_set := {x : ℝ | -1/2 < x ∧ x < 2}

-- Theorem statement
theorem quadratic_inequality_transformation 
  (h : quadratic_inequality a b = given_solution_set) :
  transformed_inequality a b = expected_solution_set := by
  sorry

end quadratic_inequality_transformation_l2681_268146


namespace infinitely_many_consecutive_almost_squares_l2681_268161

/-- A natural number is almost a square if it can be represented as a product of two numbers
    that differ by no more than one percent of the larger of them. -/
def AlmostSquare (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * b ∧ (a : ℝ) ≥ (b : ℝ) ∧ (a : ℝ) ≤ (b : ℝ) * 1.01

/-- There exist infinitely many natural numbers m such that 4m^4 - 1, 4m^4, 4m^4 + 1, and 4m^4 + 2
    are all almost squares. -/
theorem infinitely_many_consecutive_almost_squares :
  ∀ N : ℕ, ∃ m : ℕ, m > N ∧
    AlmostSquare (4 * m^4 - 1) ∧
    AlmostSquare (4 * m^4) ∧
    AlmostSquare (4 * m^4 + 1) ∧
    AlmostSquare (4 * m^4 + 2) := by
  sorry


end infinitely_many_consecutive_almost_squares_l2681_268161


namespace unscreened_percentage_l2681_268183

def tv_width : ℝ := 6
def tv_height : ℝ := 5
def screen_width : ℝ := 5
def screen_height : ℝ := 4

theorem unscreened_percentage :
  (tv_width * tv_height - screen_width * screen_height) / (tv_width * tv_height) * 100 = 100 / 3 := by
  sorry

end unscreened_percentage_l2681_268183


namespace min_value_of_S_l2681_268110

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve C representing the trajectory of the circle's center -/
def C (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- The dot product of two vectors represented by points -/
def dot_product (p1 p2 : Point) : ℝ :=
  p1.x * p2.x + p1.y * p2.y

/-- The area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- The function S to be minimized -/
noncomputable def S (a b : Point) : ℝ :=
  let o : Point := ⟨0, 0⟩
  let f : Point := ⟨1, 0⟩
  triangle_area o f a + triangle_area o a b

/-- The main theorem stating the minimum value of S -/
theorem min_value_of_S :
  ∀ a b : Point,
  C a → C b →
  dot_product a b = -4 →
  S a b ≥ 4 * Real.sqrt 3 :=
sorry

end min_value_of_S_l2681_268110


namespace printing_presses_theorem_l2681_268104

/-- The number of printing presses used in the first scenario -/
def P : ℕ := 35

/-- The time taken (in hours) by P presses to print 500,000 papers -/
def time_P : ℕ := 15

/-- The number of presses used in the second scenario -/
def presses_2 : ℕ := 25

/-- The time taken (in hours) by presses_2 to print 500,000 papers -/
def time_2 : ℕ := 21

theorem printing_presses_theorem :
  P * time_P = presses_2 * time_2 :=
sorry

end printing_presses_theorem_l2681_268104


namespace range_of_a_l2681_268148

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2)^2 * Real.exp x + a * Real.exp (-x)

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := 2 * a * |x - 2|

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  (∃ (s : Finset ℝ), s.card = 6 ∧ ∀ x ∈ s, f a x = g a x) →
  1 < a ∧ a < Real.exp 2 / (2 * Real.exp 1 - 1) :=
sorry

end range_of_a_l2681_268148


namespace balanced_colorings_count_l2681_268143

/-- Represents a color in the grid -/
inductive Color
| Red
| Blue
| Yellow
| Green

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat
  color : Color

/-- Represents the grid -/
def Grid := List Cell

/-- Checks if a 2x2 subgrid is balanced -/
def isBalanced2x2 (grid : Grid) (startRow startCol : Nat) : Bool :=
  sorry

/-- Checks if the entire grid is balanced -/
def isBalancedGrid (grid : Grid) : Bool :=
  sorry

/-- Counts the number of balanced colorings for an 8x6 grid -/
def countBalancedColorings : Nat :=
  sorry

/-- The main theorem stating the number of balanced colorings -/
theorem balanced_colorings_count :
  countBalancedColorings = 1896 :=
sorry

end balanced_colorings_count_l2681_268143


namespace range_of_a_l2681_268149

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |2*a - 1| ≤ |x + 1/x|) ↔ -1/2 ≤ a ∧ a ≤ 3/2 :=
sorry

end range_of_a_l2681_268149


namespace line_equation_proof_l2681_268152

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (p : Point) :
  given_line.a = 1 →
  given_line.b = -2 →
  given_line.c = 3 →
  p.x = -1 →
  p.y = 3 →
  ∃ (result_line : Line),
    result_line.a = 1 ∧
    result_line.b = -2 ∧
    result_line.c = 7 ∧
    pointOnLine p result_line ∧
    parallelLines given_line result_line :=
by sorry


end line_equation_proof_l2681_268152


namespace cyclist_average_speed_l2681_268169

/-- Given a cyclist who travels two segments with different distances and speeds, 
    this theorem proves that the average speed for the entire trip is 18 miles per hour. -/
theorem cyclist_average_speed : 
  ∀ (d₁ d₂ v₁ v₂ : ℝ), 
    d₁ = 45 ∧ d₂ = 15 ∧ v₁ = 15 ∧ v₂ = 45 →
    (d₁ + d₂) / ((d₁ / v₁) + (d₂ / v₂)) = 18 := by
  sorry


end cyclist_average_speed_l2681_268169


namespace consecutive_even_squares_l2681_268137

theorem consecutive_even_squares (x : ℕ) : 
  (x % 2 = 0) → (x^2 - (x-2)^2 = 2012) → x = 504 := by
  sorry

end consecutive_even_squares_l2681_268137


namespace base_conversion_property_l2681_268176

def convert_base (n : ℕ) (from_base to_base : ℕ) : ℕ :=
  sorry

def digits_to_nat (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

def nat_to_digits (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem base_conversion_property :
  ∀ b : ℕ, b ∈ [13, 12, 11] →
    let n := digits_to_nat [1, 2, 2, 1] b
    nat_to_digits (convert_base n b (b - 1)) (b - 1) = [1, 2, 2, 1] ∧
  let n₁₀ := digits_to_nat [1, 2, 2, 1] 10
  nat_to_digits (convert_base n₁₀ 10 9) 9 ≠ [1, 2, 2, 1] :=
by
  sorry

end base_conversion_property_l2681_268176


namespace square_difference_503_496_l2681_268189

theorem square_difference_503_496 : 503^2 - 496^2 = 6993 := by
  sorry

end square_difference_503_496_l2681_268189


namespace total_ants_l2681_268165

theorem total_ants (abe beth cece duke : ℕ) : 
  abe = 4 →
  beth = abe + abe / 2 →
  cece = 2 * abe →
  duke = abe / 2 →
  abe + beth + cece + duke = 20 := by
  sorry

end total_ants_l2681_268165


namespace max_ticket_types_for_specific_car_l2681_268163

/-- Represents a one-way traveling car with stations and capacity. -/
structure TravelingCar where
  num_stations : Nat
  capacity : Nat

/-- Calculates the maximum number of different ticket types that can be sold. -/
def max_ticket_types (car : TravelingCar) : Nat :=
  let total_possible_tickets := (car.num_stations - 1) * car.num_stations / 2
  let max_non_overlapping_tickets := ((car.num_stations + 1) / 2) ^ 2
  let unsellable_tickets := max_non_overlapping_tickets - car.capacity
  total_possible_tickets - unsellable_tickets

/-- Theorem stating the maximum number of different ticket types for a specific car configuration. -/
theorem max_ticket_types_for_specific_car :
  let car := TravelingCar.mk 14 25
  max_ticket_types car = 67 := by
  sorry

end max_ticket_types_for_specific_car_l2681_268163


namespace max_blocks_in_box_l2681_268113

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the problem of fitting small blocks into a larger box -/
structure BlockFittingProblem where
  box : Dimensions
  block : Dimensions

/-- Calculates the maximum number of blocks that can fit in the box based on volume -/
def maxBlocksByVolume (p : BlockFittingProblem) : ℕ :=
  (volume p.box) / (volume p.block)

/-- Determines if the arrangement of blocks is physically possible -/
def isPhysicallyPossible (p : BlockFittingProblem) (n : ℕ) : Prop :=
  (p.block.width = p.box.width) ∧
  (2 * p.block.length ≤ p.box.length) ∧
  (((n / 4) * p.block.height) ≤ p.box.height) ∧
  ((n % 4) * p.block.height ≤ p.box.height - ((n / 4) * p.block.height))

theorem max_blocks_in_box (p : BlockFittingProblem) 
  (h1 : p.box = Dimensions.mk 4 3 5)
  (h2 : p.block = Dimensions.mk 1 3 2) :
  ∃ (n : ℕ), n = 10 ∧ 
    (maxBlocksByVolume p = n) ∧ 
    (isPhysicallyPossible p n) ∧
    (∀ m : ℕ, m > n → ¬(isPhysicallyPossible p m)) := by
  sorry

end max_blocks_in_box_l2681_268113


namespace diophantine_equation_solution_l2681_268177

theorem diophantine_equation_solution :
  ∀ (a b c : ℤ), 5 * a^2 + 9 * b^2 = 13 * c^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end diophantine_equation_solution_l2681_268177


namespace probability_of_white_ball_l2681_268135

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.3 → p_black = 0.5 → p_red + p_black + p_white = 1 → p_white = 0.2 := by
  sorry

end probability_of_white_ball_l2681_268135


namespace hyperbola_parabola_intersection_l2681_268138

/-- The hyperbola and parabola intersection problem -/
theorem hyperbola_parabola_intersection
  (a : ℝ) (P F₁ F₂ : ℝ × ℝ) 
  (h_a_pos : a > 0)
  (h_hyperbola : 3 * P.1^2 - P.2^2 = 3 * a^2)
  (h_parabola : P.2^2 = 8 * a * P.1)
  (h_F₁ : F₁ = (-2*a, 0))
  (h_F₂ : F₂ = (2*a, 0))
  (h_distance : Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
                Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 12) :
  ∃ (x : ℝ), x = -2 ∧ x = -a/2 := by
  sorry


end hyperbola_parabola_intersection_l2681_268138


namespace min_value_a_plus_2b_l2681_268136

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b + 2*a*b = 8) :
  ∀ x y, x > 0 → y > 0 → x + 2*y + 2*x*y = 8 → a + 2*b ≤ x + 2*y :=
by sorry

end min_value_a_plus_2b_l2681_268136


namespace divisible_by_three_l2681_268182

theorem divisible_by_three (n : ℕ) : 
  (3 ∣ n * 2^n + 1) ↔ (∃ k : ℕ, n = 6*k + 1 ∨ n = 6*k + 2) :=
by sorry

end divisible_by_three_l2681_268182


namespace quadratic_factorization_l2681_268105

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) →
  a - b = -7 := by
sorry

end quadratic_factorization_l2681_268105


namespace log_sum_equality_l2681_268160

theorem log_sum_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 * Real.log 2 / Real.log 5 := by
  sorry

end log_sum_equality_l2681_268160


namespace snyder_income_proof_l2681_268175

/-- Mrs. Snyder's previous monthly income -/
def previous_income : ℝ := 1700

/-- Mrs. Snyder's salary increase -/
def salary_increase : ℝ := 850

/-- Percentage of income spent on rent and utilities before salary increase -/
def previous_percentage : ℝ := 0.45

/-- Percentage of income spent on rent and utilities after salary increase -/
def new_percentage : ℝ := 0.30

theorem snyder_income_proof :
  (previous_percentage * previous_income = new_percentage * (previous_income + salary_increase)) ∧
  previous_income = 1700 := by
  sorry

end snyder_income_proof_l2681_268175


namespace large_monkey_cost_is_correct_l2681_268156

/-- The cost of a large monkey doll -/
def large_monkey_cost : ℝ := 6

/-- The total amount spent on dolls -/
def total_spent : ℝ := 300

/-- The cost difference between large and small monkey dolls -/
def small_large_diff : ℝ := 2

/-- The cost difference between elephant and large monkey dolls -/
def elephant_large_diff : ℝ := 1

/-- The number of additional dolls if buying only small monkeys -/
def small_monkey_diff : ℕ := 25

/-- The number of fewer dolls if buying only elephants -/
def elephant_diff : ℕ := 15

theorem large_monkey_cost_is_correct : 
  (total_spent / (large_monkey_cost - small_large_diff) = 
   total_spent / large_monkey_cost + small_monkey_diff) ∧
  (total_spent / (large_monkey_cost + elephant_large_diff) = 
   total_spent / large_monkey_cost - elephant_diff) := by
  sorry

end large_monkey_cost_is_correct_l2681_268156


namespace max_score_is_94_l2681_268124

/-- Represents an operation that can be applied to a number -/
inductive Operation
  | Add : Operation
  | Square : Operation

/-- Applies an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Add => n + 1
  | Operation.Square => n * n

/-- Applies a sequence of operations to a starting number -/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Calculates the minimum distance from a number to any perfect square -/
def minDistanceToPerfectSquare (n : ℕ) : ℕ :=
  let sqrtFloor := (n.sqrt : ℕ)
  let sqrtCeil := sqrtFloor + 1
  min (n - sqrtFloor * sqrtFloor) (sqrtCeil * sqrtCeil - n)

/-- The main theorem -/
theorem max_score_is_94 :
  (∃ (ops : List Operation),
    ops.length = 100 ∧
    minDistanceToPerfectSquare (applyOperations 0 ops) = 94) ∧
  (∀ (ops : List Operation),
    ops.length = 100 →
    minDistanceToPerfectSquare (applyOperations 0 ops) ≤ 94) :=
  sorry


end max_score_is_94_l2681_268124


namespace sign_of_f_m_plus_one_indeterminate_l2681_268145

/-- Given a quadratic function f(x) = x^2 - x + a and the condition that f(-m) < 0,
    prove that the sign of f(m+1) cannot be determined without additional information about m. -/
theorem sign_of_f_m_plus_one_indeterminate 
  (f : ℝ → ℝ) (a m : ℝ) 
  (h1 : ∀ x, f x = x^2 - x + a) 
  (h2 : f (-m) < 0) : 
  ∃ m₁ m₂, f (m₁ + 1) > 0 ∧ f (m₂ + 1) < 0 :=
sorry

end sign_of_f_m_plus_one_indeterminate_l2681_268145


namespace right_triangle_equivalence_l2681_268129

theorem right_triangle_equivalence (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (a^3 + b^3 + c^3 = a*b*(a+b) - b*c*(b+c) + a*c*(a+c)) ↔
  (a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2) :=
sorry

end right_triangle_equivalence_l2681_268129


namespace algebraic_expression_value_l2681_268109

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 = 3 * y) : 
  x^2 - 6*x*y + 9*y^2 = 4 := by
  sorry

end algebraic_expression_value_l2681_268109


namespace point_in_fourth_quadrant_l2681_268179

/-- The complex number (1-i)·i corresponds to a point in the fourth quadrant of the complex plane. -/
theorem point_in_fourth_quadrant : ∃ (z : ℂ), z = (1 - Complex.I) * Complex.I ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end point_in_fourth_quadrant_l2681_268179


namespace greatest_three_digit_multiple_of_17_ending_4_l2681_268196

theorem greatest_three_digit_multiple_of_17_ending_4 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ n % 10 = 4 → n ≤ 204 :=
by sorry

end greatest_three_digit_multiple_of_17_ending_4_l2681_268196


namespace gourmet_smores_night_cost_l2681_268132

/-- The cost of supplies for a gourmet S'mores night -/
def cost_of_smores_night (num_people : ℕ) (smores_per_person : ℕ) 
  (graham_cracker_cost : ℚ) (marshmallow_cost : ℚ) (chocolate_cost : ℚ)
  (caramel_cost : ℚ) (toffee_cost : ℚ) : ℚ :=
  let total_smores := num_people * smores_per_person
  let cost_per_smore := graham_cracker_cost + marshmallow_cost + chocolate_cost + 
                        2 * caramel_cost + 4 * toffee_cost
  total_smores * cost_per_smore

/-- Theorem: The cost of supplies for the gourmet S'mores night is $26.40 -/
theorem gourmet_smores_night_cost :
  cost_of_smores_night 8 3 (10/100) (15/100) (25/100) (20/100) (5/100) = 2640/100 :=
by sorry

end gourmet_smores_night_cost_l2681_268132


namespace difference_of_squares_l2681_268164

theorem difference_of_squares (x y : ℕ+) 
  (sum_eq : x + y = 20)
  (product_eq : x * y = 99) :
  x^2 - y^2 = 40 := by
  sorry

end difference_of_squares_l2681_268164


namespace geometric_sequence_common_ratio_l2681_268139

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  monotone_increasing : Monotone a
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  condition1 : a 2 * a 5 = 6
  condition2 : a 3 + a 4 = 5

/-- The common ratio of the geometric sequence is 3/2 -/
theorem geometric_sequence_common_ratio (seq : GeometricSequence) :
  seq.a 2 / seq.a 1 = 3/2 := by
  sorry

end geometric_sequence_common_ratio_l2681_268139


namespace roberto_outfits_l2681_268103

/-- The number of trousers Roberto has -/
def num_trousers : ℕ := 5

/-- The number of shirts Roberto has -/
def num_shirts : ℕ := 8

/-- The number of jackets Roberto has -/
def num_jackets : ℕ := 2

/-- An outfit consists of a pair of trousers, a shirt, and a jacket -/
def outfit := ℕ × ℕ × ℕ

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_trousers * num_shirts * num_jackets

theorem roberto_outfits : total_outfits = 80 := by
  sorry

end roberto_outfits_l2681_268103


namespace lilly_fish_count_l2681_268153

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 8

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 18

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := total_fish - rosy_fish

theorem lilly_fish_count : lilly_fish = 10 := by
  sorry

end lilly_fish_count_l2681_268153


namespace f_simplification_f_specific_value_l2681_268100

noncomputable def f (α : Real) : Real :=
  (Real.sin (4 * Real.pi - α) * Real.cos (Real.pi - α) * Real.cos ((3 * Real.pi) / 2 + α) * Real.cos ((7 * Real.pi) / 2 - α)) /
  (Real.cos (Real.pi + α) * Real.sin (2 * Real.pi - α) * Real.sin (Real.pi + α) * Real.sin ((9 * Real.pi) / 2 - α))

theorem f_simplification (α : Real) : f α = Real.tan α := by sorry

theorem f_specific_value : f (-(31 / 6) * Real.pi) = -(Real.sqrt 3 / 3) := by sorry

end f_simplification_f_specific_value_l2681_268100


namespace football_game_spectators_l2681_268159

theorem football_game_spectators (total_wristbands : ℕ) (wristbands_per_person : ℕ) 
  (h1 : total_wristbands = 290)
  (h2 : wristbands_per_person = 2)
  : total_wristbands / wristbands_per_person = 145 := by
  sorry

end football_game_spectators_l2681_268159
