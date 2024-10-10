import Mathlib

namespace log_expression_equals_negative_two_l870_87079

theorem log_expression_equals_negative_two :
  (Real.log 64 / Real.log 32) / (Real.log 2 / Real.log 32) -
  (Real.log 256 / Real.log 16) / (Real.log 2 / Real.log 16) = -2 := by
  sorry

end log_expression_equals_negative_two_l870_87079


namespace complex_magnitude_l870_87062

/-- Given a complex number z satisfying z(1+i) = 1-2i, prove that |z| = √10/2 -/
theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 1 - 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_magnitude_l870_87062


namespace paths_to_2005_l870_87082

/-- Represents a position on the 5x5 grid --/
inductive Position
| Center : Position
| Side : Position
| Corner : Position
| Edge : Position

/-- Represents the possible moves on the grid --/
def possibleMoves : Position → List Position
| Position.Center => [Position.Side, Position.Corner]
| Position.Side => [Position.Side, Position.Corner, Position.Edge]
| Position.Corner => [Position.Side, Position.Edge]
| Position.Edge => []

/-- Counts the number of paths to form 2005 on the given grid --/
def countPaths : ℕ :=
  let initialSideMoves := 4
  let initialCornerMoves := 4
  let sideToSideMoves := 2
  let sideToCornerMoves := 2
  let cornerToSideMoves := 2
  let sideToEdgeMoves := 3
  let cornerToEdgeMoves := 5

  let sideSidePaths := initialSideMoves * sideToSideMoves * sideToEdgeMoves
  let sideCornerPaths := initialSideMoves * sideToCornerMoves * cornerToEdgeMoves
  let cornerSidePaths := initialCornerMoves * cornerToSideMoves * sideToEdgeMoves

  sideSidePaths + sideCornerPaths + cornerSidePaths

/-- Theorem stating that there are 88 paths to form 2005 on the given grid --/
theorem paths_to_2005 : countPaths = 88 := by sorry

end paths_to_2005_l870_87082


namespace sum_inequality_l870_87010

theorem sum_inequality (m n : ℕ+) (a : Fin m → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_range : ∀ i, a i ∈ Finset.range n)
  (h_sum : ∀ i j, i ≤ j → a i + a j ≤ n → ∃ k, a i + a j = a k) :
  (Finset.sum (Finset.range m) (λ i => a i)) / m ≥ (n + 1) / 2 := by
  sorry

end sum_inequality_l870_87010


namespace triangle_area_main_theorem_l870_87045

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the area of a specific triangle -/
theorem triangle_area (t : Triangle) 
  (h1 : t.a^2 = t.b^2 + t.c^2 - t.b * t.c) 
  (h2 : t.b * t.c = 16) : 
  (1/2) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 := by
  sorry

/-- Main theorem proving the area of the triangle -/
theorem main_theorem : 
  ∃ (t : Triangle), 
    t.a^2 = t.b^2 + t.c^2 - t.b * t.c ∧ 
    t.b * t.c = 16 ∧ 
    (1/2) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 := by
  sorry

end triangle_area_main_theorem_l870_87045


namespace vector_subtraction_l870_87024

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (-7, 0, 1)
def b : ℝ × ℝ × ℝ := (6, 2, -1)

-- State the theorem
theorem vector_subtraction :
  (a.1 - 5 * b.1, a.2.1 - 5 * b.2.1, a.2.2 - 5 * b.2.2) = (-37, -10, 6) := by
  sorry

end vector_subtraction_l870_87024


namespace negative_i_fourth_power_l870_87022

theorem negative_i_fourth_power (i : ℂ) (h : i^2 = -1) : (-i)^4 = 1 := by
  sorry

end negative_i_fourth_power_l870_87022


namespace opposite_abs_power_l870_87031

theorem opposite_abs_power (x y : ℝ) : 
  |x - 2| + |y + 3| = 0 → (x + y)^2023 = -1 := by
  sorry

end opposite_abs_power_l870_87031


namespace intersection_count_possibilities_l870_87081

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define a line
def Line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define intersection of two lines
def LinesIntersect (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ x y, Line l1.1 l1.2.1 l1.2.2 x y ∧ Line l2.1 l2.2.1 l2.2.2 x y

-- Define when a line is not tangent to the ellipse
def NotTangent (l : ℝ × ℝ × ℝ) : Prop :=
  ∃ x1 y1 x2 y2, x1 ≠ x2 ∧ y1 ≠ y2 ∧
    Line l.1 l.2.1 l.2.2 x1 y1 ∧ Line l.1 l.2.1 l.2.2 x2 y2 ∧
    Ellipse x1 y1 ∧ Ellipse x2 y2

-- Define the number of intersection points
def IntersectionCount (l1 l2 : ℝ × ℝ × ℝ) : ℕ :=
  sorry

-- Theorem statement
theorem intersection_count_possibilities
  (l1 l2 : ℝ × ℝ × ℝ)
  (h1 : LinesIntersect l1 l2)
  (h2 : NotTangent l1)
  (h3 : NotTangent l2) :
  (IntersectionCount l1 l2 = 2) ∨
  (IntersectionCount l1 l2 = 3) ∨
  (IntersectionCount l1 l2 = 4) :=
sorry

end intersection_count_possibilities_l870_87081


namespace not_sum_of_consecutive_iff_power_of_two_l870_87065

/-- A natural number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- A natural number can be expressed as the sum of consecutive natural numbers -/
def is_sum_of_consecutive (n : ℕ) : Prop :=
  ∃ (start : ℕ) (length : ℕ+), n = (length : ℕ) * (2 * start + length - 1) / 2

/-- 
Theorem: A natural number cannot be expressed as the sum of consecutive natural numbers 
if and only if it is a power of 2
-/
theorem not_sum_of_consecutive_iff_power_of_two (n : ℕ) :
  ¬(is_sum_of_consecutive n) ↔ is_power_of_two n := by sorry

end not_sum_of_consecutive_iff_power_of_two_l870_87065


namespace cinema_tickets_l870_87050

theorem cinema_tickets (x y : ℕ) : 
  x + y = 35 →
  24 * x + 18 * y = 750 →
  x = 20 ∧ y = 15 := by
sorry

end cinema_tickets_l870_87050


namespace distribute_objects_eq_144_l870_87076

-- Define the number of objects and containers
def n : ℕ := 4

-- Define the function to calculate the number of ways to distribute objects
def distribute_objects : ℕ := sorry

-- Theorem statement
theorem distribute_objects_eq_144 : distribute_objects = 144 := by sorry

end distribute_objects_eq_144_l870_87076


namespace sqrt_two_irrational_l870_87011

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop :=
  ∃ a b : ℤ, b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop :=
  ¬(IsRational x)

-- Theorem stating that √2 is irrational
theorem sqrt_two_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

end sqrt_two_irrational_l870_87011


namespace fixed_point_on_line_l870_87057

/-- The fixed point that the line (a+3)x + (2a-1)y + 7 = 0 passes through for all real a -/
def fixed_point : ℝ × ℝ := (-2, 1)

/-- The line equation as a function of a, x, and y -/
def line_equation (a x y : ℝ) : ℝ := (a + 3) * x + (2 * a - 1) * y + 7

theorem fixed_point_on_line :
  ∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2) = 0 := by
  sorry

end fixed_point_on_line_l870_87057


namespace mistaken_multiplication_l870_87036

def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (c * 10 + d : ℚ) / 99

theorem mistaken_multiplication (c d : ℕ) 
  (h1 : c < 10) (h2 : d < 10) :
  90 * repeating_decimal c d - 90 * (1 + (c * 10 + d : ℚ) / 100) = 0.9 → 
  c = 9 ∧ d = 9 := by
  sorry

end mistaken_multiplication_l870_87036


namespace solution_equation_l870_87029

theorem solution_equation (x : ℝ) (hx : x ≠ 0) :
  (9 * x)^10 = (18 * x)^5 → x = 2/9 := by
  sorry

end solution_equation_l870_87029


namespace polynomial_factorization_l870_87028

theorem polynomial_factorization (x : ℝ) : 
  x^2 - 6*x + 9 - 49*x^4 = (-7*x^2 + x - 3) * (7*x^2 + x - 3) := by
  sorry

end polynomial_factorization_l870_87028


namespace intersection_point_polar_radius_l870_87043

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 - y^2 = 4 ∧ y ≠ 0

-- Define the line l₃ in polar form
def l₃ (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) - Real.sqrt 2 = 0

-- Define the intersection point M
def M (x y : ℝ) : Prop := C x y ∧ x + y = Real.sqrt 2

-- Theorem statement
theorem intersection_point_polar_radius :
  ∀ x y : ℝ, M x y → x^2 + y^2 = 5 := by sorry

end intersection_point_polar_radius_l870_87043


namespace quadratic_equation_solution_expression_simplification_l870_87058

-- Problem 1
theorem quadratic_equation_solution (x : ℝ) :
  x * (x + 6) = 8 * (x + 3) ↔ x = 6 ∨ x = -4 := by sorry

-- Problem 2
theorem expression_simplification (a b : ℝ) (h : a ≠ b) :
  (3 * a^2 - 3 * b^2) / (a^2 * b + a * b^2) / (1 - (a^2 + b^2) / (2 * a * b)) = -6 / (a - b) := by sorry

end quadratic_equation_solution_expression_simplification_l870_87058


namespace intersection_M_N_union_complements_M_N_l870_87091

open Set

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x ≥ 1}

-- Define set N
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

-- Theorem for the intersection of M and N
theorem intersection_M_N : M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 5} := by sorry

-- Theorem for the union of complements of M and N
theorem union_complements_M_N : (U \ M) ∪ (U \ N) = {x : ℝ | x < 1 ∨ x ≥ 5} := by sorry

end intersection_M_N_union_complements_M_N_l870_87091


namespace production_days_l870_87097

theorem production_days (n : ℕ) 
  (h1 : (40 * n) / n = 40)  -- Average daily production for past n days
  (h2 : ((40 * n + 90) : ℝ) / (n + 1) = 45) : n = 9 :=
by sorry

end production_days_l870_87097


namespace pancake_diameter_l870_87055

/-- The diameter of a circular object with radius 7 centimeters is 14 centimeters. -/
theorem pancake_diameter (r : ℝ) (h : r = 7) : 2 * r = 14 := by
  sorry

end pancake_diameter_l870_87055


namespace parabola_shift_left_one_unit_l870_87033

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := 2 * p.a * h + p.b,
    c := p.a * h^2 + p.b * h + p.c }

theorem parabola_shift_left_one_unit :
  let original := Parabola.mk 1 0 2
  let shifted := shift_horizontal original (-1)
  shifted = Parabola.mk 1 2 3 := by sorry

end parabola_shift_left_one_unit_l870_87033


namespace line_perpendicular_range_l870_87074

/-- Given a line l: x - y + a = 0 and points A(-2,0) and B(2,0),
    if there exists a point P on line l such that AP ⊥ BP,
    then -2√2 ≤ a ≤ 2√2. -/
theorem line_perpendicular_range (a : ℝ) :
  (∃ (P : ℝ × ℝ), 
    (P.1 - P.2 + a = 0) ∧ 
    ((P.1 + 2) * (P.1 - 2) + P.2 * P.2 = 0)) →
  -2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 :=
by sorry


end line_perpendicular_range_l870_87074


namespace pace_difference_is_one_minute_l870_87013

-- Define the race parameters
def square_length : ℚ := 3/4
def num_laps : ℕ := 7
def this_year_time : ℚ := 42
def last_year_time : ℚ := 47.25

-- Define the total race distance
def race_distance : ℚ := square_length * num_laps

-- Define the average pace for this year and last year
def this_year_pace : ℚ := this_year_time / race_distance
def last_year_pace : ℚ := last_year_time / race_distance

-- Theorem statement
theorem pace_difference_is_one_minute :
  last_year_pace - this_year_pace = 1 := by sorry

end pace_difference_is_one_minute_l870_87013


namespace shopping_mall_problem_l870_87002

/-- Shopping mall goods purchasing and selling problem -/
theorem shopping_mall_problem 
  (cost_A_1_B_2 : ℝ) 
  (cost_A_3_B_2 : ℝ) 
  (sell_price_A : ℝ) 
  (sell_price_B : ℝ) 
  (total_units : ℕ) 
  (profit_lower : ℝ) 
  (profit_upper : ℝ) 
  (planned_units : ℕ) 
  (actual_profit : ℝ)
  (h1 : cost_A_1_B_2 = 320)
  (h2 : cost_A_3_B_2 = 520)
  (h3 : sell_price_A = 120)
  (h4 : sell_price_B = 140)
  (h5 : total_units = 50)
  (h6 : profit_lower = 1350)
  (h7 : profit_upper = 1375)
  (h8 : planned_units = 46)
  (h9 : actual_profit = 1220) :
  ∃ (cost_A cost_B : ℝ) (m : ℕ) (b : ℕ),
    cost_A = 100 ∧ 
    cost_B = 110 ∧ 
    13 ≤ m ∧ m ≤ 15 ∧
    b ≥ 32 := by sorry

end shopping_mall_problem_l870_87002


namespace equation_one_solution_equation_two_no_solution_l870_87063

-- Problem 1
theorem equation_one_solution (x : ℝ) : 
  (5 / (x - 1) = 1 / (2 * x + 1)) ↔ x = -2/3 :=
sorry

-- Problem 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℝ), (1 / (x - 2) + 2 = (1 - x) / (2 - x)) :=
sorry

end equation_one_solution_equation_two_no_solution_l870_87063


namespace area1_is_linear_area2_is_quadratic_l870_87046

-- Define the rectangles
def rectangle1_length : ℝ := 10
def rectangle1_width : ℝ := 5
def rectangle2_length : ℝ := 30
def rectangle2_width : ℝ := 20

-- Define the area functions
def area1 (x : ℝ) : ℝ := (rectangle1_length - x) * rectangle1_width
def area2 (x : ℝ) : ℝ := (rectangle2_length + x) * (rectangle2_width + x)

-- Theorem statements
theorem area1_is_linear :
  ∃ (m b : ℝ), ∀ x, area1 x = m * x + b :=
sorry

theorem area2_is_quadratic :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, area2 x = a * x^2 + b * x + c) :=
sorry

end area1_is_linear_area2_is_quadratic_l870_87046


namespace not_linear_in_M_f_expression_for_negative_range_sin_k_in_M_iff_l870_87077

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∃ (T : ℝ) (hT : T ≠ 0), ∀ x, f (x + T) = T * f x}

-- Theorem 1
theorem not_linear_in_M : ¬(λ x : ℝ => x) ∈ M := by sorry

-- Theorem 2
theorem f_expression_for_negative_range 
  (f : ℝ → ℝ) (hf : f ∈ M) (hT : ∃ T, T = 2 ∧ ∀ x, 1 < x → x < 2 → f x = x + Real.log x) :
  ∀ x, -3 < x → x < -2 → f x = (1/4) * (x + 4 + Real.log (x + 4)) := by sorry

-- Theorem 3
theorem sin_k_in_M_iff (k : ℝ) : 
  (λ x : ℝ => Real.sin (k * x)) ∈ M ↔ ∃ m : ℤ, k = m * Real.pi := by sorry

end not_linear_in_M_f_expression_for_negative_range_sin_k_in_M_iff_l870_87077


namespace daps_equivalent_to_dips_l870_87039

/-- The number of daps equivalent to one dop -/
def daps_per_dop : ℚ := 5 / 4

/-- The number of dips equivalent to one dop -/
def dips_per_dop : ℚ := 3

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 54

/-- Theorem stating the equivalence between daps and dips -/
theorem daps_equivalent_to_dips : 
  (target_dips * daps_per_dop) / dips_per_dop = 22.5 := by sorry

end daps_equivalent_to_dips_l870_87039


namespace line_parameterization_l870_87069

/-- The line equation y = (3/4)x - 2 parameterized as (x, y) = (-3, v) + u(m, -8) -/
def line_equation (x y : ℝ) : Prop :=
  y = (3/4) * x - 2

/-- The parametric form of the line -/
def parametric_form (x y u v m : ℝ) : Prop :=
  x = -3 + u * m ∧ y = v - 8 * u

/-- Theorem stating that v = -17/4 and m = -16/9 satisfy the line equation and parametric form -/
theorem line_parameterization :
  ∃ (v m : ℝ), v = -17/4 ∧ m = -16/9 ∧
  (∀ (x y u : ℝ), parametric_form x y u v m → line_equation x y) :=
sorry

end line_parameterization_l870_87069


namespace consecutive_odd_integers_count_l870_87087

/-- Represents a sequence of consecutive odd integers -/
structure ConsecutiveOddIntegers where
  n : ℕ  -- number of integers in the sequence
  first : ℤ  -- first (least) integer in the sequence
  avg : ℚ  -- average of the integers in the sequence

/-- Theorem: Given the conditions, the number of consecutive odd integers is 8 -/
theorem consecutive_odd_integers_count
  (seq : ConsecutiveOddIntegers)
  (h1 : seq.first = 407)
  (h2 : seq.avg = 414)
  : seq.n = 8 := by
  sorry

end consecutive_odd_integers_count_l870_87087


namespace tiffany_lives_l870_87025

/-- Calculates the final number of lives in a video game scenario -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Proves that for the given scenario, the final number of lives is 56 -/
theorem tiffany_lives : final_lives 43 14 27 = 56 := by
  sorry

end tiffany_lives_l870_87025


namespace max_x_minus_y_l870_87064

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ z :=
by sorry

end max_x_minus_y_l870_87064


namespace smallest_z_for_inequality_l870_87068

theorem smallest_z_for_inequality : ∃ (z : ℕ), (∀ (y : ℕ), 27 ^ y > 3 ^ 24 → z ≤ y) ∧ 27 ^ z > 3 ^ 24 := by
  sorry

end smallest_z_for_inequality_l870_87068


namespace remainder_problem_l870_87086

theorem remainder_problem (m : ℤ) (h : m % 5 = 3) : (4 * m + 5) % 5 = 2 := by
  sorry

end remainder_problem_l870_87086


namespace point_in_region_l870_87095

def point : ℝ × ℝ := (0, -2)

theorem point_in_region (x y : ℝ) (h : (x, y) = point) : 
  x + y - 1 < 0 ∧ x - y + 1 > 0 := by
  sorry

end point_in_region_l870_87095


namespace count_terminating_decimals_is_40_l870_87047

/-- 
Counts the number of integers n between 1 and 120 inclusive 
for which the decimal representation of n/120 terminates.
-/
def count_terminating_decimals : ℕ :=
  let max : ℕ := 120
  let prime_factors : Multiset ℕ := {2, 2, 2, 3, 5}
  sorry

/-- The count of terminating decimals is 40 -/
theorem count_terminating_decimals_is_40 : 
  count_terminating_decimals = 40 := by sorry

end count_terminating_decimals_is_40_l870_87047


namespace alphabet_sum_theorem_l870_87098

/-- Represents a letter in the English alphabet -/
def Letter := Fin 26

/-- Represents a sequence of 26 letters -/
def Sequence := Fin 26 → Letter

/-- The sum operation for letters -/
def letter_sum (a b : Letter) : Letter :=
  ⟨(a.val + b.val) % 26, by sorry⟩

/-- The sum operation for sequences -/
def sequence_sum (s1 s2 : Sequence) : Sequence :=
  λ i => letter_sum (s1 i) (s2 i)

/-- The standard alphabet sequence -/
def alphabet_sequence : Sequence :=
  λ i => i

/-- A permutation of the alphabet -/
def is_permutation (s : Sequence) : Prop :=
  Function.Injective s

theorem alphabet_sum_theorem (s : Sequence) (h : is_permutation s) :
  ∃ i j : Fin 26, i ≠ j ∧ sequence_sum s alphabet_sequence i = sequence_sum s alphabet_sequence j :=
sorry

end alphabet_sum_theorem_l870_87098


namespace tangent_line_through_point_l870_87016

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the tangent line equation
def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := 3 * x₀^2 * (x - x₀) + x₀^3

-- State the theorem
theorem tangent_line_through_point :
  ∃ (x₀ : ℝ), (tangent_line x₀ 1 = 1) ∧
  ((tangent_line x₀ x = 3*x - 2) ∨ (tangent_line x₀ x = 3/4*x + 1/4)) :=
sorry

end tangent_line_through_point_l870_87016


namespace manolo_total_masks_l870_87099

/-- Represents the number of face-masks Manolo can make in a given time period -/
def masks_made (rate : ℕ) (duration : ℕ) : ℕ :=
  (duration * 60) / rate

/-- Represents Manolo's six-hour shift face-mask production -/
def manolo_shift_production : ℕ :=
  masks_made 4 1 + masks_made 6 2 + masks_made 8 2

theorem manolo_total_masks :
  manolo_shift_production = 50 := by
  sorry

end manolo_total_masks_l870_87099


namespace solution_set_g_range_of_m_l870_87007

-- Define the functions f and g
def f (x : ℝ) := x^2 - 2*x - 8
def g (x : ℝ) := 2*x^2 - 4*x - 16

-- Theorem for the solution set of g(x) < 0
theorem solution_set_g (x : ℝ) : g x < 0 ↔ -2 < x ∧ x < 4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x > 2, f x ≥ (m + 2) * x - m - 15) → m ≤ 2 := by sorry

end solution_set_g_range_of_m_l870_87007


namespace max_correct_answers_is_19_l870_87044

/-- Represents the result of an exam -/
structure ExamResult where
  total_questions : Nat
  correct_answers : Nat
  wrong_answers : Nat
  unanswered : Nat
  score : Int

/-- Checks if an ExamResult is valid according to the given scoring system -/
def is_valid_result (result : ExamResult) : Prop :=
  result.total_questions = 25 ∧
  result.correct_answers + result.wrong_answers + result.unanswered = result.total_questions ∧
  4 * result.correct_answers - result.wrong_answers = result.score

/-- Theorem: The maximum number of correct answers for a score of 70 is 19 -/
theorem max_correct_answers_is_19 :
  ∀ result : ExamResult,
    is_valid_result result →
    result.score = 70 →
    result.correct_answers ≤ 19 ∧
    ∃ optimal_result : ExamResult,
      is_valid_result optimal_result ∧
      optimal_result.score = 70 ∧
      optimal_result.correct_answers = 19 :=
by sorry

#check max_correct_answers_is_19

end max_correct_answers_is_19_l870_87044


namespace brad_balloons_l870_87059

theorem brad_balloons (total red blue : ℕ) (h1 : total = 50) (h2 : red = 12) (h3 : blue = 7) :
  total - (red + blue) = 31 := by
  sorry

end brad_balloons_l870_87059


namespace prime_remainder_30_l870_87014

theorem prime_remainder_30 (p : ℕ) (h : Prime p) : 
  let r := p % 30
  Prime r ∨ r = 1 := by
sorry

end prime_remainder_30_l870_87014


namespace line_circle_intersection_l870_87042

theorem line_circle_intersection (k : ℝ) :
  ∃ (x y : ℝ), y = k * (x - 1) ∧ x^2 + y^2 = 1 := by sorry

end line_circle_intersection_l870_87042


namespace base_conversion_2869_to_base_7_l870_87078

theorem base_conversion_2869_to_base_7 :
  2869 = 1 * (7^4) + 1 * (7^3) + 2 * (7^2) + 3 * (7^1) + 6 * (7^0) :=
by sorry

end base_conversion_2869_to_base_7_l870_87078


namespace inequality_proof_l870_87008

def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2*|a - b|) := by
  sorry

end inequality_proof_l870_87008


namespace mars_radius_scientific_notation_l870_87015

theorem mars_radius_scientific_notation :
  3395000 = 3.395 * (10 ^ 6) := by sorry

end mars_radius_scientific_notation_l870_87015


namespace skittles_bought_proof_l870_87020

/-- The number of Skittles Brenda initially had -/
def initial_skittles : ℕ := 7

/-- The number of Skittles Brenda ended up with -/
def final_skittles : ℕ := 15

/-- The number of Skittles Brenda bought -/
def bought_skittles : ℕ := final_skittles - initial_skittles

theorem skittles_bought_proof :
  bought_skittles = final_skittles - initial_skittles :=
by sorry

end skittles_bought_proof_l870_87020


namespace no_integer_solution_l870_87093

theorem no_integer_solution : ¬ ∃ (a b : ℤ), a^2 + b^2 = 10^100 + 3 := by
  sorry

end no_integer_solution_l870_87093


namespace trip_length_satisfies_conditions_l870_87089

/-- Represents the total trip length in miles -/
def total_trip_length : ℝ := 180

/-- Represents the distance traveled on battery power in miles -/
def battery_distance : ℝ := 60

/-- Represents the fuel consumption rate in gallons per mile when using gasoline -/
def fuel_consumption_rate : ℝ := 0.03

/-- Represents the average fuel efficiency for the entire trip in miles per gallon -/
def average_fuel_efficiency : ℝ := 50

/-- Theorem stating that the total trip length satisfies the given conditions -/
theorem trip_length_satisfies_conditions :
  (total_trip_length / (fuel_consumption_rate * (total_trip_length - battery_distance)) = average_fuel_efficiency) ∧
  (total_trip_length > battery_distance) := by
  sorry

#check trip_length_satisfies_conditions

end trip_length_satisfies_conditions_l870_87089


namespace smallest_number_of_cubes_is_56_l870_87092

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSide := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSide) * (box.width / cubeSide) * (box.depth / cubeSide)

/-- Theorem stating that the smallest number of cubes to fill the given box is 56 -/
theorem smallest_number_of_cubes_is_56 :
  smallestNumberOfCubes ⟨35, 20, 10⟩ = 56 := by
  sorry

#eval smallestNumberOfCubes ⟨35, 20, 10⟩

end smallest_number_of_cubes_is_56_l870_87092


namespace pascal_ratio_98_l870_87032

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Pascal's Triangle property: each entry is the sum of the two entries directly above it -/
axiom pascal_property (n k : ℕ) : binomial (n + 1) k = binomial n (k - 1) + binomial n k

/-- Three consecutive entries in Pascal's Triangle -/
def consecutive_entries (n r : ℕ) : (ℕ × ℕ × ℕ) :=
  (binomial n r, binomial n (r + 1), binomial n (r + 2))

/-- Ratio of three numbers -/
def in_ratio (a b c : ℕ) (x y z : ℕ) : Prop :=
  a * y = b * x ∧ b * z = c * y

theorem pascal_ratio_98 : ∃ r : ℕ, in_ratio (binomial 98 r) (binomial 98 (r + 1)) (binomial 98 (r + 2)) 4 5 6 := by
  sorry

end pascal_ratio_98_l870_87032


namespace grid_problem_l870_87075

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Represents the grid of numbers -/
structure NumberGrid where
  row : ArithmeticSequence
  col1 : ArithmeticSequence
  col2 : ArithmeticSequence

/-- The main theorem -/
theorem grid_problem (g : NumberGrid) :
  g.row.first = 16 ∧
  g.col1.first + g.col1.diff = 10 ∧
  g.col1.first + 2 * g.col1.diff = 19 ∧
  g.col2.first + 4 * g.col2.diff = -13 ∧
  g.row.first + 6 * g.row.diff = g.col2.first + 4 * g.col2.diff →
  g.col2.first = -36.75 := by
  sorry

end grid_problem_l870_87075


namespace min_value_expression_l870_87061

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b * c = 1) (h2 : a / b = 2) :
  a^2 + 4*a*b + 9*b^2 + 8*b*c + 3*c^2 ≥ 3 * (63 : ℝ)^(1/3) := by
  sorry

end min_value_expression_l870_87061


namespace all_terms_even_l870_87080

theorem all_terms_even (p q : ℤ) (hp : Even p) (hq : Even q) :
  ∀ k : ℕ, k ≤ 8 → Even (Nat.choose 8 k * p^(8 - k) * q^k) := by sorry

end all_terms_even_l870_87080


namespace is_quadratic_equation_example_l870_87000

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation 2x^2 = 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- Theorem: The equation 2x^2 = 1 is a quadratic equation in one variable -/
theorem is_quadratic_equation_example : is_quadratic_equation f := by
  sorry

end is_quadratic_equation_example_l870_87000


namespace handshake_problem_l870_87034

theorem handshake_problem (n : ℕ) (total_handshakes : ℕ) 
  (h1 : n = 12) 
  (h2 : total_handshakes = 66) : 
  total_handshakes = n * (n - 1) / 2 ∧ 
  (total_handshakes / (n - 1) : ℚ) = 6 := by
sorry

end handshake_problem_l870_87034


namespace burt_basil_profit_l870_87090

/-- Calculate the net profit from Burt's basil plants -/
theorem burt_basil_profit :
  let seed_cost : ℕ := 200  -- in cents
  let soil_cost : ℕ := 800  -- in cents
  let total_plants : ℕ := 20
  let price_per_plant : ℕ := 500  -- in cents
  
  let total_cost : ℕ := seed_cost + soil_cost
  let total_revenue : ℕ := total_plants * price_per_plant
  let net_profit : ℤ := total_revenue - total_cost
  
  net_profit = 9000  -- 90.00 in cents
  := by sorry

end burt_basil_profit_l870_87090


namespace complex_parts_of_z_l870_87026

theorem complex_parts_of_z : ∃ z : ℂ, z = Complex.I ^ 2 + Complex.I ∧ z.re = -1 ∧ z.im = 1 := by
  sorry

end complex_parts_of_z_l870_87026


namespace sum_of_squares_l870_87051

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 22) (h2 : x * y = 40) : x^2 + y^2 = 404 := by
  sorry

end sum_of_squares_l870_87051


namespace inequality_holds_l870_87094

open Real

/-- A function satisfying the given differential equation -/
def SatisfiesDiffEq (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, x * (deriv^[2] f x) + 2 * f x = 1 / x^2

theorem inequality_holds (f : ℝ → ℝ) (hf : SatisfiesDiffEq f) :
  f 2 / 9 < f 3 / 4 := by
  sorry

end inequality_holds_l870_87094


namespace triangle_area_is_20_16_l870_87035

/-- Represents a line in 2D space --/
structure Line where
  slope : ℚ
  point : ℚ × ℚ

/-- Calculates the area of a triangle given three points --/
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

/-- Finds the intersection point of a line with the line x + y = 12 --/
def intersectionWithSum12 (l : Line) : ℚ × ℚ :=
  let (a, b) := l.point
  let m := l.slope
  let x := (12 - b + m*a) / (m + 1)
  (x, 12 - x)

theorem triangle_area_is_20_16 (l1 l2 : Line) :
  l1.point = (4, 4) →
  l2.point = (4, 4) →
  l1.slope = 2/3 →
  l2.slope = 3/2 →
  let p1 := (4, 4)
  let p2 := intersectionWithSum12 l1
  let p3 := intersectionWithSum12 l2
  triangleArea p1 p2 p3 = 20.16 := by
sorry

end triangle_area_is_20_16_l870_87035


namespace colored_dodecahedron_constructions_l870_87085

/-- The number of faces in a dodecahedron -/
def num_faces : ℕ := 12

/-- The number of rotational symmetries considered for simplification -/
def rotational_symmetries : ℕ := 5

/-- The number of distinguishable ways to construct a colored dodecahedron -/
def distinguishable_constructions : ℕ := Nat.factorial (num_faces - 1) / rotational_symmetries

/-- Theorem stating the number of distinguishable ways to construct a colored dodecahedron -/
theorem colored_dodecahedron_constructions :
  distinguishable_constructions = 7983360 := by
  sorry

#eval distinguishable_constructions

end colored_dodecahedron_constructions_l870_87085


namespace circle_radius_is_one_l870_87021

/-- The radius of a circle defined by the equation x^2 + y^2 - 2y = 0 is 1 -/
theorem circle_radius_is_one :
  let circle_eq := (fun x y : ℝ => x^2 + y^2 - 2*y = 0)
  ∃ (h k r : ℝ), r = 1 ∧ ∀ x y : ℝ, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end circle_radius_is_one_l870_87021


namespace main_theorem_l870_87072

/-- The set of real numbers c > 0 for which exactly one of two statements is true --/
def C : Set ℝ := {c | c > 0 ∧ (c ≤ 1/2 ∨ c ≥ 1)}

/-- Statement p: The function y = c^x is monotonically decreasing on ℝ --/
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

/-- Statement q: The solution set of x + |x - 2c| > 1 is ℝ --/
def q (c : ℝ) : Prop := ∀ x : ℝ, x + |x - 2*c| > 1

/-- Main theorem: c is in set C if and only if exactly one of p(c) or q(c) is true --/
theorem main_theorem (c : ℝ) : c ∈ C ↔ (p c ∧ ¬q c) ∨ (¬p c ∧ q c) := by
  sorry

end main_theorem_l870_87072


namespace kho_kho_only_players_l870_87005

theorem kho_kho_only_players (total : ℕ) (kabadi : ℕ) (both : ℕ) (kho_kho_only : ℕ) : 
  total = 45 → kabadi = 10 → both = 5 → kho_kho_only = total - kabadi + both :=
by
  sorry

end kho_kho_only_players_l870_87005


namespace modular_arithmetic_problem_l870_87071

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (7 * a) % 77 = 1 ∧ 
    (13 * b) % 77 = 1 ∧ 
    ((3 * a + 9 * b) % 77) = 10 :=
by sorry

end modular_arithmetic_problem_l870_87071


namespace nancy_clay_pots_l870_87027

/-- The number of clay pots Nancy created on Monday -/
def monday_pots : ℕ := 12

/-- The number of clay pots Nancy created on Tuesday -/
def tuesday_pots : ℕ := 2 * monday_pots

/-- The number of clay pots Nancy created on Wednesday -/
def wednesday_pots : ℕ := 14

/-- The total number of clay pots Nancy created by the end of the week -/
def total_pots : ℕ := monday_pots + tuesday_pots + wednesday_pots

theorem nancy_clay_pots : total_pots = 50 := by sorry

end nancy_clay_pots_l870_87027


namespace inequality_proof_l870_87040

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end inequality_proof_l870_87040


namespace factorial_sum_unique_solution_l870_87056

theorem factorial_sum_unique_solution :
  ∀ n a b c : ℕ, n.factorial = a.factorial + b.factorial + c.factorial →
  n = 3 ∧ a = 2 ∧ b = 2 ∧ c = 2 :=
by sorry

end factorial_sum_unique_solution_l870_87056


namespace fir_tree_needles_l870_87073

/-- Represents the number of fir trees in the forest -/
def num_trees : ℕ := 710000

/-- Represents the maximum number of needles a tree can have -/
def max_needles : ℕ := 100000

/-- Represents the minimum number of trees we want to prove have the same number of needles -/
def min_same_needles : ℕ := 7

theorem fir_tree_needles :
  ∃ (n : ℕ) (trees : Finset (Fin num_trees)),
    n ≤ max_needles ∧
    trees.card ≥ min_same_needles ∧
    ∀ t ∈ trees, (fun i => i.val) t = n :=
by sorry

end fir_tree_needles_l870_87073


namespace union_of_M_and_complement_of_N_l870_87009

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def M : Finset ℕ := {1, 3, 4}
def N : Finset ℕ := {3, 5, 6}

theorem union_of_M_and_complement_of_N :
  M ∪ (U \ N) = {1, 2, 3, 4} := by sorry

end union_of_M_and_complement_of_N_l870_87009


namespace certain_amount_proof_l870_87018

theorem certain_amount_proof (x : ℝ) (A : ℝ) : 
  x = 780 → 
  (0.25 * x) = (0.15 * 1500 - A) → 
  A = 30 := by
sorry

end certain_amount_proof_l870_87018


namespace jet_flight_time_l870_87096

/-- Given a jet flying with and against wind, calculate the time taken with tail wind -/
theorem jet_flight_time (distance : ℝ) (return_time : ℝ) (wind_speed : ℝ) 
  (h1 : distance = 2000)
  (h2 : return_time = 5)
  (h3 : wind_speed = 50) : 
  ∃ (jet_speed : ℝ) (tail_wind_time : ℝ),
    (jet_speed + wind_speed) * tail_wind_time = distance ∧
    (jet_speed - wind_speed) * return_time = distance ∧
    tail_wind_time = 4 := by
  sorry

end jet_flight_time_l870_87096


namespace divisibility_by_p_squared_l870_87019

theorem divisibility_by_p_squared (p : ℕ) (h_prime : Nat.Prime p) (h_greater_than_three : p > 3) :
  ∃ k : ℤ, (p + 1 : ℤ)^(p - 1) - 1 = k * p^2 := by
  sorry

end divisibility_by_p_squared_l870_87019


namespace star_properties_l870_87054

def star (x y : ℤ) : ℤ := (x + 2) * (y + 2) - 3

theorem star_properties :
  (∀ x y : ℤ, star x y = star y x) ∧
  (∃ x y z : ℤ, star x (y + z) ≠ star x y + star x z) ∧
  (∃ x : ℤ, star (x - 2) (x + 2) ≠ star x x - 3) ∧
  (∃ x : ℤ, star x 1 ≠ x) := by
  sorry

end star_properties_l870_87054


namespace min_a_squared_over_area_l870_87084

theorem min_a_squared_over_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  b * Real.cos C + c * Real.cos B = 3 * a * Real.cos A →
  S = (1 / 2) * b * c * Real.sin A →
  a^2 / S ≥ 2 * Real.sqrt 2 := by
  sorry

#check min_a_squared_over_area

end min_a_squared_over_area_l870_87084


namespace cone_from_sector_l870_87038

/-- Proves that a cone formed from a 300° sector of a circle with radius 8 
    has a base radius of 20/3 and a slant height of 8 -/
theorem cone_from_sector (sector_angle : Real) (circle_radius : Real) 
    (cone_base_radius : Real) (cone_slant_height : Real) : 
    sector_angle = 300 ∧ 
    circle_radius = 8 ∧ 
    cone_base_radius = 20 / 3 ∧ 
    cone_slant_height = circle_radius → 
    cone_base_radius * 2 * π = sector_angle / 360 * (2 * π * circle_radius) ∧
    cone_slant_height = circle_radius := by
  sorry

end cone_from_sector_l870_87038


namespace exchange_impossibility_l870_87037

theorem exchange_impossibility : ¬ ∃ (N : ℕ), 5 * N = 2001 := by sorry

end exchange_impossibility_l870_87037


namespace stock_price_calculation_l870_87052

theorem stock_price_calculation (initial_price : ℝ) : 
  let first_year_increase : ℝ := 1.5
  let second_year_decrease : ℝ := 0.3
  let third_year_increase : ℝ := 0.2
  let price_after_first_year : ℝ := initial_price * (1 + first_year_increase)
  let price_after_second_year : ℝ := price_after_first_year * (1 - second_year_decrease)
  let final_price : ℝ := price_after_second_year * (1 + third_year_increase)
  initial_price = 120 → final_price = 252 := by
sorry

end stock_price_calculation_l870_87052


namespace least_possible_third_side_l870_87060

theorem least_possible_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a = 8 ∧ b = 15) ∨ (a = 8 ∧ c = 15) ∨ (b = 8 ∧ c = 15) →
  a^2 + b^2 = c^2 →
  Real.sqrt 161 ≤ min a (min b c) :=
by sorry

end least_possible_third_side_l870_87060


namespace area_ratio_of_concentric_circles_l870_87023

/-- Two concentric circles with center O -/
structure ConcentricCircles where
  O : Point
  r₁ : ℝ  -- radius of smaller circle
  r₂ : ℝ  -- radius of larger circle
  h : 0 < r₁ ∧ r₁ < r₂

/-- The length of an arc on a circle -/
def arcLength (r : ℝ) (θ : ℝ) : ℝ := r * θ

theorem area_ratio_of_concentric_circles (C : ConcentricCircles) :
  arcLength C.r₁ (π/3) = arcLength C.r₂ (π/4) →
  (C.r₁^2 / C.r₂^2 : ℝ) = 9/16 := by
  sorry

#check area_ratio_of_concentric_circles

end area_ratio_of_concentric_circles_l870_87023


namespace expansion_distinct_terms_l870_87070

/-- The number of distinct terms in the expansion of a product of two sums -/
def distinctTermsInExpansion (n m : ℕ) : ℕ := n * m

/-- Theorem: The number of distinct terms in the expansion of (a+b+c)(d+e+f+g+h+i) is 18 -/
theorem expansion_distinct_terms :
  distinctTermsInExpansion 3 6 = 18 := by
  sorry

end expansion_distinct_terms_l870_87070


namespace more_men_than_women_count_l870_87049

def num_men : ℕ := 6
def num_women : ℕ := 4
def group_size : ℕ := 5

def select_group (m w : ℕ) : ℕ := Nat.choose num_men m * Nat.choose num_women w

theorem more_men_than_women_count : 
  (select_group 3 2) + (select_group 4 1) + (select_group 5 0) = 186 :=
by sorry

end more_men_than_women_count_l870_87049


namespace binary_11100_to_quaternary_l870_87053

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its quaternary representation (as a list of digits) -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary number 11100₂ -/
def binary_11100 : List Bool := [true, true, true, false, false]

theorem binary_11100_to_quaternary :
  decimal_to_quaternary (binary_to_decimal binary_11100) = [1, 3, 0] :=
sorry

end binary_11100_to_quaternary_l870_87053


namespace stacy_berries_l870_87030

theorem stacy_berries (skylar_berries steve_berries stacy_berries : ℕ) : 
  skylar_berries = 20 →
  steve_berries = skylar_berries / 2 →
  stacy_berries = 3 * steve_berries + 2 →
  stacy_berries = 32 := by
  sorry

end stacy_berries_l870_87030


namespace coefficient_of_x_l870_87041

/-- The coefficient of x in the expression 3(x - 4) + 4(7 - 2x^2 + 5x) - 8(2x - 1) is 7 -/
theorem coefficient_of_x (x : ℝ) : 
  let expr := 3*(x - 4) + 4*(7 - 2*x^2 + 5*x) - 8*(2*x - 1)
  ∃ (a b c : ℝ), expr = a*x^2 + 7*x + c :=
by
  sorry

end coefficient_of_x_l870_87041


namespace seulgi_stack_higher_l870_87083

/-- Represents the stack of boxes for each person -/
structure BoxStack where
  numBoxes : ℕ
  boxHeight : ℝ

/-- Calculates the total height of a stack of boxes -/
def totalHeight (stack : BoxStack) : ℝ :=
  stack.numBoxes * stack.boxHeight

theorem seulgi_stack_higher (hyunjeong seulgi : BoxStack)
  (h1 : hyunjeong.numBoxes = 15)
  (h2 : hyunjeong.boxHeight = 4.2)
  (h3 : seulgi.numBoxes = 20)
  (h4 : seulgi.boxHeight = 3.3) :
  totalHeight seulgi > totalHeight hyunjeong := by
  sorry

end seulgi_stack_higher_l870_87083


namespace parallelogram_area_l870_87001

/-- The area of a parallelogram with diagonals intersecting at a 60° angle
    and two sides of lengths 6 and 8 is equal to 14√3. -/
theorem parallelogram_area (a b : ℝ) : 
  (a^2 + b^2 - a*b = 36) →  -- From the side of length 6
  (a^2 + b^2 + a*b = 64) →  -- From the side of length 8
  2 * a * b * (Real.sqrt 3 / 2) = 14 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l870_87001


namespace plane_parallel_transitivity_l870_87017

-- Define the concept of planes
variable (Plane : Type)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Theorem statement
theorem plane_parallel_transitivity (α β γ : Plane) :
  (∃ γ, parallel γ α ∧ parallel γ β) → parallel α β := by
  sorry

end plane_parallel_transitivity_l870_87017


namespace cube_root_unity_product_l870_87066

theorem cube_root_unity_product (w : ℂ) : w^3 = 1 → (1 - w + w^2) * (1 + w - w^2) = 4 := by
  sorry

end cube_root_unity_product_l870_87066


namespace average_movers_to_texas_l870_87004

/-- The number of people moving to Texas over 5 days -/
def total_people : ℕ := 3500

/-- The number of days -/
def num_days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def average_per_hour : ℚ :=
  total_people / (num_days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem average_movers_to_texas :
  round_to_nearest average_per_hour = 29 := by
  sorry

end average_movers_to_texas_l870_87004


namespace symmetric_point_coordinates_l870_87012

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to x-axis
def symmetricXAxis (p q : Point2D) : Prop :=
  p.x = q.x ∧ p.y = -q.y

-- Theorem statement
theorem symmetric_point_coordinates :
  ∀ (B A : Point2D),
    B.x = 4 ∧ B.y = -1 →
    symmetricXAxis A B →
    A.x = 4 ∧ A.y = 1 := by
  sorry

end symmetric_point_coordinates_l870_87012


namespace age_difference_is_four_l870_87048

/-- Gladys' current age -/
def gladys_age : ℕ := 40 - 10

/-- Juanico's current age -/
def juanico_age : ℕ := 41 - 30

/-- The difference between half of Gladys' age and Juanico's age -/
def age_difference : ℕ := (gladys_age / 2) - juanico_age

theorem age_difference_is_four : age_difference = 4 := by
  sorry

end age_difference_is_four_l870_87048


namespace triple_sharp_40_l870_87088

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.7 * N + 2

-- State the theorem
theorem triple_sharp_40 : sharp (sharp (sharp 40)) = 18.1 := by
  sorry

end triple_sharp_40_l870_87088


namespace correct_boat_equation_l870_87003

/-- Represents the scenario of boats and students during the Qingming Festival outing. -/
structure BoatScenario where
  totalBoats : ℕ
  largeboatCapacity : ℕ
  smallboatCapacity : ℕ
  totalStudents : ℕ

/-- The equation representing the boat scenario. -/
def boatEquation (scenario : BoatScenario) (x : ℕ) : Prop :=
  scenario.largeboatCapacity * (scenario.totalBoats - x) + scenario.smallboatCapacity * x = scenario.totalStudents

/-- Theorem stating that the given equation correctly represents the boat scenario. -/
theorem correct_boat_equation (scenario : BoatScenario) (h1 : scenario.totalBoats = 8) 
    (h2 : scenario.largeboatCapacity = 6) (h3 : scenario.smallboatCapacity = 4) 
    (h4 : scenario.totalStudents = 38) : 
  boatEquation scenario = fun x => 6 * (8 - x) + 4 * x = 38 := by
  sorry


end correct_boat_equation_l870_87003


namespace additional_cars_needed_l870_87006

def current_cars : ℕ := 37
def cars_per_row : ℕ := 9

theorem additional_cars_needed :
  let next_multiple := ((current_cars + cars_per_row - 1) / cars_per_row) * cars_per_row
  next_multiple - current_cars = 8 := by
  sorry

end additional_cars_needed_l870_87006


namespace work_completion_time_l870_87067

/-- The number of days it takes for person B to complete the work alone -/
def B_days : ℝ := 60

/-- The fraction of work completed by A and B together in 6 days -/
def work_completed : ℝ := 0.25

/-- The number of days A and B work together -/
def days_together : ℝ := 6

/-- The number of days it takes for person A to complete the work alone -/
def A_days : ℝ := 40

theorem work_completion_time :
  (1 / A_days + 1 / B_days) * days_together = work_completed :=
sorry

end work_completion_time_l870_87067
