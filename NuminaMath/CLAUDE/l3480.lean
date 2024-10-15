import Mathlib

namespace NUMINAMATH_CALUDE_ratio_w_y_l3480_348063

-- Define the variables
variable (w x y z : ℚ)

-- Define the given ratios
def ratio_w_x : w / x = 5 / 4 := by sorry
def ratio_y_z : y / z = 4 / 3 := by sorry
def ratio_z_x : z / x = 1 / 8 := by sorry

-- Theorem to prove
theorem ratio_w_y (hw : w / x = 5 / 4) (hy : y / z = 4 / 3) (hz : z / x = 1 / 8) :
  w / y = 15 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_w_y_l3480_348063


namespace NUMINAMATH_CALUDE_problem_solution_l3480_348098

def f (x a : ℝ) : ℝ := |2*x - 1| + |x + a|

theorem problem_solution :
  (∀ x : ℝ, f x 1 ≥ 3 ↔ x ≥ 1 ∨ x ≤ -1) ∧
  ((∃ x : ℝ, f x a ≤ |a - 1|) → a ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3480_348098


namespace NUMINAMATH_CALUDE_roots_sum_squares_and_product_l3480_348018

theorem roots_sum_squares_and_product (α β : ℝ) : 
  (2 * α^2 - α - 4 = 0) → (2 * β^2 - β - 4 = 0) → α^2 + α*β + β^2 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_squares_and_product_l3480_348018


namespace NUMINAMATH_CALUDE_solve_a_b_l3480_348060

def U : Set ℝ := Set.univ

def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + 12*b = 0}

def B (a b : ℝ) : Set ℝ := {x | x^2 - a*x + b = 0}

theorem solve_a_b :
  ∀ a b : ℝ, 
    (2 ∈ (U \ A a b) ∩ (B a b)) → 
    (4 ∈ (A a b) ∩ (U \ B a b)) → 
    a = 8/7 ∧ b = -12/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_a_b_l3480_348060


namespace NUMINAMATH_CALUDE_expression_evaluation_l3480_348004

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  x^2 + y^2 - 3*z^2 + 2*x*y + 2*y*z - 2*x*z = -44 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3480_348004


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l3480_348043

theorem multiplication_subtraction_equality : 120 * 2400 - 20 * 2400 - 100 * 2400 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l3480_348043


namespace NUMINAMATH_CALUDE_tank_capacity_l3480_348016

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 3/4 →
  added_amount = 9 →
  final_fraction = 7/8 →
  (initial_fraction * C + added_amount = final_fraction * C) →
  C = 72 :=
by sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l3480_348016


namespace NUMINAMATH_CALUDE_second_number_solution_l3480_348003

theorem second_number_solution (x : ℝ) :
  12.1212 + x - 9.1103 = 20.011399999999995 →
  x = 18.000499999999995 := by
sorry

end NUMINAMATH_CALUDE_second_number_solution_l3480_348003


namespace NUMINAMATH_CALUDE_odd_function_product_nonpositive_l3480_348082

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_product_nonpositive (f : ℝ → ℝ) (h : odd_function f) :
  ∀ x : ℝ, f x * f (-x) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_product_nonpositive_l3480_348082


namespace NUMINAMATH_CALUDE_biography_increase_l3480_348002

theorem biography_increase (B : ℝ) (N : ℝ) (h1 : N > 0) (h2 : B > 0)
  (h3 : 0.20 * B + N = 0.30 * (B + N)) :
  (N / (0.20 * B)) * 100 = 100 / 1.4 := by
  sorry

end NUMINAMATH_CALUDE_biography_increase_l3480_348002


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3480_348099

theorem log_equality_implies_golden_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 9 = Real.log q / Real.log 12) ∧
  (Real.log p / Real.log 9 = Real.log (p + q) / Real.log 16) →
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3480_348099


namespace NUMINAMATH_CALUDE_x_n_perfect_square_iff_b_10_l3480_348071

def x_n (b n : ℕ) : ℕ :=
  let ones := (b^(2*n) - b^(n+1)) / (b - 1)
  let twos := 2 * (b^n - 1) / (b - 1)
  ones + twos + 5

theorem x_n_perfect_square_iff_b_10 (b : ℕ) (h : b > 5) :
  (∃ M : ℕ, ∀ n : ℕ, n > M → ∃ k : ℕ, x_n b n = k^2) ↔ b = 10 :=
sorry

end NUMINAMATH_CALUDE_x_n_perfect_square_iff_b_10_l3480_348071


namespace NUMINAMATH_CALUDE_repair_cost_is_5000_l3480_348083

/-- Calculates the repair cost for a machine given its purchase price, transportation cost, selling price, and profit percentage. -/
def repair_cost (purchase_price transportation_cost selling_price profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + transportation_cost + (selling_price * 100 / (100 + profit_percentage) - purchase_price - transportation_cost)
  selling_price * 100 / (100 + profit_percentage) - purchase_price - transportation_cost

/-- Theorem stating that for the given conditions, the repair cost is 5000. -/
theorem repair_cost_is_5000 :
  repair_cost 10000 1000 24000 50 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_is_5000_l3480_348083


namespace NUMINAMATH_CALUDE_f_positive_when_x_positive_smallest_a_for_g_inequality_l3480_348048

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - (2 * x) / (x + 2)

noncomputable def g (x : ℝ) : ℝ := f x - 4 / (x + 2)

theorem f_positive_when_x_positive (x : ℝ) (hx : x > 0) : f x > 0 := by
  sorry

theorem smallest_a_for_g_inequality : 
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → g x < x + a) ↔ a > -2 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_when_x_positive_smallest_a_for_g_inequality_l3480_348048


namespace NUMINAMATH_CALUDE_mythical_zoo_count_l3480_348076

theorem mythical_zoo_count (total_heads : ℕ) (total_legs : ℕ) : 
  total_heads = 300 → 
  total_legs = 798 → 
  ∃ (two_legged three_legged : ℕ), 
    two_legged + three_legged = total_heads ∧ 
    2 * two_legged + 3 * three_legged = total_legs ∧ 
    two_legged = 102 := by
sorry

end NUMINAMATH_CALUDE_mythical_zoo_count_l3480_348076


namespace NUMINAMATH_CALUDE_shirt_cost_problem_l3480_348019

theorem shirt_cost_problem (total_cost : ℕ) (num_shirts : ℕ) (num_known_shirts : ℕ) (known_shirt_cost : ℕ) (h1 : total_cost = 85) (h2 : num_shirts = 5) (h3 : num_known_shirts = 3) (h4 : known_shirt_cost = 15) :
  let remaining_shirts := num_shirts - num_known_shirts
  let remaining_cost := total_cost - (num_known_shirts * known_shirt_cost)
  remaining_cost / remaining_shirts = 20 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_problem_l3480_348019


namespace NUMINAMATH_CALUDE_volume_ratio_in_divided_tetrahedron_l3480_348097

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Represents the ratio of distances on an edge -/
def ratio (P Q R : Point3D) : ℝ := sorry

/-- Theorem: Volume ratio in a divided tetrahedron -/
theorem volume_ratio_in_divided_tetrahedron (ABCD : Tetrahedron) 
  (P : Point3D) (Q : Point3D) (R : Point3D) (S : Point3D)
  (hP : ratio P ABCD.A ABCD.B = 1)
  (hQ : ratio Q ABCD.B ABCD.D = 1/2)
  (hR : ratio R ABCD.C ABCD.D = 1/2)
  (hS : ratio S ABCD.A ABCD.C = 1)
  (V1 V2 : ℝ)
  (hV : V1 < V2)
  (hV1V2 : V1 + V2 = volume ABCD)
  : V1 / V2 = 13 / 23 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_in_divided_tetrahedron_l3480_348097


namespace NUMINAMATH_CALUDE_debby_dvd_sale_l3480_348064

theorem debby_dvd_sale (original : ℕ) (left : ℕ) (sold : ℕ) : 
  original = 13 → left = 7 → sold = original - left → sold = 6 := by sorry

end NUMINAMATH_CALUDE_debby_dvd_sale_l3480_348064


namespace NUMINAMATH_CALUDE_eightiethDigitIsOne_l3480_348055

/-- The sequence of digits formed by concatenating consecutive integers from 60 to 1 in descending order -/
def descendingSequence : List Nat := sorry

/-- The 80th digit in the descendingSequence -/
def eightiethDigit : Nat := sorry

/-- Theorem stating that the 80th digit in the sequence is 1 -/
theorem eightiethDigitIsOne : eightiethDigit = 1 := by sorry

end NUMINAMATH_CALUDE_eightiethDigitIsOne_l3480_348055


namespace NUMINAMATH_CALUDE_circle_sectors_and_square_area_l3480_348034

/-- Given a circle with radius 6 and two perpendicular diameters, 
    prove that the sum of the areas of two 120° sectors and 
    the square formed by connecting the diameter endpoints 
    is equal to 24π + 144. -/
theorem circle_sectors_and_square_area :
  let r : ℝ := 6
  let sector_angle : ℝ := 120
  let sector_area := (sector_angle / 360) * π * r^2
  let square_side := 2 * r
  let square_area := square_side^2
  2 * sector_area + square_area = 24 * π + 144 := by
sorry

end NUMINAMATH_CALUDE_circle_sectors_and_square_area_l3480_348034


namespace NUMINAMATH_CALUDE_range_of_a_l3480_348000

-- Define the sets N and M
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x + a - 2) < 0}
def M : Set ℝ := {x | -1/2 ≤ x ∧ x < 2}

-- State the theorem
theorem range_of_a :
  (∀ x, x ∈ M → x ∈ N a) → (a ≤ -1/2 ∨ a ≥ 5/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3480_348000


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3480_348051

-- Define the propositions p and q
def p (x y : ℝ) : Prop := x + y > 2 ∧ x * y > 1
def q (x y : ℝ) : Prop := x > 1 ∧ y > 1

-- Theorem statement
theorem p_necessary_not_sufficient :
  (∀ x y : ℝ, q x y → p x y) ∧
  ¬(∀ x y : ℝ, p x y → q x y) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3480_348051


namespace NUMINAMATH_CALUDE_last_digit_of_3_to_2010_l3480_348062

theorem last_digit_of_3_to_2010 (h : ∀ n : ℕ, 
  (3^n % 10) = (3^(n % 4) % 10)) : 
  3^2010 % 10 = 9 := by
sorry

end NUMINAMATH_CALUDE_last_digit_of_3_to_2010_l3480_348062


namespace NUMINAMATH_CALUDE_dane_daughters_flowers_l3480_348038

def flowers_per_basket (initial_flowers_per_daughter : ℕ) (daughters : ℕ) (new_flowers : ℕ) (dead_flowers : ℕ) (baskets : ℕ) : ℕ :=
  ((initial_flowers_per_daughter * daughters + new_flowers) - dead_flowers) / baskets

theorem dane_daughters_flowers :
  flowers_per_basket 5 2 20 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_dane_daughters_flowers_l3480_348038


namespace NUMINAMATH_CALUDE_money_at_departure_l3480_348015

def money_at_arrival : ℕ := 87
def money_difference : ℕ := 71

theorem money_at_departure : 
  money_at_arrival - money_difference = 16 := by sorry

end NUMINAMATH_CALUDE_money_at_departure_l3480_348015


namespace NUMINAMATH_CALUDE_xyz_product_l3480_348023

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 2 * y = -8)
  (eq2 : y * z + 2 * z = -8)
  (eq3 : z * x + 2 * x = -8) : 
  x * y * z = 32 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l3480_348023


namespace NUMINAMATH_CALUDE_donut_holes_problem_l3480_348021

/-- Given the number of mini-cupcakes, students, and desserts per student,
    calculate the number of donut holes needed. -/
def donut_holes_needed (mini_cupcakes : ℕ) (students : ℕ) (desserts_per_student : ℕ) : ℕ :=
  students * desserts_per_student - mini_cupcakes

/-- Theorem stating that given 14 mini-cupcakes, 13 students, and 2 desserts per student,
    the number of donut holes needed is 12. -/
theorem donut_holes_problem :
  donut_holes_needed 14 13 2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_donut_holes_problem_l3480_348021


namespace NUMINAMATH_CALUDE_store_shirts_count_l3480_348057

theorem store_shirts_count (shirts_sold : ℕ) (shirts_left : ℕ) :
  shirts_sold = 21 →
  shirts_left = 28 →
  shirts_sold + shirts_left = 49 :=
by sorry

end NUMINAMATH_CALUDE_store_shirts_count_l3480_348057


namespace NUMINAMATH_CALUDE_f_plus_a_over_e_positive_sum_less_than_two_l3480_348014

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 1) * Real.exp x

theorem f_plus_a_over_e_positive (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, f a x + a / Real.exp 1 > 0 := by sorry

theorem sum_less_than_two (x₁ x₂ : ℝ) (h1 : x₁ ≠ x₂) (h2 : f (-1/2) x₁ = f (-1/2) x₂) :
  x₁ + x₂ < 2 := by sorry

end

end NUMINAMATH_CALUDE_f_plus_a_over_e_positive_sum_less_than_two_l3480_348014


namespace NUMINAMATH_CALUDE_fifth_term_is_negative_three_l3480_348046

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 5th term of the sequence is -3 -/
theorem fifth_term_is_negative_three
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_third : a 3 = -5)
  (h_ninth : a 9 = 1) :
  a 5 = -3 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_is_negative_three_l3480_348046


namespace NUMINAMATH_CALUDE_total_cost_theorem_l3480_348033

-- Define the cost of individual items
def eraser_cost : ℝ := sorry
def pen_cost : ℝ := sorry
def marker_cost : ℝ := sorry

-- Define the conditions
axiom condition1 : eraser_cost + 3 * pen_cost + 2 * marker_cost = 240
axiom condition2 : 2 * eraser_cost + 4 * marker_cost + 5 * pen_cost = 440

-- Define the theorem to prove
theorem total_cost_theorem :
  3 * eraser_cost + 4 * pen_cost + 6 * marker_cost = 520 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l3480_348033


namespace NUMINAMATH_CALUDE_composition_value_l3480_348054

/-- Given two functions f and g, prove that g(f(3)) = 1902 -/
theorem composition_value (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = x^3 - 2) 
  (hg : ∀ x, g x = 3*x^2 + x + 2) : 
  g (f 3) = 1902 := by
  sorry

end NUMINAMATH_CALUDE_composition_value_l3480_348054


namespace NUMINAMATH_CALUDE_squares_in_figure_50_l3480_348059

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 2

/-- The sequence satisfies the given initial conditions -/
axiom initial_conditions :
  f 0 = 2 ∧ f 1 = 8 ∧ f 2 = 18 ∧ f 3 = 32

/-- The number of squares in figure 50 is 5202 -/
theorem squares_in_figure_50 : f 50 = 5202 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_figure_50_l3480_348059


namespace NUMINAMATH_CALUDE_symmetry_about_y_axis_l3480_348079

/-- Given two lines in the xy-plane, this function checks if they are symmetric with respect to the y-axis -/
def symmetric_about_y_axis (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line1 x y ↔ line2 (-x) y

/-- The original line: 3x - 4y + 5 = 0 -/
def original_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The symmetric line: 3x + 4y + 22 = 0 -/
def symmetric_line (x y : ℝ) : Prop := 3 * x + 4 * y + 22 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to the y-axis -/
theorem symmetry_about_y_axis : symmetric_about_y_axis original_line symmetric_line := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_y_axis_l3480_348079


namespace NUMINAMATH_CALUDE_correlation_coefficient_measures_linear_relationship_l3480_348067

/-- The correlation coefficient is a measure related to the relationship between two variables. -/
def correlation_coefficient : Type := sorry

/-- The strength of the linear relationship between two variables. -/
def linear_relationship_strength : Type := sorry

/-- The correlation coefficient measures the strength of the linear relationship between two variables. -/
theorem correlation_coefficient_measures_linear_relationship :
  correlation_coefficient = linear_relationship_strength := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_measures_linear_relationship_l3480_348067


namespace NUMINAMATH_CALUDE_barrel_capacity_l3480_348084

theorem barrel_capacity (original_amount : ℝ) (capacity : ℝ) : 
  (original_amount = 3 / 5 * capacity) →
  (original_amount - 18 = 0.6 * original_amount) →
  (capacity = 75) :=
by sorry

end NUMINAMATH_CALUDE_barrel_capacity_l3480_348084


namespace NUMINAMATH_CALUDE_jeans_final_price_is_correct_l3480_348065

def socks_price : ℝ := 5
def tshirt_price : ℝ := socks_price + 10
def jeans_price : ℝ := 2 * tshirt_price
def jeans_discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08

def jeans_final_price : ℝ :=
  let discounted_price := jeans_price * (1 - jeans_discount_rate)
  discounted_price * (1 + sales_tax_rate)

theorem jeans_final_price_is_correct :
  jeans_final_price = 27.54 := by sorry

end NUMINAMATH_CALUDE_jeans_final_price_is_correct_l3480_348065


namespace NUMINAMATH_CALUDE_difference_of_squares_l3480_348068

theorem difference_of_squares (y : ℝ) : y^2 - 4 = (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3480_348068


namespace NUMINAMATH_CALUDE_sum_of_squared_differences_equals_three_l3480_348045

theorem sum_of_squared_differences_equals_three (a b c : ℝ) 
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : 
  (b - c)^2 / ((a - b) * (a - c)) + 
  (c - a)^2 / ((b - c) * (b - a)) + 
  (a - b)^2 / ((c - a) * (c - b)) = 3 := by
  sorry

#check sum_of_squared_differences_equals_three

end NUMINAMATH_CALUDE_sum_of_squared_differences_equals_three_l3480_348045


namespace NUMINAMATH_CALUDE_inequality_proof_l3480_348081

theorem inequality_proof (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 1/2) :
  (k + a/(b+c)) * (k + b/(c+a)) * (k + c/(a+b)) ≥ (k + 1/2)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3480_348081


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l3480_348094

/-- Arithmetic sequence {a_n} -/
def arithmetic_sequence (n : ℕ) : ℝ := 2 * n - 2

/-- Geometric sequence {b_n} -/
def geometric_sequence (n : ℕ) : ℝ := 2^(n - 1)

/-- Sum of first n terms of geometric sequence -/
def geometric_sum (n : ℕ) : ℝ := 2^n - 1

theorem arithmetic_geometric_sequences :
  (∀ n : ℕ, arithmetic_sequence n = 2 * n - 2) ∧
  arithmetic_sequence 2 = 2 ∧
  arithmetic_sequence 5 = 8 ∧
  (∀ n : ℕ, geometric_sequence n > 0) ∧
  geometric_sequence 1 = 1 ∧
  geometric_sequence 2 + geometric_sequence 3 = arithmetic_sequence 4 ∧
  (∀ n : ℕ, geometric_sum n = 2^n - 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l3480_348094


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3480_348086

theorem sum_of_three_numbers : 0.8 + (1 / 2 : ℚ) + 0.9 = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3480_348086


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l3480_348093

theorem quadrilateral_inequality (A B P : ℝ) (θ₁ θ₂ : ℝ) 
  (hA : A > 0) (hB : B > 0) (hP : P > 0) 
  (hP_bound : P ≤ A + B)
  (h_cos : A * Real.cos θ₁ + B * Real.cos θ₂ = P) :
  A * Real.sin θ₁ + B * Real.sin θ₂ ≤ Real.sqrt ((A + B - P) * (A + B + P)) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l3480_348093


namespace NUMINAMATH_CALUDE_plane_parallel_transitivity_l3480_348053

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the parallel relation between planes
def parallel (p q : Plane) : Prop := sorry

-- State the theorem
theorem plane_parallel_transitivity (α β γ : Plane) :
  parallel γ α → parallel γ β → parallel α β := by sorry

end NUMINAMATH_CALUDE_plane_parallel_transitivity_l3480_348053


namespace NUMINAMATH_CALUDE_range_of_a_l3480_348025

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 6) * (x - 2*a - 5) > 0}
def B (a : ℝ) : Set ℝ := {x | (a^2 + 2 - x) * (2*a - x) < 0}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, 
    a > 1/2 → 
    (∀ x : ℝ, x ∈ B a → x ∈ A a) → 
    (∃ x : ℝ, x ∈ A a ∧ x ∉ B a) → 
    a > 1/2 ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3480_348025


namespace NUMINAMATH_CALUDE_golden_ratio_cosine_l3480_348085

theorem golden_ratio_cosine (golden_ratio : ℝ) (h1 : golden_ratio = (Real.sqrt 5 - 1) / 2) 
  (h2 : golden_ratio = 2 * Real.sin (18 * π / 180)) : 
  Real.cos (36 * π / 180) = (Real.sqrt 5 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_cosine_l3480_348085


namespace NUMINAMATH_CALUDE_negation_of_symmetry_for_all_l3480_348075

-- Define a type for functions
variable {α : Type*} [LinearOrder α]

-- Define symmetry about y=x
def symmetric_about_y_eq_x (f : α → α) : Prop :=
  ∀ x y, f y = x ↔ f x = y

-- State the theorem
theorem negation_of_symmetry_for_all :
  (¬ ∀ f : α → α, symmetric_about_y_eq_x f) ↔
  (∃ f : α → α, ¬ symmetric_about_y_eq_x f) :=
sorry

end NUMINAMATH_CALUDE_negation_of_symmetry_for_all_l3480_348075


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3480_348087

/-- Definition of the repeating decimal 0.3333... -/
def repeating_3 : ℚ := 1/3

/-- Definition of the repeating decimal 0.0404... -/
def repeating_04 : ℚ := 4/99

/-- Definition of the repeating decimal 0.005005... -/
def repeating_005 : ℚ := 5/999

/-- Theorem stating that the sum of the three repeating decimals equals 1135/2997 -/
theorem sum_of_repeating_decimals : 
  repeating_3 + repeating_04 + repeating_005 = 1135/2997 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3480_348087


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3480_348024

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3480_348024


namespace NUMINAMATH_CALUDE_inequality_holds_l3480_348066

theorem inequality_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3480_348066


namespace NUMINAMATH_CALUDE_inequality_implies_b_minus_a_equals_two_l3480_348006

theorem inequality_implies_b_minus_a_equals_two (a b : ℝ) :
  (∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) →
  b - a = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_b_minus_a_equals_two_l3480_348006


namespace NUMINAMATH_CALUDE_local_face_value_difference_l3480_348040

/-- The numeral we're working with -/
def numeral : ℕ := 65793

/-- The digit we're focusing on -/
def digit : ℕ := 7

/-- The place value of the digit in the numeral (hundreds) -/
def place_value : ℕ := 100

/-- The local value of the digit in the numeral -/
def local_value : ℕ := digit * place_value

/-- The face value of the digit -/
def face_value : ℕ := digit

/-- Theorem stating the difference between local value and face value -/
theorem local_face_value_difference :
  local_value - face_value = 693 := by sorry

end NUMINAMATH_CALUDE_local_face_value_difference_l3480_348040


namespace NUMINAMATH_CALUDE_strap_problem_l3480_348041

theorem strap_problem (shorter longer : ℝ) 
  (h1 : shorter + longer = 64)
  (h2 : longer = shorter + 48) :
  longer / shorter = 7 := by
  sorry

end NUMINAMATH_CALUDE_strap_problem_l3480_348041


namespace NUMINAMATH_CALUDE_paul_sold_63_books_l3480_348096

/-- The number of books Paul sold in a garage sale --/
def books_sold_in_garage_sale (initial_books donated_books exchanged_books given_to_friend remaining_books : ℕ) : ℕ :=
  initial_books - donated_books - given_to_friend - remaining_books

/-- Theorem stating that Paul sold 63 books in the garage sale --/
theorem paul_sold_63_books :
  books_sold_in_garage_sale 250 50 20 35 102 = 63 := by
  sorry

end NUMINAMATH_CALUDE_paul_sold_63_books_l3480_348096


namespace NUMINAMATH_CALUDE_crabapple_recipients_count_l3480_348030

/-- The number of students in Mrs. Crabapple's class -/
def num_students : ℕ := 12

/-- The number of times the class meets in a week -/
def class_meetings_per_week : ℕ := 5

/-- The number of possible sequences of crabapple recipients in a week -/
def crabapple_sequences : ℕ := num_students ^ class_meetings_per_week

/-- Theorem stating the number of possible sequences of crabapple recipients -/
theorem crabapple_recipients_count : crabapple_sequences = 248832 := by
  sorry

end NUMINAMATH_CALUDE_crabapple_recipients_count_l3480_348030


namespace NUMINAMATH_CALUDE_eliminate_y_by_addition_l3480_348022

/-- Given a system of two linear equations in two variables x and y,
    prove that adding the first equation to twice the second equation
    eliminates the y variable. -/
theorem eliminate_y_by_addition (a b c d e f : ℝ) :
  let eq1 := (a * x + b * y = e)
  let eq2 := (c * x + d * y = f)
  (b = -2 * d) →
  ∃ k, (a * x + b * y) + 2 * (c * x + d * y) = k * x + e + 2 * f :=
by sorry

end NUMINAMATH_CALUDE_eliminate_y_by_addition_l3480_348022


namespace NUMINAMATH_CALUDE_some_base_value_l3480_348047

theorem some_base_value (x y some_base : ℝ) 
  (h1 : x * y = 1)
  (h2 : (some_base ^ ((x + y)^2)) / (some_base ^ ((x - y)^2)) = 256) :
  some_base = 4 := by
sorry

end NUMINAMATH_CALUDE_some_base_value_l3480_348047


namespace NUMINAMATH_CALUDE_candy_probability_l3480_348061

def total_candies : ℕ := 24
def red_candies : ℕ := 12
def blue_candies : ℕ := 12
def terry_picks : ℕ := 2
def mary_picks : ℕ := 3

def same_color_probability : ℚ := 66 / 1771

theorem candy_probability :
  red_candies = blue_candies ∧
  red_candies + blue_candies = total_candies ∧
  terry_picks + mary_picks < total_candies →
  same_color_probability = (2 * (Nat.choose red_candies terry_picks * Nat.choose (red_candies - terry_picks) mary_picks)) / 
                           (Nat.choose total_candies terry_picks * Nat.choose (total_candies - terry_picks) mary_picks) :=
by sorry

end NUMINAMATH_CALUDE_candy_probability_l3480_348061


namespace NUMINAMATH_CALUDE_mean_inequalities_l3480_348009

theorem mean_inequalities (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
  (x + y + z) / 3 > (x * y * z) ^ (1/3) ∧ (x * y * z) ^ (1/3) > 3 * x * y * z / (x * y + y * z + z * x) :=
by sorry

end NUMINAMATH_CALUDE_mean_inequalities_l3480_348009


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l3480_348026

/-- Given positive real numbers a, b, and c such that the minimum value of |x - a| + |x + b| + c is 1 -/
def min_condition (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∀ x, |x - a| + |x + b| + c ≥ 1) ∧ (∃ x, |x - a| + |x + b| + c = 1)

theorem min_value_and_inequality (a b c : ℝ) (h : min_condition a b c) :
  (∀ x y z, 9*x^2 + 4*y^2 + (1/4)*z^2 ≥ 36/157) ∧
  (9*a^2 + 4*b^2 + (1/4)*c^2 = 36/157) ∧
  (Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (b^2 + b*c + c^2) + Real.sqrt (c^2 + c*a + a^2) > 3/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l3480_348026


namespace NUMINAMATH_CALUDE_sqrt_problem_l3480_348001

theorem sqrt_problem (m n a : ℝ) : 
  (∃ (x : ℝ), x^2 = m ∧ x = 3) → 
  (∃ (y z : ℝ), y^2 = n ∧ z^2 = n ∧ y = a + 4 ∧ z = 2*a - 16) →
  m = 9 ∧ n = 64 ∧ (7*m - n)^(1/3) = -1 := by sorry

end NUMINAMATH_CALUDE_sqrt_problem_l3480_348001


namespace NUMINAMATH_CALUDE_max_sequence_length_l3480_348088

/-- A sequence satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) (n m : ℕ) : Prop :=
  (∀ k, k < n → a k ≤ m) ∧
  (∀ k, 1 < k ∧ k < n - 1 → a (k - 1) ≠ a (k + 1)) ∧
  (∀ i₁ i₂ i₃ i₄, i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ < n →
    ¬(a i₁ = a i₃ ∧ a i₁ ≠ a i₂ ∧ a i₂ = a i₄))

/-- The maximum length of a valid sequence -/
def MaxSequenceLength (m : ℕ) : ℕ :=
  4 * m - 2

/-- Theorem: The maximum length of a valid sequence is 4m - 2 -/
theorem max_sequence_length (m : ℕ) (h : m > 0) :
  (∃ a n, n = MaxSequenceLength m ∧ ValidSequence a n m) ∧
  (∀ a n, ValidSequence a n m → n ≤ MaxSequenceLength m) :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_l3480_348088


namespace NUMINAMATH_CALUDE_right_triangle_area_from_sticks_l3480_348007

/-- Represents a stick of length 24 cm that can be broken into two pieces -/
structure Stick :=
  (length : ℝ := 24)
  (piece1 : ℝ)
  (piece2 : ℝ)
  (break_constraint : piece1 + piece2 = length)

/-- Represents a right triangle formed from three sticks -/
structure RightTriangle :=
  (leg1 : ℝ)
  (leg2 : ℝ)
  (hypotenuse : ℝ)
  (pythagorean : leg1^2 + leg2^2 = hypotenuse^2)

/-- Theorem stating that if a right triangle can be formed from three 24 cm sticks
    (one of which is broken), then its area is 216 square centimeters -/
theorem right_triangle_area_from_sticks 
  (s1 s2 : Stick) (s3 : Stick) (t : RightTriangle)
  (h1 : s1.length = 24 ∧ s2.length = 24 ∧ s3.length = 24)
  (h2 : t.leg1 = s1.piece1 ∧ t.leg2 = s2.length ∧ t.hypotenuse = s1.piece2 + s3.length) :
  t.leg1 * t.leg2 / 2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_from_sticks_l3480_348007


namespace NUMINAMATH_CALUDE_fraction_undefined_values_l3480_348078

def undefined_values (b : ℝ) : Prop :=
  b^2 - 9 = 0

theorem fraction_undefined_values :
  {b : ℝ | undefined_values b} = {-3, 3} := by
  sorry

end NUMINAMATH_CALUDE_fraction_undefined_values_l3480_348078


namespace NUMINAMATH_CALUDE_nine_team_league_games_l3480_348091

/-- The number of games played in a league where each team plays every other team once -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 9 teams, where each team plays every other team once, 
    the total number of games played is 36 -/
theorem nine_team_league_games :
  num_games 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_team_league_games_l3480_348091


namespace NUMINAMATH_CALUDE_sample_size_is_ten_l3480_348028

/-- Represents a collection of products -/
structure ProductCollection where
  total : Nat
  selected : Nat
  random_selection : selected ≤ total

/-- Definition of sample size for a product collection -/
def sample_size (pc : ProductCollection) : Nat := pc.selected

/-- Theorem: For a product collection with 80 total products and 10 randomly selected,
    the sample size is 10 -/
theorem sample_size_is_ten (pc : ProductCollection) 
  (h1 : pc.total = 80) 
  (h2 : pc.selected = 10) : 
  sample_size pc = 10 := by
  sorry


end NUMINAMATH_CALUDE_sample_size_is_ten_l3480_348028


namespace NUMINAMATH_CALUDE_largest_m_for_quadratic_inequality_l3480_348073

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem largest_m_for_quadratic_inequality 
  (a b c : ℝ) 
  (ha : a ≠ 0)
  (h1 : ∀ x : ℝ, f a b c (x - 4) = f a b c (2 - x))
  (h2 : ∀ x : ℝ, f a b c x ≥ x)
  (h3 : ∀ x ∈ Set.Ioo 0 2, f a b c x ≤ ((x + 1) / 2)^2)
  (h4 : ∃ x : ℝ, ∀ y : ℝ, f a b c x ≤ f a b c y)
  (h5 : ∃ x : ℝ, f a b c x = 0) :
  (∃ m : ℝ, m > 1 ∧ 
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) ∧
    (∀ m' : ℝ, m' > m → 
      ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x)) ∧
  (∀ m : ℝ, (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) → m ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_quadratic_inequality_l3480_348073


namespace NUMINAMATH_CALUDE_third_quadrant_trig_sum_l3480_348069

theorem third_quadrant_trig_sum (α : Real) : 
  π < α ∧ α < 3*π/2 → 
  |Real.sin (α/2)| / Real.sin (α/2) + |Real.cos (α/2)| / Real.cos (α/2) = 0 := by
sorry

end NUMINAMATH_CALUDE_third_quadrant_trig_sum_l3480_348069


namespace NUMINAMATH_CALUDE_function_through_point_l3480_348027

/-- Proves that if the function y = k/x passes through the point (3, -1), then k = -3 -/
theorem function_through_point (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = k / x) ∧ f 3 = -1) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_through_point_l3480_348027


namespace NUMINAMATH_CALUDE_base4_21012_to_decimal_l3480_348032

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

theorem base4_21012_to_decimal :
  base4_to_decimal [2, 1, 0, 1, 2] = 582 := by
  sorry

end NUMINAMATH_CALUDE_base4_21012_to_decimal_l3480_348032


namespace NUMINAMATH_CALUDE_fruit_basket_count_l3480_348080

/-- The number of ways to choose k items from n identical items -/
def choose_with_repetition (n : ℕ) (k : ℕ) : ℕ := (n + k - 1).choose k

/-- The number of possible fruit baskets -/
def fruit_baskets (apples oranges : ℕ) : ℕ :=
  (apples) * (choose_with_repetition (oranges + 1) 1)

theorem fruit_basket_count :
  fruit_baskets 7 12 = 91 :=
by sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l3480_348080


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3480_348029

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3480_348029


namespace NUMINAMATH_CALUDE_triangle_side_length_l3480_348049

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 3 → b = 5 → C = 2 * π / 3 → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3480_348049


namespace NUMINAMATH_CALUDE_jamies_father_weight_loss_l3480_348056

/-- Jamie's father's weight loss problem -/
theorem jamies_father_weight_loss 
  (calories_burned_per_day : ℕ)
  (calories_eaten_per_day : ℕ)
  (calories_per_pound : ℕ)
  (days_to_lose_weight : ℕ)
  (h1 : calories_burned_per_day = 2500)
  (h2 : calories_eaten_per_day = 2000)
  (h3 : calories_per_pound = 3500)
  (h4 : days_to_lose_weight = 35) :
  (days_to_lose_weight * (calories_burned_per_day - calories_eaten_per_day)) / calories_per_pound = 5 := by
  sorry


end NUMINAMATH_CALUDE_jamies_father_weight_loss_l3480_348056


namespace NUMINAMATH_CALUDE_first_triangular_year_21st_century_l3480_348010

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- First triangular number year in the 21st century -/
theorem first_triangular_year_21st_century :
  ∃ n : ℕ, triangular n = 2016 ∧ 
  (∀ m : ℕ, triangular m ≥ 2000 → triangular n ≤ triangular m) := by
  sorry

end NUMINAMATH_CALUDE_first_triangular_year_21st_century_l3480_348010


namespace NUMINAMATH_CALUDE_officer_selection_count_l3480_348092

/-- The number of ways to choose officers from a club -/
def choose_officers (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else (n - k + 1).factorial / (n - k).factorial

/-- Theorem: Choosing 5 officers from 15 members results in 360,360 possibilities -/
theorem officer_selection_count :
  choose_officers 15 5 = 360360 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_count_l3480_348092


namespace NUMINAMATH_CALUDE_absolute_value_reciprocal_graph_l3480_348036

theorem absolute_value_reciprocal_graph (x : ℝ) (x_nonzero : x ≠ 0) :
  (1 / |x|) = if x > 0 then 1 / x else -1 / x :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_reciprocal_graph_l3480_348036


namespace NUMINAMATH_CALUDE_one_third_of_nine_x_minus_three_l3480_348008

theorem one_third_of_nine_x_minus_three (x : ℝ) : (1 / 3) * (9 * x - 3) = 3 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_nine_x_minus_three_l3480_348008


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3480_348090

theorem parallel_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (2 • a - b)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3480_348090


namespace NUMINAMATH_CALUDE_page_difference_l3480_348013

/-- The number of purple books Mirella read -/
def purple_books : ℕ := 8

/-- The number of orange books Mirella read -/
def orange_books : ℕ := 7

/-- The number of pages in each purple book -/
def purple_pages : ℕ := 320

/-- The number of pages in each orange book -/
def orange_pages : ℕ := 640

/-- The difference between the total number of orange pages and purple pages read by Mirella -/
theorem page_difference : 
  orange_books * orange_pages - purple_books * purple_pages = 1920 := by
  sorry

end NUMINAMATH_CALUDE_page_difference_l3480_348013


namespace NUMINAMATH_CALUDE_total_lives_calculation_l3480_348020

theorem total_lives_calculation (initial_players additional_players lives_per_player : ℕ) :
  initial_players = 4 →
  additional_players = 5 →
  lives_per_player = 3 →
  (initial_players + additional_players) * lives_per_player = 27 :=
by sorry

end NUMINAMATH_CALUDE_total_lives_calculation_l3480_348020


namespace NUMINAMATH_CALUDE_min_chinese_score_l3480_348035

/-- Represents the scores of a student in three subjects -/
structure Scores where
  chinese : ℝ
  mathematics : ℝ
  english : ℝ

/-- The average score of the three subjects is 92 -/
def average_score (s : Scores) : Prop :=
  (s.chinese + s.mathematics + s.english) / 3 = 92

/-- Each subject has a maximum score of 100 points -/
def max_score (s : Scores) : Prop :=
  s.chinese ≤ 100 ∧ s.mathematics ≤ 100 ∧ s.english ≤ 100

/-- The Mathematics score is 4 points higher than the Chinese score -/
def math_chinese_relation (s : Scores) : Prop :=
  s.mathematics = s.chinese + 4

/-- The minimum possible score for Chinese is 86 points -/
theorem min_chinese_score (s : Scores) 
  (h1 : average_score s) 
  (h2 : max_score s) 
  (h3 : math_chinese_relation s) : 
  s.chinese ≥ 86 := by
  sorry

end NUMINAMATH_CALUDE_min_chinese_score_l3480_348035


namespace NUMINAMATH_CALUDE_complex_coordinate_l3480_348052

theorem complex_coordinate (z : ℂ) (h : Complex.I * z = 1 + 2 * Complex.I) : z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinate_l3480_348052


namespace NUMINAMATH_CALUDE_intersection_point_AB_CD_l3480_348039

def A : ℝ × ℝ × ℝ := (8, -9, 5)
def B : ℝ × ℝ × ℝ := (18, -19, 15)
def C : ℝ × ℝ × ℝ := (2, 5, -8)
def D : ℝ × ℝ × ℝ := (4, -3, 12)

def line_intersection (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

theorem intersection_point_AB_CD :
  line_intersection A B C D = (16, -19, 13) := by sorry

end NUMINAMATH_CALUDE_intersection_point_AB_CD_l3480_348039


namespace NUMINAMATH_CALUDE_ellipse_and_hyperbola_equations_l3480_348089

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of a line (asymptote) -/
structure Line where
  m : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y = m * x

def foci : (Point × Point) := ⟨⟨-5, 0⟩, ⟨5, 0⟩⟩
def intersectionPoint : Point := ⟨4, 3⟩

/-- Theorem stating the equations of the ellipse and hyperbola -/
theorem ellipse_and_hyperbola_equations 
  (e : Ellipse) 
  (h : Hyperbola) 
  (l : Line) 
  (hfoci : e.a^2 - e.b^2 = h.a^2 + h.b^2 ∧ e.a^2 - e.b^2 = 25) 
  (hpoint_on_ellipse : e.equation intersectionPoint.x intersectionPoint.y) 
  (hpoint_on_line : l.equation intersectionPoint.x intersectionPoint.y) 
  (hline_is_asymptote : l.m = h.b / h.a) :
  e.a^2 = 40 ∧ e.b^2 = 15 ∧ h.a^2 = 16 ∧ h.b^2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_and_hyperbola_equations_l3480_348089


namespace NUMINAMATH_CALUDE_other_ticket_price_l3480_348077

/-- Represents the ticket sales scenario for the Red Rose Theatre --/
def theatre_sales (other_price : ℝ) : Prop :=
  let total_tickets : ℕ := 380
  let cheap_tickets : ℕ := 205
  let cheap_price : ℝ := 4.50
  let total_revenue : ℝ := 1972.50
  (cheap_tickets : ℝ) * cheap_price + (total_tickets - cheap_tickets : ℝ) * other_price = total_revenue

/-- Theorem stating that the price of the other tickets is $6.00 --/
theorem other_ticket_price : ∃ (price : ℝ), theatre_sales price ∧ price = 6 := by
  sorry

end NUMINAMATH_CALUDE_other_ticket_price_l3480_348077


namespace NUMINAMATH_CALUDE_total_free_sides_length_l3480_348011

/-- A rectangular table with one side against a wall -/
structure RectangularTable where
  /-- Length of the side opposite the wall -/
  opposite_side : ℝ
  /-- Length of each of the other two free sides -/
  adjacent_side : ℝ
  /-- The side opposite the wall is twice the length of each adjacent side -/
  opposite_twice_adjacent : opposite_side = 2 * adjacent_side
  /-- The area of the table is 128 square feet -/
  area_is_128 : opposite_side * adjacent_side = 128

/-- The total length of the table's free sides is 32 feet -/
theorem total_free_sides_length (table : RectangularTable) :
  table.opposite_side + 2 * table.adjacent_side = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_free_sides_length_l3480_348011


namespace NUMINAMATH_CALUDE_white_pawn_on_white_square_l3480_348005

/-- Represents a chessboard with white and black pawns. -/
structure Chessboard where
  white_pawns : ℕ
  black_pawns : ℕ
  pawns_on_white_squares : ℕ
  pawns_on_black_squares : ℕ

/-- Theorem: Given a chessboard with more white pawns than black pawns,
    and more pawns on white squares than on black squares,
    there exists at least one white pawn on a white square. -/
theorem white_pawn_on_white_square (board : Chessboard)
  (h1 : board.white_pawns > board.black_pawns)
  (h2 : board.pawns_on_white_squares > board.pawns_on_black_squares) :
  ∃ (white_pawns_on_white_squares : ℕ), white_pawns_on_white_squares > 0 := by
  sorry

end NUMINAMATH_CALUDE_white_pawn_on_white_square_l3480_348005


namespace NUMINAMATH_CALUDE_books_about_sports_l3480_348050

theorem books_about_sports (total_books school_books : ℕ) 
  (h1 : total_books = 58) 
  (h2 : school_books = 19) : 
  total_books - school_books = 39 :=
sorry

end NUMINAMATH_CALUDE_books_about_sports_l3480_348050


namespace NUMINAMATH_CALUDE_range_of_a_l3480_348074

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f satisfying the given conditions -/
noncomputable def f (a : ℝ) : ℝ → ℝ
| x => if x < 0 then 9*x + a^2/x + 7 else 
       if x > 0 then 9*x + a^2/x - 7 else 0

theorem range_of_a (a : ℝ) : 
  (IsOddFunction (f a)) → 
  (∀ x ≥ 0, f a x ≥ a + 1) →
  a ≤ -8/7 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3480_348074


namespace NUMINAMATH_CALUDE_square_odd_implies_odd_l3480_348072

theorem square_odd_implies_odd (n : ℕ) : Odd (n^2) → Odd n := by
  sorry

end NUMINAMATH_CALUDE_square_odd_implies_odd_l3480_348072


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l3480_348037

theorem geometric_progression_problem :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (b / a = c / b) ∧
    (a + b + c = 65) ∧
    (a * b * c = 3375) ∧
    ((a = 5 ∧ b = 15 ∧ c = 45) ∨ (a = 45 ∧ b = 15 ∧ c = 5)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l3480_348037


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l3480_348017

/-- The function f(x) = ln x - 3/x has a zero point in the interval (2, 3) -/
theorem zero_point_in_interval (f : ℝ → ℝ) :
  (∀ x > 0, f x = Real.log x - 3 / x) →
  (∀ x > 0, StrictMono f) →
  ∃ c ∈ Set.Ioo 2 3, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l3480_348017


namespace NUMINAMATH_CALUDE_equation_transformation_l3480_348031

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^3 + x^2 - 6*x + 1 = 0 ↔ x*(x^2*y - 6) + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_transformation_l3480_348031


namespace NUMINAMATH_CALUDE_total_area_calculation_l3480_348012

def original_length : ℝ := 13
def original_width : ℝ := 18
def increase : ℝ := 2
def num_equal_rooms : ℕ := 4
def num_double_rooms : ℕ := 1

def new_length : ℝ := original_length + increase
def new_width : ℝ := original_width + increase

def room_area : ℝ := new_length * new_width

theorem total_area_calculation :
  (num_equal_rooms : ℝ) * room_area + (num_double_rooms : ℝ) * 2 * room_area = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_area_calculation_l3480_348012


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l3480_348095

theorem two_digit_number_problem (A B : ℕ) : 
  (A ≥ 1 ∧ A ≤ 9) →  -- A is a digit from 1 to 9 (tens digit)
  (B ≥ 0 ∧ B ≤ 9) →  -- B is a digit from 0 to 9 (ones digit)
  (10 * A + B) - 21 = 14 →  -- The equation AB - 21 = 14
  B = 5 := by  -- We want to prove B = 5
sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l3480_348095


namespace NUMINAMATH_CALUDE_article_cost_is_60_l3480_348070

/-- Proves that the cost of an article is 60, given the specified conditions --/
theorem article_cost_is_60 (cost : ℝ) (selling_price : ℝ) : 
  (selling_price = 1.25 * cost) →
  (0.8 * cost + 0.3 * (0.8 * cost) = selling_price - 12.6) →
  cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_is_60_l3480_348070


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3480_348042

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 → 
    4 * Real.pi * r^2 = 36 * Real.pi → 
    (4 / 3) * Real.pi * r^3 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3480_348042


namespace NUMINAMATH_CALUDE_parameterized_to_ordinary_equation_l3480_348058

theorem parameterized_to_ordinary_equation :
  ∀ (x y t : ℝ),
  (x = Real.sqrt t ∧ y = 2 * Real.sqrt (1 - t)) →
  (x^2 + y^2 / 4 = 1 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_parameterized_to_ordinary_equation_l3480_348058


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l3480_348044

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Theorem stating that for an arithmetic sequence with S_17 = 170, a_7 + a_8 + a_12 = 30 -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) (h : seq.S 17 = 170) :
  seq.a 7 + seq.a 8 + seq.a 12 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l3480_348044
