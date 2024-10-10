import Mathlib

namespace cupcake_cost_proof_l3691_369149

theorem cupcake_cost_proof (total_cupcakes : ℕ) (people : ℕ) (cost_per_person : ℚ) :
  total_cupcakes = 12 →
  people = 2 →
  cost_per_person = 9 →
  (people * cost_per_person) / total_cupcakes = 1.5 :=
by sorry

end cupcake_cost_proof_l3691_369149


namespace no_divisible_by_nine_l3691_369148

def base_n_number (n : ℕ) : ℕ := 3 + 2*n + 1*n^2 + 0*n^3 + 3*n^4 + 2*n^5

theorem no_divisible_by_nine :
  ∀ n : ℕ, 2 ≤ n → n ≤ 100 → ¬(base_n_number n % 9 = 0) := by
sorry

end no_divisible_by_nine_l3691_369148


namespace line_equation_proof_l3691_369125

/-- A parameterization of a line in R² -/
structure LineParam where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The equation of a line in the form y = mx + b -/
structure LineEquation where
  m : ℝ
  b : ℝ

/-- Given parameterization represents a line -/
axiom is_line (p : LineParam) : True

/-- The given parameterization of the line -/
def given_param : LineParam where
  x := λ t => 2 * t + 4
  y := λ t => 4 * t - 5

/-- The equation we want to prove -/
def target_equation : LineEquation where
  m := 2
  b := -13

/-- Theorem: The given parameterized line has the equation y = 2x - 13 -/
theorem line_equation_proof :
  ∀ t : ℝ, (given_param.y t) = target_equation.m * (given_param.x t) + target_equation.b :=
sorry

end line_equation_proof_l3691_369125


namespace fibonacci_square_property_l3691_369183

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Proposition: N^2 = 1 + n(N + n) iff (N, n) are consecutive Fibonacci numbers -/
theorem fibonacci_square_property (N n : ℕ) (hN : N > 0) (hn : n > 0) :
  N^2 = 1 + n * (N + n) ↔ ∃ i : ℕ, i > 0 ∧ N = fib (i + 1) ∧ n = fib i :=
sorry

end fibonacci_square_property_l3691_369183


namespace associated_functions_range_l3691_369150

/-- Two functions are associated on an interval if their difference has two distinct zeros in that interval. -/
def associated_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ b ∧ f x₁ = g x₁ ∧ f x₂ = g x₂

/-- The statement of the problem. -/
theorem associated_functions_range (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x + 4
  let g : ℝ → ℝ := λ x ↦ 2*x + m
  associated_functions f g 0 3 → -9/4 < m ∧ m ≤ -2 :=
by sorry

end associated_functions_range_l3691_369150


namespace awards_distribution_l3691_369111

/-- The number of ways to distribute n distinct awards to k students, where each student receives at least one award. -/
def distribute_awards (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

theorem awards_distribution :
  distribute_awards 5 3 = 150 :=
by
  sorry

end awards_distribution_l3691_369111


namespace males_not_listening_l3691_369137

theorem males_not_listening (males_listening : ℕ) (females_not_listening : ℕ) 
  (total_listening : ℕ) (total_not_listening : ℕ) 
  (h1 : males_listening = 45)
  (h2 : females_not_listening = 87)
  (h3 : total_listening = 115)
  (h4 : total_not_listening = 160) : 
  total_listening + total_not_listening - (males_listening + (total_listening - males_listening + females_not_listening)) = 73 :=
by sorry

end males_not_listening_l3691_369137


namespace product_of_four_consecutive_integers_divisible_by_ten_l3691_369173

theorem product_of_four_consecutive_integers_divisible_by_ten (n : ℕ) (h : n % 2 = 1) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 10 = 0 := by
  sorry

end product_of_four_consecutive_integers_divisible_by_ten_l3691_369173


namespace quadratic_integer_roots_l3691_369189

theorem quadratic_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + a*x + 2*a = 0 ∧ y^2 + a*y + 2*a = 0) ↔ (a = -1 ∨ a = 9) :=
sorry

end quadratic_integer_roots_l3691_369189


namespace nth_equation_holds_l3691_369109

theorem nth_equation_holds (n : ℕ) (hn : n > 0) : 
  (4 * n^2 : ℚ) / (2 * n - 1) - (2 * n + 1) = 1 - ((2 * n - 2) : ℚ) / (2 * n - 1) := by
  sorry

end nth_equation_holds_l3691_369109


namespace red_shirt_pairs_l3691_369182

theorem red_shirt_pairs (total_students : ℕ) (green_students : ℕ) (red_students : ℕ) 
  (total_pairs : ℕ) (green_green_pairs : ℕ) 
  (h1 : total_students = 144)
  (h2 : green_students = 65)
  (h3 : red_students = 79)
  (h4 : total_pairs = 72)
  (h5 : green_green_pairs = 27)
  (h6 : total_students = green_students + red_students)
  (h7 : total_pairs * 2 = total_students) :
  ∃ red_red_pairs : ℕ, red_red_pairs = 34 ∧ 
    red_red_pairs + green_green_pairs + (green_students - 2 * green_green_pairs) = total_pairs :=
by
  sorry


end red_shirt_pairs_l3691_369182


namespace unique_divisible_by_thirteen_l3691_369178

theorem unique_divisible_by_thirteen :
  ∀ (B : Nat),
    B < 10 →
    (2000 + 100 * B + 34) % 13 = 0 ↔ B = 6 := by
  sorry

end unique_divisible_by_thirteen_l3691_369178


namespace uncovered_area_square_in_square_l3691_369147

theorem uncovered_area_square_in_square (large_side : ℝ) (small_side : ℝ) :
  large_side = 10 →
  small_side = 4 →
  large_side ^ 2 - small_side ^ 2 = 84 := by
  sorry

end uncovered_area_square_in_square_l3691_369147


namespace median_squares_ratio_l3691_369105

/-- Given a triangle with sides a, b, c and corresponding medians ma, mb, mc,
    the ratio of the sum of squares of medians to the sum of squares of sides is 3/4 -/
theorem median_squares_ratio (a b c ma mb mc : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hma : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (hmb : mb^2 = (2*a^2 + 2*c^2 - b^2) / 4)
  (hmc : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  (ma^2 + mb^2 + mc^2) / (a^2 + b^2 + c^2) = 3/4 := by
  sorry

end median_squares_ratio_l3691_369105


namespace ellipse_major_axis_length_l3691_369165

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse in 2D space -/
structure Ellipse where
  center : Point
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- Check if four points form a trapezoid with bases parallel to x-axis -/
def isTrapezoid (p1 p2 p3 p4 : Point) : Prop :=
  (p1.y = p2.y) ∧ (p3.y = p4.y) ∧ (p1.y ≠ p3.y)

/-- Check if a point lies on the vertical bisector of a trapezoid -/
def onVerticalBisector (p : Point) (p1 p2 p3 p4 : Point) : Prop :=
  p.x = (p1.x + p2.x) / 2

theorem ellipse_major_axis_length
  (p1 p2 p3 p4 p5 : Point)
  (h1 : p1 = ⟨0, 0⟩)
  (h2 : p2 = ⟨4, 0⟩)
  (h3 : p3 = ⟨1, 3⟩)
  (h4 : p4 = ⟨3, 3⟩)
  (h5 : p5 = ⟨-1, 3/2⟩)
  (h_trapezoid : isTrapezoid p1 p2 p3 p4)
  (h_bisector : onVerticalBisector p5 p1 p2 p3 p4)
  (e : Ellipse)
  (h_on_ellipse : pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ pointOnEllipse p3 e ∧ pointOnEllipse p4 e ∧ pointOnEllipse p5 e)
  (h_axes_parallel : e.center.x = (p1.x + p2.x) / 2 ∧ e.center.y = (p1.y + p3.y) / 2) :
  2 * e.a = 5 := by
  sorry

end ellipse_major_axis_length_l3691_369165


namespace cubic_quintic_inequality_l3691_369117

theorem cubic_quintic_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := by
  sorry

end cubic_quintic_inequality_l3691_369117


namespace cos_420_degrees_l3691_369197

theorem cos_420_degrees : Real.cos (420 * π / 180) = 1 / 2 := by
  sorry

end cos_420_degrees_l3691_369197


namespace simplify_expression_l3691_369126

theorem simplify_expression : 
  Real.sqrt 27 - Real.sqrt (1/3) + Real.sqrt 12 = 14 * Real.sqrt 3 / 3 := by
  sorry

end simplify_expression_l3691_369126


namespace least_difference_nm_l3691_369130

/-- Given a triangle ABC with sides AB = x+6, AC = 4x, BC = x+12, prove that the least possible 
    value of n-m is 2.5, where m and n are defined such that 1.5 < x < 4, m = 1.5, and n = 4. -/
theorem least_difference_nm (x : ℝ) (m n : ℝ) : 
  x > 0 ∧ 
  (x + 6) + 4*x > (x + 12) ∧
  (x + 6) + (x + 12) > 4*x ∧
  4*x + (x + 12) > (x + 6) ∧
  x + 12 > x + 6 ∧
  x + 12 > 4*x ∧
  m = 1.5 ∧
  n = 4 ∧
  1.5 < x ∧
  x < 4 →
  n - m = 2.5 := by
sorry

end least_difference_nm_l3691_369130


namespace tom_fruit_purchase_l3691_369192

theorem tom_fruit_purchase (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) :
  apple_quantity = 8 →
  apple_rate = 70 →
  mango_quantity = 9 →
  mango_rate = 75 →
  apple_quantity * apple_rate + mango_quantity * mango_rate = 1235 := by
  sorry

end tom_fruit_purchase_l3691_369192


namespace tangent_line_circle_l3691_369122

-- Define the set of real numbers m+n should belong to
def tangent_range : Set ℝ :=
  {x | x ≤ 2 - 2 * Real.sqrt 2 ∨ x ≥ 2 + 2 * Real.sqrt 2}

-- Define the condition for the line to be tangent to the circle
def is_tangent (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), ((m + 1) * x + (n + 1) * y - 2 = 0) ∧
                ((x - 1)^2 + (y - 1)^2 = 1)

-- Theorem statement
theorem tangent_line_circle (m n : ℝ) :
  is_tangent m n → (m + n) ∈ tangent_range := by sorry

end tangent_line_circle_l3691_369122


namespace ellipse_major_axis_length_l3691_369167

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The length of the major axis of the ellipse -/
def major_axis_length : ℝ := 6

/-- Theorem: The length of the major axis of the ellipse defined by x^2/9 + y^2/4 = 1 is 6 -/
theorem ellipse_major_axis_length :
  ∀ x y : ℝ, is_ellipse x y → major_axis_length = 6 := by
  sorry

#check ellipse_major_axis_length

end ellipse_major_axis_length_l3691_369167


namespace solve_salary_problem_l3691_369172

def salary_problem (salaries : List ℝ) (mean : ℝ) : Prop :=
  let n : ℕ := salaries.length + 1
  let total : ℝ := mean * n
  let sum_known : ℝ := salaries.sum
  let sixth_salary : ℝ := total - sum_known
  salaries.length = 5 ∧ 
  mean = 2291.67 ∧
  sixth_salary = 2000.02

theorem solve_salary_problem (salaries : List ℝ) (mean : ℝ) 
  (h1 : salaries = [1000, 2500, 3100, 3650, 1500]) 
  (h2 : mean = 2291.67) : 
  salary_problem salaries mean := by
  sorry

end solve_salary_problem_l3691_369172


namespace arithmetic_sequence_terms_l3691_369196

/-- Prove that an arithmetic sequence starting with 13, ending with 73, 
    and having a common difference of 3 has 21 terms. -/
theorem arithmetic_sequence_terms (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 13 → aₙ = 73 → d = 3 → 
  aₙ = a₁ + (n - 1) * d → n = 21 := by
sorry

end arithmetic_sequence_terms_l3691_369196


namespace total_new_shoes_l3691_369184

def pairs_of_shoes : ℕ := 3
def shoes_per_pair : ℕ := 2

theorem total_new_shoes : pairs_of_shoes * shoes_per_pair = 6 := by
  sorry

end total_new_shoes_l3691_369184


namespace power_mod_prime_remainder_2_100_mod_101_l3691_369134

theorem power_mod_prime (p : Nat) (a : Nat) (h_prime : Nat.Prime p) (h_not_div : ¬(p ∣ a)) :
  a^(p - 1) ≡ 1 [MOD p] := by sorry

theorem remainder_2_100_mod_101 :
  2^100 ≡ 1 [MOD 101] := by
  have h_prime : Nat.Prime 101 := sorry
  have h_not_div : ¬(101 ∣ 2) := sorry
  have h_fermat := power_mod_prime 101 2 h_prime h_not_div
  sorry

end power_mod_prime_remainder_2_100_mod_101_l3691_369134


namespace sum_two_angles_greater_90_implies_acute_l3691_369133

-- Define a triangle type
structure Triangle where
  α : Real
  β : Real
  γ : Real
  angle_sum : α + β + γ = 180
  positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ

-- Define the property of sum of any two angles being greater than 90°
def sum_of_two_angles_greater_than_90 (t : Triangle) : Prop :=
  t.α + t.β > 90 ∧ t.α + t.γ > 90 ∧ t.β + t.γ > 90

-- Define an acute triangle
def is_acute_triangle (t : Triangle) : Prop :=
  t.α < 90 ∧ t.β < 90 ∧ t.γ < 90

-- Theorem statement
theorem sum_two_angles_greater_90_implies_acute (t : Triangle) :
  sum_of_two_angles_greater_than_90 t → is_acute_triangle t :=
by
  sorry


end sum_two_angles_greater_90_implies_acute_l3691_369133


namespace bottle_capacity_l3691_369118

theorem bottle_capacity (V : ℝ) 
  (h1 : V > 0) 
  (h2 : (0.12 * V - 0.24 + 0.12 / V) / V = 0.03) : 
  V = 2 := by
sorry

end bottle_capacity_l3691_369118


namespace first_investment_interest_rate_l3691_369114

/-- Proves that the annual simple interest rate of the first investment is 8.5% -/
theorem first_investment_interest_rate 
  (total_income : ℝ) 
  (total_invested : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (second_rate : ℝ) :
  total_income = 575 →
  total_invested = 8000 →
  first_investment = 3000 →
  second_investment = 5000 →
  second_rate = 0.064 →
  ∃ (first_rate : ℝ), 
    first_rate = 0.085 ∧ 
    total_income = first_investment * first_rate + second_investment * second_rate :=
by
  sorry

end first_investment_interest_rate_l3691_369114


namespace distance_between_points_l3691_369170

-- Define the complex numbers
def z_J : ℂ := 3 + 4 * Complex.I
def z_G : ℂ := 2 - 3 * Complex.I

-- Define the scaled version of Gracie's point
def scaled_z_G : ℂ := 2 * z_G

-- Theorem statement
theorem distance_between_points : Complex.abs (z_J - scaled_z_G) = Real.sqrt 101 := by
  sorry

end distance_between_points_l3691_369170


namespace right_triangle_arithmetic_progression_right_triangle_geometric_progression_l3691_369108

-- Define a right triangle with sides a, b, c
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

-- Define arithmetic progression for three numbers
def is_arithmetic_progression (x y z : ℝ) : Prop :=
  y - x = z - y

-- Define geometric progression for three numbers
def is_geometric_progression (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

theorem right_triangle_arithmetic_progression :
  ∃ t : RightTriangle, is_arithmetic_progression t.a t.b t.c ∧ t.a = 3 ∧ t.b = 4 ∧ t.c = 5 :=
sorry

theorem right_triangle_geometric_progression :
  ∃ t : RightTriangle, is_geometric_progression t.a t.b t.c ∧
    t.a = 1 ∧ t.b = Real.sqrt ((1 + Real.sqrt 5) / 2) ∧ t.c = (1 + Real.sqrt 5) / 2 :=
sorry

end right_triangle_arithmetic_progression_right_triangle_geometric_progression_l3691_369108


namespace second_part_interest_rate_l3691_369164

def total_amount : ℝ := 2500
def first_part : ℝ := 500
def first_rate : ℝ := 0.05
def total_income : ℝ := 145

theorem second_part_interest_rate :
  let second_part := total_amount - first_part
  let first_income := first_part * first_rate
  let second_income := total_income - first_income
  let second_rate := second_income / second_part
  second_rate = 0.06 := by sorry

end second_part_interest_rate_l3691_369164


namespace min_value_theorem_l3691_369194

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 2) :
  (2 / a + 4 / b) ≥ 14 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 2 ∧ 2 / a₀ + 4 / b₀ = 14 := by
  sorry

end min_value_theorem_l3691_369194


namespace minimize_sum_distances_l3691_369188

/-- Given points A(-3,8) and B(2,2), prove that M(1,0) on the x-axis minimizes |AM| + |BM| -/
theorem minimize_sum_distances (A B M : ℝ × ℝ) : 
  A = (-3, 8) → 
  B = (2, 2) → 
  M.2 = 0 → 
  M = (1, 0) → 
  ∀ P : ℝ × ℝ, P.2 = 0 → 
    Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) + Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2) ≤ 
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) :=
by
  sorry


end minimize_sum_distances_l3691_369188


namespace arithmetic_sequence_sum_l3691_369168

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_4 + a_8 = 16, a_2 + a_10 = 16 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 4 + a 8 = 16) : 
  a 2 + a 10 = 16 := by
sorry

end arithmetic_sequence_sum_l3691_369168


namespace fourth_term_value_l3691_369169

def S (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 3 * n.val

theorem fourth_term_value : ∃ (a : ℕ+ → ℤ), a 4 = 11 :=
  sorry

end fourth_term_value_l3691_369169


namespace fraction_equation_solution_l3691_369195

theorem fraction_equation_solution (x : ℚ) : 
  (x + 11) / (x - 4) = (x - 3) / (x + 6) → x = -9/4 := by
  sorry

end fraction_equation_solution_l3691_369195


namespace simplify_expression_l3691_369104

theorem simplify_expression (a b : ℝ) : -3 * (a - b) + (2 * a - 3 * b) = -a := by
  sorry

end simplify_expression_l3691_369104


namespace probability_two_not_selected_l3691_369156

theorem probability_two_not_selected (S : Finset Nat) (a b : Nat) 
  (h1 : S.card = 4) (h2 : a ∈ S) (h3 : b ∈ S) (h4 : a ≠ b) :
  (Finset.filter (λ T : Finset Nat => T.card = 2 ∧ a ∉ T ∧ b ∉ T) (S.powerset)).card / (Finset.filter (λ T : Finset Nat => T.card = 2) (S.powerset)).card = 1 / 6 :=
sorry

end probability_two_not_selected_l3691_369156


namespace electric_guitars_sold_l3691_369174

theorem electric_guitars_sold (total_guitars : ℕ) (total_revenue : ℕ) 
  (electric_price : ℕ) (acoustic_price : ℕ) 
  (h1 : total_guitars = 9)
  (h2 : total_revenue = 3611)
  (h3 : electric_price = 479)
  (h4 : acoustic_price = 339) :
  ∃ (x : ℕ), x = 4 ∧ 
    ∃ (y : ℕ), x + y = total_guitars ∧ 
    electric_price * x + acoustic_price * y = total_revenue :=
by sorry

end electric_guitars_sold_l3691_369174


namespace triangle_ABC_theorem_l3691_369159

open Real

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π

theorem triangle_ABC_theorem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_eq : a * sin B - Real.sqrt 3 * b * cos A = 0) :
  A = π / 3 ∧ 
  (a = 3 → 
    (∃ (max_area : ℝ), max_area = 9 * Real.sqrt 3 / 4 ∧
      ∀ (b' c' : ℝ), triangle_ABC 3 b' c' A B C → 
        1/2 * 3 * b' * sin A ≤ max_area ∧
        (1/2 * 3 * b' * sin A = max_area → b' = 3 ∧ c' = 3))) :=
sorry

end triangle_ABC_theorem_l3691_369159


namespace sin_180_degrees_equals_zero_l3691_369113

theorem sin_180_degrees_equals_zero : Real.sin (π) = 0 := by
  sorry

end sin_180_degrees_equals_zero_l3691_369113


namespace necklace_beads_l3691_369115

theorem necklace_beads (total blue red white silver : ℕ) : 
  total = 40 →
  blue = 5 →
  red = 2 * blue →
  white = blue + red →
  total = blue + red + white + silver →
  silver = 10 := by
sorry

end necklace_beads_l3691_369115


namespace green_minus_blue_disks_l3691_369193

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green
  | Red

/-- Represents the ratio of disks of each color -/
def colorRatio : Fin 4 → Nat
  | 0 => 3  -- Blue
  | 1 => 7  -- Yellow
  | 2 => 8  -- Green
  | 3 => 4  -- Red

/-- The total number of disks in the bag -/
def totalDisks : Nat := 176

/-- Calculates the number of disks of a given color based on the ratio and total disks -/
def disksOfColor (color : Fin 4) : Nat :=
  (colorRatio color * totalDisks) / (colorRatio 0 + colorRatio 1 + colorRatio 2 + colorRatio 3)

/-- Theorem: There are 40 more green disks than blue disks in the bag -/
theorem green_minus_blue_disks : disksOfColor 2 - disksOfColor 0 = 40 := by
  sorry

end green_minus_blue_disks_l3691_369193


namespace no_integer_solution_l3691_369129

theorem no_integer_solution :
  ¬ ∃ (x y z : ℤ), 
    (x^6 + x^3 + x^3*y + y = 147^137) ∧ 
    (x^3 + x^3*y + y^2 + y + z^9 = 157^117) := by
  sorry

end no_integer_solution_l3691_369129


namespace arithmetic_sequence_2_to_2014_l3691_369140

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The arithmetic sequence starting at 2, with common difference 4, 
    and last term 2014 has 504 terms -/
theorem arithmetic_sequence_2_to_2014 : 
  arithmetic_sequence_length 2 4 2014 = 504 := by
  sorry

end arithmetic_sequence_2_to_2014_l3691_369140


namespace male_sample_size_in_given_scenario_l3691_369187

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  female_count : ℕ
  sample_size : ℕ
  h_female_count : female_count ≤ total_population
  h_sample_size : sample_size ≤ total_population

/-- Calculates the number of male students to be drawn in a stratified sample -/
def male_sample_size (s : StratifiedSample) : ℕ :=
  ((s.total_population - s.female_count) * s.sample_size) / s.total_population

/-- Theorem stating the number of male students to be drawn in the given scenario -/
theorem male_sample_size_in_given_scenario :
  let s : StratifiedSample := {
    total_population := 900,
    female_count := 400,
    sample_size := 45,
    h_female_count := by norm_num,
    h_sample_size := by norm_num
  }
  male_sample_size s = 25 := by
  sorry

end male_sample_size_in_given_scenario_l3691_369187


namespace cos_36_degrees_l3691_369100

theorem cos_36_degrees : Real.cos (36 * Real.pi / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end cos_36_degrees_l3691_369100


namespace cartons_in_load_l3691_369145

/-- Represents the weight of vegetables in a store's delivery truck. -/
structure VegetableLoad where
  crate_weight : ℕ
  carton_weight : ℕ
  num_crates : ℕ
  total_weight : ℕ

/-- Calculates the number of cartons in a vegetable load. -/
def num_cartons (load : VegetableLoad) : ℕ :=
  (load.total_weight - load.crate_weight * load.num_crates) / load.carton_weight

/-- Theorem stating that the number of cartons in the specific load is 16. -/
theorem cartons_in_load : 
  let load : VegetableLoad := {
    crate_weight := 4,
    carton_weight := 3,
    num_crates := 12,
    total_weight := 96
  }
  num_cartons load = 16 := by
  sorry


end cartons_in_load_l3691_369145


namespace binomial_coefficient_ratio_l3691_369102

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k+1) : ℚ) = 1/2 ∧ 
  (Nat.choose n (k+1) : ℚ) / (Nat.choose n (k+2) : ℚ) = 2/3 → 
  n + k = 18 :=
by sorry

end binomial_coefficient_ratio_l3691_369102


namespace problem_1_problem_2_l3691_369191

-- Problem 1
theorem problem_1 : (-2)^2 + |-4| - 18 * (-1/3) = 14 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : 2*(3*a^2*b - 2*a*b^2) - 4*(-a*b^2 + a^2*b) = 2*a^2*b := by sorry

end problem_1_problem_2_l3691_369191


namespace chocolate_savings_bernie_savings_l3691_369163

/-- Calculates the savings when buying chocolates at a lower price over a given period -/
theorem chocolate_savings 
  (weeks : ℕ) 
  (chocolates_per_week : ℕ) 
  (price_local : ℚ) 
  (price_discount : ℚ) :
  weeks * chocolates_per_week * (price_local - price_discount) = 
  weeks * chocolates_per_week * price_local - weeks * chocolates_per_week * price_discount :=
by sorry

/-- Proves that Bernie saves $6 over three weeks by buying chocolates at the discounted store -/
theorem bernie_savings :
  let weeks : ℕ := 3
  let chocolates_per_week : ℕ := 2
  let price_local : ℚ := 3
  let price_discount : ℚ := 2
  weeks * chocolates_per_week * (price_local - price_discount) = 6 :=
by sorry

end chocolate_savings_bernie_savings_l3691_369163


namespace inequality_proof_l3691_369142

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 1) : 
  2 * (x^2 + y^2 + z^2) + 9 * x * y * z ≥ 1 ∧ 
  x * y + y * z + z * x - 3 * x * y * z ≤ 1/4 := by
  sorry

#check inequality_proof

end inequality_proof_l3691_369142


namespace ellipse_equation_l3691_369141

/-- The equation of an ellipse given specific conditions -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1) ∧ 
  (2*a*b = 4) ∧
  (∃ (c : ℝ), a^2 - b^2 = c^2 ∧ c = Real.sqrt 3) →
  (∃ (x y : ℝ), x^2/4 + y^2 = 1) :=
sorry

end ellipse_equation_l3691_369141


namespace leap_year_classification_l3691_369162

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)

theorem leap_year_classification :
  (isLeapYear 2036 = true) ∧
  (isLeapYear 1996 = true) ∧
  (isLeapYear 1998 = false) ∧
  (isLeapYear 1700 = false) := by
  sorry

end leap_year_classification_l3691_369162


namespace inverse_71_mod_83_l3691_369143

theorem inverse_71_mod_83 (h : (17⁻¹ : ZMod 83) = 53) : (71⁻¹ : ZMod 83) = 53 := by
  sorry

end inverse_71_mod_83_l3691_369143


namespace circle_chord_intersection_area_l3691_369146

theorem circle_chord_intersection_area (r : ℝ) (chord_length : ℝ) (intersection_dist : ℝ)
  (h_r : r = 30)
  (h_chord : chord_length = 50)
  (h_dist : intersection_dist = 14) :
  ∃ (m n d : ℕ), 
    (0 < m) ∧ (0 < n) ∧ (0 < d) ∧
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ d)) ∧
    (∃ (area : ℝ), area = m * Real.pi - n * Real.sqrt d) ∧
    (m + n + d = 162) :=
by sorry

end circle_chord_intersection_area_l3691_369146


namespace bill_proof_l3691_369128

/-- The number of friends who can pay -/
def paying_friends : ℕ := 9

/-- The number of friends including the one who can't pay -/
def total_friends : ℕ := 10

/-- The additional amount each paying friend contributes -/
def additional_amount : ℕ := 3

/-- The total bill amount -/
def total_bill : ℕ := 270

theorem bill_proof :
  (paying_friends : ℚ) * (total_bill / total_friends + additional_amount) = total_bill := by
  sorry

end bill_proof_l3691_369128


namespace min_perimeter_nine_square_rectangle_l3691_369131

/-- Represents a rectangle divided into nine squares with integer side lengths -/
structure NineSquareRectangle where
  squares : Fin 9 → ℕ
  width : ℕ
  height : ℕ
  is_valid : width = squares 0 + squares 1 + squares 2 ∧
             height = squares 0 + squares 3 + squares 6 ∧
             width = squares 6 + squares 7 + squares 8 ∧
             height = squares 2 + squares 5 + squares 8

/-- The perimeter of a rectangle -/
def perimeter (rect : NineSquareRectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- Theorem stating that the minimum perimeter of a valid NineSquareRectangle is 52 -/
theorem min_perimeter_nine_square_rectangle :
  ∃ (rect : NineSquareRectangle), perimeter rect = 52 ∧
  ∀ (other : NineSquareRectangle), perimeter other ≥ 52 :=
sorry

end min_perimeter_nine_square_rectangle_l3691_369131


namespace smallest_four_digit_divisible_by_35_l3691_369155

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 35 ∣ n → n ≥ 1200 := by
  sorry

end smallest_four_digit_divisible_by_35_l3691_369155


namespace book_length_proof_l3691_369157

theorem book_length_proof (pages_read : ℕ) (pages_difference : ℕ) : 
  pages_read = 2323 → pages_difference = 90 → 
  pages_read = (pages_read - pages_difference) + pages_difference → 
  pages_read + (pages_read - pages_difference) = 4556 :=
by
  sorry

end book_length_proof_l3691_369157


namespace height_less_than_sum_of_distances_l3691_369160

/-- Represents a triangle with three unequal sides -/
structure UnequalTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hab : a ≠ b
  hbc : b ≠ c
  hac : a ≠ c
  longest_side : c > max a b

/-- The height to the longest side of the triangle -/
def height_to_longest_side (t : UnequalTriangle) : ℝ := sorry

/-- Distances from a point on the longest side to the other two sides -/
def distances_to_sides (t : UnequalTriangle) : ℝ × ℝ := sorry

theorem height_less_than_sum_of_distances (t : UnequalTriangle) :
  let x := height_to_longest_side t
  let (y, z) := distances_to_sides t
  x < y + z := by sorry

end height_less_than_sum_of_distances_l3691_369160


namespace gcd_of_72_120_180_l3691_369176

theorem gcd_of_72_120_180 : Nat.gcd 72 (Nat.gcd 120 180) = 12 := by
  sorry

end gcd_of_72_120_180_l3691_369176


namespace polar_equation_is_circle_l3691_369166

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 3 * Real.sin θ * Real.cos θ

-- Define the Cartesian equation of a circle
def is_circle (x y : ℝ) : Prop := ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_equation_is_circle :
  ∀ (x y : ℝ), (∃ (r θ : ℝ), polar_equation r θ ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  is_circle x y :=
sorry

end polar_equation_is_circle_l3691_369166


namespace raft_sticks_ratio_l3691_369139

theorem raft_sticks_ratio :
  ∀ (simon_sticks gerry_sticks micky_sticks : ℕ),
    simon_sticks = 36 →
    micky_sticks = simon_sticks + gerry_sticks + 9 →
    simon_sticks + gerry_sticks + micky_sticks = 129 →
    gerry_sticks * 3 = simon_sticks * 2 :=
by
  sorry

end raft_sticks_ratio_l3691_369139


namespace ribbon_used_wendy_ribbon_problem_l3691_369198

/-- Given the total amount of ribbon and the amount left after wrapping presents,
    prove that the amount used for wrapping is the difference between the two. -/
theorem ribbon_used (total : ℕ) (leftover : ℕ) (h : leftover ≤ total) :
  total - leftover = (total - leftover : ℕ) :=
by sorry

/-- Wendy's ribbon problem -/
theorem wendy_ribbon_problem :
  let total := 84
  let leftover := 38
  total - leftover = 46 :=
by sorry

end ribbon_used_wendy_ribbon_problem_l3691_369198


namespace prime_triples_divisibility_l3691_369153

theorem prime_triples_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧
  p ∣ q^r + 1 ∧ q ∣ r^p + 1 ∧ r ∣ p^q + 1 →
  (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
by sorry

end prime_triples_divisibility_l3691_369153


namespace total_pages_purchased_l3691_369110

def total_budget : ℚ := 10
def cost_per_notepad : ℚ := 5/4
def pages_per_notepad : ℕ := 60

theorem total_pages_purchased :
  (total_budget / cost_per_notepad).floor * pages_per_notepad = 480 :=
by sorry

end total_pages_purchased_l3691_369110


namespace point_in_second_quadrant_l3691_369124

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -5
  let y : ℝ := 2
  second_quadrant x y :=
by sorry

end point_in_second_quadrant_l3691_369124


namespace pool_cover_radius_increase_l3691_369179

/-- Theorem: When a circular pool cover's circumference increases from 30 inches to 40 inches, 
    the radius increases by 5/π inches. -/
theorem pool_cover_radius_increase (r₁ r₂ : ℝ) : 
  2 * Real.pi * r₁ = 30 → 
  2 * Real.pi * r₂ = 40 → 
  r₂ - r₁ = 5 / Real.pi := by
sorry

end pool_cover_radius_increase_l3691_369179


namespace linear_function_x_axis_intersection_l3691_369190

/-- A linear function passing through (-1, 2) with y-intercept 4 -/
def f (x : ℝ) : ℝ := 2 * x + 4

theorem linear_function_x_axis_intersection :
  ∃ (x : ℝ), f x = 0 ∧ x = -2 := by
  sorry

#check linear_function_x_axis_intersection

end linear_function_x_axis_intersection_l3691_369190


namespace tops_and_chudis_problem_l3691_369180

/-- The price of tops and chudis problem -/
theorem tops_and_chudis_problem (C T : ℚ) : 
  (3 * C + 6 * T = 1500) →  -- Price of 3 chudis and 6 tops
  (C + 12 * T = 1500) →     -- Price of 1 chudi and 12 tops
  (500 / T = 5) :=          -- Number of tops for Rs. 500
by
  sorry

#check tops_and_chudis_problem

end tops_and_chudis_problem_l3691_369180


namespace geometric_sequence_sum_l3691_369161

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a n > 0) →  -- Each term is positive
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence definition
  (a 0 + a 1 = 6) →  -- Sum of first two terms is 6
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 126) →  -- Sum of first six terms is 126
  (a 0 + a 1 + a 2 + a 3 = 30) :=  -- Sum of first four terms is 30
by sorry

end geometric_sequence_sum_l3691_369161


namespace probability_sum_eight_l3691_369144

/-- A fair die with 6 faces -/
structure Die :=
  (faces : Fin 6)

/-- The result of throwing two dice -/
structure TwoDiceThrow :=
  (die1 : Die)
  (die2 : Die)

/-- The sum of the numbers on two dice -/
def diceSum (throw : TwoDiceThrow) : Nat :=
  throw.die1.faces.val + 1 + throw.die2.faces.val + 1

/-- The set of all possible throws of two dice -/
def allThrows : Finset TwoDiceThrow :=
  sorry

/-- The set of throws where the sum is 8 -/
def sumEightThrows : Finset TwoDiceThrow :=
  sorry

/-- The probability of an event occurring when throwing two fair dice -/
def probability (event : Finset TwoDiceThrow) : Rat :=
  (event.card : Rat) / (allThrows.card : Rat)

theorem probability_sum_eight :
  probability sumEightThrows = 5 / 36 :=
sorry

end probability_sum_eight_l3691_369144


namespace decimal_521_to_octal_l3691_369186

-- Define a function to convert decimal to octal
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

-- Theorem statement
theorem decimal_521_to_octal :
  decimal_to_octal 521 = [1, 0, 1, 1] := by sorry

end decimal_521_to_octal_l3691_369186


namespace ellipse_equation_l3691_369121

-- Define the ellipse parameters
variable (a b : ℝ)

-- Define the points A, B, and C
variable (A B C : ℝ × ℝ)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem ellipse_equation 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a ≠ b)
  (h4 : ∀ x y, a * x^2 + b * y^2 = 1 ↔ (x, y) = A ∨ (x, y) = B)
  (h5 : A.1 + A.2 = 1 ∧ B.1 + B.2 = 1)
  (h6 : C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h7 : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8)
  (h8 : (C.2 - O.2) / (C.1 - O.1) = Real.sqrt 2 / 2) :
  a = 1/3 ∧ b = Real.sqrt 2 / 3 :=
sorry

end ellipse_equation_l3691_369121


namespace intersection_k_value_l3691_369101

-- Define the two lines
def line1 (x y k : ℝ) : Prop := 3 * x + y = k
def line2 (x y : ℝ) : Prop := -1.2 * x + y = -20

-- Define the theorem
theorem intersection_k_value :
  ∃ (k : ℝ), ∃ (y : ℝ),
    line1 7 y k ∧ line2 7 y ∧ k = 9.4 := by
  sorry

end intersection_k_value_l3691_369101


namespace planting_area_difference_l3691_369152

/-- Given a village with wheat, rice, and corn planting areas, prove the difference between rice and corn areas. -/
theorem planting_area_difference (m : ℝ) : 
  let wheat_area : ℝ := m
  let rice_area : ℝ := 2 * wheat_area + 3
  let corn_area : ℝ := wheat_area - 5
  rice_area - corn_area = m + 8 := by
  sorry

end planting_area_difference_l3691_369152


namespace M_superset_N_l3691_369120

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | ∃ a ∈ M, x = a^2}

theorem M_superset_N : M ⊇ N := by
  sorry

end M_superset_N_l3691_369120


namespace problem_statement_l3691_369138

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi / 6) + 1 / 2

theorem problem_statement :
  ∀ x A B C a b c : ℝ,
  -- Part 1 conditions
  (x ∈ Set.Icc 0 (Real.pi / 2)) →
  (f x = 11 / 10) →
  -- Part 1 conclusion
  (Real.cos x = (4 * Real.sqrt 3 - 3) / 10) ∧
  -- Part 2 conditions
  (0 < A ∧ A < Real.pi) →
  (0 < B ∧ B < Real.pi) →
  (0 < C ∧ C < Real.pi) →
  (A + B + C = Real.pi) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (2 * b * Real.cos A ≤ 2 * c - Real.sqrt 3 * a) →
  -- Part 2 conclusion
  (f B ∈ Set.Ioc 0 (1 / 2)) :=
by sorry

end problem_statement_l3691_369138


namespace kathryn_annie_difference_l3691_369127

/-- Represents the number of pints of blueberries picked by each person -/
structure BlueberryPicks where
  annie : ℕ
  kathryn : ℕ
  ben : ℕ

/-- Theorem stating the difference between Kathryn's and Annie's blueberry picks -/
theorem kathryn_annie_difference (picks : BlueberryPicks) : 
  picks.annie = 8 →
  picks.ben = picks.kathryn - 3 →
  picks.annie + picks.kathryn + picks.ben = 25 →
  picks.kathryn - picks.annie = 2 := by
sorry

end kathryn_annie_difference_l3691_369127


namespace geometric_sequence_second_term_l3691_369151

/-- Given a geometric sequence with common ratio 2 and sum of first 4 terms equal to 60,
    prove that the second term is 8. -/
theorem geometric_sequence_second_term :
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio q = 2
  (a 1 + a 2 + a 3 + a 4 = 60) →  -- Sum of first 4 terms S_4 = 60
  a 2 = 8 := by
sorry

end geometric_sequence_second_term_l3691_369151


namespace dinner_seating_arrangements_l3691_369103

/-- The number of ways to choose and seat people at a circular table. -/
def circular_seating_arrangements (total_people : ℕ) (seats : ℕ) : ℕ :=
  total_people * (seats - 1).factorial

/-- The problem statement -/
theorem dinner_seating_arrangements :
  circular_seating_arrangements 8 7 = 5760 := by
  sorry

end dinner_seating_arrangements_l3691_369103


namespace percentage_difference_l3691_369175

theorem percentage_difference : 
  (40 * 80 / 100) - (25 * 4 / 5) = 12 := by
  sorry

end percentage_difference_l3691_369175


namespace min_value_sum_of_reciprocals_l3691_369158

theorem min_value_sum_of_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_3 : a + b + c = 3) :
  1 / (a + b)^2 + 1 / (a + c)^2 + 1 / (b + c)^2 ≥ 3 / 2 := by
  sorry

end min_value_sum_of_reciprocals_l3691_369158


namespace opposite_of_pi_l3691_369123

theorem opposite_of_pi : 
  ∃ (x : ℝ), x = -π ∧ x + π = 0 :=
by
  sorry

end opposite_of_pi_l3691_369123


namespace stationery_box_sheets_l3691_369107

/-- Represents a box of stationery -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents a person's stationery usage -/
structure Usage where
  sheetsPerLetter : ℕ
  unusedSheets : ℕ
  unusedEnvelopes : ℕ

theorem stationery_box_sheets (box : StationeryBox) (john mary : Usage) : box.sheets = 240 :=
  by
  have h1 : john.sheetsPerLetter = 2 := by sorry
  have h2 : mary.sheetsPerLetter = 4 := by sorry
  have h3 : john.unusedSheets = 40 := by sorry
  have h4 : mary.unusedEnvelopes = 40 := by sorry
  have h5 : box.sheets = john.sheetsPerLetter * box.envelopes + john.unusedSheets := by sorry
  have h6 : box.sheets = mary.sheetsPerLetter * (box.envelopes - mary.unusedEnvelopes) := by sorry
  sorry

end stationery_box_sheets_l3691_369107


namespace diophantine_equation_solutions_l3691_369136

theorem diophantine_equation_solutions :
  ∀ x y z w : ℕ, 2^x * 3^y - 5^x * 7^w = 1 ↔ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨
  (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 1) ∨
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 0) ∨
  (x = 2 ∧ y = 2 ∧ z = 1 ∧ w = 1) :=
by sorry

end diophantine_equation_solutions_l3691_369136


namespace no_power_of_three_and_five_l3691_369106

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem no_power_of_three_and_five :
  ∀ n : ℕ, ∀ α β : ℕ+, v n ≠ (3 : ℤ)^(α : ℕ) * (5 : ℤ)^(β : ℕ) :=
by sorry

end no_power_of_three_and_five_l3691_369106


namespace triangle_third_side_l3691_369181

theorem triangle_third_side (a b c : ℕ) : 
  a = 3 → b = 8 → c % 2 = 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) → 
  c ≠ 12 := by
  sorry

end triangle_third_side_l3691_369181


namespace barry_sotter_magic_l3691_369185

/-- The length factor after n days of growth -/
def length_factor (n : ℕ) : ℚ :=
  (n + 2 : ℚ) / 2

theorem barry_sotter_magic (n : ℕ) :
  length_factor n = 50 ↔ n = 98 := by
  sorry

end barry_sotter_magic_l3691_369185


namespace circle_line_intersection_l3691_369116

theorem circle_line_intersection :
  ∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 ∧ x + y = 1 ∧
  ¬(∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 ∧ x + y = 1 ∧ x = 1 ∧ y = 1) :=
by sorry

end circle_line_intersection_l3691_369116


namespace binomial_fraction_is_nat_smallest_k_binomial_fraction_is_nat_smallest_k_property_l3691_369112

/-- For any natural number m, 1/(m+1) * binomial(2m, m) is a natural number -/
theorem binomial_fraction_is_nat (m : ℕ) : ∃ (k : ℕ), k = (1 : ℚ) / (m + 1 : ℚ) * (Nat.choose (2 * m) m) := by sorry

/-- For any natural numbers m and n where n ≥ m, 
    (2m+1)/(n+m+1) * binomial(2n, n+m) is a natural number -/
theorem smallest_k_binomial_fraction_is_nat (m n : ℕ) (h : n ≥ m) : 
  ∃ (k : ℕ), k = ((2 * m + 1 : ℚ) / (n + m + 1 : ℚ)) * (Nat.choose (2 * n) (n + m)) := by sorry

/-- 2m+1 is the smallest natural number k such that 
    k/(n+m+1) * binomial(2n, n+m) is a natural number for all n ≥ m -/
theorem smallest_k_property (m : ℕ) : 
  ∀ (k : ℕ), (∀ (n : ℕ), n ≥ m → ∃ (j : ℕ), j = (k : ℚ) / (n + m + 1 : ℚ) * (Nat.choose (2 * n) (n + m))) 
  → k ≥ 2 * m + 1 := by sorry

end binomial_fraction_is_nat_smallest_k_binomial_fraction_is_nat_smallest_k_property_l3691_369112


namespace aubree_animal_count_l3691_369135

/-- The total number of animals Aubree saw in a day, given the initial counts and changes --/
def total_animals (initial_beavers initial_chipmunks : ℕ) : ℕ :=
  let morning_total := initial_beavers + initial_chipmunks
  let afternoon_beavers := 2 * initial_beavers
  let afternoon_chipmunks := initial_chipmunks - 10
  let afternoon_total := afternoon_beavers + afternoon_chipmunks
  morning_total + afternoon_total

/-- Theorem stating that the total number of animals seen is 130 --/
theorem aubree_animal_count : total_animals 20 40 = 130 := by
  sorry

end aubree_animal_count_l3691_369135


namespace two_diamonds_balance_three_dots_l3691_369154

-- Define the symbols
variable (triangle diamond dot : ℕ)

-- Define the balance relationships
axiom balance1 : 3 * triangle + diamond = 9 * dot
axiom balance2 : triangle = diamond + dot

-- Theorem to prove
theorem two_diamonds_balance_three_dots : 2 * diamond = 3 * dot := by
  sorry

end two_diamonds_balance_three_dots_l3691_369154


namespace classroom_benches_l3691_369119

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- The number of students that can be seated in the classroom -/
def studentsInBase7 : Nat := 321

/-- The number of students that sit on one bench -/
def studentsPerBench : Nat := 3

/-- The number of benches in the classroom -/
def numberOfBenches : Nat := base7ToBase10 studentsInBase7 / studentsPerBench

theorem classroom_benches :
  numberOfBenches = 54 := by
  sorry

end classroom_benches_l3691_369119


namespace angle_C_60_not_sufficient_for_similarity_l3691_369132

-- Define triangles ABC and A'B'C'
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angles of a triangle
def angle (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

-- State the given conditions
axiom triangle_ABC : Triangle
axiom triangle_A'B'C' : Triangle

axiom angle_B_is_right : angle triangle_ABC 1 = 90
axiom angle_B'_is_right : angle triangle_A'B'C' 1 = 90
axiom angle_A_is_30 : angle triangle_ABC 0 = 30

-- Define triangle similarity
def similar (t1 t2 : Triangle) : Prop :=
  sorry

-- State the theorem
theorem angle_C_60_not_sufficient_for_similarity :
  ¬(∀ (ABC A'B'C' : Triangle),
    angle ABC 1 = 90 →
    angle A'B'C' 1 = 90 →
    angle ABC 0 = 30 →
    angle ABC 2 = 60 →
    similar ABC A'B'C') :=
  sorry

end angle_C_60_not_sufficient_for_similarity_l3691_369132


namespace lindas_savings_l3691_369199

-- Define the problem parameters
def furniture_ratio : ℚ := 5/8
def tv_ratio : ℚ := 1/4
def tv_discount : ℚ := 15/100
def furniture_discount : ℚ := 10/100
def initial_tv_cost : ℚ := 320
def exchange_rate : ℚ := 11/10

-- Define the theorem
theorem lindas_savings : 
  ∃ (savings : ℚ),
    savings * tv_ratio * (1 - tv_discount) = initial_tv_cost * (1 - tv_discount) ∧
    savings * furniture_ratio * (1 - furniture_discount) * exchange_rate = 
      savings * furniture_ratio * (1 - furniture_discount) ∧
    savings = 1088 :=
sorry

end lindas_savings_l3691_369199


namespace parallel_lines_theorem_l3691_369171

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  (l1.a * l2.b = l1.b * l2.a) ∧ (l1.a ≠ 0 ∨ l1.b ≠ 0) ∧ (l2.a ≠ 0 ∨ l2.b ≠ 0)

/-- The main theorem -/
theorem parallel_lines_theorem (k : ℝ) :
  let l1 : Line := { a := k - 2, b := 4 - k, c := 1 }
  let l2 : Line := { a := 2 * (k - 2), b := -2, c := 3 }
  are_parallel l1 l2 ↔ k = 2 ∨ k = 5 := by
  sorry

end parallel_lines_theorem_l3691_369171


namespace car_braking_distance_l3691_369177

/-- Represents the distance traveled by a car during braking -/
def distance_traveled (initial_speed : ℕ) (deceleration : ℕ) : ℕ :=
  let stopping_time := initial_speed / deceleration
  (initial_speed * stopping_time) - (deceleration * stopping_time * (stopping_time - 1) / 2)

/-- Theorem: A car with initial speed 40 ft/s and deceleration 10 ft/s² travels 100 ft before stopping -/
theorem car_braking_distance :
  distance_traveled 40 10 = 100 := by
  sorry

end car_braking_distance_l3691_369177
