import Mathlib

namespace NUMINAMATH_GPT_problem_am_hm_l1811_181196

open Real

theorem problem_am_hm (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 2) :
  ∃ S : Set ℝ, (∀ s ∈ S, (2 ≤ s)) ∧ (∀ z, (2 ≤ z) → (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 ∧ z = 1/x + 1/y))
  ∧ (S = {z | 2 ≤ z}) := sorry

end NUMINAMATH_GPT_problem_am_hm_l1811_181196


namespace NUMINAMATH_GPT_fractional_eq_no_solution_l1811_181163

theorem fractional_eq_no_solution (m : ℝ) :
  ¬ ∃ x, (x - 2) / (x + 2) - (m * x) / (x^2 - 4) = 1 ↔ m = -4 :=
by
  sorry

end NUMINAMATH_GPT_fractional_eq_no_solution_l1811_181163


namespace NUMINAMATH_GPT_sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C_l1811_181151

theorem sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C
  {A B C : ℝ}
  (h : A + B + C = π) :
  Real.sin (4 * A) + Real.sin (4 * B) + Real.sin (4 * C) = -4 * Real.sin (2 * A) * Real.sin (2 * B) * Real.sin (2 * C) :=
sorry

end NUMINAMATH_GPT_sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C_l1811_181151


namespace NUMINAMATH_GPT_find_total_amount_l1811_181150

theorem find_total_amount (x : ℝ) (h₁ : 1.5 * x = 40) : x + 1.5 * x + 0.5 * x = 80.01 :=
by
  sorry

end NUMINAMATH_GPT_find_total_amount_l1811_181150


namespace NUMINAMATH_GPT_exists_perpendicular_line_l1811_181141

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure DirectionVector :=
  (dx : ℝ)
  (dy : ℝ)
  (dz : ℝ)

noncomputable def parametric_line_through_point 
  (P : Point3D) 
  (d : DirectionVector) : Prop :=
  ∀ t : ℝ, ∃ x y z : ℝ, 
  x = P.x + d.dx * t ∧
  y = P.y + d.dy * t ∧
  z = P.z + d.dz * t

theorem exists_perpendicular_line : 
  ∃ d : DirectionVector, 
    (d.dx * 2 + d.dy * 3 - d.dz = 0) ∧ 
    (d.dx * 4 - d.dy * -1 + d.dz * 3 = 0) ∧ 
    parametric_line_through_point 
      ⟨3, -2, 1⟩ d :=
  sorry

end NUMINAMATH_GPT_exists_perpendicular_line_l1811_181141


namespace NUMINAMATH_GPT_value_of_expression_l1811_181193

theorem value_of_expression 
  (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 + 49 * x7 = 1) 
  (h2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 + 64 * x7 = 12) 
  (h3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 + 81 * x7 = 123) 
  : 16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 + 100 * x7 = 334 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1811_181193


namespace NUMINAMATH_GPT_f_nested_seven_l1811_181102

-- Definitions for the given conditions
variables (f : ℝ → ℝ) (odd_f : ∀ x, f (-x) = -f x)
variables (period_f : ∀ x, f (x + 4) = f x)
variables (f_one : f 1 = 4)

theorem f_nested_seven (f : ℝ → ℝ)
  (odd_f : ∀ x, f (-x) = -f x)
  (period_f : ∀ x, f (x + 4) = f x)
  (f_one : f 1 = 4) :
  f (f 7) = 0 :=
sorry

end NUMINAMATH_GPT_f_nested_seven_l1811_181102


namespace NUMINAMATH_GPT_positive_integer_cases_l1811_181195

theorem positive_integer_cases (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℤ, (abs (x^2 - abs x)) / x = n ∧ n > 0) ↔ (∃ m : ℤ, (x = m) ∧ (m > 1 ∨ m < -1)) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_cases_l1811_181195


namespace NUMINAMATH_GPT_John_surveyed_total_people_l1811_181137

theorem John_surveyed_total_people :
  ∃ P D : ℝ, 
  0 ≤ P ∧ 
  D = 0.868 * P ∧ 
  21 = 0.457 * D ∧ 
  P = 53 :=
by
  sorry

end NUMINAMATH_GPT_John_surveyed_total_people_l1811_181137


namespace NUMINAMATH_GPT_smallest_possible_X_l1811_181148

theorem smallest_possible_X (T : ℕ) (h1 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) (h2 : T % 24 = 0) :
  ∃ (X : ℕ), X = T / 24 ∧ X = 4625 :=
  sorry

end NUMINAMATH_GPT_smallest_possible_X_l1811_181148


namespace NUMINAMATH_GPT_jessica_total_cost_l1811_181165

def price_of_cat_toy : ℝ := 10.22
def price_of_cage : ℝ := 11.73
def price_of_cat_food : ℝ := 5.65
def price_of_catnip : ℝ := 2.30
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.07

def discounted_price_of_cat_toy : ℝ := price_of_cat_toy * (1 - discount_rate)
def total_cost_before_tax : ℝ := discounted_price_of_cat_toy + price_of_cage + price_of_cat_food + price_of_catnip
def sales_tax : ℝ := total_cost_before_tax * tax_rate
def total_cost_after_discount_and_tax : ℝ := total_cost_before_tax + sales_tax

theorem jessica_total_cost : total_cost_after_discount_and_tax = 30.90 := by
  sorry

end NUMINAMATH_GPT_jessica_total_cost_l1811_181165


namespace NUMINAMATH_GPT_distance_between_cities_l1811_181154

variable (a b : Nat)

theorem distance_between_cities :
  (a = (10 * a + b) - (10 * b + a)) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → 10 * a + b = 98 := by
  sorry

end NUMINAMATH_GPT_distance_between_cities_l1811_181154


namespace NUMINAMATH_GPT_proof_product_eq_l1811_181108

theorem proof_product_eq (a b c d : ℚ) (h1 : 2 * a + 3 * b + 5 * c + 7 * d = 42)
    (h2 : 4 * (d + c) = b) (h3 : 2 * b + 2 * c = a) (h4 : c - 2 = d) :
    a * b * c * d = -26880 / 729 := by
  sorry

end NUMINAMATH_GPT_proof_product_eq_l1811_181108


namespace NUMINAMATH_GPT_percent_pension_participation_l1811_181144

-- Define the conditions provided
def total_first_shift_members : ℕ := 60
def total_second_shift_members : ℕ := 50
def total_third_shift_members : ℕ := 40

def first_shift_pension_percentage : ℚ := 20 / 100
def second_shift_pension_percentage : ℚ := 40 / 100
def third_shift_pension_percentage : ℚ := 10 / 100

-- Calculate participation in the pension program for each shift
def first_shift_pension_members := total_first_shift_members * first_shift_pension_percentage
def second_shift_pension_members := total_second_shift_members * second_shift_pension_percentage
def third_shift_pension_members := total_third_shift_members * third_shift_pension_percentage

-- Calculate total participation in the pension program and total number of workers
def total_pension_members := first_shift_pension_members + second_shift_pension_members + third_shift_pension_members
def total_workers := total_first_shift_members + total_second_shift_members + total_third_shift_members

-- Lean proof statement
theorem percent_pension_participation : (total_pension_members / total_workers * 100) = 24 := by
  sorry

end NUMINAMATH_GPT_percent_pension_participation_l1811_181144


namespace NUMINAMATH_GPT_intersection_M_N_l1811_181173

-- Define the sets M and N based on given conditions
def M : Set ℝ := { x : ℝ | x^2 < 4 }
def N : Set ℝ := { x : ℝ | x < 1 }

-- State the theorem to prove the intersection of M and N
theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1811_181173


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1811_181126

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (a_mono : ∀ n, a n < a (n+1))
    (a2a5_eq_6 : a 2 * a 5 = 6)
    (a3a4_eq_5 : a 3 + a 4 = 5) 
    (q : ℝ) (hq : ∀ n, a n = a 1 * q ^ (n - 1)) :
    q = 3 / 2 :=
by
    sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1811_181126


namespace NUMINAMATH_GPT_max_c_friendly_value_l1811_181109

def is_c_friendly (c : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → |f x - f y| ≤ c * |x - y|

theorem max_c_friendly_value (c : ℝ) (f : ℝ → ℝ) (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  c > 1 → is_c_friendly c f → |f x - f y| ≤ (c + 1) / 2 :=
sorry

end NUMINAMATH_GPT_max_c_friendly_value_l1811_181109


namespace NUMINAMATH_GPT_Rachel_spent_on_lunch_fraction_l1811_181124

variable {MoneyEarned MoneySpentOnDVD MoneyLeft MoneySpentOnLunch : ℝ}

-- Given conditions
axiom Rachel_earnings : MoneyEarned = 200
axiom Rachel_spent_on_DVD : MoneySpentOnDVD = MoneyEarned / 2
axiom Rachel_leftover : MoneyLeft = 50
axiom Rachel_total_spent : MoneyEarned - MoneyLeft = MoneySpentOnLunch + MoneySpentOnDVD

-- Prove that Rachel spent 1/4 of her money on lunch
theorem Rachel_spent_on_lunch_fraction :
  MoneySpentOnLunch / MoneyEarned = 1 / 4 :=
sorry

end NUMINAMATH_GPT_Rachel_spent_on_lunch_fraction_l1811_181124


namespace NUMINAMATH_GPT_correct_statement_C_l1811_181177

theorem correct_statement_C : ∀ y : ℝ, y^2 = 25 ↔ (y = 5 ∨ y = -5) := 
by sorry

end NUMINAMATH_GPT_correct_statement_C_l1811_181177


namespace NUMINAMATH_GPT_inequality_proof_l1811_181199

theorem inequality_proof (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^4 / (y-1)^2) + (y^4 / (z-1)^2) + (z^4 / (x-1)^2) ≥ 48 := 
by
  sorry -- The actual proof is omitted

end NUMINAMATH_GPT_inequality_proof_l1811_181199


namespace NUMINAMATH_GPT_general_term_formula_exponential_seq_l1811_181133

variable (n : ℕ)

def exponential_sequence (a1 r : ℕ) (n : ℕ) : ℕ := a1 * r^(n-1)

theorem general_term_formula_exponential_seq :
  exponential_sequence 2 3 n = 2 * 3^(n-1) :=
by
  sorry

end NUMINAMATH_GPT_general_term_formula_exponential_seq_l1811_181133


namespace NUMINAMATH_GPT_jonathans_and_sisters_total_letters_l1811_181123

theorem jonathans_and_sisters_total_letters:
  (jonathan_first: Nat) = 8 ∧
  (jonathan_surname: Nat) = 10 ∧
  (sister_first: Nat) = 5 ∧
  (sister_surname: Nat) = 10 →
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  intros
  sorry

end NUMINAMATH_GPT_jonathans_and_sisters_total_letters_l1811_181123


namespace NUMINAMATH_GPT_quadratic_increasing_implies_m_gt_1_l1811_181112

theorem quadratic_increasing_implies_m_gt_1 (m : ℝ) (x : ℝ) 
(h1 : x > 1) 
(h2 : ∀ x, (y = x^2 + (m-3) * x + m + 1) → (∀ z > x, y < z^2 + (m-3) * z + m + 1)) 
: m > 1 := 
sorry

end NUMINAMATH_GPT_quadratic_increasing_implies_m_gt_1_l1811_181112


namespace NUMINAMATH_GPT_find_constants_l1811_181175

theorem find_constants (a b c : ℝ) (h_neq_0_a : a ≠ 0) (h_neq_0_b : b ≠ 0) 
(h_neq_0_c : c ≠ 0) 
(h_eq1 : a * b = 3 * (a + b)) 
(h_eq2 : b * c = 4 * (b + c)) 
(h_eq3 : a * c = 5 * (a + c)) : 
a = 120 / 17 ∧ b = 120 / 23 ∧ c = 120 / 7 := 
  sorry

end NUMINAMATH_GPT_find_constants_l1811_181175


namespace NUMINAMATH_GPT_parallel_lines_sufficient_not_necessary_condition_l1811_181188

theorem parallel_lines_sufficient_not_necessary_condition {a : ℝ} :
  (a = 4) → (∀ x y : ℝ, (a * x + 8 * y - 3 = 0) ↔ (2 * x + a * y - a = 0)) :=
by sorry

end NUMINAMATH_GPT_parallel_lines_sufficient_not_necessary_condition_l1811_181188


namespace NUMINAMATH_GPT_evaluate_expression_l1811_181191

-- Definitions based on conditions
def a : ℤ := 5
def b : ℤ := -3
def c : ℤ := 2

-- Theorem to be proved: evaluate the expression
theorem evaluate_expression : (3 : ℚ) / (a + b + c) = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1811_181191


namespace NUMINAMATH_GPT_Lisa_types_correctly_l1811_181198

-- Given conditions
def Rudy_wpm : ℕ := 64
def Joyce_wpm : ℕ := 76
def Gladys_wpm : ℕ := 91
def Mike_wpm : ℕ := 89
def avg_wpm : ℕ := 80
def num_employees : ℕ := 5

-- Define the hypothesis about Lisa's typing speaking
def Lisa_wpm : ℕ := (num_employees * avg_wpm) - Rudy_wpm - Joyce_wpm - Gladys_wpm - Mike_wpm

-- The statement to prove
theorem Lisa_types_correctly :
  Lisa_wpm = 140 := by
  sorry

end NUMINAMATH_GPT_Lisa_types_correctly_l1811_181198


namespace NUMINAMATH_GPT_second_markdown_percentage_l1811_181162

theorem second_markdown_percentage (P : ℝ) (h1 : P > 0)
    (h2 : ∃ x : ℝ, x = 0.50 * P) -- First markdown
    (h3 : ∃ y : ℝ, y = 0.45 * P) -- Final price
    : ∃ X : ℝ, X = 10 := 
sorry

end NUMINAMATH_GPT_second_markdown_percentage_l1811_181162


namespace NUMINAMATH_GPT_math_problem_l1811_181125

theorem math_problem : 3 * 13 + 3 * 14 + 3 * 17 + 11 = 143 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1811_181125


namespace NUMINAMATH_GPT_evaluate_expression_l1811_181178

-- Definition of the function f
def f (x : ℤ) : ℤ := 3 * x^2 - 5 * x + 8

-- Theorems and lemmas
theorem evaluate_expression : 3 * f 4 + 2 * f (-4) = 260 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1811_181178


namespace NUMINAMATH_GPT_ducks_and_chickens_l1811_181189

theorem ducks_and_chickens : 
  (∃ ducks chickens : ℕ, ducks = 7 ∧ chickens = 6 ∧ ducks + chickens = 13) :=
by
  sorry

end NUMINAMATH_GPT_ducks_and_chickens_l1811_181189


namespace NUMINAMATH_GPT_becky_necklaces_count_l1811_181166

-- Define the initial conditions
def initial_necklaces := 50
def broken_necklaces := 3
def new_necklaces := 5
def given_away_necklaces := 15

-- Define the final number of necklaces
def final_necklaces (initial : Nat) (broken : Nat) (bought : Nat) (given_away : Nat) : Nat :=
  initial - broken + bought - given_away

-- The theorem stating that after performing the series of operations,
-- Becky should have 37 necklaces.
theorem becky_necklaces_count :
  final_necklaces initial_necklaces broken_necklaces new_necklaces given_away_necklaces = 37 :=
  by
    -- This proof is just a placeholder to ensure the code can be built successfully.
    -- Actual proof logic needs to be filled in to complete the theorem.
    sorry

end NUMINAMATH_GPT_becky_necklaces_count_l1811_181166


namespace NUMINAMATH_GPT_volume_of_largest_sphere_from_cube_l1811_181101

theorem volume_of_largest_sphere_from_cube : 
  (∃ (V : ℝ), 
    (∀ (l : ℝ), l = 1 → (V = (4 / 3) * π * ((l / 2)^3)) → V = π / 6)) :=
sorry

end NUMINAMATH_GPT_volume_of_largest_sphere_from_cube_l1811_181101


namespace NUMINAMATH_GPT_sum_of_fifth_powers_l1811_181172

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end NUMINAMATH_GPT_sum_of_fifth_powers_l1811_181172


namespace NUMINAMATH_GPT_lines_are_perpendicular_l1811_181145

noncomputable def line1 := {x : ℝ | ∃ y : ℝ, x + y - 1 = 0}
noncomputable def line2 := {x : ℝ | ∃ y : ℝ, x - y + 1 = 0}

theorem lines_are_perpendicular : 
  let slope1 := -1
  let slope2 := 1
  slope1 * slope2 = -1 := sorry

end NUMINAMATH_GPT_lines_are_perpendicular_l1811_181145


namespace NUMINAMATH_GPT_find_m_l1811_181114

theorem find_m {m : ℝ} :
  (∃ x y : ℝ, y = x + 1 ∧ y = -x ∧ y = mx + 3) → m = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1811_181114


namespace NUMINAMATH_GPT_value_of_a8_l1811_181168

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

noncomputable def arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n : ℕ, ∃ d : α, a (n + 1) = a n + d

variable {a : ℕ → ℝ}

axiom seq_is_arithmetic : arithmetic_sequence a

axiom initial_condition :
  a 1 + 3 * a 8 + a 15 = 120

axiom arithmetic_property :
  a 1 + a 15 = 2 * a 8

theorem value_of_a8 : a 8 = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_a8_l1811_181168


namespace NUMINAMATH_GPT_jeremy_age_l1811_181149

theorem jeremy_age
  (A J C : ℕ)
  (h1 : A + J + C = 132)
  (h2 : A = 1 / 3 * J)
  (h3 : C = 2 * A) :
  J = 66 :=
sorry

end NUMINAMATH_GPT_jeremy_age_l1811_181149


namespace NUMINAMATH_GPT_range_of_a_l1811_181142

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1811_181142


namespace NUMINAMATH_GPT_points_3_units_away_from_origin_l1811_181139

theorem points_3_units_away_from_origin (a : ℝ) (h : |a| = 3) : a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_GPT_points_3_units_away_from_origin_l1811_181139


namespace NUMINAMATH_GPT_number_of_true_propositions_l1811_181185

-- Define the original proposition
def prop (x: Real) : Prop := x^2 > 1 → x > 1

-- Define converse, inverse, contrapositive
def converse (x: Real) : Prop := x > 1 → x^2 > 1
def inverse (x: Real) : Prop := x^2 ≤ 1 → x ≤ 1
def contrapositive (x: Real) : Prop := x ≤ 1 → x^2 ≤ 1

-- Define the proposition we want to prove: the number of true propositions among them
theorem number_of_true_propositions :
  (converse 2 = True) ∧ (inverse 2 = True) ∧ (contrapositive 2 = False) → 2 = 2 :=
by sorry

end NUMINAMATH_GPT_number_of_true_propositions_l1811_181185


namespace NUMINAMATH_GPT_range_of_independent_variable_l1811_181156

theorem range_of_independent_variable (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_independent_variable_l1811_181156


namespace NUMINAMATH_GPT_total_amount_spent_l1811_181176

variables (P J I T : ℕ)

-- Given conditions
def Pauline_dress : P = 30 := sorry
def Jean_dress : J = P - 10 := sorry
def Ida_dress : I = J + 30 := sorry
def Patty_dress : T = I + 10 := sorry

-- Theorem to prove the total amount spent
theorem total_amount_spent :
  P + J + I + T = 160 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_total_amount_spent_l1811_181176


namespace NUMINAMATH_GPT_k_ge_1_l1811_181106

theorem k_ge_1 (k : ℝ) : 
  (∀ x : ℝ, 2 * x + 9 > 6 * x + 1 ∧ x - k < 1 → x < 2) → k ≥ 1 :=
by 
  sorry

end NUMINAMATH_GPT_k_ge_1_l1811_181106


namespace NUMINAMATH_GPT_find_a_l1811_181122

noncomputable def l1 (a : ℝ) (x y : ℝ) : ℝ := a * x + (a + 1) * y + 1
noncomputable def l2 (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 2

def perp_lines (a : ℝ) : Prop :=
  let m1 := -a
  let m2 := -1 / a
  m1 * m2 = -1

theorem find_a (a : ℝ) : (perp_lines a) ↔ (a = 0 ∨ a = -2) := 
sorry

end NUMINAMATH_GPT_find_a_l1811_181122


namespace NUMINAMATH_GPT_correct_statements_l1811_181182

variables {n : ℕ}
noncomputable def S (n : ℕ) : ℝ := (n + 1) / n
noncomputable def T (n : ℕ) : ℝ := (n + 1)
noncomputable def a (n : ℕ) : ℝ := if n = 1 then 2 else (-(1:ℝ)) / (n * (n - 1))

theorem correct_statements (n : ℕ) (hn : n ≠ 0) :
  (S n + T n = S n * T n) ∧ (a 1 = 2) ∧ (∀ n, ∃ d, ∀ m, T (n + m) - T n = m * d) ∧ (S n = (n + 1) / n) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l1811_181182


namespace NUMINAMATH_GPT_inequality_a_cube_less_b_cube_l1811_181159

theorem inequality_a_cube_less_b_cube (a b : ℝ) (ha : a < 0) (hb : b > 0) : a^3 < b^3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_a_cube_less_b_cube_l1811_181159


namespace NUMINAMATH_GPT_Alice_has_3_more_dimes_than_quarters_l1811_181104

-- Definitions of the conditions given in the problem
variable (n d : ℕ) -- number of 5-cent and 10-cent coins
def q : ℕ := 10
def total_coins : ℕ := 30
def total_value : ℕ := 435
def extra_dimes : ℕ := 6

-- Conditions translated to Lean
axiom total_coin_count : n + d + q = total_coins
axiom total_value_count : 5 * n + 10 * d + 25 * q = total_value
axiom dime_difference : d = n + extra_dimes

-- The theorem that needs to be proven: Alice has 3 more 10-cent coins than 25-cent coins.
theorem Alice_has_3_more_dimes_than_quarters :
  d - q = 3 :=
sorry

end NUMINAMATH_GPT_Alice_has_3_more_dimes_than_quarters_l1811_181104


namespace NUMINAMATH_GPT_find_number_l1811_181121

theorem find_number (n x : ℕ) (h1 : n * (x - 1) = 21) (h2 : x = 4) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1811_181121


namespace NUMINAMATH_GPT_margaret_spends_on_croissants_l1811_181152

theorem margaret_spends_on_croissants :
  (∀ (people : ℕ) (sandwiches_per_person : ℕ) (croissants_per_sandwich : ℕ) (croissants_per_set : ℕ) (cost_per_set : ℝ),
    people = 24 →
    sandwiches_per_person = 2 →
    croissants_per_sandwich = 1 →
    croissants_per_set = 12 →
    cost_per_set = 8 →
    (people * sandwiches_per_person * croissants_per_sandwich) / croissants_per_set * cost_per_set = 32) := sorry

end NUMINAMATH_GPT_margaret_spends_on_croissants_l1811_181152


namespace NUMINAMATH_GPT_shaded_square_area_l1811_181131

theorem shaded_square_area (a b s : ℝ) (h : a * b = 40) :
  ∃ s, s^2 = 2500 / 441 :=
by
  sorry

end NUMINAMATH_GPT_shaded_square_area_l1811_181131


namespace NUMINAMATH_GPT_circle_equation_exists_l1811_181171

noncomputable def point (α : Type*) := {p : α × α // ∃ x y : α, p = (x, y)}

structure Circle (α : Type*) :=
(center : α × α)
(radius : α)

def passes_through (c : Circle ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1) ^ 2 + (p.2 - c.center.2) ^ 2 = c.radius ^ 2

theorem circle_equation_exists :
  ∃ (c : Circle ℝ),
    c.center = (-4, 3) ∧ c.radius = 5 ∧ passes_through c (-1, -1) ∧ passes_through c (-8, 0) ∧ passes_through c (0, 6) :=
by { sorry }

end NUMINAMATH_GPT_circle_equation_exists_l1811_181171


namespace NUMINAMATH_GPT_parallel_line_slope_y_intercept_l1811_181170

theorem parallel_line_slope_y_intercept (x y : ℝ) (h : 3 * x - 6 * y = 12) :
  ∃ (m b : ℝ), m = 1 / 2 ∧ b = -2 := 
by { sorry }

end NUMINAMATH_GPT_parallel_line_slope_y_intercept_l1811_181170


namespace NUMINAMATH_GPT_area_of_triangle_AEB_l1811_181120

noncomputable def rectangle_area_AEB : ℝ :=
  let AB := 8
  let BC := 4
  let DF := 2
  let GC := 2
  let FG := 8 - DF - GC -- DC (8 units) minus DF and GC.
  let ratio := AB / FG
  let altitude_AEB := BC * ratio
  let area_AEB := 0.5 * AB * altitude_AEB
  area_AEB

theorem area_of_triangle_AEB : rectangle_area_AEB = 32 :=
by
  -- placeholder for detailed proof
  sorry

end NUMINAMATH_GPT_area_of_triangle_AEB_l1811_181120


namespace NUMINAMATH_GPT_difference_in_amount_paid_l1811_181116

variable (P Q : ℝ)

def original_price := P
def intended_quantity := Q

def new_price := P * 1.10
def new_quantity := Q * 0.80

theorem difference_in_amount_paid :
  ((new_price P * new_quantity Q) - (original_price P * intended_quantity Q)) = -0.12 * (original_price P * intended_quantity Q) :=
by
  sorry

end NUMINAMATH_GPT_difference_in_amount_paid_l1811_181116


namespace NUMINAMATH_GPT_no_nat_m_n_square_diff_2014_l1811_181164

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end NUMINAMATH_GPT_no_nat_m_n_square_diff_2014_l1811_181164


namespace NUMINAMATH_GPT_tori_needs_more_correct_answers_l1811_181118

theorem tori_needs_more_correct_answers :
  let total_questions := 80
  let arithmetic_questions := 20
  let algebra_questions := 25
  let geometry_questions := 35
  let arithmetic_correct := 0.60 * arithmetic_questions
  let algebra_correct := Float.round (0.50 * algebra_questions)
  let geometry_correct := Float.round (0.70 * geometry_questions)
  let correct_answers := arithmetic_correct + algebra_correct + geometry_correct
  let passing_percentage := 0.65
  let required_correct := passing_percentage * total_questions
-- assertion
  required_correct - correct_answers = 2 := 
by 
  sorry

end NUMINAMATH_GPT_tori_needs_more_correct_answers_l1811_181118


namespace NUMINAMATH_GPT_initial_apples_l1811_181110

-- Defining the conditions
def apples_handed_out := 8
def pies_made := 6
def apples_per_pie := 9
def apples_for_pies := pies_made * apples_per_pie

-- Prove the initial number of apples
theorem initial_apples : apples_handed_out + apples_for_pies = 62 :=
by
  sorry

end NUMINAMATH_GPT_initial_apples_l1811_181110


namespace NUMINAMATH_GPT_total_hamburger_varieties_l1811_181134

def num_condiments : ℕ := 9
def num_condiment_combinations : ℕ := 2 ^ num_condiments
def num_patties_choices : ℕ := 4
def num_bread_choices : ℕ := 2

theorem total_hamburger_varieties : num_condiment_combinations * num_patties_choices * num_bread_choices = 4096 :=
by
  -- conditions
  have h1 : num_condiments = 9 := rfl
  have h2 : num_condiment_combinations = 2 ^ num_condiments := rfl
  have h3 : num_patties_choices = 4 := rfl
  have h4 : num_bread_choices = 2 := rfl

  -- correct answer
  sorry

end NUMINAMATH_GPT_total_hamburger_varieties_l1811_181134


namespace NUMINAMATH_GPT_prove_moles_of_C2H6_l1811_181115

def moles_of_CCl4 := 4
def moles_of_Cl2 := 14
def moles_of_C2H6 := 2

theorem prove_moles_of_C2H6
  (h1 : moles_of_Cl2 = 14)
  (h2 : moles_of_CCl4 = 4)
  : moles_of_C2H6 = 2 := 
sorry

end NUMINAMATH_GPT_prove_moles_of_C2H6_l1811_181115


namespace NUMINAMATH_GPT_range_of_g_l1811_181161

noncomputable def g (x : ℝ) : ℝ := (3 * x + 8 - 2 * x ^ 2) / (x + 4)

theorem range_of_g : 
  (∀ y : ℝ, ∃ x : ℝ, x ≠ -4 ∧ y = (3 * x + 8 - 2 * x^2) / (x + 4)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_g_l1811_181161


namespace NUMINAMATH_GPT_market_value_of_stock_l1811_181174

theorem market_value_of_stock (dividend_rate : ℝ) (yield_rate : ℝ) (face_value : ℝ) :
  dividend_rate = 0.12 → yield_rate = 0.08 → face_value = 100 → (dividend_rate * face_value / yield_rate * 100) = 150 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_market_value_of_stock_l1811_181174


namespace NUMINAMATH_GPT_algebraic_expression_value_l1811_181105

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x + 3 = 7) : 3 * x^2 + 3 * x + 7 = 19 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1811_181105


namespace NUMINAMATH_GPT_union_A_B_m_eq_3_range_of_m_l1811_181107

def A (x : ℝ) : Prop := x^2 - x - 12 ≤ 0
def B (x m : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem union_A_B_m_eq_3 :
  A x ∨ B x 3 ↔ (-3 : ℝ) ≤ x ∧ x ≤ 5 := sorry

theorem range_of_m (h : ∀ x, A x ∨ B x m ↔ A x) : m ≤ (5 / 2) := sorry

end NUMINAMATH_GPT_union_A_B_m_eq_3_range_of_m_l1811_181107


namespace NUMINAMATH_GPT_dislike_both_tv_and_video_games_l1811_181169

theorem dislike_both_tv_and_video_games (total_people : ℕ) (percent_dislike_tv : ℝ) (percent_dislike_tv_and_games : ℝ) :
  let people_dislike_tv := percent_dislike_tv * total_people
  let people_dislike_both := percent_dislike_tv_and_games * people_dislike_tv
  total_people = 1800 ∧ percent_dislike_tv = 0.4 ∧ percent_dislike_tv_and_games = 0.25 →
  people_dislike_both = 180 :=
by {
  sorry
}

end NUMINAMATH_GPT_dislike_both_tv_and_video_games_l1811_181169


namespace NUMINAMATH_GPT_largest_three_digit_divisible_and_prime_sum_l1811_181140

theorem largest_three_digit_divisible_and_prime_sum :
  ∃ n : ℕ, 900 ≤ n ∧ n < 1000 ∧
           (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ≠ 0 ∧ n % d = 0) ∧
           Prime (n / 100 + (n / 10) % 10 + n % 10) ∧
           n = 963 ∧
           ∀ m : ℕ, 900 ≤ m ∧ m < 1000 ∧
           (∀ d ∈ [m / 100, (m / 10) % 10, m % 10], d ≠ 0 ∧ m % d = 0) ∧
           Prime (m / 100 + (m / 10) % 10 + m % 10) →
           m ≤ 963 :=
by
  sorry

end NUMINAMATH_GPT_largest_three_digit_divisible_and_prime_sum_l1811_181140


namespace NUMINAMATH_GPT_binomial_coeff_coprime_l1811_181179

def binom (a b : ℕ) : ℕ := Nat.factorial a / (Nat.factorial b * Nat.factorial (a - b))

theorem binomial_coeff_coprime (p a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hp : Nat.Prime p) 
  (hbase_p_a : ∀ i, (a / p^i % p) ≥ (b / p^i % p)) 
  : Nat.gcd (binom a b) p = 1 :=
by sorry

end NUMINAMATH_GPT_binomial_coeff_coprime_l1811_181179


namespace NUMINAMATH_GPT_c_share_l1811_181186

theorem c_share (a b c : ℕ) (k : ℕ) 
    (h1 : a + b + c = 1010)
    (h2 : a - 25 = 3 * k) 
    (h3 : b - 10 = 2 * k) 
    (h4 : c - 15 = 5 * k) 
    : c = 495 := 
sorry

end NUMINAMATH_GPT_c_share_l1811_181186


namespace NUMINAMATH_GPT_probability_of_rolling_2_4_6_l1811_181138

open Set Classical

noncomputable def fair_six_sided_die_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

def successful_outcomes : Finset ℕ := {2, 4, 6}

theorem probability_of_rolling_2_4_6 : 
  (successful_outcomes.card : ℚ) / (fair_six_sided_die_outcomes.card : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_rolling_2_4_6_l1811_181138


namespace NUMINAMATH_GPT_max_side_range_of_triangle_l1811_181146

-- Define the requirement on the sides a and b
def side_condition (a b : ℝ) : Prop :=
  |a - 3| + (b - 7)^2 = 0

-- Prove the range of side c
theorem max_side_range_of_triangle (a b c : ℝ) (h : side_condition a b) (hc : c = max a (max b c)) :
  7 ≤ c ∧ c < 10 :=
sorry

end NUMINAMATH_GPT_max_side_range_of_triangle_l1811_181146


namespace NUMINAMATH_GPT_range_of_a_l1811_181167

noncomputable def f (a x : ℝ) : ℝ := Real.log (a * x ^ 2 + 2 * x + 1)

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, a * x ^ 2 + 2 * x + 1 > 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1811_181167


namespace NUMINAMATH_GPT_provisions_initial_days_l1811_181190

theorem provisions_initial_days (D : ℕ) (P : ℕ) (Q : ℕ) (X : ℕ) (Y : ℕ)
  (h1 : P = 300) 
  (h2 : X = 30) 
  (h3 : Y = 90) 
  (h4 : Q = 200) 
  (h5 : P * D = P * X + Q * Y) : D + X = 120 :=
by
  -- We need to prove that the initial number of days the provisions were meant to last is 120.
  sorry

end NUMINAMATH_GPT_provisions_initial_days_l1811_181190


namespace NUMINAMATH_GPT_find_fraction_l1811_181183

theorem find_fraction (F N : ℝ) 
  (h1 : F * (1 / 4 * N) = 15)
  (h2 : (3 / 10) * N = 54) : 
  F = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_find_fraction_l1811_181183


namespace NUMINAMATH_GPT_gcd_324_243_l1811_181157

-- Define the numbers involved in the problem.
def a : ℕ := 324
def b : ℕ := 243

-- State the theorem that the GCD of a and b is 81.
theorem gcd_324_243 : Nat.gcd a b = 81 := by
  sorry

end NUMINAMATH_GPT_gcd_324_243_l1811_181157


namespace NUMINAMATH_GPT_tangent_line_equation_l1811_181187

theorem tangent_line_equation (e x y : ℝ) (h_curve : y = x^3 / e) (h_point : x = e ∧ y = e^2) :
  3 * e * x - y - 2 * e^2 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_equation_l1811_181187


namespace NUMINAMATH_GPT_times_faster_l1811_181127

theorem times_faster (A B W : ℝ) (h1 : A = 3 * B) (h2 : (A + B) * 21 = A * 28) : A = 3 * B :=
by sorry

end NUMINAMATH_GPT_times_faster_l1811_181127


namespace NUMINAMATH_GPT_anita_smallest_number_of_candies_l1811_181128

theorem anita_smallest_number_of_candies :
  ∃ x : ℕ, x ≡ 5 [MOD 6] ∧ x ≡ 3 [MOD 8] ∧ x ≡ 7 [MOD 9] ∧ ∀ y : ℕ,
  (y ≡ 5 [MOD 6] ∧ y ≡ 3 [MOD 8] ∧ y ≡ 7 [MOD 9]) → x ≤ y :=
  ⟨203, by sorry⟩

end NUMINAMATH_GPT_anita_smallest_number_of_candies_l1811_181128


namespace NUMINAMATH_GPT_tangent_circle_locus_l1811_181192

-- Definitions for circle C1 and circle C2
def Circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def Circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Definition of being tangent to a circle
def ExternallyTangent (cx cy cr : ℝ) : Prop := (cx - 0)^2 + (cy - 0)^2 = (cr + 1)^2
def InternallyTangent (cx cy cr : ℝ) : Prop := (cx - 3)^2 + (cy - 0)^2 = (3 - cr)^2

-- Definition of locus L where (a,b) are centers of circles tangent to both C1 and C2
def Locus (a b : ℝ) : Prop := 28 * a^2 + 64 * b^2 - 84 * a - 49 = 0

-- The theorem to be proved
theorem tangent_circle_locus (a b r : ℝ) :
  (ExternallyTangent a b r) → (InternallyTangent a b r) → Locus a b :=
by {
  sorry
}

end NUMINAMATH_GPT_tangent_circle_locus_l1811_181192


namespace NUMINAMATH_GPT_log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6_l1811_181158

theorem log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6 
  (h1 : 5^0.6 > 1)
  (h2 : 0 < 0.6^5 ∧ 0.6^5 < 1)
  (h3 : Real.logb 0.6 5 < 0) :
  Real.logb 0.6 5 < 0.6^5 ∧ 0.6^5 < 5^0.6 :=
sorry

end NUMINAMATH_GPT_log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6_l1811_181158


namespace NUMINAMATH_GPT_missing_score_and_variance_l1811_181119

theorem missing_score_and_variance (score_A score_B score_D score_E : ℕ) (avg_score : ℕ)
  (h_scores : score_A = 81 ∧ score_B = 79 ∧ score_D = 80 ∧ score_E = 82)
  (h_avg : avg_score = 80):
  ∃ (score_C variance : ℕ), score_C = 78 ∧ variance = 2 := by
  sorry

end NUMINAMATH_GPT_missing_score_and_variance_l1811_181119


namespace NUMINAMATH_GPT_fill_pipe_half_cistern_time_l1811_181100

theorem fill_pipe_half_cistern_time (time_to_fill_half : ℕ) 
  (H : time_to_fill_half = 10) : 
  time_to_fill_half = 10 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_fill_pipe_half_cistern_time_l1811_181100


namespace NUMINAMATH_GPT_circle_radius_increase_l1811_181184

theorem circle_radius_increase (r r' : ℝ) (h : π * r'^2 = (25.44 / 100 + 1) * π * r^2) : 
  (r' - r) / r * 100 = 12 :=
by sorry

end NUMINAMATH_GPT_circle_radius_increase_l1811_181184


namespace NUMINAMATH_GPT_gcd_lcm_product_l1811_181160

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 90) (h₂ : b = 135) : 
  Nat.gcd a b * Nat.lcm a b = 12150 :=
by
  -- Using given assumptions
  rw [h₁, h₂]
  -- Lean's definition of gcd and lcm in Nat
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l1811_181160


namespace NUMINAMATH_GPT_city_growth_rate_order_l1811_181147

theorem city_growth_rate_order 
  (Dover Eden Fairview : Type) 
  (highest lowest : Type)
  (h1 : Dover = highest → ¬(Eden = highest) ∧ (Fairview = lowest))
  (h2 : ¬(Dover = highest) ∧ Eden = highest ∧ Fairview = lowest → Eden = highest ∧ Dover = lowest ∧ Fairview = highest)
  (h3 : ¬(Fairview = lowest) → ¬(Eden = highest) ∧ ¬(Dover = highest)) : 
  Eden = highest ∧ Dover = lowest ∧ Fairview = highest ∧ Eden ≠ lowest :=
by
  sorry

end NUMINAMATH_GPT_city_growth_rate_order_l1811_181147


namespace NUMINAMATH_GPT_orlie_age_l1811_181111

theorem orlie_age (O R : ℕ) (h1 : R = 9) (h2 : R = (3 * O) / 4)
  (h3 : R - 4 = ((O - 4) / 2) + 1) : O = 12 :=
by
  sorry

end NUMINAMATH_GPT_orlie_age_l1811_181111


namespace NUMINAMATH_GPT_form_triangle_condition_right_angled_triangle_condition_l1811_181129

def vector (α : Type*) := α × α
noncomputable def oa : vector ℝ := ⟨2, -1⟩
noncomputable def ob : vector ℝ := ⟨3, 2⟩
noncomputable def oc (m : ℝ) : vector ℝ := ⟨m, 2 * m + 1⟩

def vector_sub (v1 v2 : vector ℝ) : vector ℝ := ⟨v1.1 - v2.1, v1.2 - v2.2⟩
def vector_dot (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem form_triangle_condition (m : ℝ) : 
  ¬ ((vector_sub ob oa).1 * (vector_sub (oc m) oa).2 = (vector_sub ob oa).2 * (vector_sub (oc m) oa).1) ↔ m ≠ 8 :=
sorry

theorem right_angled_triangle_condition (m : ℝ) : 
  (vector_dot (vector_sub ob oa) (vector_sub (oc m) oa) = 0 ∨ 
   vector_dot (vector_sub ob oa) (vector_sub (oc m) ob) = 0 ∨ 
   vector_dot (vector_sub (oc m) oa) (vector_sub (oc m) ob) = 0) ↔ 
  (m = -4/7 ∨ m = 6/7) :=
sorry

end NUMINAMATH_GPT_form_triangle_condition_right_angled_triangle_condition_l1811_181129


namespace NUMINAMATH_GPT_arrangements_three_events_l1811_181180

theorem arrangements_three_events (volunteers : ℕ) (events : ℕ) (h_vol : volunteers = 5) (h_events : events = 3) : 
  ∃ n : ℕ, n = (events^volunteers - events * 2^volunteers + events * 1^volunteers) ∧ n = 150 := 
by
  sorry

end NUMINAMATH_GPT_arrangements_three_events_l1811_181180


namespace NUMINAMATH_GPT_minimum_value_of_f_l1811_181132

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) * (x^2 - 4 * x + 3)

theorem minimum_value_of_f : ∃ m : ℝ, m = -16 ∧ ∀ x : ℝ, f x ≥ m :=
by
  use -16
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1811_181132


namespace NUMINAMATH_GPT_length_of_string_C_l1811_181181

theorem length_of_string_C (A B C : ℕ) (h1 : A = 6 * C) (h2 : A = 5 * B) (h3 : B = 12) : C = 10 :=
sorry

end NUMINAMATH_GPT_length_of_string_C_l1811_181181


namespace NUMINAMATH_GPT_mk97_x_eq_one_l1811_181197

noncomputable def mk97_initial_number (x : ℝ) : Prop := 
  x ≠ 0 ∧ 4 * (x^2 - x) = 0

theorem mk97_x_eq_one (x : ℝ) (h : mk97_initial_number x) : x = 1 := by
  sorry

end NUMINAMATH_GPT_mk97_x_eq_one_l1811_181197


namespace NUMINAMATH_GPT_positive_difference_complementary_angles_l1811_181117

theorem positive_difference_complementary_angles (a b : ℝ) 
  (h1 : a + b = 90) 
  (h2 : 3 * b = a) :
  |a - b| = 45 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_complementary_angles_l1811_181117


namespace NUMINAMATH_GPT_smallest_number_after_operations_n_111_smallest_number_after_operations_n_110_l1811_181113

theorem smallest_number_after_operations_n_111 :
  ∀ (n : ℕ), n = 111 → 
  (∃ (f : List ℕ → ℕ), -- The function f represents the sequence of operations
     (∀ (l : List ℕ), l = List.range 111 →
       (f l) = 0)) :=
by 
  sorry

theorem smallest_number_after_operations_n_110 :
  ∀ (n : ℕ), n = 110 → 
  (∃ (f : List ℕ → ℕ), -- The function f represents the sequence of operations
     (∀ (l : List ℕ), l = List.range 110 →
       (f l) = 1)) :=
by 
  sorry

end NUMINAMATH_GPT_smallest_number_after_operations_n_111_smallest_number_after_operations_n_110_l1811_181113


namespace NUMINAMATH_GPT_part1_part2_l1811_181103

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 2) - abs (x + 1)

theorem part1 (x : ℝ) : f x ≤ 3 ↔ -2/3 ≤ x ∧ x ≤ 6 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, f x ≤ abs (x + 1) + a^2) ↔ a ≤ -2 ∨ 2 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1811_181103


namespace NUMINAMATH_GPT_same_terminal_side_angle_l1811_181155

theorem same_terminal_side_angle (k : ℤ) : 
  0 ≤ (k * 360 - 35) ∧ (k * 360 - 35) < 360 → (k * 360 - 35) = 325 :=
by
  sorry

end NUMINAMATH_GPT_same_terminal_side_angle_l1811_181155


namespace NUMINAMATH_GPT_simplify_expression_l1811_181136

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (b * c + a * c + a * b) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1811_181136


namespace NUMINAMATH_GPT_regular_polygon_sides_l1811_181135

theorem regular_polygon_sides (θ : ℝ) (hθ : θ = 45) : 360 / θ = 8 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1811_181135


namespace NUMINAMATH_GPT_complement_intersection_example_l1811_181153

open Set

theorem complement_intersection_example
  (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3, 4})
  (hB : B = {2, 3}) :
  (U \ A) ∩ B = {2} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_example_l1811_181153


namespace NUMINAMATH_GPT_situationD_not_represented_l1811_181143

def situationA := -2 + 10 = 8

def situationB := -2 + 10 = 8

def situationC := 10 - 2 = 8 ∧ -2 + 10 = 8

def situationD := |10 - (-2)| = 12

theorem situationD_not_represented : ¬ (|10 - (-2)| = -2 + 10) := 
by
  sorry

end NUMINAMATH_GPT_situationD_not_represented_l1811_181143


namespace NUMINAMATH_GPT_price_difference_pc_sm_l1811_181130

-- Definitions based on given conditions
def S : ℕ := 300
def x : ℕ := sorry -- This is what we are trying to find
def PC : ℕ := S + x
def AT : ℕ := S + PC
def total_cost : ℕ := S + PC + AT

-- Theorem to be proved
theorem price_difference_pc_sm (h : total_cost = 2200) : x = 500 :=
by
  -- We would prove the theorem here
  sorry

end NUMINAMATH_GPT_price_difference_pc_sm_l1811_181130


namespace NUMINAMATH_GPT_vacation_cost_l1811_181194

theorem vacation_cost (n : ℕ) (h : 480 / n + 40 = 120) : n = 6 :=
sorry

end NUMINAMATH_GPT_vacation_cost_l1811_181194
