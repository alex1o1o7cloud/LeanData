import Mathlib

namespace NUMINAMATH_GPT_find_f_20_l677_67720

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_20 :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, f x = f (2 - x)) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x - 1 / 2) →
  f 20 = - 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_f_20_l677_67720


namespace NUMINAMATH_GPT_even_of_even_square_sqrt_two_irrational_l677_67735

-- Problem 1: Prove that if \( p^2 \) is even, then \( p \) is even given \( p \in \mathbb{Z} \).
theorem even_of_even_square (p : ℤ) (h : Even (p * p)) : Even p := 
sorry 

-- Problem 2: Prove that \( \sqrt{2} \) is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ Int.gcd a b = 1 ∧ (a : ℝ) / b = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_even_of_even_square_sqrt_two_irrational_l677_67735


namespace NUMINAMATH_GPT_elizabeth_husband_weight_l677_67783

-- Defining the variables for weights of the three wives
variable (s : ℝ) -- Weight of Simona
def elizabeta_weight : ℝ := s + 5
def georgetta_weight : ℝ := s + 10

-- Condition: The total weight of all wives
def total_wives_weight : ℝ := s + elizabeta_weight s + georgetta_weight s

-- Given: The total weight of all wives is 171 kg
def total_wives_weight_cond : Prop := total_wives_weight s = 171

-- Given:
-- Leon weighs the same as his wife.
-- Victor weighs one and a half times more than his wife.
-- Maurice weighs twice as much as his wife.

-- Given: Elizabeth's weight relationship
def elizabeth_weight_cond : Prop := (s + 5 * 1.5) = 85.5

-- Main proof problem:
theorem elizabeth_husband_weight (s : ℝ) (h1: total_wives_weight_cond s) : elizabeth_weight_cond s :=
by
  sorry

end NUMINAMATH_GPT_elizabeth_husband_weight_l677_67783


namespace NUMINAMATH_GPT_geometric_sequence_divisible_l677_67719

theorem geometric_sequence_divisible (a1 a2 : ℝ) (h1 : a1 = 5 / 8) (h2 : a2 = 25) :
  ∃ n : ℕ, n = 7 ∧ (40^(n-1) * (5/8)) % 10^7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_divisible_l677_67719


namespace NUMINAMATH_GPT_problem_1_problem_2a_problem_2b_l677_67733

noncomputable def v_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def v_b : ℝ × ℝ := (3, -Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (v_a x).1 * (v_b).1 + (v_a x).2 * (v_b).2

theorem problem_1 (x : ℝ) (h : x ∈ Set.Icc 0 Real.pi) : 
  (v_a x).1 * (v_b).2 = (v_a x).2 * (v_b).1 → x = (5 * Real.pi / 6) :=
by
  sorry

theorem problem_2a : 
  ∃ x ∈ Set.Icc 0 Real.pi, f x = 3 ∧ ∀ y ∈ Set.Icc 0 Real.pi, f y ≤ 3 :=
by
  sorry

theorem problem_2b :
  ∃ x ∈ Set.Icc 0 Real.pi, f x = -2 * Real.sqrt 3 ∧ ∀ y ∈ Set.Icc 0 Real.pi, f y ≥ -2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2a_problem_2b_l677_67733


namespace NUMINAMATH_GPT_find_value_of_2a_plus_c_l677_67797

theorem find_value_of_2a_plus_c (a b c : ℝ) (h1 : 3 * a + b + 2 * c = 3) (h2 : a + 3 * b + 2 * c = 1) :
  2 * a + c = 2 :=
sorry

end NUMINAMATH_GPT_find_value_of_2a_plus_c_l677_67797


namespace NUMINAMATH_GPT_years_to_earn_house_l677_67788

-- Defining the variables
variables (E S H : ℝ)

-- Defining the assumptions
def annual_expenses_savings_relation (E S : ℝ) : Prop :=
  8 * E = 12 * S

def annual_income_relation (H E S : ℝ) : Prop :=
  H / 24 = E + S

-- Theorem stating that it takes 60 years to earn the amount needed to buy the house
theorem years_to_earn_house (E S H : ℝ) 
  (h1 : annual_expenses_savings_relation E S) 
  (h2 : annual_income_relation H E S) : 
  H / S = 60 :=
by
  sorry

end NUMINAMATH_GPT_years_to_earn_house_l677_67788


namespace NUMINAMATH_GPT_least_number_to_add_l677_67727

theorem least_number_to_add (n : ℕ) (d : ℕ) (h1 : n = 907223) (h2 : d = 577) : (d - (n % d) = 518) := 
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_least_number_to_add_l677_67727


namespace NUMINAMATH_GPT_isosceles_triangle_leg_length_l677_67701

theorem isosceles_triangle_leg_length
  (P : ℝ) (base : ℝ) (L : ℝ)
  (h_isosceles : true)
  (h_perimeter : P = 24)
  (h_base : base = 10)
  (h_perimeter_formula : P = base + 2 * L) :
  L = 7 := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_leg_length_l677_67701


namespace NUMINAMATH_GPT_sequence_expression_l677_67717

noncomputable def a_n (n : ℕ) : ℤ :=
if n = 1 then -1 else 1 - 2^n

def S_n (a_n : ℕ → ℤ) (n : ℕ) : ℤ :=
2 * a_n n + n

theorem sequence_expression :
  ∀ n : ℕ, n > 0 → (a_n n = 1 - 2^n) :=
by
  intro n hn
  sorry

end NUMINAMATH_GPT_sequence_expression_l677_67717


namespace NUMINAMATH_GPT_find_other_integer_l677_67780

theorem find_other_integer (x y : ℤ) (h1 : 3*x + 4*y = 103) (h2 : x = 19 ∨ y = 19) : x = 9 ∨ y = 9 :=
by sorry

end NUMINAMATH_GPT_find_other_integer_l677_67780


namespace NUMINAMATH_GPT_simplify_eval_expr_l677_67786

noncomputable def a : ℝ := (Real.sqrt 2) + 1
noncomputable def b : ℝ := (Real.sqrt 2) - 1

theorem simplify_eval_expr (a b : ℝ) (ha : a = (Real.sqrt 2) + 1) (hb : b = (Real.sqrt 2) - 1) : 
  (a^2 - b^2) / a / (a + (2 * a * b + b^2) / a) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_eval_expr_l677_67786


namespace NUMINAMATH_GPT_sum_of_first_eight_terms_l677_67745

-- Define the first term, common ratio, and the number of terms
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 8

-- Sum of the first n terms of a geometric sequence
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Proof statement
theorem sum_of_first_eight_terms : geometric_sum a r n = 3280 / 6561 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_eight_terms_l677_67745


namespace NUMINAMATH_GPT_positive_difference_of_y_l677_67777

theorem positive_difference_of_y (y : ℝ) (h : (50 + y) / 2 = 35) : |50 - y| = 30 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_y_l677_67777


namespace NUMINAMATH_GPT_divisibility_theorem_l677_67791

theorem divisibility_theorem {a m x n : ℕ} : (m ∣ n) ↔ (x^m - a^m ∣ x^n - a^n) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_theorem_l677_67791


namespace NUMINAMATH_GPT_problem_solution_l677_67715

theorem problem_solution (a b c d e f g : ℝ) 
  (h1 : a + b + e = 7)
  (h2 : b + c + f = 10)
  (h3 : c + d + g = 6)
  (h4 : e + f + g = 9) : 
  a + d + g = 6 := 
sorry

end NUMINAMATH_GPT_problem_solution_l677_67715


namespace NUMINAMATH_GPT_max_abs_sum_on_circle_l677_67732

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_max_abs_sum_on_circle_l677_67732


namespace NUMINAMATH_GPT_horner_value_at_2_l677_67705

noncomputable def f (x : ℝ) := 2 * x^5 - 3 * x^3 + 2 * x^2 + x - 3

theorem horner_value_at_2 : f 2 = 12 := sorry

end NUMINAMATH_GPT_horner_value_at_2_l677_67705


namespace NUMINAMATH_GPT_alexa_pages_left_l677_67775

theorem alexa_pages_left 
  (total_pages : ℕ) 
  (first_day_read : ℕ) 
  (next_day_read : ℕ) 
  (total_pages_val : total_pages = 95) 
  (first_day_read_val : first_day_read = 18) 
  (next_day_read_val : next_day_read = 58) : 
  total_pages - (first_day_read + next_day_read) = 19 := by
  sorry

end NUMINAMATH_GPT_alexa_pages_left_l677_67775


namespace NUMINAMATH_GPT_max_value_of_sum_l677_67796

open Real

theorem max_value_of_sum (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : a + b + c = 3) :
  (ab / (a + b) + bc / (b + c) + ca / (c + a)) ≤ 3 / 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_sum_l677_67796


namespace NUMINAMATH_GPT_jeans_price_difference_l677_67714

variable (x : Real)

theorem jeans_price_difference
  (hx : 0 < x) -- Assuming x > 0 for a positive cost
  (r := 1.40 * x)
  (c := 1.30 * r) :
  c = 1.82 * x :=
by
  sorry

end NUMINAMATH_GPT_jeans_price_difference_l677_67714


namespace NUMINAMATH_GPT_range_of_m_l677_67795

theorem range_of_m (m : ℝ) (x : ℝ) (h₁ : x^2 - 8*x - 20 ≤ 0) 
  (h₂ : (x - 1 - m) * (x - 1 + m) ≤ 0) (h₃ : 0 < m) : 
  m ≤ 3 := sorry

end NUMINAMATH_GPT_range_of_m_l677_67795


namespace NUMINAMATH_GPT_cupcakes_left_over_l677_67742

def total_cupcakes := 40
def ms_delmont_class := 18
def mrs_donnelly_class := 16
def ms_delmont := 1
def mrs_donnelly := 1
def school_nurse := 1
def school_principal := 1

def total_given_away := ms_delmont_class + mrs_donnelly_class + ms_delmont + mrs_donnelly + school_nurse + school_principal

theorem cupcakes_left_over : total_cupcakes - total_given_away = 2 := by
  sorry

end NUMINAMATH_GPT_cupcakes_left_over_l677_67742


namespace NUMINAMATH_GPT_HCl_yield_l677_67736

noncomputable def total_moles_HCl (moles_C2H6 moles_Cl2 yield1 yield2 : ℝ) : ℝ :=
  let theoretical_yield1 := if moles_C2H6 ≤ moles_Cl2 then moles_C2H6 else moles_Cl2
  let actual_yield1 := theoretical_yield1 * yield1
  let theoretical_yield2 := actual_yield1
  let actual_yield2 := theoretical_yield2 * yield2
  actual_yield1 + actual_yield2

theorem HCl_yield (moles_C2H6 moles_Cl2 : ℝ) (yield1 yield2 : ℝ) :
  moles_C2H6 = 3 → moles_Cl2 = 3 → yield1 = 0.85 → yield2 = 0.70 →
  total_moles_HCl moles_C2H6 moles_Cl2 yield1 yield2 = 4.335 :=
by
  intros h1 h2 h3 h4
  simp [total_moles_HCl, h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_HCl_yield_l677_67736


namespace NUMINAMATH_GPT_min_value_expr_ge_52_l677_67771

open Real

theorem min_value_expr_ge_52 (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  (sin x + 3 * (1 / sin x)) ^ 2 + (cos x + 3 * (1 / cos x)) ^ 2 ≥ 52 := 
by
  sorry

end NUMINAMATH_GPT_min_value_expr_ge_52_l677_67771


namespace NUMINAMATH_GPT_max_value_of_q_l677_67751

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end NUMINAMATH_GPT_max_value_of_q_l677_67751


namespace NUMINAMATH_GPT_A_B_finish_l677_67764

theorem A_B_finish (A B C : ℕ → ℝ) (h1 : A + B + C = 1 / 6) (h2 : C = 1 / 10) :
  1 / (A + B) = 15 :=
by
  sorry

end NUMINAMATH_GPT_A_B_finish_l677_67764


namespace NUMINAMATH_GPT_find_f_neg_one_l677_67725

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, f (x^2 + y) = f x + f (y^2)

theorem find_f_neg_one : f (-1) = 0 := sorry

end NUMINAMATH_GPT_find_f_neg_one_l677_67725


namespace NUMINAMATH_GPT_probability_non_defective_pens_l677_67794

theorem probability_non_defective_pens :
  let total_pens := 12
  let defective_pens := 6
  let non_defective_pens := total_pens - defective_pens
  let probability_first_non_defective := non_defective_pens / total_pens
  let probability_second_non_defective := (non_defective_pens - 1) / (total_pens - 1)
  (probability_first_non_defective * probability_second_non_defective = 5 / 22) :=
by
  rfl

end NUMINAMATH_GPT_probability_non_defective_pens_l677_67794


namespace NUMINAMATH_GPT_cos_alpha_value_l677_67753

open Real

theorem cos_alpha_value (α : ℝ) (h_cos : cos (α - π/6) = 15/17) (h_range : π/6 < α ∧ α < π/2) : 
  cos α = (15 * Real.sqrt 3 - 8) / 34 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_value_l677_67753


namespace NUMINAMATH_GPT_total_time_spent_l677_67723

def one_round_time : ℕ := 30
def saturday_initial_rounds : ℕ := 1
def saturday_additional_rounds : ℕ := 10
def sunday_rounds : ℕ := 15

theorem total_time_spent :
  one_round_time * (saturday_initial_rounds + saturday_additional_rounds + sunday_rounds) = 780 := by
  sorry

end NUMINAMATH_GPT_total_time_spent_l677_67723


namespace NUMINAMATH_GPT_borrowed_amount_l677_67700

theorem borrowed_amount (P : ℝ) 
    (borrow_rate : ℝ := 4) 
    (lend_rate : ℝ := 6) 
    (borrow_time : ℝ := 2) 
    (lend_time : ℝ := 2) 
    (gain_per_year : ℝ := 140) 
    (h₁ : ∀ (P : ℝ), P / 8.333 - P / 12.5 = 280) 
    : P = 7000 := 
sorry

end NUMINAMATH_GPT_borrowed_amount_l677_67700


namespace NUMINAMATH_GPT_triangular_pyramid_surface_area_l677_67704

theorem triangular_pyramid_surface_area
  (base_area : ℝ)
  (side_area : ℝ) :
  base_area = 3 ∧ side_area = 6 → base_area + 3 * side_area = 21 :=
by
  sorry

end NUMINAMATH_GPT_triangular_pyramid_surface_area_l677_67704


namespace NUMINAMATH_GPT_angle_of_inclination_range_l677_67789

noncomputable def curve (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

noncomputable def tangent_slope (x : ℝ) : ℝ := 
  -4 * Real.exp x / (Real.exp x + 1) ^ 2

theorem angle_of_inclination_range (x : ℝ) (a : ℝ) 
  (hx : tangent_slope x = Real.tan a) : 
  (3 * Real.pi / 4 ≤ a ∧ a < Real.pi) :=
by 
  sorry

end NUMINAMATH_GPT_angle_of_inclination_range_l677_67789


namespace NUMINAMATH_GPT_conversion_200_meters_to_kilometers_l677_67778

noncomputable def meters_to_kilometers (meters : ℕ) : ℝ :=
  meters / 1000

theorem conversion_200_meters_to_kilometers :
  meters_to_kilometers 200 = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_conversion_200_meters_to_kilometers_l677_67778


namespace NUMINAMATH_GPT_distinct_real_solutions_l677_67784

open Real Nat

noncomputable def p_n : ℕ → ℝ → ℝ 
| 0, x => x
| (n+1), x => (p_n n (x^2 - 2))

theorem distinct_real_solutions (n : ℕ) : 
  ∃ S : Finset ℝ, S.card = 2^n ∧ ∀ x ∈ S, p_n n x = x ∧ (∀ y ∈ S, x ≠ y → x ≠ y) := 
sorry

end NUMINAMATH_GPT_distinct_real_solutions_l677_67784


namespace NUMINAMATH_GPT_number_of_trees_is_correct_l677_67706

-- Define the conditions
def length_of_plot := 120
def width_of_plot := 70
def distance_between_trees := 5

-- Define the calculated number of intervals along each side
def intervals_along_length := length_of_plot / distance_between_trees
def intervals_along_width := width_of_plot / distance_between_trees

-- Define the number of trees along each side including the boundaries
def trees_along_length := intervals_along_length + 1
def trees_along_width := intervals_along_width + 1

-- Define the total number of trees
def total_number_of_trees := trees_along_length * trees_along_width

-- The theorem we want to prove
theorem number_of_trees_is_correct : total_number_of_trees = 375 :=
by sorry

end NUMINAMATH_GPT_number_of_trees_is_correct_l677_67706


namespace NUMINAMATH_GPT_triangle_inequality_l677_67712

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l677_67712


namespace NUMINAMATH_GPT_quadratic_inequality_solution_is_interval_l677_67746

noncomputable def quadratic_inequality_solution : Set ℝ :=
  { x : ℝ | -3*x^2 + 9*x + 12 > 0 }

theorem quadratic_inequality_solution_is_interval :
  quadratic_inequality_solution = { x : ℝ | -1 < x ∧ x < 4 } :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_is_interval_l677_67746


namespace NUMINAMATH_GPT_boat_speed_still_water_l677_67743

theorem boat_speed_still_water (V_s : ℝ) (T_u T_d : ℝ) 
  (h1 : V_s = 24) 
  (h2 : T_u = 2 * T_d) 
  (h3 : (V_b - V_s) * T_u = (V_b + V_s) * T_d) : 
  V_b = 72 := 
sorry

end NUMINAMATH_GPT_boat_speed_still_water_l677_67743


namespace NUMINAMATH_GPT_first_set_cost_l677_67724

theorem first_set_cost (F S : ℕ) (hS : S = 50) (h_equation : 2 * F + 3 * S = 220) 
: 3 * F + S = 155 := 
sorry

end NUMINAMATH_GPT_first_set_cost_l677_67724


namespace NUMINAMATH_GPT_compute_problem_l677_67757

theorem compute_problem : (19^12 / 19^8)^2 = 130321 := by
  sorry

end NUMINAMATH_GPT_compute_problem_l677_67757


namespace NUMINAMATH_GPT_line_intersection_l677_67779

-- Parameters for the first line
def line1_param (s : ℝ) : ℝ × ℝ := (1 - 2 * s, 4 + 3 * s)

-- Parameters for the second line
def line2_param (v : ℝ) : ℝ × ℝ := (-v, 5 + 6 * v)

-- Statement of the intersection point
theorem line_intersection :
  ∃ (s v : ℝ), line1_param s = (-1 / 9, 17 / 3) ∧ line2_param v = (-1 / 9, 17 / 3) :=
by
  -- Placeholder for the proof, which we are not providing as per instructions
  sorry

end NUMINAMATH_GPT_line_intersection_l677_67779


namespace NUMINAMATH_GPT_last_digit_to_appear_is_6_l677_67787

def modified_fib (n : ℕ) : ℕ :=
match n with
| 1 => 2
| 2 => 3
| n + 3 => modified_fib (n + 2) + modified_fib (n + 1)
| _ => 0 -- To silence the "missing cases" warning; won't be hit.

theorem last_digit_to_appear_is_6 :
  ∃ N : ℕ, ∀ n : ℕ, (n < N → ∃ d, d < 10 ∧ 
    (∀ m < n, (modified_fib m) % 10 ≠ d) ∧ d = 6) := sorry

end NUMINAMATH_GPT_last_digit_to_appear_is_6_l677_67787


namespace NUMINAMATH_GPT_increase_in_lines_l677_67770

variable (L : ℝ)
variable (h1 : L + (1 / 3) * L = 240)

theorem increase_in_lines : (240 - L) = 60 := by
  sorry

end NUMINAMATH_GPT_increase_in_lines_l677_67770


namespace NUMINAMATH_GPT_M_subset_N_l677_67754

def M : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 2 * a + 2}
def N : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4 * b + 5}

theorem M_subset_N : M ⊆ N := 
by 
  sorry

end NUMINAMATH_GPT_M_subset_N_l677_67754


namespace NUMINAMATH_GPT_minimize_quadratic_sum_l677_67766

theorem minimize_quadratic_sum (a b : ℝ) : 
  ∃ x : ℝ, y = (x-a)^2 + (x-b)^2 ∧ (∀ x', (x'-a)^2 + (x'-b)^2 ≥ y) ∧ x = (a + b) / 2 := 
sorry

end NUMINAMATH_GPT_minimize_quadratic_sum_l677_67766


namespace NUMINAMATH_GPT_correct_conclusions_l677_67769

noncomputable def f1 (x : ℝ) : ℝ := 2^x - 1
noncomputable def f2 (x : ℝ) : ℝ := x^3
noncomputable def f3 (x : ℝ) : ℝ := x
noncomputable def f4 (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2

theorem correct_conclusions :
  ((∀ x, 0 < x ∧ x < 1 → f4 x > f1 x ∧ f4 x > f2 x ∧ f4 x > f3 x) ∧
  (∀ x, x > 1 → f4 x < f1 x ∧ f4 x < f2 x ∧ f4 x < f3 x)) ∧
  (∀ x, ¬(f3 x > f1 x ∧ f3 x > f2 x ∧ f3 x > f4 x) ∧
        ¬(f3 x < f1 x ∧ f3 x < f2 x ∧ f3 x < f4 x)) ∧
  (∃ x, x > 0 ∧ ∀ y, y > x → f1 y > f2 y ∧ f1 y > f3 y ∧ f1 y > f4 y) := by
  sorry

end NUMINAMATH_GPT_correct_conclusions_l677_67769


namespace NUMINAMATH_GPT_two_cos_45_eq_sqrt_2_l677_67758

theorem two_cos_45_eq_sqrt_2 : 2 * Real.cos (pi / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_two_cos_45_eq_sqrt_2_l677_67758


namespace NUMINAMATH_GPT_binary_mul_correct_l677_67747

def binary_mul (a b : ℕ) : ℕ :=
  a * b

def binary_to_nat (s : String) : ℕ :=
  String.foldr (λ c acc => if c = '1' then acc * 2 + 1 else acc * 2) 0 s

def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0"
  else let rec aux (n : ℕ) (acc : String) :=
         if n = 0 then acc
         else aux (n / 2) (if n % 2 = 0 then "0" ++ acc else "1" ++ acc)
       aux n ""

theorem binary_mul_correct :
  nat_to_binary (binary_mul (binary_to_nat "1101") (binary_to_nat "111")) = "10001111" :=
by
  sorry

end NUMINAMATH_GPT_binary_mul_correct_l677_67747


namespace NUMINAMATH_GPT_a_n_formula_b_n_formula_l677_67767

namespace SequenceFormulas

theorem a_n_formula (n : ℕ) (h_pos : 0 < n) : 
  (∃ S : ℕ → ℕ, S n = 2 * n^2 + 2 * n) → ∃ a : ℕ → ℕ, a n = 4 * n :=
by
  sorry

theorem b_n_formula (n : ℕ) (h_pos : 0 < n) : 
  (∃ T : ℕ → ℕ, T n = 2 - (if n > 1 then T (n-1) else 1)) → ∃ b : ℕ → ℝ, b n = (1/2)^(n-1) :=
by
  sorry

end SequenceFormulas


end NUMINAMATH_GPT_a_n_formula_b_n_formula_l677_67767


namespace NUMINAMATH_GPT_trenton_earning_goal_l677_67734

-- Parameters
def fixed_weekly_earnings : ℝ := 190
def commission_rate : ℝ := 0.04
def sales_amount : ℝ := 7750
def goal : ℝ := 500

-- Proof statement
theorem trenton_earning_goal :
  fixed_weekly_earnings + (commission_rate * sales_amount) = goal :=
by
  sorry

end NUMINAMATH_GPT_trenton_earning_goal_l677_67734


namespace NUMINAMATH_GPT_largest_n_polynomials_l677_67739

theorem largest_n_polynomials :
  ∃ (P : ℕ → (ℝ → ℝ)), (∀ i j, i ≠ j → ∀ x, P i x + P j x ≠ 0) ∧ (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ∃ x, P i x + P j x + P k x = 0) ↔ n = 3 := 
sorry

end NUMINAMATH_GPT_largest_n_polynomials_l677_67739


namespace NUMINAMATH_GPT_volume_of_cuboid_l677_67773

theorem volume_of_cuboid (l w h : ℝ) (hlw: l * w = 120) (hwh: w * h = 72) (hhl: h * l = 60) : l * w * h = 720 :=
  sorry

end NUMINAMATH_GPT_volume_of_cuboid_l677_67773


namespace NUMINAMATH_GPT_snack_eaters_initial_count_l677_67711

-- Define all variables and conditions used in the problem
variables (S : ℕ) (initial_people : ℕ) (new_outsiders_1 : ℕ) (new_outsiders_2 : ℕ) (left_after_first_half : ℕ) (left_after_second_half : ℕ) (remaining_snack_eaters : ℕ)

-- Assign the specific values according to conditions
def conditions := 
  initial_people = 200 ∧
  new_outsiders_1 = 20 ∧
  new_outsiders_2 = 10 ∧
  left_after_first_half = (S + new_outsiders_1) / 2 ∧
  left_after_second_half = left_after_first_half + new_outsiders_2 - 30 ∧
  remaining_snack_eaters = left_after_second_half / 2 ∧
  remaining_snack_eaters = 20

-- State the theorem to prove
theorem snack_eaters_initial_count (S : ℕ) (initial_people new_outsiders_1 new_outsiders_2 left_after_first_half left_after_second_half remaining_snack_eaters : ℕ) :
  conditions S initial_people new_outsiders_1 new_outsiders_2 left_after_first_half left_after_second_half remaining_snack_eaters → S = 100 :=
by sorry

end NUMINAMATH_GPT_snack_eaters_initial_count_l677_67711


namespace NUMINAMATH_GPT_range_of_a_l677_67768

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def increasing_on_negative (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → y ≤ 0 → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (ha : even_function f) (hb : increasing_on_negative f) 
  (hc : ∀ a : ℝ, f a ≤ f (2 - a)) : ∀ a : ℝ, a < 1 → false :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l677_67768


namespace NUMINAMATH_GPT_maria_trip_distance_l677_67740

theorem maria_trip_distance (D : ℝ) 
  (h1 : D / 2 + ((D / 2) / 4) + 150 = D) 
  (h2 : 150 = 3 * D / 8) : 
  D = 400 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_maria_trip_distance_l677_67740


namespace NUMINAMATH_GPT_total_maggots_served_l677_67721

-- Define the conditions in Lean
def maggots_first_attempt : ℕ := 10
def maggots_second_attempt : ℕ := 10

-- Define the statement to prove
theorem total_maggots_served : maggots_first_attempt + maggots_second_attempt = 20 :=
by 
  sorry

end NUMINAMATH_GPT_total_maggots_served_l677_67721


namespace NUMINAMATH_GPT_seven_k_plus_four_l677_67738

theorem seven_k_plus_four (k m n : ℕ) (h1 : 4 * k + 5 = m^2) (h2 : 9 * k + 4 = n^2) (hk : k = 5) : 
  7 * k + 4 = 39 :=
by 
  -- assume conditions
  have h1' := h1
  have h2' := h2
  have hk' := hk
  sorry

end NUMINAMATH_GPT_seven_k_plus_four_l677_67738


namespace NUMINAMATH_GPT_ice_cream_cost_proof_l677_67713

-- Assume the cost of the ice cream and toppings
def cost_of_ice_cream : ℝ := 2 -- Ice cream cost in dollars
def cost_per_topping : ℝ := 0.5 -- Cost per topping in dollars
def total_cost_of_sundae_with_10_toppings : ℝ := 7 -- Total cost in dollars

theorem ice_cream_cost_proof :
  (∀ (cost_of_ice_cream : ℝ), 
    total_cost_of_sundae_with_10_toppings = cost_of_ice_cream + 10 * cost_per_topping) →
  cost_of_ice_cream = 2 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_cost_proof_l677_67713


namespace NUMINAMATH_GPT_ratio_of_work_completed_by_a_l677_67710

theorem ratio_of_work_completed_by_a (A B W : ℝ) (ha : (A + B) * 6 = W) :
  (A * 3) / W = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_work_completed_by_a_l677_67710


namespace NUMINAMATH_GPT_fifth_number_in_tenth_row_l677_67765

def nth_number_in_row (n k : ℕ) : ℕ :=
  7 * n - (7 - k)

theorem fifth_number_in_tenth_row : nth_number_in_row 10 5 = 68 :=
by
  sorry

end NUMINAMATH_GPT_fifth_number_in_tenth_row_l677_67765


namespace NUMINAMATH_GPT_valid_pairs_l677_67776

def valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def valid_number (n : ℕ) : Prop :=
  let digits := [5, 3, 2, 9, n / 10 % 10, n % 10]
  (n % 2 = 0) ∧ (digits.sum % 3 = 0)

theorem valid_pairs (d₀ d₁ : ℕ) :
  valid_digit d₀ →
  valid_digit d₁ →
  (d₀ % 2 = 0) →
  valid_number (53290 * 10 + d₀ * 10 + d₁) →
  (d₀, d₁) = (0, 3) ∨ (d₀, d₁) = (2, 0) ∨ (d₀, d₁) = (2, 3) ∨ (d₀, d₁) = (2, 6) ∨
  (d₀, d₁) = (2, 9) ∨ (d₀, d₁) = (4, 1) ∨ (d₀, d₁) = (4, 4) ∨ (d₀, d₁) = (4, 7) ∨
  (d₀, d₁) = (6, 2) ∨ (d₀, d₁) = (6, 5) ∨ (d₀, d₁) = (6, 8) ∨ (d₀, d₁) = (8, 0) :=
by sorry

end NUMINAMATH_GPT_valid_pairs_l677_67776


namespace NUMINAMATH_GPT_positive_root_in_range_l677_67792

theorem positive_root_in_range : ∃ x > 0, (x^2 - 2 * x - 1 = 0) ∧ (2 < x ∧ x < 3) :=
by
  sorry

end NUMINAMATH_GPT_positive_root_in_range_l677_67792


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l677_67730

theorem problem1_solution (x : ℝ) (h : 5 / (x - 1) = 1 / (2 * x + 1)) : x = -2 / 3 := sorry

theorem problem2_solution (x : ℝ) (h : 1 / (x - 2) + 2 = (1 - x) / (2 - x)) : false := sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l677_67730


namespace NUMINAMATH_GPT_smallest_y_in_arithmetic_series_l677_67799

theorem smallest_y_in_arithmetic_series (x y z : ℝ) (h1 : x < y) (h2 : y < z) (h3 : (x * y * z) = 216) : y = 6 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_y_in_arithmetic_series_l677_67799


namespace NUMINAMATH_GPT_min_value_of_expression_l677_67750

theorem min_value_of_expression (x y : ℝ) (h : 2 * x - y = 4) : ∃ z : ℝ, (z = 4^x + (1/2)^y) ∧ z = 8 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l677_67750


namespace NUMINAMATH_GPT_right_triangle_area_l677_67703

variable {AB BC AC : ℕ}

theorem right_triangle_area : ∀ (AB BC AC : ℕ), (AC = 50) → (AB + BC = 70) → (AB^2 + BC^2 = AC^2) → (1 / 2) * AB * BC = 300 :=
by
  intros AB BC AC h1 h2 h3
  -- Proof steps will be added here
  sorry

end NUMINAMATH_GPT_right_triangle_area_l677_67703


namespace NUMINAMATH_GPT_geometric_sequence_and_sum_l677_67782

theorem geometric_sequence_and_sum (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ)
  (h_a1 : a 1 = 3/2)
  (h_a_recur : ∀ n : ℕ, a (n + 1) = 3 * a n - 1)
  (h_b_def : ∀ n : ℕ, b n = a n - 1/2) :
  (∀ n : ℕ, b (n + 1) = 3 * b n ∧ b 1 = 1) ∧ 
  (∀ n : ℕ, S n = (3^n + n - 1) / 2) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_and_sum_l677_67782


namespace NUMINAMATH_GPT_smaller_circle_circumference_l677_67737

-- Definitions based on the conditions given in the problem
def AB : ℝ := 24
def BC : ℝ := 45
def CD : ℝ := 28
def DA : ℝ := 53
def smaller_circle_diameter : ℝ := AB

-- Main statement to prove
theorem smaller_circle_circumference :
  let r : ℝ := smaller_circle_diameter / 2
  let circumference := 2 * Real.pi * r
  circumference = 24 * Real.pi := by
  sorry

end NUMINAMATH_GPT_smaller_circle_circumference_l677_67737


namespace NUMINAMATH_GPT_smallest_possible_b_l677_67748

theorem smallest_possible_b (a b c : ℚ) (h1 : a < b) (h2 : b < c)
    (arithmetic_seq : 2 * b = a + c) (geometric_seq : c^2 = a * b) :
    b = 1 / 2 :=
by
  let a := 4 * b
  let c := 2 * b - a
  -- rewrite and derived equations will be done in the proof
  sorry

end NUMINAMATH_GPT_smallest_possible_b_l677_67748


namespace NUMINAMATH_GPT_part_a_part_b_l677_67744

-- Define what it means for a number to be "surtido"
def is_surtido (A : ℕ) : Prop :=
  ∀ n, (1 ≤ n → n ≤ (A.digits 10).sum → ∃ B : ℕ, n = (B.digits 10).sum) 

-- Part (a): Prove that if 1, 2, 3, 4, 5, 6, 7, and 8 can be expressed as sums of digits in A, then A is "surtido".
theorem part_a (A : ℕ)
  (h1 : ∃ B1 : ℕ, 1 = (B1.digits 10).sum)
  (h2 : ∃ B2 : ℕ, 2 = (B2.digits 10).sum)
  (h3 : ∃ B3 : ℕ, 3 = (B3.digits 10).sum)
  (h4 : ∃ B4 : ℕ, 4 = (B4.digits 10).sum)
  (h5 : ∃ B5 : ℕ, 5 = (B5.digits 10).sum)
  (h6 : ∃ B6 : ℕ, 6 = (B6.digits 10).sum)
  (h7 : ∃ B7 : ℕ, 7 = (B7.digits 10).sum)
  (h8 : ∃ B8 : ℕ, 8 = (B8.digits 10).sum) : is_surtido A :=
sorry

-- Part (b): Determine if having the sums 1, 2, 3, 4, 5, 6, and 7 as sums of digits in A implies that A is "surtido".
theorem part_b (A : ℕ)
  (h1 : ∃ B1 : ℕ, 1 = (B1.digits 10).sum)
  (h2 : ∃ B2 : ℕ, 2 = (B2.digits 10).sum)
  (h3 : ∃ B3 : ℕ, 3 = (B3.digits 10).sum)
  (h4 : ∃ B4 : ℕ, 4 = (B4.digits 10).sum)
  (h5 : ∃ B5 : ℕ, 5 = (B5.digits 10).sum)
  (h6 : ∃ B6 : ℕ, 6 = (B6.digits 10).sum)
  (h7 : ∃ B7 : ℕ, 7 = (B7.digits 10).sum) : ¬is_surtido A :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l677_67744


namespace NUMINAMATH_GPT_red_light_probability_l677_67774

theorem red_light_probability (n : ℕ) (p_r : ℚ) (waiting_time_for_two_red : ℚ) 
    (prob_two_red : ℚ) :
    n = 4 →
    p_r = (1/3 : ℚ) →
    waiting_time_for_two_red = 4 →
    prob_two_red = (8/27 : ℚ) :=
by
  intros hn hp hw
  sorry

end NUMINAMATH_GPT_red_light_probability_l677_67774


namespace NUMINAMATH_GPT_wheat_acres_l677_67731

def cultivate_crops (x y : ℕ) : Prop :=
  (42 * x + 30 * y = 18600) ∧ (x + y = 500) 

theorem wheat_acres : ∃ y, ∃ x, 
  cultivate_crops x y ∧ y = 200 :=
by {sorry}

end NUMINAMATH_GPT_wheat_acres_l677_67731


namespace NUMINAMATH_GPT_correct_operation_l677_67728

variables (a : ℝ)

-- defining the expressions to be compared
def lhs := 2 * a^2 * a^4
def rhs := 2 * a^6

theorem correct_operation : lhs a = rhs a := 
by sorry

end NUMINAMATH_GPT_correct_operation_l677_67728


namespace NUMINAMATH_GPT_cody_final_money_l677_67709

-- Definitions for the initial conditions
def original_money : ℝ := 45
def birthday_money : ℝ := 9
def game_price : ℝ := 19
def discount_rate : ℝ := 0.10
def friend_owes : ℝ := 12

-- Calculate the final amount Cody has
def final_amount : ℝ := original_money + birthday_money - (game_price * (1 - discount_rate)) + friend_owes

-- The theorem to prove the amount of money Cody has now
theorem cody_final_money :
  final_amount = 48.90 :=
by sorry

end NUMINAMATH_GPT_cody_final_money_l677_67709


namespace NUMINAMATH_GPT_car_initial_time_l677_67760

variable (t : ℝ)

theorem car_initial_time (h : 80 = 720 / (3/2 * t)) : t = 6 :=
sorry

end NUMINAMATH_GPT_car_initial_time_l677_67760


namespace NUMINAMATH_GPT_divisor_of_425904_l677_67762

theorem divisor_of_425904 :
  ∃ (d : ℕ), d = 7 ∧ ∃ (n : ℕ), n = 425897 + 7 ∧ 425904 % d = 0 :=
by
  sorry

end NUMINAMATH_GPT_divisor_of_425904_l677_67762


namespace NUMINAMATH_GPT_benjamin_speed_l677_67749

-- Define the problem conditions
def distance : ℕ := 800 -- Distance in kilometers
def time : ℕ := 10 -- Time in hours

-- Define the main statement
theorem benjamin_speed : distance / time = 80 := by
  sorry

end NUMINAMATH_GPT_benjamin_speed_l677_67749


namespace NUMINAMATH_GPT_max_donation_amount_l677_67772

theorem max_donation_amount (x : ℝ) : 
  (500 * x + 1500 * (x / 2) = 0.4 * 3750000) → x = 1200 :=
by 
  sorry

end NUMINAMATH_GPT_max_donation_amount_l677_67772


namespace NUMINAMATH_GPT_polynomial_transformation_l677_67702

noncomputable def p : ℝ → ℝ := sorry

variable (k : ℕ)

axiom ax1 (x : ℝ) : p (2 * x) = 2^(k - 1) * (p x + p (x + 1/2))

theorem polynomial_transformation (k : ℕ) (p : ℝ → ℝ)
  (h_p : ∀ x : ℝ, p (2 * x) = 2^(k - 1) * (p x + p (x + 1/2))) :
  ∀ x : ℝ, p (3 * x) = 3^(k - 1) * (p x + p (x + 1/3) + p (x + 2/3)) := sorry

end NUMINAMATH_GPT_polynomial_transformation_l677_67702


namespace NUMINAMATH_GPT_dog_bones_remaining_l677_67793

noncomputable def initial_bones : ℕ := 350
noncomputable def factor : ℕ := 9
noncomputable def found_bones : ℕ := factor * initial_bones
noncomputable def total_bones : ℕ := initial_bones + found_bones
noncomputable def bones_given_away : ℕ := 120
noncomputable def bones_remaining : ℕ := total_bones - bones_given_away

theorem dog_bones_remaining : bones_remaining = 3380 :=
by
  sorry

end NUMINAMATH_GPT_dog_bones_remaining_l677_67793


namespace NUMINAMATH_GPT_sum_ak_div_k2_ge_sum_inv_k_l677_67752

open BigOperators

theorem sum_ak_div_k2_ge_sum_inv_k
  (n : ℕ)
  (a : Fin n → ℕ)
  (hpos : ∀ k, 0 < a k)
  (hdist : Function.Injective a) :
  ∑ k : Fin n, (a k : ℝ) / (k + 1 : ℝ)^2 ≥ ∑ k : Fin n, 1 / (k + 1 : ℝ) := sorry

end NUMINAMATH_GPT_sum_ak_div_k2_ge_sum_inv_k_l677_67752


namespace NUMINAMATH_GPT_solution_set_of_inequality1_solution_set_of_inequality2_l677_67708

-- First inequality problem
theorem solution_set_of_inequality1 :
  {x : ℝ | x^2 + 3*x + 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
sorry

-- Second inequality problem
theorem solution_set_of_inequality2 :
  {x : ℝ | -3*x^2 + 2*x + 2 < 0} =
  {x : ℝ | x ∈ Set.Iio ((1 - Real.sqrt 7) / 3) ∪ Set.Ioi ((1 + Real.sqrt 7) / 3)} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality1_solution_set_of_inequality2_l677_67708


namespace NUMINAMATH_GPT_neg_p_implies_neg_q_l677_67759

variable {x : ℝ}

def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x^2

theorem neg_p_implies_neg_q (h : ¬ p x) : ¬ q x :=
sorry

end NUMINAMATH_GPT_neg_p_implies_neg_q_l677_67759


namespace NUMINAMATH_GPT_find_c_of_parabola_l677_67718

theorem find_c_of_parabola (a b c : ℝ) (h_vertex : ∀ x, y = a * (x - 3)^2 - 5)
                           (h_point : ∀ x y, (x = 1) → (y = -3) → y = a * (x - 3)^2 - 5)
                           (h_standard_form : ∀ x, y = a * x^2 + b * x + c) :
  c = -0.5 :=
sorry

end NUMINAMATH_GPT_find_c_of_parabola_l677_67718


namespace NUMINAMATH_GPT_cell_phone_total_cost_l677_67763

def base_cost : ℕ := 25
def text_cost_per_message : ℕ := 3
def extra_minute_cost_per_minute : ℕ := 15
def included_hours : ℕ := 40
def messages_sent_in_february : ℕ := 200
def hours_talked_in_february : ℕ := 41

theorem cell_phone_total_cost :
  base_cost + (messages_sent_in_february * text_cost_per_message) / 100 + 
  ((hours_talked_in_february - included_hours) * 60 * extra_minute_cost_per_minute) / 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_cell_phone_total_cost_l677_67763


namespace NUMINAMATH_GPT_quadrilateral_is_parallelogram_l677_67781

theorem quadrilateral_is_parallelogram
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 - 2 * a * c - 2 * b * d = 0) :
  (a = c) ∧ (b = d) :=
by {
  sorry
}

end NUMINAMATH_GPT_quadrilateral_is_parallelogram_l677_67781


namespace NUMINAMATH_GPT_rectangle_diagonal_length_l677_67741

theorem rectangle_diagonal_length (P : ℝ) (L W D : ℝ) 
  (hP : P = 72) 
  (h_ratio : 3 * W = 2 * L) 
  (h_perimeter : 2 * (L + W) = P) :
  D = Real.sqrt (L * L + W * W) :=
sorry

end NUMINAMATH_GPT_rectangle_diagonal_length_l677_67741


namespace NUMINAMATH_GPT_height_of_triangle_on_parabola_l677_67722

open Real

theorem height_of_triangle_on_parabola
  (x0 x1 : ℝ)
  (y0 y1 : ℝ)
  (hA : y0 = x0^2)
  (hB : y0 = (-x0)^2)
  (hC : y1 = x1^2)
  (hypotenuse_parallel : y0 = y1 + 1):
  y0 - y1 = 1 := 
by
  sorry

end NUMINAMATH_GPT_height_of_triangle_on_parabola_l677_67722


namespace NUMINAMATH_GPT_sum_of_products_lt_zero_l677_67790

theorem sum_of_products_lt_zero (a b c d e f : ℤ) (h : ∃ (i : ℕ), i ≤ 6 ∧ i ≠ 6 ∧ (∀ i ∈ [a, b, c, d, e, f], i < 0 → i ≤ i)) :
  ab + cdef < 0 :=
sorry

end NUMINAMATH_GPT_sum_of_products_lt_zero_l677_67790


namespace NUMINAMATH_GPT_find_b_find_area_l677_67798

open Real

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := A + π / 2
noncomputable def a : ℝ := 3
noncomputable def cos_A : ℝ := sqrt 6 / 3
noncomputable def b : ℝ := 3 * sqrt 2
noncomputable def area : ℝ := 3 * sqrt 2 / 2

theorem find_b (A : ℝ) (H1 : a = 3) (H2 : cos A = sqrt 6 / 3) (H3 : B = A + π / 2) : 
  b = 3 * sqrt 2 := 
  sorry

theorem find_area (A : ℝ) (H1 : a = 3) (H2 : cos A = sqrt 6 / 3) (H3 : B = A + π / 2) : 
  area = 3 * sqrt 2 / 2 := 
  sorry

end NUMINAMATH_GPT_find_b_find_area_l677_67798


namespace NUMINAMATH_GPT_percentage_calculation_l677_67785

def part : ℝ := 12.356
def whole : ℝ := 12356
def expected_percentage : ℝ := 0.1

theorem percentage_calculation (p w : ℝ) (h_p : p = part) (h_w : w = whole) : 
  (p / w) * 100 = expected_percentage :=
sorry

end NUMINAMATH_GPT_percentage_calculation_l677_67785


namespace NUMINAMATH_GPT_traceable_edges_l677_67726

-- Define the vertices of the rectangle
def vertex (x y : ℕ) : ℕ × ℕ := (x, y)

-- Define the edges of the rectangle
def edges : List (ℕ × ℕ) :=
  [vertex 0 0, vertex 0 1,    -- vertical edges
   vertex 1 0, vertex 1 1,
   vertex 2 0, vertex 2 1,
   vertex 0 0, vertex 1 0,    -- horizontal edges
   vertex 1 0, vertex 2 0,
   vertex 0 1, vertex 1 1,
   vertex 1 1, vertex 2 1]

-- Define the theorem to be proved
theorem traceable_edges :
  ∃ (count : ℕ), count = 61 :=
by
  sorry

end NUMINAMATH_GPT_traceable_edges_l677_67726


namespace NUMINAMATH_GPT_can_capacity_l677_67761

-- Definitions of the conditions
variable (M W : ℕ) -- initial amounts of milk and water
variable (M' : ℕ := M + 2) -- new amount of milk after adding 2 liters
variable (ratio_initial : M / W = 1 / 5)
variable (ratio_new : M' / W = 3 / 5)

theorem can_capacity (M W : ℕ) (h_ratio_initial : M / W = 1 / 5) (h_ratio_new : (M + 2) / W = 3 / 5) : (M + W + 2) = 8 := 
by
  sorry

end NUMINAMATH_GPT_can_capacity_l677_67761


namespace NUMINAMATH_GPT_order_of_real_numbers_l677_67729

noncomputable def a : ℝ := Real.arcsin (3 / 4)
noncomputable def b : ℝ := Real.arccos (1 / 5)
noncomputable def c : ℝ := 1 + Real.arctan (2 / 3)

theorem order_of_real_numbers : a < b ∧ b < c :=
by sorry

end NUMINAMATH_GPT_order_of_real_numbers_l677_67729


namespace NUMINAMATH_GPT_time_worked_together_l677_67755

noncomputable def combined_rate (P_rate Q_rate : ℝ) : ℝ :=
  P_rate + Q_rate

theorem time_worked_together (P_rate Q_rate : ℝ) (t additional_time job_completed : ℝ) :
  P_rate = 1 / 4 ∧ Q_rate = 1 / 15 ∧ additional_time = 1 / 5 ∧ job_completed = (additional_time * P_rate) →
  (t * combined_rate P_rate Q_rate + job_completed = 1) → 
  t = 3 :=
sorry

end NUMINAMATH_GPT_time_worked_together_l677_67755


namespace NUMINAMATH_GPT_min_value_S_max_value_m_l677_67716

noncomputable def S (x : ℝ) : ℝ := abs (x - 2) + abs (x - 4)

theorem min_value_S : ∃ x, S x = 2 ∧ ∀ x, S x ≥ 2 := by
  sorry

theorem max_value_m : ∀ x y, S x ≥ m * (-y^2 + 2*y) → 0 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_GPT_min_value_S_max_value_m_l677_67716


namespace NUMINAMATH_GPT_minimum_value_of_f_l677_67756

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 4

theorem minimum_value_of_f : ∃ x : ℝ, f x = -5 ∧ ∀ y : ℝ, f y ≥ -5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l677_67756


namespace NUMINAMATH_GPT_triangle_is_right_triangle_l677_67707

variable (A B C : ℝ) (a b c : ℝ)

-- Conditions definitions
def condition1 : Prop := A + B = C
def condition2 : Prop := a / b = 3 / 4 ∧ b / c = 4 / 5 ∧ a / c = 3 / 5
def condition3 : Prop := A = 90 - B

-- Proof problem
theorem triangle_is_right_triangle (h1 : condition1 A B C) (h2 : condition2 a b c) (h3 : condition3 A B) : C = 90 := 
sorry

end NUMINAMATH_GPT_triangle_is_right_triangle_l677_67707
