import Mathlib

namespace NUMINAMATH_GPT_complement_correct_l230_23061

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}

theorem complement_correct : (U \ A) = {2, 4} := by
  sorry

end NUMINAMATH_GPT_complement_correct_l230_23061


namespace NUMINAMATH_GPT_exists_sequence_of_ten_numbers_l230_23045

theorem exists_sequence_of_ten_numbers :
  ∃ a : Fin 10 → ℝ,
    (∀ i : Fin 6,    a i + a ⟨i.1 + 1, sorry⟩ + a ⟨i.1 + 2, sorry⟩ + a ⟨i.1 + 3, sorry⟩ + a ⟨i.1 + 4, sorry⟩ > 0) ∧
    (∀ j : Fin 4, a j + a ⟨j.1 + 1, sorry⟩ + a ⟨j.1 + 2, sorry⟩ + a ⟨j.1 + 3, sorry⟩ + a ⟨j.1 + 4, sorry⟩ + a ⟨j.1 + 5, sorry⟩ + a ⟨j.1 + 6, sorry⟩ < 0) :=
sorry

end NUMINAMATH_GPT_exists_sequence_of_ten_numbers_l230_23045


namespace NUMINAMATH_GPT_percent_increase_bike_helmet_l230_23050

theorem percent_increase_bike_helmet :
  let old_bike_cost := 160
  let old_helmet_cost := 40
  let bike_increase_rate := 0.05
  let helmet_increase_rate := 0.10
  let new_bike_cost := old_bike_cost * (1 + bike_increase_rate)
  let new_helmet_cost := old_helmet_cost * (1 + helmet_increase_rate)
  let old_total_cost := old_bike_cost + old_helmet_cost
  let new_total_cost := new_bike_cost + new_helmet_cost
  let increase_amount := new_total_cost - old_total_cost
  let percent_increase := (increase_amount / old_total_cost) * 100
  percent_increase = 6 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_bike_helmet_l230_23050


namespace NUMINAMATH_GPT_solve_for_x_l230_23071

theorem solve_for_x (x : ℝ) (h : (x / 5) + 3 = 4) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l230_23071


namespace NUMINAMATH_GPT_angle_A_condition_area_range_condition_l230_23053

/-- Given a triangle ABC with sides opposite to internal angles A, B, and C labeled as a, b, and c respectively. 
Given the condition a * cos C + sqrt 3 * a * sin C = b + c.
Prove that angle A = π / 3.
-/
theorem angle_A_condition
  (a b c : ℝ) (C : ℝ) (h : a * Real.cos C + Real.sqrt 3 * a * Real.sin C = b + c) :
  A = Real.pi / 3 := sorry
  
/-- Given an acute triangle ABC with b = 2 and angle A = π / 3,
find the range of possible values for the area of the triangle ABC.
-/
theorem area_range_condition
  (a c : ℝ) (A : ℝ) (b : ℝ) (C B : ℝ)
  (h1 : b = 2)
  (h2 : A = Real.pi / 3)
  (h3 : 0 < B) (h4 : B < Real.pi / 2)
  (h5 : 0 < C) (h6 : C < Real.pi / 2)
  (h7 : A + C = 2 * Real.pi / 3) :
  Real.sqrt 3 / 2 < (1 / 2) * a * b * Real.sin C ∧
  (1 / 2) * a * b * Real.sin C < 2 * Real.sqrt 3 := sorry

end NUMINAMATH_GPT_angle_A_condition_area_range_condition_l230_23053


namespace NUMINAMATH_GPT_find_common_ratio_l230_23002

noncomputable def geometric_seq_sum (a₁ q : ℂ) (n : ℕ) :=
if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem find_common_ratio (a₁ q : ℂ) :
(geometric_seq_sum a₁ q 8) / (geometric_seq_sum a₁ q 4) = 2 → q = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_common_ratio_l230_23002


namespace NUMINAMATH_GPT_arrangement_possible_l230_23096

noncomputable def exists_a_b : Prop :=
  ∃ a b : ℝ, a + 2*b > 0 ∧ 7*a + 13*b < 0

theorem arrangement_possible : exists_a_b := by
  sorry

end NUMINAMATH_GPT_arrangement_possible_l230_23096


namespace NUMINAMATH_GPT_minimum_value_ineq_l230_23055

variable (m n : ℝ)

noncomputable def minimum_value := (1 / (2 * m)) + (1 / n)

theorem minimum_value_ineq (h1 : m > 0) (h2 : n > 0) (h3 : m + 2 * n = 1) : minimum_value m n = 9 / 2 := 
sorry

end NUMINAMATH_GPT_minimum_value_ineq_l230_23055


namespace NUMINAMATH_GPT_main_l230_23003

def M (x : ℝ) : Prop := x^2 - 5 * x ≤ 0
def N (x : ℝ) (p : ℝ) : Prop := p < x ∧ x < 6
def intersection (x : ℝ) (q : ℝ) : Prop := 2 < x ∧ x ≤ q

theorem main (p q : ℝ) (hM : ∀ x, M x → 0 ≤ x ∧ x ≤ 5) (hN : ∀ x, N x p → p < x ∧ x < 6) (hMN : ∀ x, (M x ∧ N x p) ↔ intersection x q) :
  p + q = 7 :=
by
  sorry

end NUMINAMATH_GPT_main_l230_23003


namespace NUMINAMATH_GPT_number_of_chickens_free_ranging_l230_23026

-- Defining the conditions
def chickens_in_coop : ℕ := 14
def chickens_in_run (coop_chickens : ℕ) : ℕ := 2 * coop_chickens
def chickens_free_ranging (run_chickens : ℕ) : ℕ := 2 * run_chickens - 4

-- Proving the number of chickens free ranging
theorem number_of_chickens_free_ranging : chickens_free_ranging (chickens_in_run chickens_in_coop) = 52 := by
  -- Lean will be able to infer
  sorry  -- proof is not required

end NUMINAMATH_GPT_number_of_chickens_free_ranging_l230_23026


namespace NUMINAMATH_GPT_seq_2016_2017_l230_23044

-- Define the sequence a_n
def seq (n : ℕ) : ℚ := sorry

-- Given conditions
axiom a1_cond : seq 1 = 1/2
axiom a2_cond : seq 2 = 1/3
axiom seq_rec : ∀ n : ℕ, seq n * seq (n + 2) = 1

-- The main goal
theorem seq_2016_2017 : seq 2016 + seq 2017 = 7/2 := sorry

end NUMINAMATH_GPT_seq_2016_2017_l230_23044


namespace NUMINAMATH_GPT_simplify_expression_l230_23040

theorem simplify_expression (a c d x y z : ℝ) :
  (cx * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + dz * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (cx + dz) =
  a^3 * x^3 + c^3 * z^3 + (3 * cx * a^3 * y^3 / (cx + dz)) + (3 * dz * c^3 * x^3 / (cx + dz)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l230_23040


namespace NUMINAMATH_GPT_solve_textbook_by_12th_l230_23004

/-!
# Problem Statement
There are 91 problems in a textbook. Yura started solving them on September 6 and solves one problem less each subsequent morning. By the evening of September 8, there are 46 problems left to solve.

We need to prove that Yura finishes solving all the problems by September 12.
-/

def initial_problems : ℕ := 91
def problems_left_by_evening_of_8th : ℕ := 46

def problems_solved_by_evening_of_8th : ℕ :=
  initial_problems - problems_left_by_evening_of_8th

def z : ℕ := (problems_solved_by_evening_of_8th / 3)

theorem solve_textbook_by_12th 
    (total_problems : ℕ)
    (problems_left : ℕ)
    (solved_by_evening_8th : ℕ)
    (daily_problem_count : ℕ) :
    (total_problems = initial_problems) →
    (problems_left = problems_left_by_evening_of_8th) →
    (solved_by_evening_8th = problems_solved_by_evening_of_8th) →
    (daily_problem_count = z) →
    ∃ (finishing_date : ℕ), finishing_date = 12 :=
  by
    intros _ _ _ _
    sorry

end NUMINAMATH_GPT_solve_textbook_by_12th_l230_23004


namespace NUMINAMATH_GPT_abs_neg_three_l230_23060

theorem abs_neg_three : abs (-3) = 3 :=
sorry

end NUMINAMATH_GPT_abs_neg_three_l230_23060


namespace NUMINAMATH_GPT_line_equation_in_slope_intercept_form_l230_23081

variable {x y : ℝ}

theorem line_equation_in_slope_intercept_form :
  (3 * (x - 2) - 4 * (y - 8) = 0) → (y = (3 / 4) * x + 6.5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_line_equation_in_slope_intercept_form_l230_23081


namespace NUMINAMATH_GPT_sugar_cone_count_l230_23015

theorem sugar_cone_count (ratio_sugar_waffle : ℕ → ℕ → Prop) (sugar_waffle_ratio : ratio_sugar_waffle 5 4) 
(w : ℕ) (h_w : w = 36) : ∃ s : ℕ, ratio_sugar_waffle s w ∧ s = 45 :=
by
  sorry

end NUMINAMATH_GPT_sugar_cone_count_l230_23015


namespace NUMINAMATH_GPT_fg_eq_gf_condition_l230_23018

/-- Definitions of the functions f and g --/
def f (m n c x : ℝ) : ℝ := m * x + n + c
def g (p q c x : ℝ) : ℝ := p * x + q + c

/-- The main theorem stating the equivalence of the condition for f(g(x)) = g(f(x)) --/
theorem fg_eq_gf_condition (m n p q c x : ℝ) :
  f m n c (g p q c x) = g p q c (f m n c x) ↔ n * (1 - p) - q * (1 - m) + c * (m - p) = 0 := by
  sorry

end NUMINAMATH_GPT_fg_eq_gf_condition_l230_23018


namespace NUMINAMATH_GPT_john_twice_james_l230_23063

def john_age : ℕ := 39
def years_ago : ℕ := 3
def years_future : ℕ := 6
def age_difference : ℕ := 4

theorem john_twice_james {J : ℕ} (h : 39 - years_ago = 2 * (J + years_future)) : 
  (J + age_difference = 16) :=
by
  sorry  -- Proof steps here

end NUMINAMATH_GPT_john_twice_james_l230_23063


namespace NUMINAMATH_GPT_sum_x_y_z_l230_23074
open Real

theorem sum_x_y_z (a b : ℝ) (h1 : a / b = 98 / 63) (x y z : ℕ) (h2 : (sqrt a) / (sqrt b) = (x * sqrt y) / z) : x + y + z = 18 := 
by
  sorry

end NUMINAMATH_GPT_sum_x_y_z_l230_23074


namespace NUMINAMATH_GPT_negative_expression_l230_23082

noncomputable def U : ℝ := -2.5
noncomputable def V : ℝ := -0.8
noncomputable def W : ℝ := 0.4
noncomputable def X : ℝ := 1.0
noncomputable def Y : ℝ := 2.2

theorem negative_expression :
  (U - V < 0) ∧ ¬(U * V < 0) ∧ ¬((X / V) * U < 0) ∧ ¬(W / (U * V) < 0) ∧ ¬((X + Y) / W < 0) :=
by
  sorry

end NUMINAMATH_GPT_negative_expression_l230_23082


namespace NUMINAMATH_GPT_cube_volume_increase_l230_23091

theorem cube_volume_increase (s : ℝ) (h : s > 0) :
  let new_volume := (1.4 * s) ^ 3
  let original_volume := s ^ 3
  let increase_percentage := ((new_volume - original_volume) / original_volume) * 100
  increase_percentage = 174.4 := by
  sorry

end NUMINAMATH_GPT_cube_volume_increase_l230_23091


namespace NUMINAMATH_GPT_tangent_line_circle_p_l230_23029

theorem tangent_line_circle_p (p : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 6 * x + 8 = 0 → (x = -p/2 ∨ y = 0)) → 
  (p = 4 ∨ p = 8) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_circle_p_l230_23029


namespace NUMINAMATH_GPT_tiger_distance_proof_l230_23000

-- Declare the problem conditions
def tiger_initial_speed : ℝ := 25
def tiger_initial_time : ℝ := 3
def tiger_slow_speed : ℝ := 10
def tiger_slow_time : ℝ := 4
def tiger_chase_speed : ℝ := 50
def tiger_chase_time : ℝ := 0.5

-- Compute individual distances
def distance1 := tiger_initial_speed * tiger_initial_time
def distance2 := tiger_slow_speed * tiger_slow_time
def distance3 := tiger_chase_speed * tiger_chase_time

-- Compute the total distance
def total_distance := distance1 + distance2 + distance3

-- The final theorem to prove
theorem tiger_distance_proof : total_distance = 140 := by
  sorry

end NUMINAMATH_GPT_tiger_distance_proof_l230_23000


namespace NUMINAMATH_GPT_sin_value_l230_23076

theorem sin_value (x : ℝ) (h : Real.sin (x + π / 3) = Real.sqrt 3 / 3) :
  Real.sin (2 * π / 3 - x) = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_value_l230_23076


namespace NUMINAMATH_GPT_total_toothpicks_correct_l230_23049

def number_of_horizontal_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
(height + 1) * width

def number_of_vertical_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
(height) * (width + 1)

def total_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
number_of_horizontal_toothpicks height width + number_of_vertical_toothpicks height width

theorem total_toothpicks_correct:
  total_toothpicks 30 15 = 945 :=
by
  sorry

end NUMINAMATH_GPT_total_toothpicks_correct_l230_23049


namespace NUMINAMATH_GPT_smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6_l230_23098

theorem smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6 :
  ∃ n : ℤ, n = 3323 ∧ n > (Real.sqrt 5 + Real.sqrt 3)^6 ∧ ∀ m : ℤ, m > (Real.sqrt 5 + Real.sqrt 3)^6 → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6_l230_23098


namespace NUMINAMATH_GPT_fixed_monthly_fee_l230_23009

theorem fixed_monthly_fee (x y : ℝ)
  (h1 : x + y = 15.80)
  (h2 : x + 3 * y = 28.62) :
  x = 9.39 :=
sorry

end NUMINAMATH_GPT_fixed_monthly_fee_l230_23009


namespace NUMINAMATH_GPT_terminal_zeros_of_product_l230_23046

noncomputable def prime_factors (n : ℕ) : List (ℕ × ℕ) := sorry

theorem terminal_zeros_of_product (n m : ℕ) (hn : prime_factors n = [(2, 1), (5, 2)])
 (hm : prime_factors m = [(2, 3), (3, 2), (5, 1)]) : 
  (∃ k, n * m = 10^k) ∧ k = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_terminal_zeros_of_product_l230_23046


namespace NUMINAMATH_GPT_otimes_calc_1_otimes_calc_2_otimes_calc_3_l230_23013

def otimes (a b : Int) : Int :=
  a^2 - Int.natAbs b

theorem otimes_calc_1 : otimes (-2) 3 = 1 :=
by
  sorry

theorem otimes_calc_2 : otimes 5 (-4) = 21 :=
by
  sorry

theorem otimes_calc_3 : otimes (-3) (-1) = 8 :=
by
  sorry

end NUMINAMATH_GPT_otimes_calc_1_otimes_calc_2_otimes_calc_3_l230_23013


namespace NUMINAMATH_GPT_neil_baked_cookies_l230_23039

theorem neil_baked_cookies (total_cookies : ℕ) (given_to_friend : ℕ) (cookies_left : ℕ)
    (h1 : given_to_friend = (2 / 5) * total_cookies)
    (h2 : cookies_left = (3 / 5) * total_cookies)
    (h3 : cookies_left = 12) : total_cookies = 20 :=
by
  sorry

end NUMINAMATH_GPT_neil_baked_cookies_l230_23039


namespace NUMINAMATH_GPT_stickers_after_birthday_l230_23069

-- Definitions based on conditions
def initial_stickers : Nat := 39
def birthday_stickers : Nat := 22

-- Theorem stating the problem we aim to prove
theorem stickers_after_birthday : initial_stickers + birthday_stickers = 61 :=
by 
  sorry

end NUMINAMATH_GPT_stickers_after_birthday_l230_23069


namespace NUMINAMATH_GPT_tangent_line_problem_l230_23054

theorem tangent_line_problem 
  (x1 x2 : ℝ)
  (h1 : (1 / x1) = Real.exp x2)
  (h2 : Real.log x1 - 1 = Real.exp x2 * (1 - x2)) :
  (2 / (x1 - 1) + x2 = -1) :=
by 
  sorry

end NUMINAMATH_GPT_tangent_line_problem_l230_23054


namespace NUMINAMATH_GPT_number_of_real_solutions_l230_23024

noncomputable def greatest_integer (x: ℝ) : ℤ :=
  ⌊x⌋

def equation (x: ℝ) :=
  4 * x^2 - 40 * (greatest_integer x : ℝ) + 51 = 0

theorem number_of_real_solutions : 
  ∃ (x1 x2 x3 x4: ℝ), 
  equation x1 ∧ equation x2 ∧ equation x3 ∧ equation x4 ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 := 
sorry

end NUMINAMATH_GPT_number_of_real_solutions_l230_23024


namespace NUMINAMATH_GPT_compute_y_geometric_series_l230_23084

theorem compute_y_geometric_series :
  let S1 := (∑' n : ℕ, (1 / 3)^n)
  let S2 := (∑' n : ℕ, (-1)^n * (1 / 3)^n)
  (S1 * S2 = ∑' n : ℕ, (1 / 9)^n) → 
  S1 = 3 / 2 →
  S2 = 3 / 4 →
  (∑' n : ℕ, (1 / y)^n) = 9 / 8 →
  y = 9 := 
by
  intros S1 S2 h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_compute_y_geometric_series_l230_23084


namespace NUMINAMATH_GPT_expand_polynomial_l230_23085

theorem expand_polynomial (x : ℝ) : 
  3 * (x - 2) * (x^2 + x + 1) = 3 * x^3 - 3 * x^2 - 3 * x - 6 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l230_23085


namespace NUMINAMATH_GPT_parabola_sequence_l230_23012

theorem parabola_sequence (m: ℝ) (n: ℕ):
  (∀ t s: ℝ, t * s = -1/4) →
  (∀ x y: ℝ, y^2 = (1/(3^n)) * m * (x - (m / 4) * (1 - (1/(3^n))))) :=
sorry

end NUMINAMATH_GPT_parabola_sequence_l230_23012


namespace NUMINAMATH_GPT_rate_percent_simple_interest_l230_23058

theorem rate_percent_simple_interest:
  ∀ (P SI T R : ℝ), SI = 400 → P = 1000 → T = 4 → (SI = P * R * T / 100) → R = 10 :=
by
  intros P SI T R h_si h_p h_t h_formula
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_rate_percent_simple_interest_l230_23058


namespace NUMINAMATH_GPT_grid_sum_21_proof_l230_23077

-- Define the condition that the sum of the horizontal and vertical lines are 21
def valid_grid (nums : List ℕ) (x : ℕ) : Prop :=
  nums ≠ [] ∧ (((nums.sum + x) = 42) ∧ (21 + 21 = 42))

-- Define the main theorem to prove x = 7
theorem grid_sum_21_proof (nums : List ℕ) (h : valid_grid nums 7) : 7 ∈ nums :=
  sorry

end NUMINAMATH_GPT_grid_sum_21_proof_l230_23077


namespace NUMINAMATH_GPT_greatest_prime_factor_15_factorial_plus_17_factorial_l230_23094

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end NUMINAMATH_GPT_greatest_prime_factor_15_factorial_plus_17_factorial_l230_23094


namespace NUMINAMATH_GPT_union_A_B_l230_23030

def setA : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def setB : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_A_B : setA ∪ setB = {x | -1 < x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_l230_23030


namespace NUMINAMATH_GPT_problem_statement_l230_23035

variable (x : ℝ)
def A := ({-3, x^2, x + 1} : Set ℝ)
def B := ({x - 3, 2 * x - 1, x^2 + 1} : Set ℝ)

theorem problem_statement (hx : A x ∩ B x = {-3}) : 
  x = -1 ∧ A x ∪ B x = ({-4, -3, 0, 1, 2} : Set ℝ) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l230_23035


namespace NUMINAMATH_GPT_quiz_score_of_dropped_student_l230_23017

theorem quiz_score_of_dropped_student 
    (avg_all : ℝ) (num_all : ℕ) (new_avg_remaining : ℝ) (num_remaining : ℕ)
    (total_all : ℝ := num_all * avg_all) (total_remaining : ℝ := num_remaining * new_avg_remaining) :
    avg_all = 61.5 → num_all = 16 → new_avg_remaining = 64 → num_remaining = 15 → (total_all - total_remaining = 24) :=
by
  intros h_avg_all h_num_all h_new_avg_remaining h_num_remaining
  rw [h_avg_all, h_new_avg_remaining, h_num_all, h_num_remaining]
  sorry

end NUMINAMATH_GPT_quiz_score_of_dropped_student_l230_23017


namespace NUMINAMATH_GPT_grace_pennies_l230_23001

theorem grace_pennies (dime_value nickel_value : ℕ) (dimes nickels : ℕ) 
  (h₁ : dime_value = 10) (h₂ : nickel_value = 5) (h₃ : dimes = 10) (h₄ : nickels = 10) : 
  dimes * dime_value + nickels * nickel_value = 150 := 
by 
  sorry

end NUMINAMATH_GPT_grace_pennies_l230_23001


namespace NUMINAMATH_GPT_grid_values_equal_l230_23016

theorem grid_values_equal (f : ℤ × ℤ → ℕ) (h : ∀ (i j : ℤ), 
  f (i, j) = 1 / 4 * (f (i + 1, j) + f (i - 1, j) + f (i, j + 1) + f (i, j - 1))) :
  ∀ (i j i' j' : ℤ), f (i, j) = f (i', j') :=
by
  sorry

end NUMINAMATH_GPT_grid_values_equal_l230_23016


namespace NUMINAMATH_GPT_chocolates_difference_l230_23041

/-!
We are given that:
- Robert ate 10 chocolates
- Nickel ate 5 chocolates

We need to prove that Robert ate 5 more chocolates than Nickel.
-/

def robert_chocolates := 10
def nickel_chocolates := 5

theorem chocolates_difference : robert_chocolates - nickel_chocolates = 5 :=
by
  -- Proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_chocolates_difference_l230_23041


namespace NUMINAMATH_GPT_susan_gave_sean_8_apples_l230_23051

theorem susan_gave_sean_8_apples (initial_apples total_apples apples_given : ℕ) 
  (h1 : initial_apples = 9)
  (h2 : total_apples = 17)
  (h3 : apples_given = total_apples - initial_apples) : 
  apples_given = 8 :=
by
  sorry

end NUMINAMATH_GPT_susan_gave_sean_8_apples_l230_23051


namespace NUMINAMATH_GPT_candy_left_l230_23065

-- Define the given conditions
def KatieCandy : ℕ := 8
def SisterCandy : ℕ := 23
def AteCandy : ℕ := 8

-- The theorem stating the total number of candy left
theorem candy_left (k : ℕ) (s : ℕ) (e : ℕ) (hk : k = KatieCandy) (hs : s = SisterCandy) (he : e = AteCandy) : 
  (k + s) - e = 23 :=
by
  -- (Proof will be inserted here, but we include a placeholder "sorry" for now)
  sorry

end NUMINAMATH_GPT_candy_left_l230_23065


namespace NUMINAMATH_GPT_preceding_integer_binary_l230_23006

theorem preceding_integer_binary (M : ℕ) (h : M = 0b110101) : 
  (M - 1) = 0b110100 :=
by
  sorry

end NUMINAMATH_GPT_preceding_integer_binary_l230_23006


namespace NUMINAMATH_GPT_total_balls_estimation_l230_23008

theorem total_balls_estimation
  (n : ℕ)  -- Let n be the total number of balls in the bag
  (yellow_balls : ℕ)  -- Let yellow_balls be the number of yellow balls
  (frequency : ℝ)  -- Let frequency be the stabilized frequency of drawing a yellow ball
  (h1 : yellow_balls = 6)
  (h2 : frequency = 0.3)
  (h3 : (yellow_balls : ℝ) / (n : ℝ) = frequency) :
  n = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_balls_estimation_l230_23008


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l230_23066

theorem sum_of_reciprocals_of_roots {r1 r2 : ℚ} (h1 : r1 + r2 = 15) (h2 : r1 * r2 = 6) :
  (1 / r1 + 1 / r2) = 5 / 2 := 
by sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l230_23066


namespace NUMINAMATH_GPT_geometric_seq_not_sufficient_necessary_l230_23025

theorem geometric_seq_not_sufficient_necessary (a_n : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a_n (n+1) = a_n n * q) : 
  ¬ ((∃ q > 1, ∀ n, a_n (n+1) > a_n n) ∧ (∀ q > 1, ∀ n, a_n (n+1) > a_n n)) := 
sorry

end NUMINAMATH_GPT_geometric_seq_not_sufficient_necessary_l230_23025


namespace NUMINAMATH_GPT_ratio_of_areas_l230_23038

theorem ratio_of_areas (b : ℝ) (h1 : 0 < b) (h2 : b < 4) 
  (h3 : (9 : ℝ) / 25 = (4 - b) / b * (4 : ℝ)) : b = 2.5 := 
sorry

end NUMINAMATH_GPT_ratio_of_areas_l230_23038


namespace NUMINAMATH_GPT_dot_product_focus_hyperbola_l230_23023

-- Definitions related to the problem of the hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - y^2 = 1

def is_focus (c : ℝ) (x y : ℝ) : Prop := (x = c ∧ y = 0) ∨ (x = -c ∧ y = 0)

-- Problem conditions
def point_on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

def triangle_area (a b c : ℝ × ℝ) (area : ℝ) : Prop :=
  0.5 * (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)) = area

def foci_of_hyperbola : (ℝ × ℝ) × (ℝ × ℝ) := ((2, 0), (-2, 0))

-- Main statement to prove
theorem dot_product_focus_hyperbola
  (m n : ℝ)
  (hP : point_on_hyperbola (m, n))
  (hArea : triangle_area (2, 0) (m, n) (-2, 0) 2) :
  ((-2 - m) * (2 - m) + (-n) * (-n)) = 3 :=
sorry

end NUMINAMATH_GPT_dot_product_focus_hyperbola_l230_23023


namespace NUMINAMATH_GPT_determine_peter_and_liar_l230_23052

structure Brothers where
  names : Fin 2 → String
  tells_truth : Fin 2 → Bool -- true if the brother tells the truth, false if lies
  (unique_truth_teller : ∃! (i : Fin 2), tells_truth i)
  (one_is_peter : ∃ (i : Fin 2), names i = "Péter")

theorem determine_peter_and_liar (B : Brothers) : 
  ∃ (peter liar : Fin 2), B.names peter = "Péter" ∧ B.tells_truth liar = false ∧
    ∀ (p q : Fin 2), B.names p = "Péter" → B.tells_truth q = false → p = peter ∧ q = liar :=
by
  sorry

end NUMINAMATH_GPT_determine_peter_and_liar_l230_23052


namespace NUMINAMATH_GPT_cards_sum_l230_23034

theorem cards_sum (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
by
  sorry

end NUMINAMATH_GPT_cards_sum_l230_23034


namespace NUMINAMATH_GPT_min_value_of_expression_l230_23093

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 1) :
  (1 / (2 * a) + 2 / b) = 8 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l230_23093


namespace NUMINAMATH_GPT_track_width_eight_l230_23011

theorem track_width_eight (r1 r2 : ℝ) (h : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 16 * Real.pi) : r1 - r2 = 8 := 
sorry

end NUMINAMATH_GPT_track_width_eight_l230_23011


namespace NUMINAMATH_GPT_recipe_sugar_amount_l230_23083

theorem recipe_sugar_amount (F_total F_added F_additional F_needed S : ℕ)
  (h1 : F_total = 9)
  (h2 : F_added = 2)
  (h3 : F_additional = S + 1)
  (h4 : F_needed = F_total - F_added)
  (h5 : F_needed = F_additional) :
  S = 6 := 
sorry

end NUMINAMATH_GPT_recipe_sugar_amount_l230_23083


namespace NUMINAMATH_GPT_geometric_sum_first_six_terms_l230_23005

theorem geometric_sum_first_six_terms : 
  let a := (1 : ℚ) / 2
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 4095 / 6144 :=
by
  -- Definitions and properties of geometric series
  sorry

end NUMINAMATH_GPT_geometric_sum_first_six_terms_l230_23005


namespace NUMINAMATH_GPT_total_money_calculation_l230_23027

theorem total_money_calculation (N50 N500 Total_money : ℕ) 
( h₁ : N50 = 37 ) 
( h₂ : N50 + N500 = 54 ) :
Total_money = N50 * 50 + N500 * 500 ↔ Total_money = 10350 := 
by 
  sorry

end NUMINAMATH_GPT_total_money_calculation_l230_23027


namespace NUMINAMATH_GPT_second_half_takes_200_percent_longer_l230_23078

noncomputable def time_take (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

variable (total_distance : ℕ := 640)
variable (first_half_speed : ℕ := 80)
variable (average_speed : ℕ := 40)

theorem second_half_takes_200_percent_longer :
  let first_half_distance := total_distance / 2;
  let first_half_time := time_take first_half_distance first_half_speed;
  let total_time := time_take total_distance average_speed;
  let second_half_time := total_time - first_half_time;
  let time_increase := second_half_time - first_half_time;
  let percentage_increase := (time_increase * 100) / first_half_time;
  percentage_increase = 200 :=
by
  sorry

end NUMINAMATH_GPT_second_half_takes_200_percent_longer_l230_23078


namespace NUMINAMATH_GPT_xiao_ming_completion_days_l230_23072

/-
  Conditions:
  1. The total number of pages is 960.
  2. The planned number of days to finish the book is 20.
  3. Xiao Ming actually read 12 more pages per day than planned.

  Question:
  How many days did it actually take Xiao Ming to finish the book?

  Answer:
  The actual number of days to finish the book is 16 days.
-/

open Nat

theorem xiao_ming_completion_days :
  let total_pages := 960
  let planned_days := 20
  let additional_pages_per_day := 12
  let planned_pages_per_day := total_pages / planned_days
  let actual_pages_per_day := planned_pages_per_day + additional_pages_per_day
  let actual_days := total_pages / actual_pages_per_day
  actual_days = 16 :=
by
  let total_pages := 960
  let planned_days := 20
  let additional_pages_per_day := 12
  let planned_pages_per_day := total_pages / planned_days
  let actual_pages_per_day := planned_pages_per_day + additional_pages_per_day
  let actual_days := total_pages / actual_pages_per_day
  show actual_days = 16
  sorry

end NUMINAMATH_GPT_xiao_ming_completion_days_l230_23072


namespace NUMINAMATH_GPT_plane_equation_of_points_l230_23056

theorem plane_equation_of_points :
  ∃ A B C D : ℤ, A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 ∧
  ∀ x y z : ℤ, (15 * x + 7 * y + 17 * z - 26 = 0) ↔
  (A * x + B * y + C * z + D = 0) :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_of_points_l230_23056


namespace NUMINAMATH_GPT_avg_age_across_rooms_l230_23007

namespace AverageAgeProof

def Room := Type

-- Conditions
def people_in_room_a : ℕ := 8
def avg_age_room_a : ℕ := 35

def people_in_room_b : ℕ := 5
def avg_age_room_b : ℕ := 30

def people_in_room_c : ℕ := 7
def avg_age_room_c : ℕ := 25

-- Combined Calculations
def total_people := people_in_room_a + people_in_room_b + people_in_room_c
def total_age := (people_in_room_a * avg_age_room_a) + (people_in_room_b * avg_age_room_b) + (people_in_room_c * avg_age_room_c)

noncomputable def average_age : ℚ := total_age / total_people

-- Proof that the average age of all the people across the three rooms is 30.25
theorem avg_age_across_rooms : average_age = 30.25 := 
sorry

end AverageAgeProof

end NUMINAMATH_GPT_avg_age_across_rooms_l230_23007


namespace NUMINAMATH_GPT_initial_nickels_proof_l230_23095

def initial_nickels (N : ℕ) (D : ℕ) (total_value : ℝ) : Prop :=
  D = 3 * N ∧
  total_value = (N + 2 * N) * 0.05 + 3 * N * 0.10 ∧
  total_value = 9

theorem initial_nickels_proof : ∃ N, ∃ D, (initial_nickels N D 9) → (N = 20) :=
by
  sorry

end NUMINAMATH_GPT_initial_nickels_proof_l230_23095


namespace NUMINAMATH_GPT_find_number_of_flowers_l230_23028
open Nat

theorem find_number_of_flowers (F : ℕ) (h_candles : choose 4 2 = 6) (h_groupings : 6 * choose F 8 = 54) : F = 9 :=
sorry

end NUMINAMATH_GPT_find_number_of_flowers_l230_23028


namespace NUMINAMATH_GPT_positive_difference_sum_even_odd_l230_23020

theorem positive_difference_sum_even_odd :
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  sum_30_even - sum_25_odd = 305 :=
by
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  show sum_30_even - sum_25_odd = 305
  sorry

end NUMINAMATH_GPT_positive_difference_sum_even_odd_l230_23020


namespace NUMINAMATH_GPT_sugar_amount_l230_23010

variables (S F B : ℝ)

-- Conditions
def condition1 : Prop := S / F = 5 / 2
def condition2 : Prop := F / B = 10 / 1
def condition3 : Prop := F / (B + 60) = 8 / 1

-- Theorem to prove
theorem sugar_amount (h1 : condition1 S F) (h2 : condition2 F B) (h3 : condition3 F B) : S = 6000 :=
sorry

end NUMINAMATH_GPT_sugar_amount_l230_23010


namespace NUMINAMATH_GPT_fraction_of_earth_surface_inhabitable_l230_23022

theorem fraction_of_earth_surface_inhabitable (f_land : ℚ) (f_inhabitable_land : ℚ)
  (h1 : f_land = 1 / 3)
  (h2 : f_inhabitable_land = 2 / 3) :
  f_land * f_inhabitable_land = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_earth_surface_inhabitable_l230_23022


namespace NUMINAMATH_GPT_find_fourth_number_in_proportion_l230_23070

-- Define the given conditions
def x : ℝ := 0.39999999999999997
def proportion (y : ℝ) := 0.60 / x = 6 / y

-- State the theorem to be proven
theorem find_fourth_number_in_proportion :
  proportion y → y = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_fourth_number_in_proportion_l230_23070


namespace NUMINAMATH_GPT_cone_cylinder_volume_ratio_l230_23048

theorem cone_cylinder_volume_ratio (h_cyl r_cyl: ℝ) (h_cone: ℝ) :
  h_cyl = 10 → r_cyl = 5 → h_cone = 5 →
  (1/3 * (Real.pi * r_cyl^2 * h_cone)) / (Real.pi * r_cyl^2 * h_cyl) = 1/6 :=
by
  intros h_cyl_eq r_cyl_eq h_cone_eq
  rw [h_cyl_eq, r_cyl_eq, h_cone_eq]
  sorry

end NUMINAMATH_GPT_cone_cylinder_volume_ratio_l230_23048


namespace NUMINAMATH_GPT_ratio_traditionalists_progressives_l230_23086

variables (T P C : ℝ)

-- Conditions from the problem
-- There are 6 provinces and each province has the same number of traditionalists
-- The fraction of the country that is traditionalist is 0.6
def country_conditions (T P C : ℝ) :=
  (6 * T = 0.6 * C) ∧
  (C = P + 6 * T)

-- Theorem that needs to be proven
theorem ratio_traditionalists_progressives (T P C : ℝ) (h : country_conditions T P C) :
  T / P = 1 / 4 :=
by
  -- Setup conditions from the hypothesis h
  rcases h with ⟨h1, h2⟩
  -- Start the proof (Proof content is not required as per instructions)
  sorry

end NUMINAMATH_GPT_ratio_traditionalists_progressives_l230_23086


namespace NUMINAMATH_GPT_jake_split_shots_l230_23042

theorem jake_split_shots (shot_volume : ℝ) (purity : ℝ) (alcohol_consumed : ℝ) 
    (h1 : shot_volume = 1.5) (h2 : purity = 0.50) (h3 : alcohol_consumed = 3) : 
    2 * (alcohol_consumed / (purity * shot_volume)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_jake_split_shots_l230_23042


namespace NUMINAMATH_GPT_no_line_normal_to_both_curves_l230_23090

theorem no_line_normal_to_both_curves :
  ¬ ∃ a b : ℝ, ∃ (l : ℝ → ℝ),
    -- normal to y = cosh x at x = a
    (∀ x : ℝ, l x = -1 / (Real.sinh a) * (x - a) + Real.cosh a) ∧
    -- normal to y = sinh x at x = b
    (∀ x : ℝ, l x = -1 / (Real.cosh b) * (x - b) + Real.sinh b) := 
  sorry

end NUMINAMATH_GPT_no_line_normal_to_both_curves_l230_23090


namespace NUMINAMATH_GPT_solve_inequality_l230_23014

theorem solve_inequality (x : ℝ) : 2 * x + 6 > 5 * x - 3 → x < 3 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_solve_inequality_l230_23014


namespace NUMINAMATH_GPT_deepak_speed_proof_l230_23032

noncomputable def deepak_speed (circumference : ℝ) (meeting_time : ℝ) (wife_speed_kmh : ℝ) : ℝ :=
  let wife_speed_mpm := wife_speed_kmh * 1000 / 60
  let wife_distance := wife_speed_mpm * meeting_time
  let deepak_speed_mpm := ((circumference - wife_distance) / meeting_time)
  deepak_speed_mpm * 60 / 1000

theorem deepak_speed_proof :
  deepak_speed 726 5.28 3.75 = 4.5054 :=
by
  -- The functions and definitions used here come from the problem statement
  -- Conditions:
  -- circumference = 726
  -- meeting_time = 5.28 minutes
  -- wife_speed_kmh = 3.75 km/hr
  sorry

end NUMINAMATH_GPT_deepak_speed_proof_l230_23032


namespace NUMINAMATH_GPT_Eva_is_6_l230_23033

def ages : Set ℕ := {2, 4, 6, 8, 10}

def conditions : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a + b = 12 ∧
  b ≠ 2 ∧ b ≠ 10 ∧ a ≠ 2 ∧ a ≠ 10 ∧
  (∃ c d, c ∈ ages ∧ d ∈ ages ∧ c = 2 ∧ d = 10 ∧
           (∃ e, e ∈ ages ∧ e = 4 ∧
           ∃ eva, eva ∈ ages ∧ eva ≠ 2 ∧ eva ≠ 4 ∧ eva ≠ 8 ∧ eva ≠ 10 ∧ eva = 6))

theorem Eva_is_6 (h : conditions) : ∃ eva, eva ∈ ages ∧ eva = 6 := sorry

end NUMINAMATH_GPT_Eva_is_6_l230_23033


namespace NUMINAMATH_GPT_scientific_notation_11580000_l230_23079

theorem scientific_notation_11580000 :
  11580000 = 1.158 * 10^7 :=
sorry

end NUMINAMATH_GPT_scientific_notation_11580000_l230_23079


namespace NUMINAMATH_GPT_statement_c_false_l230_23059

theorem statement_c_false : ¬ ∃ (x y : ℝ), x^2 + y^2 < 0 := by
  sorry

end NUMINAMATH_GPT_statement_c_false_l230_23059


namespace NUMINAMATH_GPT_correct_inequality_l230_23099

variable (a b c d : ℝ)
variable (h₁ : a > b)
variable (h₂ : b > 0)
variable (h₃ : 0 > c)
variable (h₄ : c > d)

theorem correct_inequality :
  (c / a) - (d / b) > 0 :=
by sorry

end NUMINAMATH_GPT_correct_inequality_l230_23099


namespace NUMINAMATH_GPT_distance_AB_l230_23057

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_distance_AB_l230_23057


namespace NUMINAMATH_GPT_solve_inequality_l230_23031

theorem solve_inequality (x : ℝ) :
  (abs ((6 - x) / 4) < 3) ∧ (2 ≤ x) ↔ (2 ≤ x) ∧ (x < 18) := 
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l230_23031


namespace NUMINAMATH_GPT_choose_7_starters_with_at_least_one_quadruplet_l230_23062

-- Given conditions
variable (n : ℕ := 18) -- total players
variable (k : ℕ := 7)  -- number of starters
variable (q : ℕ := 4)  -- number of quadruplets

-- Lean statement
theorem choose_7_starters_with_at_least_one_quadruplet 
  (h : n = 18) 
  (h1 : k = 7) 
  (h2 : q = 4) :
  (Nat.choose 18 7 - Nat.choose 14 7) = 28392 :=
by
  sorry

end NUMINAMATH_GPT_choose_7_starters_with_at_least_one_quadruplet_l230_23062


namespace NUMINAMATH_GPT_weights_in_pile_l230_23073

theorem weights_in_pile (a b c : ℕ) (h1 : a + b + c = 100) (h2 : a + 10 * b + 50 * c = 500) : 
  a = 60 ∧ b = 39 ∧ c = 1 :=
sorry

end NUMINAMATH_GPT_weights_in_pile_l230_23073


namespace NUMINAMATH_GPT_find_valid_pairs_l230_23089

theorem find_valid_pairs :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 12 →
  (∃ C : ℤ, ∀ (n : ℕ), 0 < n → (a^n + b^(n+9)) % 13 = C % 13) ↔
  (a, b) = (1, 1) ∨ (a, b) = (4, 4) ∨ (a, b) = (10, 10) ∨ (a, b) = (12, 12) := 
by
  sorry

end NUMINAMATH_GPT_find_valid_pairs_l230_23089


namespace NUMINAMATH_GPT_largest_constant_D_l230_23088

theorem largest_constant_D (D : ℝ) 
  (h : ∀ (x y : ℝ), x^2 + y^2 + 4 ≥ D * (x + y)) : 
  D ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_largest_constant_D_l230_23088


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l230_23047

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1/2) : x^2 * (x - 1) - x * (x^2 + x - 1) = 0 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l230_23047


namespace NUMINAMATH_GPT_father_age_when_sum_100_l230_23021

/-- Given the current ages of the mother and father, prove that the father's age will be 51 years old when the sum of their ages is 100. -/
theorem father_age_when_sum_100 (M F : ℕ) (hM : M = 42) (hF : F = 44) :
  ∃ X : ℕ, (M + X) + (F + X) = 100 ∧ F + X = 51 :=
by
  sorry

end NUMINAMATH_GPT_father_age_when_sum_100_l230_23021


namespace NUMINAMATH_GPT_salmon_trip_l230_23064

theorem salmon_trip (male_female_sum : 712261 + 259378 = 971639) : 
  712261 + 259378 = 971639 := 
by 
  exact male_female_sum

end NUMINAMATH_GPT_salmon_trip_l230_23064


namespace NUMINAMATH_GPT_flat_path_time_l230_23087

/-- Malcolm's walking time problem -/
theorem flat_path_time (x : ℕ) (h1 : 6 + 12 + 6 = 24)
                       (h2 : 3 * x = 24 + 18) : x = 14 := 
by
  sorry

end NUMINAMATH_GPT_flat_path_time_l230_23087


namespace NUMINAMATH_GPT_coefficient_fifth_term_expansion_l230_23097

theorem coefficient_fifth_term_expansion :
  let a := (2 : ℝ)
  let b := -(1 : ℝ)
  let n := 6
  let k := 4
  Nat.choose n k * (a ^ (n - k)) * (b ^ k) = 60 := by
  -- We can assume x to be any nonzero real, but it is not needed in the theorem itself.
  sorry

end NUMINAMATH_GPT_coefficient_fifth_term_expansion_l230_23097


namespace NUMINAMATH_GPT_cats_weigh_more_by_5_kg_l230_23068

def puppies_weight (num_puppies : ℕ) (weight_per_puppy : ℝ) : ℝ :=
  num_puppies * weight_per_puppy

def cats_weight (num_cats : ℕ) (weight_per_cat : ℝ) : ℝ :=
  num_cats * weight_per_cat

theorem cats_weigh_more_by_5_kg :
  puppies_weight 4 7.5  = 30 ∧ cats_weight 14 2.5 = 35 → (cats_weight 14 2.5 - puppies_weight 4 7.5 = 5) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_cats_weigh_more_by_5_kg_l230_23068


namespace NUMINAMATH_GPT_combined_area_of_four_removed_triangles_l230_23075

noncomputable def combined_area_of_removed_triangles (s x y: ℝ) : Prop :=
  x + y = s ∧ s - 2 * x = 15 ∧ s - 2 * y = 9 ∧
  4 * (1 / 2 * x * y) = 67.5

-- Statement of the problem
theorem combined_area_of_four_removed_triangles (s x y: ℝ) :
  combined_area_of_removed_triangles s x y :=
  by
    sorry

end NUMINAMATH_GPT_combined_area_of_four_removed_triangles_l230_23075


namespace NUMINAMATH_GPT_total_valid_votes_l230_23043

theorem total_valid_votes (V : ℝ) (h1 : 0.70 * V - 0.30 * V = 176) : V = 440 :=
by sorry

end NUMINAMATH_GPT_total_valid_votes_l230_23043


namespace NUMINAMATH_GPT_how_many_children_l230_23037

-- Definitions based on conditions
def total_spectators : ℕ := 10000
def men : ℕ := 7000
def others : ℕ := total_spectators - men -- women + children
def children_per_woman : ℕ := 5

-- Variables
variable (W C : ℕ)

-- Conditions as Lean equalities
def condition1 : W + C = others := by sorry
def condition2 : C = children_per_woman * W := by sorry

-- Theorem statement to prove the number of children
theorem how_many_children (h1 : W + C = others) (h2 : C = children_per_woman * W) : C = 2500 :=
by sorry

end NUMINAMATH_GPT_how_many_children_l230_23037


namespace NUMINAMATH_GPT_evaluate_expression_l230_23080

def x : ℚ := 1 / 4
def y : ℚ := 1 / 3
def z : ℚ := 12

theorem evaluate_expression : x^3 * y^4 * z = 1 / 432 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l230_23080


namespace NUMINAMATH_GPT_lcm_factor_l230_23067

-- Define the variables and conditions
variables (A B H L x : ℕ)
variable (hcf_23 : Nat.gcd A B = 23)
variable (larger_number_391 : A = 391)
variable (lcm_hcf_mult_factors : L = Nat.lcm A B)
variable (lcm_factors : L = 23 * x * 17)

-- The proof statement
theorem lcm_factor (hcf_23 : Nat.gcd A B = 23) (larger_number_391 : A = 391) (lcm_hcf_mult_factors : L = Nat.lcm A B) (lcm_factors : L = 23 * x * 17) :
  x = 17 :=
sorry

end NUMINAMATH_GPT_lcm_factor_l230_23067


namespace NUMINAMATH_GPT_retailer_profit_percentage_l230_23019

theorem retailer_profit_percentage
  (cost_price : ℝ)
  (marked_percent : ℝ)
  (discount_percent : ℝ)
  (selling_price : ℝ)
  (marked_price : ℝ)
  (profit_percent : ℝ) :
  marked_percent = 60 →
  discount_percent = 25 →
  marked_price = cost_price * (1 + marked_percent / 100) →
  selling_price = marked_price * (1 - discount_percent / 100) →
  profit_percent = ((selling_price - cost_price) / cost_price) * 100 →
  profit_percent = 20 :=
by
  sorry

end NUMINAMATH_GPT_retailer_profit_percentage_l230_23019


namespace NUMINAMATH_GPT_first_day_exceeding_100_paperclips_l230_23036

def paperclips_day (k : ℕ) : ℕ := 3 * 2^k

theorem first_day_exceeding_100_paperclips :
  ∃ (k : ℕ), paperclips_day k > 100 ∧ k = 6 := by
  sorry

end NUMINAMATH_GPT_first_day_exceeding_100_paperclips_l230_23036


namespace NUMINAMATH_GPT_number_of_chickens_l230_23092

theorem number_of_chickens (c k : ℕ) (h1 : c + k = 120) (h2 : 2 * c + 4 * k = 350) : c = 65 :=
by sorry

end NUMINAMATH_GPT_number_of_chickens_l230_23092
