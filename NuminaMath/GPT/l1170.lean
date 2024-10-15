import Mathlib

namespace NUMINAMATH_GPT_arithmetic_seq_problem_l1170_117092

theorem arithmetic_seq_problem
  (a : ℕ → ℤ)  -- sequence a_n is an arithmetic sequence
  (h0 : ∃ (a1 d : ℤ), ∀ (n : ℕ), a n = a1 + n * d)  -- exists a1 and d such that a_n = a1 + n * d
  (h1 : a 0 + 3 * a 7 + a 14 = 120) :                -- given a1 + 3a8 + a15 = 120
  3 * a 8 - a 10 = 48 :=                             -- prove 3a9 - a11 = 48
sorry

end NUMINAMATH_GPT_arithmetic_seq_problem_l1170_117092


namespace NUMINAMATH_GPT_train_cross_time_l1170_117013

noncomputable def time_to_cross_pole (length: ℝ) (speed_kmh: ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  length / speed_ms

theorem train_cross_time :
  let length := 100
  let speed := 126
  abs (time_to_cross_pole length speed - 2.8571) < 0.0001 :=
by
  let length := 100
  let speed := 126
  have h1 : abs (time_to_cross_pole length speed - 2.8571) < 0.0001
  sorry
  exact h1

end NUMINAMATH_GPT_train_cross_time_l1170_117013


namespace NUMINAMATH_GPT_range_of_x_in_function_l1170_117094

theorem range_of_x_in_function (x : ℝ) :
  (x - 1 ≥ 0) ∧ (x - 2 ≠ 0) → (x ≥ 1 ∧ x ≠ 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_x_in_function_l1170_117094


namespace NUMINAMATH_GPT_a_takes_30_minutes_more_l1170_117027

noncomputable def speed_ratio := 3 / 4
noncomputable def time_A := 2 -- 2 hours
noncomputable def time_diff (b_time : ℝ) := time_A - b_time

theorem a_takes_30_minutes_more (b_time : ℝ) 
  (h_ratio : speed_ratio = 3 / 4)
  (h_a : time_A = 2) :
  time_diff b_time = 0.5 →  -- because 0.5 hours = 30 minutes
  time_diff b_time * 60 = 30 :=
by sorry

end NUMINAMATH_GPT_a_takes_30_minutes_more_l1170_117027


namespace NUMINAMATH_GPT_negate_exactly_one_even_l1170_117007

variable (a b c : ℕ)

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_exactly_one_even :
  ¬(is_even a ∧ is_odd b ∧ is_odd c ∨ is_odd a ∧ is_even b ∧ is_odd c ∨ is_odd a ∧ is_odd b ∧ is_even c) ↔
  (is_even a ∧ is_even b ∨ is_even a ∧ is_even c ∨ is_even b ∧ is_even c ∨ is_odd a ∧ is_odd b ∧ is_odd c) := sorry

end NUMINAMATH_GPT_negate_exactly_one_even_l1170_117007


namespace NUMINAMATH_GPT_sequence_increasing_range_of_a_l1170_117090

theorem sequence_increasing_range_of_a :
  ∀ {a : ℝ}, (∀ n : ℕ, 
    (n ≤ 7 → (4 - a) * n - 10 ≤ (4 - a) * (n + 1) - 10) ∧ 
    (7 < n → a^(n - 6) ≤ a^(n - 5))
  ) → 2 < a ∧ a < 4 :=
by
  sorry

end NUMINAMATH_GPT_sequence_increasing_range_of_a_l1170_117090


namespace NUMINAMATH_GPT_radishes_in_first_basket_l1170_117046

theorem radishes_in_first_basket :
  ∃ x : ℕ, ∃ y : ℕ, x + y = 88 ∧ y = x + 14 ∧ x = 37 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_radishes_in_first_basket_l1170_117046


namespace NUMINAMATH_GPT_num_letters_with_line_no_dot_l1170_117073

theorem num_letters_with_line_no_dot :
  ∀ (total_letters with_dot_and_line : ℕ) (with_dot_only with_line_only : ℕ),
    (total_letters = 60) →
    (with_dot_and_line = 20) →
    (with_dot_only = 4) →
    (total_letters = with_dot_and_line + with_dot_only + with_line_only) →
    with_line_only = 36 :=
by
  intros total_letters with_dot_and_line with_dot_only with_line_only
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_num_letters_with_line_no_dot_l1170_117073


namespace NUMINAMATH_GPT_find_k_l1170_117044

theorem find_k (a b : ℕ) (k : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (a^2 + b^2) = k * (a * b - 1)) :
  k = 5 :=
sorry

end NUMINAMATH_GPT_find_k_l1170_117044


namespace NUMINAMATH_GPT_classroom_width_perimeter_ratio_l1170_117041

theorem classroom_width_perimeter_ratio
  (L : Real) (W : Real) (P : Real)
  (hL : L = 15) (hW : W = 10)
  (hP : P = 2 * (L + W)) :
  W / P = 1 / 5 :=
sorry

end NUMINAMATH_GPT_classroom_width_perimeter_ratio_l1170_117041


namespace NUMINAMATH_GPT_find_integer_n_l1170_117063

theorem find_integer_n (n : ℤ) : 
  (∃ m : ℤ, n = 35 * m + 24) ↔ (5 ∣ (3 * n - 2) ∧ 7 ∣ (2 * n + 1)) :=
by sorry

end NUMINAMATH_GPT_find_integer_n_l1170_117063


namespace NUMINAMATH_GPT_problem_statement_l1170_117057

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l1170_117057


namespace NUMINAMATH_GPT_solve_cubic_equation_l1170_117008

theorem solve_cubic_equation (x : ℝ) (h : 4 * x^(1/3) - 2 * (x / x^(2/3)) = 7 + x^(1/3)) : x = 343 := by
  sorry

end NUMINAMATH_GPT_solve_cubic_equation_l1170_117008


namespace NUMINAMATH_GPT_original_number_of_people_l1170_117099

theorem original_number_of_people (x : ℕ) (h1 : x - x / 3 + (x / 3) * 3/4 = x * 1/4 + 15) : x = 30 :=
sorry

end NUMINAMATH_GPT_original_number_of_people_l1170_117099


namespace NUMINAMATH_GPT_calculate_expression_solve_quadratic_l1170_117011

-- Problem 1
theorem calculate_expression (x : ℝ) (hx : x > 0) :
  (2 / 3) * Real.sqrt (9 * x) + 6 * Real.sqrt (x / 4) - x * Real.sqrt (1 / x) = 4 * Real.sqrt x :=
sorry

-- Problem 2
theorem solve_quadratic (x : ℝ) (h : x^2 - 4 * x + 1 = 0) :
  x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_calculate_expression_solve_quadratic_l1170_117011


namespace NUMINAMATH_GPT_chemist_sons_ages_l1170_117095

theorem chemist_sons_ages 
    (a b c w : ℕ)
    (h1 : a * b * c = 36)
    (h2 : a + b + c = w)
    (h3 : ∃! x, x = max a (max b c)) :
    (a = 2 ∧ b = 2 ∧ c = 9) ∨ 
    (a = 2 ∧ b = 9 ∧ c = 2) ∨ 
    (a = 9 ∧ b = 2 ∧ c = 2) :=
  sorry

end NUMINAMATH_GPT_chemist_sons_ages_l1170_117095


namespace NUMINAMATH_GPT_josh_found_marbles_l1170_117012

theorem josh_found_marbles :
  ∃ (F : ℕ), (F + 14 = 23) → (F = 9) :=
by
  existsi 9
  intro h
  linarith

end NUMINAMATH_GPT_josh_found_marbles_l1170_117012


namespace NUMINAMATH_GPT_tangent_line_MP_l1170_117049

theorem tangent_line_MP
  (O : Type)
  (circle : O → O → Prop)
  (K M N P L : O)
  (is_tangent : O → O → Prop)
  (is_diameter : O → O → O)
  (K_tangent : is_tangent K M)
  (eq_segments : ∀ {P Q R}, circle P Q → circle Q R → circle P R → (P, Q) = (Q, R))
  (diam_opposite : L = is_diameter K L)
  (line_intrsc : ∀ {X Y}, is_tangent X Y → circle X Y → (Y = Y) → P = Y)
  (circ : ∀ {X Y}, circle X Y) :
  is_tangent M P :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_MP_l1170_117049


namespace NUMINAMATH_GPT_max_initial_number_l1170_117019

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end NUMINAMATH_GPT_max_initial_number_l1170_117019


namespace NUMINAMATH_GPT_triangle_perimeter_l1170_117045

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 3) 
  (h2 : b = 5) 
  (hc : c ^ 2 - 3 * c = c - 3) 
  (h3 : 3 + 3 > 5) 
  (h4 : 3 + 5 > 3) 
  (h5 : 5 + 3 > 3) : 
  a + b + c = 11 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1170_117045


namespace NUMINAMATH_GPT_percentage_of_students_on_trip_l1170_117062

-- Define the problem context
variable (total_students : ℕ)
variable (students_more_100 : ℕ)
variable (students_on_trip : ℕ)

-- Define the conditions as per the problem
def condition_1 : Prop := students_more_100 = total_students * 15 / 100
def condition_2 : Prop := students_more_100 = students_on_trip * 25 / 100

-- Define the problem statement
theorem percentage_of_students_on_trip
  (h1 : condition_1 total_students students_more_100)
  (h2 : condition_2 students_more_100 students_on_trip) :
  students_on_trip = total_students * 60 / 100 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_students_on_trip_l1170_117062


namespace NUMINAMATH_GPT_ceil_square_count_ceil_x_eq_15_l1170_117024

theorem ceil_square_count_ceil_x_eq_15 : 
  ∀ (x : ℝ), ( ⌈x⌉ = 15 ) → ∃ n : ℕ, n = 29 ∧ ∀ k : ℕ, k = ⌈x^2⌉ → 197 ≤ k ∧ k ≤ 225 :=
sorry

end NUMINAMATH_GPT_ceil_square_count_ceil_x_eq_15_l1170_117024


namespace NUMINAMATH_GPT_problem_statement_l1170_117086

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 - x else 2 - (x % 2)

theorem problem_statement : 
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, f (x + 1) + f x = 3) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2 - x) →
  f (-2007.5) = 1.5 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1170_117086


namespace NUMINAMATH_GPT_find_remainder_when_q_divided_by_x_plus_2_l1170_117002

noncomputable def q (x : ℝ) (D E F : ℝ) := D * x^4 + E * x^2 + F * x + 5

theorem find_remainder_when_q_divided_by_x_plus_2 (D E F : ℝ) :
  q 2 D E F = 15 → q (-2) D E F = 15 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_remainder_when_q_divided_by_x_plus_2_l1170_117002


namespace NUMINAMATH_GPT_harrys_mothers_age_l1170_117025

theorem harrys_mothers_age 
  (h : ℕ)  -- Harry's age
  (f : ℕ)  -- Father's age
  (m : ℕ)  -- Mother's age
  (h_age : h = 50)
  (f_age : f = h + 24)
  (m_age : m = f - h / 25) 
  : (m - h = 22) := 
by
  sorry

end NUMINAMATH_GPT_harrys_mothers_age_l1170_117025


namespace NUMINAMATH_GPT_range_eq_domain_l1170_117004

def f (x : ℝ) : ℝ := |x - 2| - 2

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem range_eq_domain : (Set.range f) = M :=
by
  sorry

end NUMINAMATH_GPT_range_eq_domain_l1170_117004


namespace NUMINAMATH_GPT_parallelogram_area_l1170_117037

theorem parallelogram_area {a b : ℝ} (h₁ : a = 9) (h₂ : b = 12) (angle : ℝ) (h₃ : angle = 150) : 
  ∃ (area : ℝ), area = 54 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1170_117037


namespace NUMINAMATH_GPT_john_average_speed_l1170_117029

theorem john_average_speed
  (uphill_distance : ℝ)
  (uphill_time : ℝ)
  (downhill_distance : ℝ)
  (downhill_time : ℝ)
  (uphill_time_is_45_minutes : uphill_time = 45)
  (downhill_time_is_15_minutes : downhill_time = 15)
  (uphill_distance_is_3_km : uphill_distance = 3)
  (downhill_distance_is_3_km : downhill_distance = 3)
  : (uphill_distance + downhill_distance) / ((uphill_time + downhill_time) / 60) = 6 := 
by
  sorry

end NUMINAMATH_GPT_john_average_speed_l1170_117029


namespace NUMINAMATH_GPT_range_of_fx_a_eq_2_range_of_a_increasing_fx_l1170_117043

-- Part (1)
theorem range_of_fx_a_eq_2 (x : ℝ) (h : x ∈ Set.Icc (-2 : ℝ) (3 : ℝ)) :
  ∃ y ∈ Set.Icc (-21 / 4 : ℝ) (15 : ℝ), y = x^2 + 3 * x - 3 :=
sorry

-- Part (2)
theorem range_of_a_increasing_fx (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) (3 : ℝ) → 2 * x + 2 * a - 1 ≥ 0) ↔ a ∈ Set.Ici (3 / 2 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_of_fx_a_eq_2_range_of_a_increasing_fx_l1170_117043


namespace NUMINAMATH_GPT_sqrt_inequality_l1170_117034

theorem sqrt_inequality (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
  sorry

end NUMINAMATH_GPT_sqrt_inequality_l1170_117034


namespace NUMINAMATH_GPT_total_notebooks_l1170_117079

-- Definitions from the conditions
def Yoongi_notebooks : Nat := 3
def Jungkook_notebooks : Nat := 3
def Hoseok_notebooks : Nat := 3

-- The proof problem
theorem total_notebooks : Yoongi_notebooks + Jungkook_notebooks + Hoseok_notebooks = 9 := 
by 
  sorry

end NUMINAMATH_GPT_total_notebooks_l1170_117079


namespace NUMINAMATH_GPT_max_a_no_lattice_point_l1170_117026

theorem max_a_no_lattice_point (a : ℚ) : a = 35 / 51 ↔ 
  (∀ (m : ℚ), (2 / 3 < m ∧ m < a) → 
    (∀ (x : ℤ), (0 < x ∧ x ≤ 50) → 
      ¬ ∃ (y : ℤ), y = m * x + 5)) :=
sorry

end NUMINAMATH_GPT_max_a_no_lattice_point_l1170_117026


namespace NUMINAMATH_GPT_perfect_square_condition_l1170_117076

def is_perfect_square (x : ℤ) : Prop := ∃ k : ℤ, k^2 = x

noncomputable def a_n (n : ℕ) : ℤ := (10^n - 1) / 9

theorem perfect_square_condition (n b : ℕ) (h1 : 0 < b) (h2 : b < 10) :
  is_perfect_square ((a_n (2 * n)) - b * (a_n n)) ↔ (b = 2 ∨ (b = 7 ∧ n = 1)) := by
  sorry

end NUMINAMATH_GPT_perfect_square_condition_l1170_117076


namespace NUMINAMATH_GPT_largest_sum_of_distinct_factors_l1170_117015

theorem largest_sum_of_distinct_factors (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) (h_product : A * B * C = 3003) :
  A + B + C ≤ 105 :=
sorry  -- Proof is not required, just the statement.

end NUMINAMATH_GPT_largest_sum_of_distinct_factors_l1170_117015


namespace NUMINAMATH_GPT_range_of_m_l1170_117069

-- Define sets A and B
def A := {x : ℝ | x ≤ 1}
def B (m : ℝ) := {x : ℝ | x ≤ m}

-- Statement: Prove the range of m such that B ⊆ A
theorem range_of_m (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ↔ (m ≤ 1) :=
by sorry

end NUMINAMATH_GPT_range_of_m_l1170_117069


namespace NUMINAMATH_GPT_find_ks_l1170_117033

def is_valid_function (f : ℕ → ℤ) (k : ℤ) : Prop :=
  ∀ x y : ℕ, f (x * y) = f x + f y + k * f (Nat.gcd x y)

theorem find_ks (f : ℕ → ℤ) :
  (f 2006 = 2007) →
  is_valid_function f k →
  k = 0 ∨ k = -1 :=
sorry

end NUMINAMATH_GPT_find_ks_l1170_117033


namespace NUMINAMATH_GPT_largest_angle_in_triangle_PQR_l1170_117081

-- Definitions
def is_isosceles_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  (α = β) ∨ (β = γ) ∨ (γ = α)

def is_obtuse_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

variables (P Q R : ℝ)
variables (angleP angleQ angleR : ℝ)

-- Condition: PQR is an obtuse and isosceles triangle, and angle P measures 30 degrees
axiom h1 : is_isosceles_triangle P Q R angleP angleQ angleR
axiom h2 : is_obtuse_triangle P Q R angleP angleQ angleR
axiom h3 : angleP = 30

-- Theorem: The measure of the largest interior angle of triangle PQR is 120 degrees
theorem largest_angle_in_triangle_PQR : max angleP (max angleQ angleR) = 120 :=
  sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_PQR_l1170_117081


namespace NUMINAMATH_GPT_largest_divisor_of_n_l1170_117028

theorem largest_divisor_of_n (n : ℕ) (hn : 0 < n) (h : 50 ∣ n^2) : 5 ∣ n :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_n_l1170_117028


namespace NUMINAMATH_GPT_find_angle_between_planes_l1170_117009

noncomputable def angle_between_planes (α β : ℝ) : ℝ := Real.arcsin ((Real.sqrt 6 + 1) / 5)

theorem find_angle_between_planes (α β : ℝ) (h : α = β) : 
  (∃ (cube : Type) (A B C D A₁ B₁ C₁ D₁ : cube),
    α = Real.arcsin ((Real.sqrt 6 - 1) / 5) ∨ α = Real.arcsin ((Real.sqrt 6 + 1) / 5)) 
    :=
sorry

end NUMINAMATH_GPT_find_angle_between_planes_l1170_117009


namespace NUMINAMATH_GPT_min_value_p_plus_q_l1170_117020

-- Definitions related to the conditions.
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def satisfies_equations (a b p q : ℕ) : Prop :=
  20 * a + 17 * b = p ∧ 17 * a + 20 * b = q ∧ is_prime p ∧ is_prime q

def distinct_positive_integers (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a ≠ b

-- The main proof problem.
theorem min_value_p_plus_q (a b p q : ℕ) :
  distinct_positive_integers a b →
  satisfies_equations a b p q →
  p + q = 296 :=
by
  sorry

end NUMINAMATH_GPT_min_value_p_plus_q_l1170_117020


namespace NUMINAMATH_GPT_parallel_lines_condition_l1170_117071

theorem parallel_lines_condition (k_1 k_2 : ℝ) :
  (k_1 = k_2) ↔ (∀ x y : ℝ, k_1 * x + y + 1 = 0 → k_2 * x + y - 1 = 0) :=
sorry

end NUMINAMATH_GPT_parallel_lines_condition_l1170_117071


namespace NUMINAMATH_GPT_ratio_eq_one_l1170_117059

variable {a b : ℝ}

theorem ratio_eq_one (h1 : 7 * a = 8 * b) (h2 : a * b ≠ 0) : (a / 8) / (b / 7) = 1 := 
by
  sorry

end NUMINAMATH_GPT_ratio_eq_one_l1170_117059


namespace NUMINAMATH_GPT_ratio_and_lcm_l1170_117058

noncomputable def common_factor (a b : ℕ) := ∃ x : ℕ, a = 3 * x ∧ b = 4 * x

theorem ratio_and_lcm (a b : ℕ) (h1 : common_factor a b) (h2 : Nat.lcm a b = 180) (h3 : a = 60) : b = 45 :=
by sorry

end NUMINAMATH_GPT_ratio_and_lcm_l1170_117058


namespace NUMINAMATH_GPT_cos_2alpha_2beta_l1170_117053

variables (α β : ℝ)

open Real

theorem cos_2alpha_2beta (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) : cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end NUMINAMATH_GPT_cos_2alpha_2beta_l1170_117053


namespace NUMINAMATH_GPT_unique_solution_pair_l1170_117096

theorem unique_solution_pair (x p : ℕ) (hp : Nat.Prime p) (hx : x ≥ 0) (hp2 : p ≥ 2) :
  x * (x + 1) * (x + 2) * (x + 3) = 1679 ^ (p - 1) + 1680 ^ (p - 1) + 1681 ^ (p - 1) ↔ (x = 4 ∧ p = 2) := 
by
  sorry

end NUMINAMATH_GPT_unique_solution_pair_l1170_117096


namespace NUMINAMATH_GPT_find_k_hyperbola_l1170_117065

-- Define the given conditions
variables (k : ℝ)
def condition1 : Prop := k < 0
def condition2 : Prop := 2 * k^2 + k - 2 = -1

-- State the proof goal
theorem find_k_hyperbola (h1 : condition1 k) (h2 : condition2 k) : k = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_hyperbola_l1170_117065


namespace NUMINAMATH_GPT_angle_A_sides_b_c_l1170_117022

noncomputable def triangle_angles (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.sin C - Real.sqrt 3 * c * Real.cos A = 0

theorem angle_A (a b c A B C : ℝ) (h1 : triangle_angles a b c A B C) :
  A = Real.pi / 3 :=
by sorry

noncomputable def triangle_area (a b c S : ℝ) : Prop :=
  S = Real.sqrt 3 ∧ a = 2

theorem sides_b_c (a b c S : ℝ) (h : triangle_area a b c S) :
  b = 2 ∧ c = 2 :=
by sorry

end NUMINAMATH_GPT_angle_A_sides_b_c_l1170_117022


namespace NUMINAMATH_GPT_proportion_of_boys_correct_l1170_117047

noncomputable def proportion_of_boys : ℚ :=
  let p_boy := 1 / 2
  let p_girl := 1 / 2
  let expected_children := 3 -- (2 boys and 1 girl)
  let expected_boys := 2 -- Expected number of boys in each family
  
  expected_boys / expected_children

theorem proportion_of_boys_correct : proportion_of_boys = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_proportion_of_boys_correct_l1170_117047


namespace NUMINAMATH_GPT_exists_constant_C_inequality_for_difference_l1170_117005

theorem exists_constant_C (a : ℕ → ℝ) (C : ℝ) (hC : 0 < C) :
  (a 1 = 1) →
  (a 2 = 8) →
  (∀ n : ℕ, 2 ≤ n → a (n + 1) = a (n - 1) + (4 / n) * a n) →
  (∀ n : ℕ, a n ≤ C * n^2) := sorry

theorem inequality_for_difference (a : ℕ → ℝ) :
  (a 1 = 1) →
  (a 2 = 8) →
  (∀ n : ℕ, 2 ≤ n → a (n + 1) = a (n - 1) + (4 / n) * a n) →
  (∀ n : ℕ, a (n + 1) - a n ≤ 4 * n + 3) := sorry

end NUMINAMATH_GPT_exists_constant_C_inequality_for_difference_l1170_117005


namespace NUMINAMATH_GPT_cube_of_odd_number_minus_itself_divisible_by_24_l1170_117000

theorem cube_of_odd_number_minus_itself_divisible_by_24 (n : ℤ) : 
  24 ∣ ((2 * n + 1) ^ 3 - (2 * n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_cube_of_odd_number_minus_itself_divisible_by_24_l1170_117000


namespace NUMINAMATH_GPT_max_height_of_rock_l1170_117023

theorem max_height_of_rock : 
    ∃ t_max : ℝ, (∀ t : ℝ, -5 * t^2 + 25 * t + 10 ≤ -5 * t_max^2 + 25 * t_max + 10) ∧ (-5 * t_max^2 + 25 * t_max + 10 = 165 / 4) := 
sorry

end NUMINAMATH_GPT_max_height_of_rock_l1170_117023


namespace NUMINAMATH_GPT_cannot_determine_both_correct_l1170_117016

-- Definitions
def total_students : ℕ := 40
def answered_q1_correctly : ℕ := 30
def did_not_take_test : ℕ := 10

-- Assertion that the number of students answering both questions correctly cannot be determined
theorem cannot_determine_both_correct (answered_q2_correctly : ℕ) :
  (∃ (both_correct : ℕ), both_correct ≤ answered_q1_correctly ∧ both_correct ≤ answered_q2_correctly)  ↔ answered_q2_correctly > 0 :=
by 
 sorry

end NUMINAMATH_GPT_cannot_determine_both_correct_l1170_117016


namespace NUMINAMATH_GPT_find_constants_l1170_117060

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
if x < 3 then a * x^2 + b else 10 - 2 * x

theorem find_constants (a b : ℝ)
  (H : ∀ x, f a b (f a b x) = x) :
  a + b = 13 / 3 := by 
  sorry

end NUMINAMATH_GPT_find_constants_l1170_117060


namespace NUMINAMATH_GPT_solve_for_sum_l1170_117097

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := -1
noncomputable def c : ℝ := Real.sqrt 26

theorem solve_for_sum :
  (a * (a - 4) = 5) ∧ (b * (b - 4) = 5) ∧ (c * (c - 4) = 5) ∧ 
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a^2 + b^2 = c^2) → (a + b + c = 4 + Real.sqrt 26) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_sum_l1170_117097


namespace NUMINAMATH_GPT_arc_length_of_sector_l1170_117072

theorem arc_length_of_sector (r : ℝ) (θ : ℝ) (h : r = Real.pi ∧ θ = 120) : 
  r * θ / 180 * Real.pi = 2 * Real.pi * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_arc_length_of_sector_l1170_117072


namespace NUMINAMATH_GPT_min_value_c_plus_d_l1170_117048

theorem min_value_c_plus_d (c d : ℤ) (h : c * d = 144) : c + d = -145 :=
sorry

end NUMINAMATH_GPT_min_value_c_plus_d_l1170_117048


namespace NUMINAMATH_GPT_smallest_perfect_square_divisible_by_5_and_6_l1170_117052

-- 1. Define the gcd and lcm functionality
def lcm (a b : ℕ) : ℕ :=
  (a * b) / Nat.gcd a b

-- 2. Define the condition that a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- 3. State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 : ∃ n : ℕ, is_perfect_square n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (is_perfect_square m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m :=
  sorry

end NUMINAMATH_GPT_smallest_perfect_square_divisible_by_5_and_6_l1170_117052


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l1170_117080

-- Define the necessary conditions for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  k > 5 ∨ k < -2

-- Define the condition for k
axiom k_in_real (k : ℝ) : Prop

-- The proof statement
theorem necessary_not_sufficient_condition (k : ℝ) (hk : k_in_real k) :
  (∃ (k_val : ℝ), k_val > 5 ∧ k = k_val) → represents_hyperbola k ∧ ¬ (represents_hyperbola k → k > 5) :=
by
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l1170_117080


namespace NUMINAMATH_GPT_part_a_l1170_117050

theorem part_a (a b c : ℝ) : 
  (∀ n : ℝ, (n + 2)^2 = a * (n + 1)^2 + b * n^2 + c * (n - 1)^2) ↔ (a = 3 ∧ b = -3 ∧ c = 1) :=
by 
  sorry

end NUMINAMATH_GPT_part_a_l1170_117050


namespace NUMINAMATH_GPT_tangent_line_through_P_line_through_P_chord_length_8_l1170_117018

open Set

def circle (x y : ℝ) : Prop := x^2 + y^2 = 25

def point_P : ℝ × ℝ := (3, 4)

def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y - 25 = 0

def line_m_case1 (x : ℝ) : Prop := x = 3

def line_m_case2 (x y : ℝ) : Prop := 7 * x - 24 * y + 75 = 0

theorem tangent_line_through_P :
  tangent_line point_P.1 point_P.2 :=
sorry

theorem line_through_P_chord_length_8 :
  (∀ x y, circle x y → line_m_case1 x ∨ line_m_case2 x y) :=
sorry

end NUMINAMATH_GPT_tangent_line_through_P_line_through_P_chord_length_8_l1170_117018


namespace NUMINAMATH_GPT_distinct_banners_l1170_117036

inductive Color
| red
| white
| blue
| green
| yellow

def adjacent_different (a b : Color) : Prop := a ≠ b

theorem distinct_banners : 
  ∃ n : ℕ, n = 320 ∧ ∀ strips : Fin 4 → Color, 
    adjacent_different (strips 0) (strips 1) ∧ 
    adjacent_different (strips 1) (strips 2) ∧ 
    adjacent_different (strips 2) (strips 3) :=
sorry

end NUMINAMATH_GPT_distinct_banners_l1170_117036


namespace NUMINAMATH_GPT_monthly_food_expense_l1170_117066

-- Definitions based on the given conditions
def E : ℕ := 6000
def R : ℕ := 640
def EW : ℕ := E / 4
def I : ℕ := E / 5
def L : ℕ := 2280

-- Define the monthly food expense F
def F : ℕ := E - (R + EW + I) - L

-- The theorem stating that the monthly food expense is 380
theorem monthly_food_expense : F = 380 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_monthly_food_expense_l1170_117066


namespace NUMINAMATH_GPT_evaluate_expression_l1170_117091

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 2) : 4 * x^y + 5 * y^x = 76 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1170_117091


namespace NUMINAMATH_GPT_oliver_more_money_l1170_117006

noncomputable def totalOliver : ℕ := 10 * 20 + 3 * 5
noncomputable def totalWilliam : ℕ := 15 * 10 + 4 * 5

theorem oliver_more_money : totalOliver - totalWilliam = 45 := by
  sorry

end NUMINAMATH_GPT_oliver_more_money_l1170_117006


namespace NUMINAMATH_GPT_avg_speed_of_car_l1170_117056

noncomputable def average_speed (distance1 distance2 : ℕ) (time1 time2 : ℕ) : ℕ :=
  (distance1 + distance2) / (time1 + time2)

theorem avg_speed_of_car :
  average_speed 65 45 1 1 = 55 := by
  sorry

end NUMINAMATH_GPT_avg_speed_of_car_l1170_117056


namespace NUMINAMATH_GPT_harmonica_value_l1170_117087

theorem harmonica_value (x : ℕ) (h1 : ∃ k : ℕ, ∃ r : ℕ, x = 12 * k + r ∧ r ≠ 0 
                                                   ∧ r ≠ 6 ∧ r ≠ 9 
                                                   ∧ r ≠ 10 ∧ r ≠ 11)
                         (h2 : ¬ (x * x % 12 = 0)) : 
                         4 = 4 :=
by 
  sorry

end NUMINAMATH_GPT_harmonica_value_l1170_117087


namespace NUMINAMATH_GPT_petya_recover_x_y_l1170_117030

theorem petya_recover_x_y (x y a b c d : ℝ)
    (hx_pos : x > 0) (hy_pos : y > 0)
    (ha : a = x + y) (hb : b = x - y) (hc : c = x / y) (hd : d = x * y) :
    ∃! (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ a = x' + y' ∧ b = x' - y' ∧ c = x' / y' ∧ d = x' * y' :=
sorry

end NUMINAMATH_GPT_petya_recover_x_y_l1170_117030


namespace NUMINAMATH_GPT_quadratic_completes_square_l1170_117038

theorem quadratic_completes_square (b c : ℤ) :
  (∃ b c : ℤ, (∀ x : ℤ, x^2 - 12 * x + 49 = (x + b)^2 + c) ∧ b + c = 7) :=
sorry

end NUMINAMATH_GPT_quadratic_completes_square_l1170_117038


namespace NUMINAMATH_GPT_right_triangle_properties_l1170_117078

theorem right_triangle_properties (a b c h : ℝ)
  (ha: a = 5) (hb: b = 12) (h_right_angle: a^2 + b^2 = c^2)
  (h_area: (1/2) * a * b = (1/2) * c * h) :
  c = 13 ∧ h = 60 / 13 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_properties_l1170_117078


namespace NUMINAMATH_GPT_geometric_sequence_m_solution_l1170_117055

theorem geometric_sequence_m_solution (m : ℝ) (h : ∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 4 ∧ a * c = b^2) :
  m = 2 ∨ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_m_solution_l1170_117055


namespace NUMINAMATH_GPT_sum_of_valid_m_values_l1170_117031

-- Variables and assumptions
variable (m x : ℝ)

-- Conditions from the given problem
def inequality_system (m x : ℝ) : Prop :=
  (x - 4) / 3 < x - 4 ∧ (m - x) / 5 < 0

def solution_set_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, inequality_system m x → x > 4

def fractional_equation (m x : ℝ) : Prop :=
  6 / (x - 3) + 1 = (m * x - 3) / (x - 3)

-- Lean statement to prove the sum of integers satisfying the conditions
theorem sum_of_valid_m_values : 
  (∀ m : ℝ, solution_set_condition m ∧ 
            (∃ x : ℝ, x > 0 ∧ x ≠ 3 ∧ fractional_equation m x) →
            (∃ (k : ℕ), k = 2 ∨ k = 4) → 
            2 + 4 = 6) :=
sorry

end NUMINAMATH_GPT_sum_of_valid_m_values_l1170_117031


namespace NUMINAMATH_GPT_non_congruent_parallelograms_l1170_117021

def side_lengths_sum (a b : ℕ) : Prop :=
  a + b = 25

def is_congruent (a b : ℕ) (a' b' : ℕ) : Prop :=
  (a = a' ∧ b = b') ∨ (a = b' ∧ b = a')

def non_congruent_count (n : ℕ) : Prop :=
  ∀ (a b : ℕ), side_lengths_sum a b → 
  ∃! (m : ℕ), is_congruent a b m b

theorem non_congruent_parallelograms :
  ∃ (n : ℕ), non_congruent_count n ∧ n = 13 :=
sorry

end NUMINAMATH_GPT_non_congruent_parallelograms_l1170_117021


namespace NUMINAMATH_GPT_night_shift_hours_l1170_117088

theorem night_shift_hours
  (hours_first_guard : ℕ := 3)
  (hours_last_guard : ℕ := 2)
  (hours_each_middle_guard : ℕ := 2) :
  hours_first_guard + 2 * hours_each_middle_guard + hours_last_guard = 9 :=
by 
  sorry

end NUMINAMATH_GPT_night_shift_hours_l1170_117088


namespace NUMINAMATH_GPT_chipmunks_initial_count_l1170_117042

variable (C : ℕ) (total : ℕ) (morning_beavers : ℕ) (afternoon_beavers : ℕ) (decrease_chipmunks : ℕ)

axiom chipmunks_count : morning_beavers = 20 
axiom beavers_double : afternoon_beavers = 2 * morning_beavers
axiom decrease_chipmunks_initial : decrease_chipmunks = 10
axiom total_animals : total = 130

theorem chipmunks_initial_count : 
  20 + C + (2 * 20) + (C - 10) = 130 → C = 40 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_chipmunks_initial_count_l1170_117042


namespace NUMINAMATH_GPT_compare_groups_l1170_117082

noncomputable def mean (scores : List ℝ) : ℝ :=
  (scores.sum / scores.length)

noncomputable def variance (scores : List ℝ) : ℝ :=
  let m := mean scores
  (scores.map (λ x => (x - m) ^ 2)).sum / scores.length

noncomputable def stddev (scores : List ℝ) : ℝ :=
  (variance scores).sqrt

def groupA_scores : List ℝ := [88, 100, 95, 86, 95, 91, 84, 74, 92, 83]
def groupB_scores : List ℝ := [93, 89, 81, 77, 96, 78, 77, 85, 89, 86]

theorem compare_groups :
  mean groupA_scores > mean groupB_scores ∧ stddev groupA_scores > stddev groupB_scores :=
by
  sorry

end NUMINAMATH_GPT_compare_groups_l1170_117082


namespace NUMINAMATH_GPT_find_cost_prices_l1170_117085

-- These represent the given selling prices of the items.
def SP_computer_table : ℝ := 3600
def SP_office_chair : ℝ := 5000
def SP_bookshelf : ℝ := 1700

-- These represent the percentage markups and discounts as multipliers.
def markup_computer_table : ℝ := 1.20
def markup_office_chair : ℝ := 1.25
def discount_bookshelf : ℝ := 0.85

-- The problem requires us to find the cost prices. We will define these as variables.
variable (C O B : ℝ)

theorem find_cost_prices :
  (SP_computer_table = C * markup_computer_table) ∧
  (SP_office_chair = O * markup_office_chair) ∧
  (SP_bookshelf = B * discount_bookshelf) →
  (C = 3000) ∧ (O = 4000) ∧ (B = 2000) :=
by
  sorry

end NUMINAMATH_GPT_find_cost_prices_l1170_117085


namespace NUMINAMATH_GPT_inequality_cannot_hold_l1170_117068

variable (a b : ℝ)
variable (h : a < b ∧ b < 0)

theorem inequality_cannot_hold (h : a < b ∧ b < 0) : ¬ (1 / (a - b) > 1 / a) := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_cannot_hold_l1170_117068


namespace NUMINAMATH_GPT_trig_evaluation_l1170_117051

noncomputable def sin30 := 1 / 2
noncomputable def cos45 := Real.sqrt 2 / 2
noncomputable def tan30 := Real.sqrt 3 / 3
noncomputable def sin60 := Real.sqrt 3 / 2

theorem trig_evaluation : 4 * sin30 - Real.sqrt 2 * cos45 - Real.sqrt 3 * tan30 + 2 * sin60 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_trig_evaluation_l1170_117051


namespace NUMINAMATH_GPT_sum_of_x_intercepts_l1170_117035

theorem sum_of_x_intercepts (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (5 : ℤ) * (3 : ℤ) = (a : ℤ) * (b : ℤ)) : 
  ((-5 : ℤ) / (a : ℤ)) + ((-5 : ℤ) / (3 : ℤ)) + ((-1 : ℤ) / (1 : ℤ)) + ((-1 : ℤ) / (15 : ℤ)) = -8 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_x_intercepts_l1170_117035


namespace NUMINAMATH_GPT_first_discount_percentage_l1170_117098

theorem first_discount_percentage
  (list_price : ℝ)
  (second_discount : ℝ)
  (third_discount : ℝ)
  (tax_rate : ℝ)
  (final_price : ℝ)
  (D1 : ℝ)
  (h_list_price : list_price = 150)
  (h_second_discount : second_discount = 12)
  (h_third_discount : third_discount = 5)
  (h_tax_rate : tax_rate = 10)
  (h_final_price : final_price = 105) :
  100 - 100 * (final_price / (list_price * (1 - D1 / 100) * (1 - second_discount / 100) * (1 - third_discount / 100) * (1 + tax_rate / 100))) = 24.24 :=
by
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l1170_117098


namespace NUMINAMATH_GPT_cubic_function_increasing_l1170_117039

noncomputable def f (a x : ℝ) := x ^ 3 + a * x ^ 2 + 7 * a * x

theorem cubic_function_increasing (a : ℝ) (h : 0 ≤ a ∧ a ≤ 21) :
    ∀ x y : ℝ, x ≤ y → f a x ≤ f a y :=
sorry

end NUMINAMATH_GPT_cubic_function_increasing_l1170_117039


namespace NUMINAMATH_GPT_same_solution_k_value_l1170_117064

theorem same_solution_k_value 
  (x : ℝ)
  (k : ℝ)
  (m : ℝ)
  (h₁ : 2 * x + 4 = 4 * (x - 2))
  (h₂ : k * x + m = 2 * x - 1) 
  (h₃ : k = 17) : 
  k = 17 ∧ m = -91 :=
by
  sorry

end NUMINAMATH_GPT_same_solution_k_value_l1170_117064


namespace NUMINAMATH_GPT_peach_trees_count_l1170_117017

theorem peach_trees_count : ∀ (almond_trees: ℕ), almond_trees = 300 → 2 * almond_trees - 30 = 570 :=
by
  intros
  sorry

end NUMINAMATH_GPT_peach_trees_count_l1170_117017


namespace NUMINAMATH_GPT_socks_cost_5_l1170_117067

theorem socks_cost_5
  (jeans t_shirt socks : ℕ)
  (h1 : jeans = 2 * t_shirt)
  (h2 : t_shirt = socks + 10)
  (h3 : jeans = 30) :
  socks = 5 :=
by
  sorry

end NUMINAMATH_GPT_socks_cost_5_l1170_117067


namespace NUMINAMATH_GPT_common_ratio_is_63_98_l1170_117003

/-- Define the terms of the geometric series -/
def term (n : Nat) : ℚ := 
  match n with
  | 0 => 4 / 7
  | 1 => 18 / 49
  | 2 => 162 / 343
  | _ => sorry  -- For simplicity, we can define more terms if needed, but it's irrelevant for our proof

/-- Define the common ratio of the geometric series -/
def common_ratio (a b : ℚ) : ℚ := b / a

/-- The problem states that the common ratio of first two terms of the given series is equal to 63/98 -/
theorem common_ratio_is_63_98 : common_ratio (term 0) (term 1) = 63 / 98 :=
by
  -- leave the proof as sorry for now
  sorry

end NUMINAMATH_GPT_common_ratio_is_63_98_l1170_117003


namespace NUMINAMATH_GPT_contrapositive_statement_l1170_117010

theorem contrapositive_statement (x : ℝ) : (x ≤ -3 → x < 0) → (x ≥ 0 → x > -3) := 
by
  sorry

end NUMINAMATH_GPT_contrapositive_statement_l1170_117010


namespace NUMINAMATH_GPT_find_f_2017_l1170_117084

theorem find_f_2017 (f : ℤ → ℤ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 3) = f x) (h_f_neg1 : f (-1) = 1) : 
  f 2017 = -1 :=
sorry

end NUMINAMATH_GPT_find_f_2017_l1170_117084


namespace NUMINAMATH_GPT_quadratic_term_free_polynomial_l1170_117083

theorem quadratic_term_free_polynomial (m : ℤ) (h : 36 + 12 * m = 0) : m^3 = -27 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_quadratic_term_free_polynomial_l1170_117083


namespace NUMINAMATH_GPT_product_102_108_l1170_117061

theorem product_102_108 : (102 = 105 - 3) → (108 = 105 + 3) → (102 * 108 = 11016) := by
  sorry

end NUMINAMATH_GPT_product_102_108_l1170_117061


namespace NUMINAMATH_GPT_perpendicular_lines_l1170_117001

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y - 1 = 0 → x + 2 * y = 0) →
  (a = 1) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l1170_117001


namespace NUMINAMATH_GPT_one_gallon_fills_one_cubic_foot_l1170_117070

theorem one_gallon_fills_one_cubic_foot
  (total_water : ℕ)
  (drinking_cooking : ℕ)
  (shower_water : ℕ)
  (num_showers : ℕ)
  (pool_length : ℕ)
  (pool_width : ℕ)
  (pool_height : ℕ)
  (h_total_water : total_water = 1000)
  (h_drinking_cooking : drinking_cooking = 100)
  (h_shower_water : shower_water = 20)
  (h_num_showers : num_showers = 15)
  (h_pool_length : pool_length = 10)
  (h_pool_width : pool_width = 10)
  (h_pool_height : pool_height = 6) :
  (pool_length * pool_width * pool_height) / 
  (total_water - drinking_cooking - num_showers * shower_water) = 1 := by
  sorry

end NUMINAMATH_GPT_one_gallon_fills_one_cubic_foot_l1170_117070


namespace NUMINAMATH_GPT_exists_arithmetic_seq_2003_terms_perfect_powers_no_infinite_arithmetic_seq_perfect_powers_l1170_117032

-- Part (a): Proving the existence of such an arithmetic sequence with 2003 terms.
theorem exists_arithmetic_seq_2003_terms_perfect_powers :
  ∃ (a : ℕ) (d : ℕ), ∀ n : ℕ, n ≤ 2002 → ∃ (k m : ℕ), m > 1 ∧ a + n * d = k ^ m :=
by
  sorry

-- Part (b): Proving the non-existence of such an infinite arithmetic sequence.
theorem no_infinite_arithmetic_seq_perfect_powers :
  ¬ ∃ (a : ℕ) (d : ℕ), ∀ n : ℕ, ∃ (k m : ℕ), m > 1 ∧ a + n * d = k ^ m :=
by
  sorry

end NUMINAMATH_GPT_exists_arithmetic_seq_2003_terms_perfect_powers_no_infinite_arithmetic_seq_perfect_powers_l1170_117032


namespace NUMINAMATH_GPT_Nicole_has_69_clothes_l1170_117093

def clothingDistribution : Prop :=
  let nicole_clothes := 15
  let first_sister_clothes := nicole_clothes / 3
  let second_sister_clothes := nicole_clothes + 5
  let third_sister_clothes := 2 * first_sister_clothes
  let average_clothes := (nicole_clothes + first_sister_clothes + second_sister_clothes + third_sister_clothes) / 4
  let oldest_sister_clothes := 1.5 * average_clothes
  let total_clothes := nicole_clothes + first_sister_clothes + second_sister_clothes + third_sister_clothes + oldest_sister_clothes
  total_clothes = 69

theorem Nicole_has_69_clothes : clothingDistribution :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_Nicole_has_69_clothes_l1170_117093


namespace NUMINAMATH_GPT_total_views_correct_l1170_117089

-- Definitions based on the given conditions
def initial_views : ℕ := 4000
def views_increase := 10 * initial_views
def additional_views := 50000
def total_views_after_6_days := initial_views + views_increase + additional_views

-- The theorem we are going to state
theorem total_views_correct :
  total_views_after_6_days = 94000 :=
sorry

end NUMINAMATH_GPT_total_views_correct_l1170_117089


namespace NUMINAMATH_GPT_infinite_series_sum_l1170_117014

theorem infinite_series_sum :
  ∑' (n : ℕ), (n + 1) / 10^(n + 1) = 10 / 81 :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_l1170_117014


namespace NUMINAMATH_GPT_total_fencing_cost_l1170_117040

-- Definitions based on the conditions
def cost_per_side : ℕ := 69
def number_of_sides : ℕ := 4

-- The proof problem statement
theorem total_fencing_cost : number_of_sides * cost_per_side = 276 := by
  sorry

end NUMINAMATH_GPT_total_fencing_cost_l1170_117040


namespace NUMINAMATH_GPT_arith_seq_largest_portion_l1170_117077

theorem arith_seq_largest_portion (a1 d : ℝ) (h_d_pos : d > 0) 
  (h_sum : 5 * a1 + 10 * d = 100)
  (h_ratio : (3 * a1 + 9 * d) / 7 = 2 * a1 + d) : 
  a1 + 4 * d = 115 / 3 := by
  sorry

end NUMINAMATH_GPT_arith_seq_largest_portion_l1170_117077


namespace NUMINAMATH_GPT_range_of_a_l1170_117074

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - 2 * x ^ 2

theorem range_of_a (a : ℝ) :
  (∀ x0 : ℝ, 0 < x0 ∧ x0 < 1 →
  (0 < (deriv (fun x => f a x - x)) x0)) →
  a > (4 / Real.exp (3 / 4)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1170_117074


namespace NUMINAMATH_GPT_pablo_days_to_complete_all_puzzles_l1170_117054

def average_pieces_per_hour : ℕ := 100
def puzzles_300_pieces : ℕ := 8
def puzzles_500_pieces : ℕ := 5
def pieces_per_300_puzzle : ℕ := 300
def pieces_per_500_puzzle : ℕ := 500
def max_hours_per_day : ℕ := 7

theorem pablo_days_to_complete_all_puzzles :
  let total_pieces := (puzzles_300_pieces * pieces_per_300_puzzle) + (puzzles_500_pieces * pieces_per_500_puzzle)
  let pieces_per_day := max_hours_per_day * average_pieces_per_hour
  let days_to_complete := total_pieces / pieces_per_day
  days_to_complete = 7 :=
by
  sorry

end NUMINAMATH_GPT_pablo_days_to_complete_all_puzzles_l1170_117054


namespace NUMINAMATH_GPT_set_D_is_empty_l1170_117075

-- Definitions based on the conditions from the original problem
def set_A : Set ℝ := {x | x + 3 = 3}
def set_B : Set (ℝ × ℝ) := {(x, y) | y^2 = -x^2}
def set_C : Set ℝ := {x | x^2 ≤ 0}
def set_D : Set ℝ := {x | x^2 - x + 1 = 0}

-- The theorem statement
theorem set_D_is_empty : set_D = ∅ :=
sorry

end NUMINAMATH_GPT_set_D_is_empty_l1170_117075
