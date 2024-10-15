import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l188_18812

/-
If x is equal to the sum of the even integers from 40 to 60 inclusive,
y is the number of even integers from 40 to 60 inclusive,
and z is the sum of the odd integers from 41 to 59 inclusive,
prove that x + y + z = 1061.
-/
theorem problem_statement :
  let x := (11 / 2) * (40 + 60)
  let y := 11
  let z := (10 / 2) * (41 + 59)
  x + y + z = 1061 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l188_18812


namespace NUMINAMATH_GPT_exist_ai_for_xij_l188_18894

theorem exist_ai_for_xij (n : ℕ) (x : Fin n → Fin n → ℝ)
  (h : ∀ i j k : Fin n, x i j + x j k + x k i = 0) :
  ∃ a : Fin n → ℝ, ∀ i j : Fin n, x i j = a i - a j :=
by
  sorry

end NUMINAMATH_GPT_exist_ai_for_xij_l188_18894


namespace NUMINAMATH_GPT_expenditure_ratio_l188_18822

theorem expenditure_ratio (I_A I_B E_A E_B : ℝ) (h1 : I_A / I_B = 5 / 6)
  (h2 : I_B = 7200) (h3 : 1800 = I_A - E_A) (h4 : 1600 = I_B - E_B) :
  E_A / E_B = 3 / 4 :=
sorry

end NUMINAMATH_GPT_expenditure_ratio_l188_18822


namespace NUMINAMATH_GPT_towel_bleach_percentage_decrease_l188_18807

theorem towel_bleach_percentage_decrease :
  ∀ (L B : ℝ), (L > 0) → (B > 0) → 
  let L' := 0.70 * L 
  let B' := 0.75 * B 
  let A := L * B 
  let A' := L' * B' 
  (A - A') / A * 100 = 47.5 :=
by sorry

end NUMINAMATH_GPT_towel_bleach_percentage_decrease_l188_18807


namespace NUMINAMATH_GPT_total_number_of_numbers_l188_18814

theorem total_number_of_numbers (avg : ℝ) (sum1 sum2 sum3 : ℝ) (N : ℝ) :
  avg = 3.95 →
  sum1 = 2 * 3.8 →
  sum2 = 2 * 3.85 →
  sum3 = 2 * 4.200000000000001 →
  avg = (sum1 + sum2 + sum3) / N →
  N = 6 :=
by
  intros h_avg h_sum1 h_sum2 h_sum3 h_total
  sorry

end NUMINAMATH_GPT_total_number_of_numbers_l188_18814


namespace NUMINAMATH_GPT_value_of_m_plus_n_l188_18890

noncomputable def exponential_function (a x m n : ℝ) : ℝ :=
  a^(x - m) + n - 3

theorem value_of_m_plus_n (a x m n y : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : exponential_function a 3 m n = 2) : m + n = 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_plus_n_l188_18890


namespace NUMINAMATH_GPT_find_incorrect_statement_l188_18867

variable (q n x y : ℚ)

theorem find_incorrect_statement :
  (∀ q, q < -1 → q < 1/q) ∧
  (∀ n, n ≥ 0 → -n ≥ n) ∧
  (∀ x, x < 0 → x^3 < x) ∧
  (∀ y, y < 0 → y^2 > y) →
  (∃ x, x < 0 ∧ ¬ (x^3 < x)) :=
by
  sorry

end NUMINAMATH_GPT_find_incorrect_statement_l188_18867


namespace NUMINAMATH_GPT_children_eating_porridge_today_l188_18825

theorem children_eating_porridge_today
  (eat_every_day : ℕ)
  (eat_every_other_day : ℕ)
  (ate_yesterday : ℕ) :
  eat_every_day = 5 →
  eat_every_other_day = 7 →
  ate_yesterday = 9 →
  (eat_every_day + (eat_every_other_day - (ate_yesterday - eat_every_day)) = 8) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_children_eating_porridge_today_l188_18825


namespace NUMINAMATH_GPT_find_a_b_and_range_of_c_l188_18895

noncomputable def f (x a b c : ℝ) : ℝ := x^3 - a * x^2 + b * x + c

theorem find_a_b_and_range_of_c (c : ℝ) (h1 : ∀ x, 3 * x^2 - 2 * 3 * x - 9 = 0 → x = -1 ∨ x = 3)
    (h2 : ∀ x, x ∈ Set.Icc (-2 : ℝ) 6 → f x 3 (-9) c < c^2 + 4 * c) : 
    (a = 3 ∧ b = -9) ∧ (c > 6 ∨ c < -9) := by
  sorry

end NUMINAMATH_GPT_find_a_b_and_range_of_c_l188_18895


namespace NUMINAMATH_GPT_sally_combinations_l188_18898

theorem sally_combinations :
  let wall_colors := 4
  let flooring_types := 3
  wall_colors * flooring_types = 12 := by
  sorry

end NUMINAMATH_GPT_sally_combinations_l188_18898


namespace NUMINAMATH_GPT_sin_of_angle_l188_18846

theorem sin_of_angle (θ : ℝ) (h : Real.cos (θ + Real.pi) = -1/3) :
  Real.sin (2*θ + Real.pi/2) = -7/9 :=
by
  sorry

end NUMINAMATH_GPT_sin_of_angle_l188_18846


namespace NUMINAMATH_GPT_intersection_complement_l188_18818

open Set

noncomputable def N := {x : ℕ | true}

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}
def C_N (B : Set ℕ) : Set ℕ := {n ∈ N | n ∉ B}

theorem intersection_complement :
  A ∩ (C_N B) = {1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l188_18818


namespace NUMINAMATH_GPT_D_coordinates_l188_18880

namespace Parallelogram

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 1, y := 2 }
def C : Point := { x := 3, y := 1 }

theorem D_coordinates :
  ∃ D : Point, D = { x := 2, y := -1 } ∧ ∀ A B C D : Point, 
    (B.x - A.x, B.y - A.y) = (D.x - C.x, D.y - C.y) := by
  sorry

end Parallelogram

end NUMINAMATH_GPT_D_coordinates_l188_18880


namespace NUMINAMATH_GPT_part_I_part_II_l188_18841

noncomputable def f (x : ℝ) := Real.cos (x + Real.pi / 4)

-- Part I
theorem part_I : f (Real.pi / 6) + f (-Real.pi / 6) = Real.sqrt 6 / 2 :=
by
  sorry

-- Part II
theorem part_II (x : ℝ) (h : f x = Real.sqrt 2 / 3) : Real.sin (2 * x) = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l188_18841


namespace NUMINAMATH_GPT_value_of_b_l188_18838

theorem value_of_b (a b : ℕ) (r : ℝ) (h₁ : a = 2020) (h₂ : r = a / b) (h₃ : r = 0.5) : b = 4040 := 
by
  -- Hint: The proof takes steps to transform the conditions using basic algebraic manipulations.
  sorry

end NUMINAMATH_GPT_value_of_b_l188_18838


namespace NUMINAMATH_GPT_divisibility_by_7_l188_18860

theorem divisibility_by_7 (n : ℕ) (h : 0 < n) : 7 ∣ (3 ^ (2 * n + 2) - 2 ^ (n + 1)) :=
sorry

end NUMINAMATH_GPT_divisibility_by_7_l188_18860


namespace NUMINAMATH_GPT_find_first_parrot_weight_l188_18800

def cats_weights := [7, 10, 13, 15]
def cats_sum := List.sum cats_weights
def dog1 := cats_sum - 2
def dog2 := cats_sum + 7
def dog3 := (dog1 + dog2) / 2
def dogs_sum := dog1 + dog2 + dog3
def total_parrots_weight := 2 / 3 * dogs_sum

noncomputable def parrot1 := 2 / 5 * total_parrots_weight
noncomputable def parrot2 := 3 / 5 * total_parrots_weight

theorem find_first_parrot_weight : parrot1 = 38 :=
by
  sorry

end NUMINAMATH_GPT_find_first_parrot_weight_l188_18800


namespace NUMINAMATH_GPT_rational_numbers_satisfying_conditions_l188_18824

theorem rational_numbers_satisfying_conditions :
  (∃ n : ℕ, n = 166 ∧ ∀ (m : ℚ),
  abs m < 500 → (∃ x : ℤ, 3 * x^2 + m * x + 25 = 0) ↔ n = 166)
:=
sorry

end NUMINAMATH_GPT_rational_numbers_satisfying_conditions_l188_18824


namespace NUMINAMATH_GPT_coprime_divisible_l188_18885

theorem coprime_divisible (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : a ∣ b * c) : a ∣ c :=
by
  sorry

end NUMINAMATH_GPT_coprime_divisible_l188_18885


namespace NUMINAMATH_GPT_length_of_CB_l188_18887

noncomputable def length_CB (CD DA CF : ℕ) (DF_parallel_AB : Prop) := 9 * (CD + DA) / CD

theorem length_of_CB {CD DA CF : ℕ} (DF_parallel_AB : Prop):
  CD = 3 → DA = 12 → CF = 9 → CB = 9 * 5 := by
  sorry

end NUMINAMATH_GPT_length_of_CB_l188_18887


namespace NUMINAMATH_GPT_three_at_five_l188_18875

def op_at (a b : ℤ) : ℤ := 3 * a - 3 * b

theorem three_at_five : op_at 3 5 = -6 :=
by
  sorry

end NUMINAMATH_GPT_three_at_five_l188_18875


namespace NUMINAMATH_GPT_probability_of_Y_l188_18831

variable (P_X : ℝ) (P_X_and_Y : ℝ) (P_Y : ℝ)

theorem probability_of_Y (h1 : P_X = 1 / 7)
                         (h2 : P_X_and_Y = 0.031746031746031744) :
  P_Y = 0.2222222222222222 :=
sorry

end NUMINAMATH_GPT_probability_of_Y_l188_18831


namespace NUMINAMATH_GPT_find_y_l188_18821

variable (a b y : ℝ)
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(3 * b) = a^b * y^b)

theorem find_y : y = 27 * a^2 :=
  by sorry

end NUMINAMATH_GPT_find_y_l188_18821


namespace NUMINAMATH_GPT_book_original_price_l188_18852

noncomputable def original_price : ℝ := 420 / 1.40

theorem book_original_price (new_price : ℝ) (percentage_increase : ℝ) : 
  new_price = 420 → percentage_increase = 0.40 → original_price = 300 :=
by
  intros h1 h2
  exact sorry

end NUMINAMATH_GPT_book_original_price_l188_18852


namespace NUMINAMATH_GPT_find_x_y_sum_squared_l188_18802

theorem find_x_y_sum_squared (x y : ℝ) (h1 : x * y = 6) (h2 : (1 / x^2) + (1 / y^2) = 7) (h3 : x - y = Real.sqrt 10) :
  (x + y)^2 = 264 := sorry

end NUMINAMATH_GPT_find_x_y_sum_squared_l188_18802


namespace NUMINAMATH_GPT_facemasks_per_box_l188_18848

theorem facemasks_per_box (x : ℝ) :
  (3 * x * 0.50) - 15 = 15 → x = 20 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_facemasks_per_box_l188_18848


namespace NUMINAMATH_GPT_race_distance_l188_18844

theorem race_distance 
  (D : ℝ) 
  (A_time : ℝ) (B_time : ℝ) 
  (A_beats_B_by : ℝ) 
  (A_time_eq : A_time = 36)
  (B_time_eq : B_time = 45)
  (A_beats_B_by_eq : A_beats_B_by = 24) :
  ((D / A_time) * B_time = D + A_beats_B_by) -> D = 24 := 
by 
  sorry

end NUMINAMATH_GPT_race_distance_l188_18844


namespace NUMINAMATH_GPT_students_per_row_l188_18826

theorem students_per_row (x : ℕ) : 45 = 11 * x + 1 → x = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_students_per_row_l188_18826


namespace NUMINAMATH_GPT_max_annual_profit_l188_18813

noncomputable def R (x : ℝ) : ℝ :=
  if x < 40 then 10 * x^2 + 300 * x
  else (901 * x^2 - 9450 * x + 10000) / x

noncomputable def W (x : ℝ) : ℝ :=
  if x < 40 then -10 * x^2 + 600 * x - 260
  else -x + 9190 - 10000 / x

theorem max_annual_profit : ∃ x : ℝ, W 100 = 8990 :=
by {
  use 100,
  sorry
}

end NUMINAMATH_GPT_max_annual_profit_l188_18813


namespace NUMINAMATH_GPT_sabrina_cookies_l188_18856

theorem sabrina_cookies :
  let S0 : ℕ := 28
  let S1 : ℕ := S0 - 10
  let S2 : ℕ := S1 + 3 * 10
  let S3 : ℕ := S2 - S2 / 3
  let S4 : ℕ := S3 + 16 / 4
  let S5 : ℕ := S4 - S4 / 2
  S5 = 18 := 
by
  -- begin proof here
  sorry

end NUMINAMATH_GPT_sabrina_cookies_l188_18856


namespace NUMINAMATH_GPT_smallest_x_for_perfect_cube_l188_18884

theorem smallest_x_for_perfect_cube :
  ∃ (x : ℕ) (h : x > 0), x = 36 ∧ (∃ (k : ℕ), 1152 * x = k ^ 3) := by
  sorry

end NUMINAMATH_GPT_smallest_x_for_perfect_cube_l188_18884


namespace NUMINAMATH_GPT_gcd_fact_8_10_l188_18872

theorem gcd_fact_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = 40320 := by
  -- No proof needed
  sorry

end NUMINAMATH_GPT_gcd_fact_8_10_l188_18872


namespace NUMINAMATH_GPT_arithmetic_geometric_progressions_l188_18805

theorem arithmetic_geometric_progressions (a b : ℕ → ℕ) (d r : ℕ) 
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = r * b n)
  (h_comm_ratio : r = 2)
  (h_eq1 : a 1 + d - 2 * (b 1) = a 1 + 2 * d - 4 * (b 1))
  (h_eq2 : a 1 + d - 2 * (b 1) = 8 * (b 1) - (a 1 + 3 * d)) :
  (a 1 = b 1) ∧ (∃ n, ∀ k, 1 ≤ k ∧ k ≤ 10 → (b (k + 1) = a (1 + n * d) + a 1)) := by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_progressions_l188_18805


namespace NUMINAMATH_GPT_deceased_member_income_l188_18877

theorem deceased_member_income (A B C : ℝ) (h1 : (A + B + C) / 3 = 735) (h2 : (A + B) / 2 = 650) : 
  C = 905 :=
by
  sorry

end NUMINAMATH_GPT_deceased_member_income_l188_18877


namespace NUMINAMATH_GPT_ratio_of_area_to_breadth_is_15_l188_18851

-- Definitions for our problem
def breadth := 5
def length := 15 -- since l - b = 10 and b = 5

-- Given conditions
axiom area_is_ktimes_breadth (k : ℝ) : length * breadth = k * breadth
axiom length_breadth_difference : length - breadth = 10

-- The proof statement
theorem ratio_of_area_to_breadth_is_15 : (length * breadth) / breadth = 15 := by
  sorry

end NUMINAMATH_GPT_ratio_of_area_to_breadth_is_15_l188_18851


namespace NUMINAMATH_GPT_range_of_x_plus_one_over_x_l188_18830

theorem range_of_x_plus_one_over_x (x : ℝ) (h : x < 0) : x + 1/x ≤ -2 := by
  sorry

end NUMINAMATH_GPT_range_of_x_plus_one_over_x_l188_18830


namespace NUMINAMATH_GPT_tom_catches_up_in_60_minutes_l188_18873

-- Definitions of the speeds and initial distance
def lucy_speed : ℝ := 4  -- Lucy's speed in miles per hour
def tom_speed : ℝ := 6   -- Tom's speed in miles per hour
def initial_distance : ℝ := 2  -- Initial distance between Tom and Lucy in miles

-- Conclusion that needs to be proved
theorem tom_catches_up_in_60_minutes :
  (initial_distance / (tom_speed - lucy_speed)) * 60 = 60 :=
by
  sorry

end NUMINAMATH_GPT_tom_catches_up_in_60_minutes_l188_18873


namespace NUMINAMATH_GPT_pow_mod_eq_l188_18854

theorem pow_mod_eq : (6 ^ 2040) % 50 = 26 := by
  sorry

end NUMINAMATH_GPT_pow_mod_eq_l188_18854


namespace NUMINAMATH_GPT_compute_alpha_l188_18828

-- Define the main hypothesis with complex numbers
variable (α γ : ℂ)
variable (h1 : γ = 4 + 3 * Complex.I)
variable (h2 : ∃r1 r2: ℝ, r1 > 0 ∧ r2 > 0 ∧ (α + γ = r1) ∧ (Complex.I * (α - 3 * γ) = r2))

-- The main theorem
theorem compute_alpha : α = 12 + 3 * Complex.I :=
by
  sorry

end NUMINAMATH_GPT_compute_alpha_l188_18828


namespace NUMINAMATH_GPT_solve_for_x_l188_18881

theorem solve_for_x (x : ℝ) : 3^(3 * x) = Real.sqrt 81 -> x = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l188_18881


namespace NUMINAMATH_GPT_min_value_of_F_on_negative_half_l188_18853

variable (f g : ℝ → ℝ)
variable (a b : ℝ)

def F (x : ℝ) := a * f x + b * g x + 2

def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem min_value_of_F_on_negative_half
  (h_f : is_odd f) (h_g : is_odd g)
  (max_F_positive_half : ∃ x, x > 0 ∧ F f g a b x = 5) :
  ∃ x, x < 0 ∧ F f g a b x = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_of_F_on_negative_half_l188_18853


namespace NUMINAMATH_GPT_find_X_l188_18899

theorem find_X (X : ℝ) 
  (h : 2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1600.0000000000002) : 
  X = 1.25 := 
sorry

end NUMINAMATH_GPT_find_X_l188_18899


namespace NUMINAMATH_GPT_solve_equation_l188_18896

theorem solve_equation : ∀ (x : ℝ), x ≠ 2 → -2 * x^2 = (4 * x + 2) / (x - 2) → x = 1 :=
by
  intros x hx h_eq
  sorry

end NUMINAMATH_GPT_solve_equation_l188_18896


namespace NUMINAMATH_GPT_correct_operation_l188_18871

variable (a b : ℝ)

theorem correct_operation :
  -a^6 / a^3 = -a^3 := by
  sorry

end NUMINAMATH_GPT_correct_operation_l188_18871


namespace NUMINAMATH_GPT_polynomial_even_or_odd_polynomial_divisible_by_3_l188_18845

theorem polynomial_even_or_odd (p q : ℤ) :
  (∀ x : ℤ, (x^2 + p * x + q) % 2 = 0 ↔ (q % 2 = 0) ∧ (p % 2 = 1)) ∧
  (∀ x : ℤ, (x^2 + p * x + q) % 2 = 1 ↔ (q % 2 = 1) ∧ (p % 2 = 1)) := 
sorry

theorem polynomial_divisible_by_3 (p q : ℤ) :
  (∀ x : ℤ, (x^3 + p * x + q) % 3 = 0) ↔ (q % 3 = 0) ∧ (p % 3 = 2) := 
sorry

end NUMINAMATH_GPT_polynomial_even_or_odd_polynomial_divisible_by_3_l188_18845


namespace NUMINAMATH_GPT_spencer_total_distance_l188_18839

-- Definitions for the given conditions
def distance_house_to_library : ℝ := 0.3
def distance_library_to_post_office : ℝ := 0.1
def distance_post_office_to_home : ℝ := 0.4

-- Define the total distance based on the given conditions
def total_distance : ℝ := distance_house_to_library + distance_library_to_post_office + distance_post_office_to_home

-- Statement to prove
theorem spencer_total_distance : total_distance = 0.8 := by
  sorry

end NUMINAMATH_GPT_spencer_total_distance_l188_18839


namespace NUMINAMATH_GPT_abs_value_expression_l188_18849

theorem abs_value_expression : abs (3 * Real.pi - abs (3 * Real.pi - 10)) = 6 * Real.pi - 10 :=
by sorry

end NUMINAMATH_GPT_abs_value_expression_l188_18849


namespace NUMINAMATH_GPT_problem1_problem2_l188_18874

variables (a b : ℝ)

theorem problem1 : ((a^2)^3 / (-a)^2) = a^4 :=
sorry

theorem problem2 : ((a + 2 * b) * (a + b) - 3 * a * (a + b)) = -2 * a^2 + 2 * b^2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l188_18874


namespace NUMINAMATH_GPT_min_value_f_l188_18836

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x)^2 / (Real.cos x * Real.sin x - (Real.sin x)^2)

theorem min_value_f :
  ∃ x : ℝ, 0 < x ∧ x < Real.pi / 4 ∧ f x = 4 := 
sorry

end NUMINAMATH_GPT_min_value_f_l188_18836


namespace NUMINAMATH_GPT_simplify_expression_l188_18866

theorem simplify_expression (y : ℝ) : (y - 2) ^ 2 + 2 * (y - 2) * (4 + y) + (4 + y) ^ 2 = 4 * (y + 1) ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l188_18866


namespace NUMINAMATH_GPT_z_in_third_quadrant_l188_18857

def i := Complex.I

def z := i + 2 * (i^2) + 3 * (i^3)

theorem z_in_third_quadrant : 
    let z_real := Complex.re z
    let z_imag := Complex.im z
    z_real < 0 ∧ z_imag < 0 :=
by
  sorry

end NUMINAMATH_GPT_z_in_third_quadrant_l188_18857


namespace NUMINAMATH_GPT_inequality_for_five_real_numbers_l188_18829

open Real

theorem inequality_for_five_real_numbers
  (a1 a2 a3 a4 a5 : ℝ)
  (h1 : 1 < a1)
  (h2 : 1 < a2)
  (h3 : 1 < a3)
  (h4 : 1 < a4)
  (h5 : 1 < a5) :
  16 * (a1 * a2 * a3 * a4 * a5 + 1) ≥ (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) * (1 + a5) := 
sorry

end NUMINAMATH_GPT_inequality_for_five_real_numbers_l188_18829


namespace NUMINAMATH_GPT_fixed_point_coordinates_l188_18891

theorem fixed_point_coordinates (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, 4) ∧ ∀ x, P = (x, a^(x-1) + 3) :=
by
  use (1, 4)
  sorry

end NUMINAMATH_GPT_fixed_point_coordinates_l188_18891


namespace NUMINAMATH_GPT_correct_flowchart_requirement_l188_18855

def flowchart_requirement (option : String) : Prop := 
  option = "From left to right, from top to bottom" ∨
  option = "From right to left, from top to bottom" ∨
  option = "From left to right, from bottom to top" ∨
  option = "From right to left, from bottom to top"

theorem correct_flowchart_requirement : 
  (∀ option, flowchart_requirement option → option = "From left to right, from top to bottom") :=
by
  sorry

end NUMINAMATH_GPT_correct_flowchart_requirement_l188_18855


namespace NUMINAMATH_GPT_find_x_l188_18819

noncomputable def isCorrectValue (x : ℝ) : Prop :=
  ⌊x⌋ + x = 13.4

theorem find_x (x : ℝ) (h : isCorrectValue x) : x = 6.4 :=
  sorry

end NUMINAMATH_GPT_find_x_l188_18819


namespace NUMINAMATH_GPT_evaluate_series_l188_18868

theorem evaluate_series : 1 + (1 / 2) + (1 / 4) + (1 / 8) = 15 / 8 := by
  sorry

end NUMINAMATH_GPT_evaluate_series_l188_18868


namespace NUMINAMATH_GPT_probability_even_heads_after_60_flips_l188_18834

noncomputable def P_n (n : ℕ) : ℝ :=
  if n = 0 then 1
  else (3 / 4) - (1 / 2) * P_n (n - 1)

theorem probability_even_heads_after_60_flips :
  P_n 60 = 1 / 2 * (1 + 1 / 2^60) :=
sorry

end NUMINAMATH_GPT_probability_even_heads_after_60_flips_l188_18834


namespace NUMINAMATH_GPT_range_of_x_sq_add_y_sq_l188_18861

theorem range_of_x_sq_add_y_sq (x y : ℝ) (h : x^2 + y^2 = 4 * x) : 
  ∃ (a b : ℝ), a ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ b ∧ a = 0 ∧ b = 16 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_sq_add_y_sq_l188_18861


namespace NUMINAMATH_GPT_estimate_flight_time_around_earth_l188_18815

theorem estimate_flight_time_around_earth 
  (radius : ℝ) 
  (speed : ℝ)
  (h_radius : radius = 6000) 
  (h_speed : speed = 600) 
  : abs (20 * Real.pi - 63) < 1 :=
by
  sorry

end NUMINAMATH_GPT_estimate_flight_time_around_earth_l188_18815


namespace NUMINAMATH_GPT_parabola_vertex_l188_18862

theorem parabola_vertex (x y : ℝ) :
  (x^2 - 4 * x + 3 * y + 8 = 0) → (x, y) = (2, -4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l188_18862


namespace NUMINAMATH_GPT_contrapositive_example_l188_18832

theorem contrapositive_example (a b : ℝ) :
  (a > b → a - 1 > b - 2) ↔ (a - 1 ≤ b - 2 → a ≤ b) := 
by
  sorry

end NUMINAMATH_GPT_contrapositive_example_l188_18832


namespace NUMINAMATH_GPT_sum_greater_than_product_l188_18809

theorem sum_greater_than_product (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b > a * b) ↔ (a = 1 ∨ b = 1) := 
by { sorry }

end NUMINAMATH_GPT_sum_greater_than_product_l188_18809


namespace NUMINAMATH_GPT_quadrilateral_count_l188_18863

-- Define the number of points
def num_points := 9

-- Define the number of vertices in a quadrilateral
def vertices_in_quadrilateral := 4

-- Use a combination function to find the number of ways to choose 4 points out of 9
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem that asserts the number of quadrilaterals that can be formed
theorem quadrilateral_count : combination num_points vertices_in_quadrilateral = 126 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_quadrilateral_count_l188_18863


namespace NUMINAMATH_GPT_triangle_angle_sixty_degrees_l188_18806

theorem triangle_angle_sixty_degrees (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) : 
  ∃ (θ : ℝ), θ = 60 ∧ ∃ (a b c : ℝ), a * b * c ≠ 0 ∧ ∀ {α β γ : ℝ}, (a + b + c = α + β + γ + θ) := 
sorry

end NUMINAMATH_GPT_triangle_angle_sixty_degrees_l188_18806


namespace NUMINAMATH_GPT_only_natural_number_solution_l188_18803

theorem only_natural_number_solution (n : ℕ) :
  (∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = n * x * y * z) ↔ (n = 3) := 
sorry

end NUMINAMATH_GPT_only_natural_number_solution_l188_18803


namespace NUMINAMATH_GPT_min_value_expr_l188_18878

theorem min_value_expr (x y : ℝ) : 
  ∃ min_val, min_val = 2 ∧ min_val ≤ (x + y)^2 + (x - 1/y)^2 :=
sorry

end NUMINAMATH_GPT_min_value_expr_l188_18878


namespace NUMINAMATH_GPT_find_b_l188_18801

theorem find_b
  (b : ℝ)
  (hx : ∃ y : ℝ, 4 * 3 + 2 * y = b ∧ 3 * 3 + 4 * y = 3 * b) :
  b = -15 :=
sorry

end NUMINAMATH_GPT_find_b_l188_18801


namespace NUMINAMATH_GPT_Louie_monthly_payment_l188_18897

noncomputable def monthly_payment (P : ℕ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  let A := P * (1 + r) ^ n
  A / 3

theorem Louie_monthly_payment : 
  monthly_payment 1000 0.10 3 (3 / 12) = 444 := 
by
  -- computation and rounding
  sorry

end NUMINAMATH_GPT_Louie_monthly_payment_l188_18897


namespace NUMINAMATH_GPT_remainder_of_sum_l188_18869

theorem remainder_of_sum (a b c : ℕ) (h₁ : a % 15 = 11) (h₂ : b % 15 = 12) (h₃ : c % 15 = 13) : 
  (a + b + c) % 15 = 6 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l188_18869


namespace NUMINAMATH_GPT_find_prime_power_solutions_l188_18842

theorem find_prime_power_solutions (p n m : ℕ) (hp : Nat.Prime p) (hn : n > 0) (hm : m > 0) 
  (h : p^n + 144 = m^2) :
  (p = 2 ∧ n = 9 ∧ m = 36) ∨ (p = 3 ∧ n = 4 ∧ m = 27) :=
by sorry

end NUMINAMATH_GPT_find_prime_power_solutions_l188_18842


namespace NUMINAMATH_GPT_expected_babies_is_1008_l188_18843

noncomputable def babies_expected_after_loss
  (num_kettles : ℕ)
  (pregnancies_per_kettle : ℕ)
  (babies_per_pregnancy : ℕ)
  (loss_percentage : ℤ) : ℤ :=
  let total_babies := num_kettles * pregnancies_per_kettle * babies_per_pregnancy
  let survival_rate := (100 - loss_percentage) / 100
  total_babies * survival_rate

theorem expected_babies_is_1008 :
  babies_expected_after_loss 12 20 6 30 = 1008 :=
by
  sorry

end NUMINAMATH_GPT_expected_babies_is_1008_l188_18843


namespace NUMINAMATH_GPT_min_unattainable_score_l188_18811

theorem min_unattainable_score : ∀ (score : ℕ), (¬ ∃ (a b c : ℕ), 
  (a = 1 ∨ a = 3 ∨ a = 8 ∨ a = 12 ∨ a = 0) ∧ 
  (b = 1 ∨ b = 3 ∨ b = 8 ∨ b = 12 ∨ b = 0) ∧ 
  (c = 1 ∨ c = 3 ∨ c = 8 ∨ c = 12 ∨ c = 0) ∧ 
  score = a + b + c) ↔ score = 22 := 
by
  sorry

end NUMINAMATH_GPT_min_unattainable_score_l188_18811


namespace NUMINAMATH_GPT_cubic_inequality_l188_18888

theorem cubic_inequality :
  {x : ℝ | x^3 - 12*x^2 + 47*x - 60 < 0} = {x : ℝ | 3 < x ∧ x < 5} :=
by
  sorry

end NUMINAMATH_GPT_cubic_inequality_l188_18888


namespace NUMINAMATH_GPT_find_initial_money_l188_18876

def initial_money (s1 s2 s3 : ℝ) : ℝ :=
  let after_store_1 := s1 - (0.4 * s1 + 4)
  let after_store_2 := after_store_1 - (0.5 * after_store_1 + 5)
  let after_store_3 := after_store_2 - (0.6 * after_store_2 + 6)
  after_store_3

theorem find_initial_money (s1 s2 s3 : ℝ) (hs3 : initial_money s1 s2 s3 = 2) : s1 = 90 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_initial_money_l188_18876


namespace NUMINAMATH_GPT_part1_part2_l188_18850

-- Prove Part (1)
theorem part1 (M : ℕ) (N : ℕ) (h : M = 9) (h2 : N - 4 + 6 = M) : N = 7 :=
sorry

-- Prove Part (2)
theorem part2 (M : ℕ) (h : M = 9) : M - 4 = 5 ∨ M + 4 = 13 :=
sorry

end NUMINAMATH_GPT_part1_part2_l188_18850


namespace NUMINAMATH_GPT_part_I_min_value_part_II_nonexistence_l188_18817

theorem part_I_min_value (a b : ℝ) (hab : a > 0 ∧ b > 0 ∧ a + 4 * b = (a * b)^(3/2)) : a^2 + 16 * b^2 ≥ 32 :=
by
  sorry

theorem part_II_nonexistence (a b : ℝ) (hab : a > 0 ∧ b > 0 ∧ a + 4 * b = (a * b)^(3/2)) : ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 6 :=
by
  sorry

end NUMINAMATH_GPT_part_I_min_value_part_II_nonexistence_l188_18817


namespace NUMINAMATH_GPT_combined_distance_correct_l188_18859

-- Define the conditions
def wheelA_rotations_per_minute := 20
def wheelA_distance_per_rotation_cm := 35
def wheelB_rotations_per_minute := 30
def wheelB_distance_per_rotation_cm := 50

-- Calculate distances in meters
def wheelA_distance_per_minute_m :=
  (wheelA_rotations_per_minute * wheelA_distance_per_rotation_cm) / 100

def wheelB_distance_per_minute_m :=
  (wheelB_rotations_per_minute * wheelB_distance_per_rotation_cm) / 100

def wheelA_distance_per_hour_m :=
  wheelA_distance_per_minute_m * 60

def wheelB_distance_per_hour_m :=
  wheelB_distance_per_minute_m * 60

def combined_distance_per_hour_m :=
  wheelA_distance_per_hour_m + wheelB_distance_per_hour_m

theorem combined_distance_correct : combined_distance_per_hour_m = 1320 := by
  -- skip the proof here with sorry
  sorry

end NUMINAMATH_GPT_combined_distance_correct_l188_18859


namespace NUMINAMATH_GPT_ants_need_more_hours_l188_18892

theorem ants_need_more_hours (initial_sugar : ℕ) (removal_rate : ℕ) (hours_spent : ℕ) : 
  initial_sugar = 24 ∧ removal_rate = 4 ∧ hours_spent = 3 → 
  (initial_sugar - removal_rate * hours_spent) / removal_rate = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ants_need_more_hours_l188_18892


namespace NUMINAMATH_GPT_function_always_negative_iff_l188_18865

theorem function_always_negative_iff (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ -4 < k ∧ k ≤ 0 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_function_always_negative_iff_l188_18865


namespace NUMINAMATH_GPT_intersection_eq_l188_18804

-- Given conditions
def M : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def N : Set ℝ := { x | x > 1 }

-- Statement of the problem to be proved
theorem intersection_eq : M ∩ N = { x | 1 < x ∧ x < 3 } :=
sorry

end NUMINAMATH_GPT_intersection_eq_l188_18804


namespace NUMINAMATH_GPT_stamp_blocks_inequalities_l188_18889

noncomputable def b (n : ℕ) : ℕ := sorry

theorem stamp_blocks_inequalities (n : ℕ) (m : ℕ) (hn : 0 < n) :
  ∃ c d : ℝ, c = 2 / 7 ∧ d = (4 * m^2 + 4 * m + 40) / 5 ∧
    (1 / 7 : ℝ) * n^2 - c * n ≤ b n ∧ 
    b n ≤ (1 / 5 : ℝ) * n^2 + d * n := 
  sorry

end NUMINAMATH_GPT_stamp_blocks_inequalities_l188_18889


namespace NUMINAMATH_GPT_compute_division_l188_18810

variable (a b c : ℕ)
variable (ha : a = 3)
variable (hb : b = 2)
variable (hc : c = 2)

theorem compute_division : (c * a^3 + c * b^3) / (a^2 - a * b + b^2) = 10 := by
  sorry

end NUMINAMATH_GPT_compute_division_l188_18810


namespace NUMINAMATH_GPT_solve_for_x_l188_18816

theorem solve_for_x :
  ∃ x : ℝ, ((17.28 / x) / (3.6 * 0.2)) = 2 ∧ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l188_18816


namespace NUMINAMATH_GPT_negation_of_universal_l188_18879

theorem negation_of_universal (h : ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) : ∃ x : ℝ, x^2 + 2 * x + 5 = 0 :=
sorry

end NUMINAMATH_GPT_negation_of_universal_l188_18879


namespace NUMINAMATH_GPT_find_m_n_sum_l188_18858

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem find_m_n_sum (x₀ : ℝ) (m n : ℤ) 
  (hmn_adj : n = m + 1) 
  (hx₀_zero : f x₀ = 0) 
  (hx₀_interval : (m : ℝ) < x₀ ∧ x₀ < (n : ℝ)) :
  m + n = 1 :=
sorry

end NUMINAMATH_GPT_find_m_n_sum_l188_18858


namespace NUMINAMATH_GPT_simplify_expression_l188_18827

theorem simplify_expression (a b : ℝ) : -3 * (a - b) + (2 * a - 3 * b) = -a :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l188_18827


namespace NUMINAMATH_GPT_bob_grade_is_35_l188_18870

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

theorem bob_grade_is_35 : bob_grade = 35 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_bob_grade_is_35_l188_18870


namespace NUMINAMATH_GPT_smallest_n_for_factorable_quadratic_l188_18808

open Int

theorem smallest_n_for_factorable_quadratic : ∃ n : ℤ, (∀ A B : ℤ, 3 * A * B = 72 → 3 * B + A = n) ∧ n = 35 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_factorable_quadratic_l188_18808


namespace NUMINAMATH_GPT_length_of_BC_l188_18820

-- Definitions of given conditions
def AB : ℝ := 4
def AC : ℝ := 3
def dot_product_AC_BC : ℝ := 1

-- Hypothesis used in the problem
axiom nonneg_AC (AC : ℝ) : AC ≥ 0
axiom nonneg_AB (AB : ℝ) : AB ≥ 0

-- Statement to be proved
theorem length_of_BC (AB AC dot_product_AC_BC : ℝ)
  (h1 : AB = 4) (h2 : AC = 3) (h3 : dot_product_AC_BC = 1) : exists (BC : ℝ), BC = 3 := by
  sorry

end NUMINAMATH_GPT_length_of_BC_l188_18820


namespace NUMINAMATH_GPT_rectangle_area_at_stage_8_l188_18847

-- Declare constants for the conditions.
def square_side_length : ℕ := 4
def number_of_stages : ℕ := 8
def area_of_single_square : ℕ := square_side_length * square_side_length

-- The statement to prove
theorem rectangle_area_at_stage_8 : number_of_stages * area_of_single_square = 128 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_at_stage_8_l188_18847


namespace NUMINAMATH_GPT_travel_time_proportion_l188_18882

theorem travel_time_proportion (D V : ℝ) (hV_pos : V > 0) :
  let Time1 := D / (16 * V)
  let Time2 := 3 * D / (4 * V)
  let TimeTotal := Time1 + Time2
  (Time1 / TimeTotal) = 1 / 13 :=
by
  sorry

end NUMINAMATH_GPT_travel_time_proportion_l188_18882


namespace NUMINAMATH_GPT_decipher_numbers_l188_18833

variable (K I S : Nat)

theorem decipher_numbers
  (h1: 1 ≤ K ∧ K < 5)
  (h2: I ≠ 0)
  (h3: I ≠ K)
  (h_eq: K * 100 + I * 10 + S + K * 10 + S * 10 + I = I * 100 + S * 10 + K):
  (K, I, S) = (4, 9, 5) :=
by sorry

end NUMINAMATH_GPT_decipher_numbers_l188_18833


namespace NUMINAMATH_GPT_num_of_nickels_l188_18837

theorem num_of_nickels (n : ℕ) (h1 : n = 17) (h2 : (17 * n) - 1 = 18 * (n - 1)) : n = 17 → 17 * n = 289 → ∃ k, k = 2 :=
by 
  intros hn hv
  sorry

end NUMINAMATH_GPT_num_of_nickels_l188_18837


namespace NUMINAMATH_GPT_prob_at_least_one_head_is_7_over_8_l188_18886

-- Define the event and probability calculation
def probability_of_tails_all_three_tosses : ℚ :=
  (1 / 2) ^ 3

def probability_of_at_least_one_head : ℚ :=
  1 - probability_of_tails_all_three_tosses

-- Prove the probability of at least one head is 7/8
theorem prob_at_least_one_head_is_7_over_8 : probability_of_at_least_one_head = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_prob_at_least_one_head_is_7_over_8_l188_18886


namespace NUMINAMATH_GPT_final_price_correct_l188_18840

-- Definitions that follow the given conditions
def initial_price : ℝ := 150
def increase_percentage_year1 : ℝ := 1.5
def decrease_percentage_year2 : ℝ := 0.3

-- Compute intermediate values
noncomputable def price_end_year1 : ℝ := initial_price + (increase_percentage_year1 * initial_price)
noncomputable def price_end_year2 : ℝ := price_end_year1 - (decrease_percentage_year2 * price_end_year1)

-- The final theorem stating the price at the end of the second year
theorem final_price_correct : price_end_year2 = 262.5 := by
  sorry

end NUMINAMATH_GPT_final_price_correct_l188_18840


namespace NUMINAMATH_GPT_inverse_f_of_7_l188_18864

def f (x : ℝ) : ℝ := 2 * x^2 + 3

theorem inverse_f_of_7:
  ∀ y : ℝ, f (7) = y ↔ y = 101 :=
by
  sorry

end NUMINAMATH_GPT_inverse_f_of_7_l188_18864


namespace NUMINAMATH_GPT_translate_B_to_origin_l188_18893

structure Point where
  x : ℝ
  y : ℝ

def translate_right (p : Point) (d : ℕ) : Point := 
  { x := p.x + d, y := p.y }

theorem translate_B_to_origin :
  ∀ (A B : Point) (d : ℕ),
  A = { x := -4, y := 0 } →
  B = { x := 0, y := 2 } →
  (translate_right A d).x = 0 →
  translate_right B d = { x := 4, y := 2 } :=
by
  intros A B d hA hB hA'
  sorry

end NUMINAMATH_GPT_translate_B_to_origin_l188_18893


namespace NUMINAMATH_GPT_cordelia_bleaching_l188_18883

noncomputable def bleaching_time (B : ℝ) : Prop :=
  B + 4 * B + B / 3 = 10

theorem cordelia_bleaching : ∃ B : ℝ, bleaching_time B ∧ B = 1.875 :=
by {
  sorry
}

end NUMINAMATH_GPT_cordelia_bleaching_l188_18883


namespace NUMINAMATH_GPT_total_pets_count_l188_18823

/-- Taylor and his six friends have a total of 45 pets, given the specified conditions about the number of each type of pet they have. -/
theorem total_pets_count
  (Taylor_cats : ℕ := 4)
  (Friend1_pets : ℕ := 8 * 3)
  (Friend2_dogs : ℕ := 3)
  (Friend2_birds : ℕ := 1)
  (Friend3_dogs : ℕ := 5)
  (Friend3_cats : ℕ := 2)
  (Friend4_reptiles : ℕ := 2)
  (Friend4_birds : ℕ := 3)
  (Friend4_cats : ℕ := 1) :
  Taylor_cats + Friend1_pets + Friend2_dogs + Friend2_birds + Friend3_dogs + Friend3_cats + Friend4_reptiles + Friend4_birds + Friend4_cats = 45 :=
sorry

end NUMINAMATH_GPT_total_pets_count_l188_18823


namespace NUMINAMATH_GPT_a_minus_b_eq_one_l188_18835

variable (a b : ℕ)

theorem a_minus_b_eq_one
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : Real.sqrt 18 = a * Real.sqrt 2) 
  (h4 : Real.sqrt 8 = 2 * Real.sqrt b) : 
  a - b = 1 := 
sorry

end NUMINAMATH_GPT_a_minus_b_eq_one_l188_18835
