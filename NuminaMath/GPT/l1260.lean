import Mathlib

namespace NUMINAMATH_GPT_sin_X_value_l1260_126057

variables (a b X : ℝ)

-- Conditions
def conditions :=
  (1/2 * a * b * Real.sin X = 100) ∧ (Real.sqrt (a * b) = 15)

theorem sin_X_value (h : conditions a b X) : Real.sin X = 8 / 9 := by
  sorry

end NUMINAMATH_GPT_sin_X_value_l1260_126057


namespace NUMINAMATH_GPT_number_when_added_by_5_is_30_l1260_126004

theorem number_when_added_by_5_is_30 (x: ℕ) (h: x - 10 = 15) : x + 5 = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_when_added_by_5_is_30_l1260_126004


namespace NUMINAMATH_GPT_f_at_1_over_11_l1260_126087

noncomputable def f : (ℝ → ℝ) := sorry

axiom f_domain : ∀ x, 0 < x → 0 < f x

axiom f_eq : ∀ x y, 0 < x → 0 < y → 10 * ((x + y) / (x * y)) = (f x) * (f y) - f (x * y) - 90

theorem f_at_1_over_11 : f (1 / 11) = 21 := by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_f_at_1_over_11_l1260_126087


namespace NUMINAMATH_GPT_find_constants_to_satisfy_equation_l1260_126033

-- Define the condition
def equation_condition (x : ℝ) (A B C : ℝ) :=
  -2 * x^2 + 5 * x - 6 = A * (x^2 + 1) + (B * x + C) * x

-- Define the proof problem as a Lean 4 statement
theorem find_constants_to_satisfy_equation (A B C : ℝ) :
  A = -6 ∧ B = 4 ∧ C = 5 ↔ ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 → equation_condition x A B C := 
by
  sorry

end NUMINAMATH_GPT_find_constants_to_satisfy_equation_l1260_126033


namespace NUMINAMATH_GPT_sin_cos_solution_count_l1260_126047

-- Statement of the problem
theorem sin_cos_solution_count : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin (3 * x) = Real.cos (x / 2)) ∧ s.card = 6 := by
  sorry

end NUMINAMATH_GPT_sin_cos_solution_count_l1260_126047


namespace NUMINAMATH_GPT_leo_third_part_time_l1260_126072

theorem leo_third_part_time :
  ∃ (T3 : ℕ), 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 3 → T = 25 * k) →
  T1 = 25 →
  T2 = 50 →
  Break1 = 10 →
  Break2 = 15 →
  TotalTime = 2 * 60 + 30 →
  (TotalTime - (T1 + Break1 + T2 + Break2) = T3) →
  T3 = 50 := 
sorry

end NUMINAMATH_GPT_leo_third_part_time_l1260_126072


namespace NUMINAMATH_GPT_max_marks_l1260_126022

theorem max_marks (M S : ℕ) :
  (267 + 45 = 312) ∧ (312 = (45 * M) / 100) ∧ (292 + 38 = 330) ∧ (330 = (50 * S) / 100) →
  (M + S = 1354) :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l1260_126022


namespace NUMINAMATH_GPT_greatest_natural_number_l1260_126095

theorem greatest_natural_number (n q r : ℕ) (h1 : n = 91 * q + r)
  (h2 : r = q^2) (h3 : r < 91) : n = 900 :=
sorry

end NUMINAMATH_GPT_greatest_natural_number_l1260_126095


namespace NUMINAMATH_GPT_Gina_gave_fraction_to_mom_l1260_126009

variable (M : ℝ)

theorem Gina_gave_fraction_to_mom :
  (∃ M, M + (1/8 : ℝ) * 400 + (1/5 : ℝ) * 400 + 170 = 400) →
  M / 400 = 1/4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Gina_gave_fraction_to_mom_l1260_126009


namespace NUMINAMATH_GPT_max_side_of_triangle_l1260_126036

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end NUMINAMATH_GPT_max_side_of_triangle_l1260_126036


namespace NUMINAMATH_GPT_find_age_of_B_l1260_126037

-- Define A and B as natural numbers (assuming ages are non-negative integers)
variables (A B : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := A + 10 = 2 * (B - 10)
def condition2 : Prop := A = B + 6

-- The goal is to prove that B = 36 given the conditions
theorem find_age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 36 :=
sorry

end NUMINAMATH_GPT_find_age_of_B_l1260_126037


namespace NUMINAMATH_GPT_problem_solution_l1260_126040

-- Definitions of odd function and given conditions.
variables {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) (h_eq : f 3 - f 2 = 1)

-- Proof statement of the math problem.
theorem problem_solution : f (-2) - f (-3) = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1260_126040


namespace NUMINAMATH_GPT_polygon_sides_l1260_126080

theorem polygon_sides (n : ℕ) 
  (H : (n * (n - 3)) / 2 = 3 * n) : n = 9 := 
sorry

end NUMINAMATH_GPT_polygon_sides_l1260_126080


namespace NUMINAMATH_GPT_min_x1_x2_squared_l1260_126046

theorem min_x1_x2_squared (x1 x2 m : ℝ) (hm : (m + 3)^2 ≥ 0) 
  (h_sum : x1 + x2 = -(m + 1)) 
  (h_prod : x1 * x2 = 2 * m - 2) : 
  (x1^2 + x2^2 = (m - 1)^2 + 4) ∧ ∃ m, m = 1 → x1^2 + x2^2 = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_x1_x2_squared_l1260_126046


namespace NUMINAMATH_GPT_sum_of_series_l1260_126005

noncomputable def sum_term (k : ℕ) : ℝ :=
  (7 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))

theorem sum_of_series : (∑' k : ℕ, sum_term (k + 1)) = 7 / 4 := by
  sorry

end NUMINAMATH_GPT_sum_of_series_l1260_126005


namespace NUMINAMATH_GPT_math_problem_l1260_126015

theorem math_problem (x y : ℕ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1260_126015


namespace NUMINAMATH_GPT_log_expression_value_l1260_126061

theorem log_expression_value (x : ℝ) (hx : x < 1) (h : (Real.log x / Real.log 10)^3 - 2 * (Real.log (x^3) / Real.log 10) = 150) :
  (Real.log x / Real.log 10)^4 - (Real.log (x^4) / Real.log 10) = 645 := 
sorry

end NUMINAMATH_GPT_log_expression_value_l1260_126061


namespace NUMINAMATH_GPT_bread_cost_l1260_126064

theorem bread_cost (H C B : ℕ) (h₁ : H = 150) (h₂ : C = 200) (h₃ : H + B = C) : B = 50 :=
by
  sorry

end NUMINAMATH_GPT_bread_cost_l1260_126064


namespace NUMINAMATH_GPT_sum_consecutive_powers_of_2_divisible_by_6_l1260_126028

theorem sum_consecutive_powers_of_2_divisible_by_6 (n : ℕ) :
  ∃ k : ℕ, 2^n + 2^(n+1) = 6 * k :=
sorry

end NUMINAMATH_GPT_sum_consecutive_powers_of_2_divisible_by_6_l1260_126028


namespace NUMINAMATH_GPT_circumcenter_coords_l1260_126075

-- Define the given points A, B, and C
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (-5, 1)
def C : ℝ × ℝ := (3, -5)

-- The target statement to prove
theorem circumcenter_coords :
  ∃ x y : ℝ, (x - 2)^2 + (y - 2)^2 = (x + 5)^2 + (y - 1)^2 ∧
             (x - 2)^2 + (y - 2)^2 = (x - 3)^2 + (y + 5)^2 ∧
             x = -1 ∧ y = -2 :=
by
  sorry

end NUMINAMATH_GPT_circumcenter_coords_l1260_126075


namespace NUMINAMATH_GPT_ab_value_l1260_126035

theorem ab_value (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 29) : a * b = 2 :=
by
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_ab_value_l1260_126035


namespace NUMINAMATH_GPT_no_positive_real_roots_l1260_126068

theorem no_positive_real_roots (x : ℝ) : (x^3 + 6 * x^2 + 11 * x + 6 = 0) → x < 0 :=
sorry

end NUMINAMATH_GPT_no_positive_real_roots_l1260_126068


namespace NUMINAMATH_GPT_value_of_3Y5_l1260_126089

def Y (a b : ℤ) : ℤ := b + 10 * a - a^2 - b^2

theorem value_of_3Y5 : Y 3 5 = 1 := sorry

end NUMINAMATH_GPT_value_of_3Y5_l1260_126089


namespace NUMINAMATH_GPT_sages_success_l1260_126083

-- Assume we have a finite type representing our 1000 colors
inductive Color
| mk : Fin 1000 → Color

open Color

-- Define the sages
def Sage : Type := Fin 11

-- Define the problem conditions into a Lean structure
structure Problem :=
  (sages : Fin 11)
  (colors : Fin 1000)
  (assignments : Sage → Color)
  (strategies : Sage → (Fin 1024 → Fin 2))

-- Define the success condition
def success (p : Problem) : Prop :=
  ∃ (strategies : Sage → (Fin 1024 → Fin 2)),
    ∀ (assignment : Sage → Color),
      ∃ (color_guesses : Sage → Color),
        (∀ s, color_guesses s = assignment s)

-- The sages will succeed in determining the colors of their hats.
theorem sages_success : ∀ (p : Problem), success p := by
  sorry

end NUMINAMATH_GPT_sages_success_l1260_126083


namespace NUMINAMATH_GPT_rectangle_no_shaded_square_l1260_126073

noncomputable def total_rectangles (cols : ℕ) : ℕ :=
  (cols + 1) * (cols + 1 - 1) / 2

noncomputable def shaded_rectangles (cols : ℕ) : ℕ :=
  cols + 1 - 1

noncomputable def probability_no_shaded (cols : ℕ) : ℚ :=
  let n := total_rectangles cols
  let m := shaded_rectangles cols
  1 - (m / n)

theorem rectangle_no_shaded_square :
  probability_no_shaded 2003 = 2002 / 2003 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_no_shaded_square_l1260_126073


namespace NUMINAMATH_GPT_count_m_in_A_l1260_126067

def A : Set ℕ := { 
  x | ∃ (a0 a1 a2 a3 : ℕ), a0 ∈ Finset.range 8 ∧ 
                           a1 ∈ Finset.range 8 ∧ 
                           a2 ∈ Finset.range 8 ∧ 
                           a3 ∈ Finset.range 8 ∧ 
                           a3 ≠ 0 ∧ 
                           x = a0 + a1 * 8 + a2 * 8^2 + a3 * 8^3 }

theorem count_m_in_A (m n : ℕ) (hA_m : m ∈ A) (hA_n : n ∈ A) (h_sum : m + n = 2018) (h_m_gt_n : m > n) :
  ∃! (count : ℕ), count = 497 := 
sorry

end NUMINAMATH_GPT_count_m_in_A_l1260_126067


namespace NUMINAMATH_GPT_solve_inequality_l1260_126065

theorem solve_inequality : { x : ℝ // (x < -1) ∨ (-2/3 < x) } :=
sorry

end NUMINAMATH_GPT_solve_inequality_l1260_126065


namespace NUMINAMATH_GPT_round_robin_games_l1260_126010

theorem round_robin_games (x : ℕ) (h : ∃ (n : ℕ), n = 15) : (x * (x - 1)) / 2 = 15 :=
sorry

end NUMINAMATH_GPT_round_robin_games_l1260_126010


namespace NUMINAMATH_GPT_sum_of_coefficients_l1260_126079

-- Define the polynomial expansion and the target question
theorem sum_of_coefficients
  (x : ℝ)
  (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℝ)
  (h : (5 * x - 2) ^ 6 = b_6 * x ^ 6 + b_5 * x ^ 5 + b_4 * x ^ 4 + 
                        b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0) :
  (b_6 + b_5 + b_4 + b_3 + b_2 + b_1 + b_0) = 729 :=
by {
  -- We substitute x = 1 and show that the polynomial equals 729
  sorry
}

end NUMINAMATH_GPT_sum_of_coefficients_l1260_126079


namespace NUMINAMATH_GPT_inequality_range_of_k_l1260_126038

theorem inequality_range_of_k 
  (a b k : ℝ)
  (h : ∀ a b : ℝ, a^2 + b^2 ≥ 2 * k * a * b) : k ∈ Set.Icc (-1 : ℝ) (1 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_inequality_range_of_k_l1260_126038


namespace NUMINAMATH_GPT_bird_families_flew_away_for_winter_l1260_126019

def bird_families_africa : ℕ := 38
def bird_families_asia : ℕ := 80
def total_bird_families_flew_away : ℕ := bird_families_africa + bird_families_asia

theorem bird_families_flew_away_for_winter : total_bird_families_flew_away = 118 := by
  -- proof goes here (not required)
  sorry

end NUMINAMATH_GPT_bird_families_flew_away_for_winter_l1260_126019


namespace NUMINAMATH_GPT_original_number_is_142857_l1260_126032

-- Definitions based on conditions
def six_digit_number (x : ℕ) : ℕ := 100000 + x
def moved_digit_number (x : ℕ) : ℕ := 10 * x + 1

-- Lean statement of the equivalent problem
theorem original_number_is_142857 : ∃ x, six_digit_number x = 142857 ∧ moved_digit_number x = 3 * six_digit_number x :=
  sorry

end NUMINAMATH_GPT_original_number_is_142857_l1260_126032


namespace NUMINAMATH_GPT_greenville_height_of_boxes_l1260_126099

theorem greenville_height_of_boxes:
  ∃ h : ℝ, 
    (20 * 20 * h) * (2160000 / (20 * 20 * h)) * 0.40 = 180 ∧ 
    400 * h = 2160000 / (2160000 / (20 * 20 * h)) ∧
    400 * h = 5400 ∧
    h = 12 :=
    sorry

end NUMINAMATH_GPT_greenville_height_of_boxes_l1260_126099


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l1260_126049

theorem right_triangle_hypotenuse (A : ℝ) (h height : ℝ) :
  A = 320 ∧ height = 16 →
  ∃ c : ℝ, c = 4 * Real.sqrt 116 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l1260_126049


namespace NUMINAMATH_GPT_Vanya_number_thought_of_l1260_126076

theorem Vanya_number_thought_of :
  ∃ m n : ℕ, m < 10 ∧ n < 10 ∧ (10 * m + n = 81 ∧ (10 * n + m)^2 = 4 * (10 * m + n)) :=
sorry

end NUMINAMATH_GPT_Vanya_number_thought_of_l1260_126076


namespace NUMINAMATH_GPT_base_of_1987_with_digit_sum_25_l1260_126050

theorem base_of_1987_with_digit_sum_25 (b a c : ℕ) (h₀ : a * b^2 + b * b + c = 1987)
(h₁ : a + b + c = 25) (h₂ : 1 ≤ b ∧ b ≤ 45) : b = 19 :=
sorry

end NUMINAMATH_GPT_base_of_1987_with_digit_sum_25_l1260_126050


namespace NUMINAMATH_GPT_customers_who_didnt_tip_l1260_126023

theorem customers_who_didnt_tip:
  ∀ (total_customers tips_per_customer total_tips : ℕ),
  total_customers = 10 →
  tips_per_customer = 3 →
  total_tips = 15 →
  (total_customers - total_tips / tips_per_customer) = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_customers_who_didnt_tip_l1260_126023


namespace NUMINAMATH_GPT_solve_system_l1260_126088

theorem solve_system 
    (x y z : ℝ) 
    (h1 : x + y - 2 + 4 * x * y = 0) 
    (h2 : y + z - 2 + 4 * y * z = 0) 
    (h3 : z + x - 2 + 4 * z * x = 0) :
    (x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
sorry

end NUMINAMATH_GPT_solve_system_l1260_126088


namespace NUMINAMATH_GPT_range_of_m_l1260_126021

theorem range_of_m (m : ℝ) :
  ( ∀ x : ℝ, |x + m| ≤ 4 → -2 ≤ x ∧ x ≤ 8) ↔ -4 ≤ m ∧ m ≤ -2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1260_126021


namespace NUMINAMATH_GPT_perimeter_increase_ratio_of_sides_l1260_126012

def width_increase (a : ℝ) : ℝ := 1.1 * a
def length_increase (b : ℝ) : ℝ := 1.2 * b
def original_perimeter (a b : ℝ) : ℝ := 2 * (a + b)
def new_perimeter (a b : ℝ) : ℝ := 2 * (1.1 * a + 1.2 * b)

theorem perimeter_increase : ∀ a b : ℝ, 
  (a > 0) → (b > 0) → 
  (new_perimeter a b - original_perimeter a b) / (original_perimeter a b) * 100 < 20 := 
by
  sorry

theorem ratio_of_sides (a b : ℝ) (h : new_perimeter a b = 1.18 * original_perimeter a b) : a / b = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_increase_ratio_of_sides_l1260_126012


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1260_126056

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2 * x1 + m = 0) ∧ (x2^2 - 2 * x2 + m = 0)) ↔ (m < 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1260_126056


namespace NUMINAMATH_GPT_min_value_reciprocal_l1260_126069

theorem min_value_reciprocal (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_eq : 2 * a + b = 4) : 
  (∀ (x : ℝ), (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 4 -> x ≥ 1 / (2 * a * b)) -> x ≥ 1 / 2) := 
by
  sorry

end NUMINAMATH_GPT_min_value_reciprocal_l1260_126069


namespace NUMINAMATH_GPT_correct_simplification_l1260_126090

theorem correct_simplification (x y : ℝ) (hy : y ≠ 0):
  3 * x^4 * y / (x^2 * y) = 3 * x^2 :=
by
  sorry

end NUMINAMATH_GPT_correct_simplification_l1260_126090


namespace NUMINAMATH_GPT_initial_number_of_quarters_l1260_126078

theorem initial_number_of_quarters 
  (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (half_dollars : ℕ) (dollar_coins : ℕ) 
  (two_dollar_coins : ℕ) (quarters : ℕ)
  (cost_per_sundae : ℝ) 
  (special_topping_cost : ℝ)
  (featured_flavor_discount : ℝ)
  (members_with_special_topping : ℕ)
  (members_with_featured_flavor : ℕ)
  (left_over : ℝ)
  (expected_quarters : ℕ) :
  pennies = 123 ∧
  nickels = 85 ∧
  dimes = 35 ∧
  half_dollars = 15 ∧
  dollar_coins = 5 ∧
  quarters = expected_quarters ∧
  two_dollar_coins = 4 ∧
  cost_per_sundae = 5.25 ∧
  special_topping_cost = 0.50 ∧
  featured_flavor_discount = 0.25 ∧
  members_with_special_topping = 3 ∧
  members_with_featured_flavor = 5 ∧
  left_over = 0.97 →
  expected_quarters = 54 :=
  by
  sorry

end NUMINAMATH_GPT_initial_number_of_quarters_l1260_126078


namespace NUMINAMATH_GPT_max_value_of_ratio_l1260_126016

theorem max_value_of_ratio (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) : 
  ∃ z, z = (x / y) ∧ z ≤ 1 := sorry

end NUMINAMATH_GPT_max_value_of_ratio_l1260_126016


namespace NUMINAMATH_GPT_matrix_determinant_zero_implies_sum_of_squares_l1260_126026

theorem matrix_determinant_zero_implies_sum_of_squares (a b : ℝ)
  (h : (Matrix.det ![![a - Complex.I, b - 2 * Complex.I],
                       ![1, 1 + Complex.I]]) = 0) :
  a^2 + b^2 = 1 :=
sorry

end NUMINAMATH_GPT_matrix_determinant_zero_implies_sum_of_squares_l1260_126026


namespace NUMINAMATH_GPT_attendees_not_from_A_B_C_D_l1260_126062

theorem attendees_not_from_A_B_C_D
  (num_A : ℕ) (num_B : ℕ) (num_C : ℕ) (num_D : ℕ) (total_attendees : ℕ)
  (hA : num_A = 30)
  (hB : num_B = 2 * num_A)
  (hC : num_C = num_A + 10)
  (hD : num_D = num_C - 5)
  (hTotal : total_attendees = 185)
  : total_attendees - (num_A + num_B + num_C + num_D) = 20 := by
  sorry

end NUMINAMATH_GPT_attendees_not_from_A_B_C_D_l1260_126062


namespace NUMINAMATH_GPT_initial_wine_volume_l1260_126041

theorem initial_wine_volume (x : ℝ) 
  (h₁ : ∀ k : ℝ, k = x → ∀ n : ℕ, n = 3 → 
    (∀ y : ℝ, y = k - 4 * (1 - ((k - 4) / k) ^ n) + 2.5)) :
  x = 16 := by
  sorry

end NUMINAMATH_GPT_initial_wine_volume_l1260_126041


namespace NUMINAMATH_GPT_distance_from_origin_to_midpoint_l1260_126093

theorem distance_from_origin_to_midpoint :
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 10) → (y1 = 20) → (x2 = -10) → (y2 = -20) → 
  dist (0 : ℝ × ℝ) ((x1 + x2) / 2, (y1 + y2) / 2) = 0 := 
by
  intros x1 y1 x2 y2 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- remaining proof goes here
  sorry

end NUMINAMATH_GPT_distance_from_origin_to_midpoint_l1260_126093


namespace NUMINAMATH_GPT_airplane_rows_l1260_126066

theorem airplane_rows (R : ℕ) 
  (h1 : ∀ n, n = 5) 
  (h2 : ∀ s, s = 7) 
  (h3 : ∀ f, f = 2) 
  (h4 : ∀ p, p = 1400):
  (2 * 5 * 7 * R = 1400) → R = 20 :=
by
  -- Assuming the given equation 2 * 5 * 7 * R = 1400
  sorry

end NUMINAMATH_GPT_airplane_rows_l1260_126066


namespace NUMINAMATH_GPT_walnut_trees_total_l1260_126013

theorem walnut_trees_total : 33 + 44 = 77 :=
by
  sorry

end NUMINAMATH_GPT_walnut_trees_total_l1260_126013


namespace NUMINAMATH_GPT_increasing_iff_positive_difference_l1260_126030

variable (a : ℕ → ℝ) (d : ℝ)

def arithmetic_sequence (aₙ : ℕ → ℝ) (d : ℝ) := ∃ (a₁ : ℝ), ∀ n : ℕ, aₙ n = a₁ + n * d

theorem increasing_iff_positive_difference (a : ℕ → ℝ) (d : ℝ) (h : arithmetic_sequence a d) :
  (∀ n, a (n+1) > a n) ↔ d > 0 :=
by
  sorry

end NUMINAMATH_GPT_increasing_iff_positive_difference_l1260_126030


namespace NUMINAMATH_GPT_sequence_expression_l1260_126059

theorem sequence_expression (n : ℕ) (h : n ≥ 2) (T : ℕ → ℕ) (a : ℕ → ℕ)
  (hT : ∀ k : ℕ, T k = 2 * k^2)
  (ha : ∀ k : ℕ, k ≥ 2 → a k = T k / T (k - 1)) :
  a n = (n / (n - 1))^2 := 
sorry

end NUMINAMATH_GPT_sequence_expression_l1260_126059


namespace NUMINAMATH_GPT_part_I_part_II_l1260_126070

def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem part_I (x : ℝ) : (f x > 4) ↔ (x < -1.5 ∨ x > 2.5) :=
by
  sorry

theorem part_II (x : ℝ) : ∀ x : ℝ, f x ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1260_126070


namespace NUMINAMATH_GPT_vectors_not_coplanar_l1260_126052

def vector_a : Fin 3 → ℤ := ![1, 5, 2]
def vector_b : Fin 3 → ℤ := ![-1, 1, -1]
def vector_c : Fin 3 → ℤ := ![1, 1, 1]

def scalar_triple_product (a b c : Fin 3 → ℤ) : ℤ :=
  a 0 * (b 1 * c 2 - b 2 * c 1) -
  a 1 * (b 0 * c 2 - b 2 * c 0) +
  a 2 * (b 0 * c 1 - b 1 * c 0)

theorem vectors_not_coplanar :
  scalar_triple_product vector_a vector_b vector_c ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_vectors_not_coplanar_l1260_126052


namespace NUMINAMATH_GPT_mean_age_correct_l1260_126098

def children_ages : List ℕ := [6, 6, 9, 12]

def number_of_children : ℕ := 4

def sum_of_ages (ages : List ℕ) : ℕ := ages.sum

def mean_age (ages : List ℕ) (num_children : ℕ) : ℚ :=
  sum_of_ages ages / num_children

theorem mean_age_correct :
  mean_age children_ages number_of_children = 8.25 := by
  sorry

end NUMINAMATH_GPT_mean_age_correct_l1260_126098


namespace NUMINAMATH_GPT_min_vitamins_sold_l1260_126058

theorem min_vitamins_sold (n : ℕ) (h1 : n % 11 = 0) (h2 : n % 23 = 0) (h3 : n % 37 = 0) : n = 9361 :=
by
  sorry

end NUMINAMATH_GPT_min_vitamins_sold_l1260_126058


namespace NUMINAMATH_GPT_bus_ride_cost_l1260_126042

theorem bus_ride_cost (B T : ℝ) 
  (h1 : T = B + 6.85)
  (h2 : T + B = 9.65)
  (h3 : ∃ n : ℤ, B = 0.35 * n ∧ ∃ m : ℤ, T = 0.35 * m) : 
  B = 1.40 := 
by
  sorry

end NUMINAMATH_GPT_bus_ride_cost_l1260_126042


namespace NUMINAMATH_GPT_abcd_sum_l1260_126003

theorem abcd_sum : 
  ∃ (a b c d : ℕ), 
    (∃ x y : ℝ, x + y = 5 ∧ 2 * x * y = 6 ∧ 
      (x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d)) →
    a + b + c + d = 21 :=
by
  sorry

end NUMINAMATH_GPT_abcd_sum_l1260_126003


namespace NUMINAMATH_GPT_swimming_class_attendance_l1260_126017

theorem swimming_class_attendance (total_students : ℕ) (chess_percentage : ℝ) (swimming_percentage : ℝ) 
  (H1 : total_students = 1000) 
  (H2 : chess_percentage = 0.20) 
  (H3 : swimming_percentage = 0.10) : 
  200 * 0.10 = 20 := 
by sorry

end NUMINAMATH_GPT_swimming_class_attendance_l1260_126017


namespace NUMINAMATH_GPT_abc_inequality_l1260_126006

theorem abc_inequality (x y z : ℝ) (a b c : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : a = (x * (y - z) ^ 2) ^ 2) (h2 : b = (y * (z - x) ^ 2) ^ 2) (h3 : c = (z * (x - y) ^ 2) ^ 2) :
  a^2 + b^2 + c^2 ≥ 2 * (a * b + b * c + c * a) :=
by {
  sorry
}

end NUMINAMATH_GPT_abc_inequality_l1260_126006


namespace NUMINAMATH_GPT_sum_of_possible_values_of_g1_l1260_126092

def g (x : ℝ) : ℝ := sorry

axiom g_prop : ∀ x y : ℝ, g (g (x - y)) = g x * g y - g x + g y - x^2 * y^2

theorem sum_of_possible_values_of_g1 : g 1 = -1 := by sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_g1_l1260_126092


namespace NUMINAMATH_GPT_divisibility_equivalence_l1260_126043

theorem divisibility_equivalence (a b c d : ℤ) (h : a ≠ c) :
  (a - c) ∣ (a * b + c * d) ↔ (a - c) ∣ (a * d + b * c) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_equivalence_l1260_126043


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1260_126039

theorem isosceles_triangle_perimeter (a b : ℕ) (h₀ : a = 3 ∨ a = 4) (h₁ : b = 3 ∨ b = 4) (h₂ : a ≠ b) :
  (a = 3 ∧ b = 4 ∧ 4 ∈ [b]) ∨ (a = 4 ∧ b = 3 ∧ 4 ∈ [a]) → 
  (a + a + b = 10) ∨ (a + b + b = 11) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1260_126039


namespace NUMINAMATH_GPT_josh_remaining_marbles_l1260_126054

theorem josh_remaining_marbles : 
  let initial_marbles := 19 
  let lost_marbles := 11
  initial_marbles - lost_marbles = 8 := by
  sorry

end NUMINAMATH_GPT_josh_remaining_marbles_l1260_126054


namespace NUMINAMATH_GPT_correct_calculation_l1260_126044

variable (a b : ℝ)

theorem correct_calculation :
  -(a - b) = -a + b := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1260_126044


namespace NUMINAMATH_GPT_soccer_team_lineups_l1260_126007

noncomputable def num_starting_lineups (n k t g : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) k)

theorem soccer_team_lineups :
  num_starting_lineups 18 9 1 1 = 3501120 := by
    sorry

end NUMINAMATH_GPT_soccer_team_lineups_l1260_126007


namespace NUMINAMATH_GPT_dorothy_will_be_twice_as_old_l1260_126094

-- Define some variables
variables (D S Y : ℕ)

-- Hypothesis
def dorothy_age_condition (D S : ℕ) : Prop := D = 3 * S
def dorothy_current_age (D : ℕ) : Prop := D = 15

-- Theorems we want to prove
theorem dorothy_will_be_twice_as_old (D S Y : ℕ) 
  (h1 : dorothy_age_condition D S)
  (h2 : dorothy_current_age D)
  (h3 : D = 15)
  (h4 : S = 5)
  (h5 : D + Y = 2 * (S + Y)) : Y = 5 := 
sorry

end NUMINAMATH_GPT_dorothy_will_be_twice_as_old_l1260_126094


namespace NUMINAMATH_GPT_grant_earnings_l1260_126014

theorem grant_earnings 
  (baseball_cards_sale : ℕ) 
  (baseball_bat_sale : ℕ) 
  (baseball_glove_price : ℕ) 
  (baseball_glove_discount : ℕ) 
  (baseball_cleats_sale : ℕ) : 
  baseball_cards_sale + baseball_bat_sale + (baseball_glove_price - baseball_glove_discount) + 2 * baseball_cleats_sale = 79 :=
by
  let baseball_cards_sale := 25
  let baseball_bat_sale := 10
  let baseball_glove_price := 30
  let baseball_glove_discount := (30 * 20) / 100
  let baseball_cleats_sale := 10
  sorry

end NUMINAMATH_GPT_grant_earnings_l1260_126014


namespace NUMINAMATH_GPT_new_rectangle_area_eq_a_squared_l1260_126034

theorem new_rectangle_area_eq_a_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let d := Real.sqrt (a^2 + b^2)
  let base := 2 * (d + b)
  let height := (d - b) / 2
  base * height = a^2 := by
  sorry

end NUMINAMATH_GPT_new_rectangle_area_eq_a_squared_l1260_126034


namespace NUMINAMATH_GPT_number_of_people_speaking_both_languages_l1260_126027

theorem number_of_people_speaking_both_languages
  (total : ℕ) (L : ℕ) (F : ℕ) (N : ℕ) (B : ℕ) :
  total = 25 → L = 13 → F = 15 → N = 6 → total = L + F - B + N → B = 9 :=
by
  intros h_total h_L h_F h_N h_inclusion_exclusion
  sorry

end NUMINAMATH_GPT_number_of_people_speaking_both_languages_l1260_126027


namespace NUMINAMATH_GPT_expression_evaluation_l1260_126091

def eval_expression : Int := 
  let a := -2 ^ 3
  let b := abs (2 - 3)
  let c := -2 * (-1) ^ 2023
  a + b + c

theorem expression_evaluation :
  eval_expression = -5 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1260_126091


namespace NUMINAMATH_GPT_probability_intersection_inside_nonagon_correct_l1260_126002

def nonagon_vertices : ℕ := 9

def total_pairs_of_points := Nat.choose nonagon_vertices 2

def sides_of_nonagon : ℕ := nonagon_vertices

def diagonals_of_nonagon := total_pairs_of_points - sides_of_nonagon

def pairs_of_diagonals := Nat.choose diagonals_of_nonagon 2

def sets_of_intersecting_diagonals := Nat.choose nonagon_vertices 4

noncomputable def probability_intersection_inside_nonagon : ℚ :=
  sets_of_intersecting_diagonals / pairs_of_diagonals

theorem probability_intersection_inside_nonagon_correct :
  probability_intersection_inside_nonagon = 14 / 39 := 
  sorry

end NUMINAMATH_GPT_probability_intersection_inside_nonagon_correct_l1260_126002


namespace NUMINAMATH_GPT_actual_distance_traveled_l1260_126008

theorem actual_distance_traveled (T : ℝ) :
  ∀ D : ℝ, (D = 4 * T) → (D + 6 = 5 * T) → D = 24 :=
by
  intro D h1 h2
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l1260_126008


namespace NUMINAMATH_GPT_area_of_square_l1260_126031

-- Defining the points A and B as given in the conditions.
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (4, 6)

-- Theorem statement: proving that the area of the square given the endpoints A and B is 12.5.
theorem area_of_square : 
  ∀ (A B : ℝ × ℝ),
  A = (1, 2) → B = (4, 6) → 
  ∃ (area : ℝ), area = 12.5 := 
by
  intros A B hA hB
  sorry

end NUMINAMATH_GPT_area_of_square_l1260_126031


namespace NUMINAMATH_GPT_find_f_inv_128_l1260_126084

noncomputable def f : ℕ → ℕ := sorry

axiom f_at_5 : f 5 = 2
axiom f_doubling : ∀ x : ℕ, f (2 * x) = 2 * f x

theorem find_f_inv_128 : f 320 = 128 :=
by sorry

end NUMINAMATH_GPT_find_f_inv_128_l1260_126084


namespace NUMINAMATH_GPT_number_of_people_l1260_126024

-- Definitions based on the conditions
def average_age (T : ℕ) (n : ℕ) := T / n = 30
def youngest_age := 3
def average_age_when_youngest_born (T : ℕ) (n : ℕ) := (T - youngest_age) / (n - 1) = 27

theorem number_of_people (T n : ℕ) (h1 : average_age T n) (h2 : average_age_when_youngest_born T n) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_l1260_126024


namespace NUMINAMATH_GPT_range_of_a_l1260_126086

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 0 then (x - a) ^ 2 else x + (1 / x) + a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ f 0 a) ↔ 0 ≤ a ∧ a ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1260_126086


namespace NUMINAMATH_GPT_percent_of_b_l1260_126029

variables (a b c : ℝ)

theorem percent_of_b (h1 : c = 0.30 * a) (h2 : b = 1.20 * a) : c = 0.25 * b :=
by sorry

end NUMINAMATH_GPT_percent_of_b_l1260_126029


namespace NUMINAMATH_GPT_thirty_percent_more_than_80_is_one_fourth_less_l1260_126060

-- Translating the mathematical equivalency conditions into Lean definitions and theorems

def thirty_percent_more (n : ℕ) : ℕ :=
  n + (n * 30 / 100)

def one_fourth_less (x : ℕ) : ℕ :=
  x - (x / 4)

theorem thirty_percent_more_than_80_is_one_fourth_less (x : ℕ) :
  thirty_percent_more 80 = one_fourth_less x → x = 139 :=
by
  sorry

end NUMINAMATH_GPT_thirty_percent_more_than_80_is_one_fourth_less_l1260_126060


namespace NUMINAMATH_GPT_range_of_a_l1260_126011

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ 2 - a * x
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) : ℝ := Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ f x a = h x) →
  1 ≤ a ∧ a ≤ Real.exp 1 + 1 / Real.exp 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1260_126011


namespace NUMINAMATH_GPT_percentage_increase_in_area_l1260_126000

variable (L W : Real)

theorem percentage_increase_in_area (hL : L > 0) (hW : W > 0) :
  ((1 + 0.25) * L * (1 + 0.25) * W - L * W) / (L * W) * 100 = 56.25 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_area_l1260_126000


namespace NUMINAMATH_GPT_large_pretzel_cost_l1260_126053

theorem large_pretzel_cost : 
  ∀ (P S : ℕ), 
  P = 3 * S ∧ 7 * P + 4 * S = 4 * P + 7 * S + 12 → 
  P = 6 :=
by sorry

end NUMINAMATH_GPT_large_pretzel_cost_l1260_126053


namespace NUMINAMATH_GPT_original_price_of_house_l1260_126001

theorem original_price_of_house (P: ℝ) (sold_price: ℝ) (profit: ℝ) (commission: ℝ):
  sold_price = 100000 ∧ profit = 0.20 ∧ commission = 0.05 → P = 86956.52 :=
by
  sorry -- Proof not provided

end NUMINAMATH_GPT_original_price_of_house_l1260_126001


namespace NUMINAMATH_GPT_find_j_l1260_126018

theorem find_j (n j : ℕ) (h1 : n % j = 28) (h2 : (n : ℝ) / j = 142.07) : j = 400 :=
by
  sorry

end NUMINAMATH_GPT_find_j_l1260_126018


namespace NUMINAMATH_GPT_find_largest_x_and_compute_ratio_l1260_126025

theorem find_largest_x_and_compute_ratio (a b c d : ℤ) (h : x = (a + b * Real.sqrt c) / d)
   (cond : (5 * x / 7) + 1 = 3 / x) : a * c * d / b = -70 :=
by
  sorry

end NUMINAMATH_GPT_find_largest_x_and_compute_ratio_l1260_126025


namespace NUMINAMATH_GPT_cats_awake_l1260_126020

theorem cats_awake (total_cats asleep_cats cats_awake : ℕ) (h1 : total_cats = 98) (h2 : asleep_cats = 92) (h3 : cats_awake = total_cats - asleep_cats) : cats_awake = 6 :=
by
  -- Definitions and conditions
  subst h1
  subst h2
  subst h3
  -- The statement we need to prove
  sorry

end NUMINAMATH_GPT_cats_awake_l1260_126020


namespace NUMINAMATH_GPT_highest_monthly_profit_max_average_profit_l1260_126048

noncomputable def profit (x : ℕ) : ℤ :=
if 1 ≤ x ∧ x ≤ 5 then 26 * x - 56
else if 5 < x ∧ x ≤ 12 then 210 - 20 * x
else 0

noncomputable def average_profit (x : ℕ) : ℝ :=
if 1 ≤ x ∧ x ≤ 5 then (13 * ↑x - 43 : ℤ) / ↑x
else if 5 < x ∧ x ≤ 12 then (-10 * ↑x + 200 - 640 / ↑x : ℝ)
else 0

theorem highest_monthly_profit :
  ∃ m p, m = 6 ∧ p = 90 ∧ profit m = p :=
by sorry

theorem max_average_profit (x : ℕ) :
  1 ≤ x ∧ x ≤ 12 →
  average_profit x ≤ 40 ∧ (average_profit 8 = 40 → x = 8) :=
by sorry

end NUMINAMATH_GPT_highest_monthly_profit_max_average_profit_l1260_126048


namespace NUMINAMATH_GPT_choices_of_N_l1260_126045

def base7_representation (N : ℕ) : ℕ := 
  (N / 49) * 100 + ((N % 49) / 7) * 10 + (N % 7)

def base8_representation (N : ℕ) : ℕ := 
  (N / 64) * 100 + ((N % 64) / 8) * 10 + (N % 8)

theorem choices_of_N : 
  ∃ (N_set : Finset ℕ), 
    (∀ N ∈ N_set, 100 ≤ N ∧ N < 1000 ∧ 
      ((base7_representation N * base8_representation N) % 100 = (3 * N) % 100)) 
    ∧ N_set.card = 15 :=
by
  sorry

end NUMINAMATH_GPT_choices_of_N_l1260_126045


namespace NUMINAMATH_GPT_find_clique_of_size_6_l1260_126097

-- Defining the conditions of the graph G
variable (G : SimpleGraph (Fin 12))

-- Condition: For any subset of 9 vertices, there exists a subset of 5 vertices that form a complete subgraph K_5.
def condition (s : Finset (Fin 12)) : Prop :=
  s.card = 9 → ∃ t : Finset (Fin 12), t ⊆ s ∧ t.card = 5 ∧ (∀ u v : Fin 12, u ∈ t → v ∈ t → u ≠ v → G.Adj u v)

-- The theorem to prove given the conditions
theorem find_clique_of_size_6 (h : ∀ s : Finset (Fin 12), condition G s) : 
  ∃ t : Finset (Fin 12), t.card = 6 ∧ (∀ u v : Fin 12, u ∈ t → v ∈ t → u ≠ v → G.Adj u v) :=
sorry

end NUMINAMATH_GPT_find_clique_of_size_6_l1260_126097


namespace NUMINAMATH_GPT_lucy_total_fish_l1260_126077

variable (current_fish additional_fish : ℕ)

def total_fish (current_fish additional_fish : ℕ) : ℕ :=
  current_fish + additional_fish

theorem lucy_total_fish (h1 : current_fish = 212) (h2 : additional_fish = 68) : total_fish current_fish additional_fish = 280 :=
by
  sorry

end NUMINAMATH_GPT_lucy_total_fish_l1260_126077


namespace NUMINAMATH_GPT_negation_of_no_vegetarian_students_eat_at_cafeteria_l1260_126071

variable (Student : Type) 
variable (isVegetarian : Student → Prop)
variable (eatsAtCafeteria : Student → Prop)

theorem negation_of_no_vegetarian_students_eat_at_cafeteria :
  (∀ x, isVegetarian x → ¬ eatsAtCafeteria x) →
  (∃ x, isVegetarian x ∧ eatsAtCafeteria x) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_no_vegetarian_students_eat_at_cafeteria_l1260_126071


namespace NUMINAMATH_GPT_total_trophies_correct_l1260_126063

-- Define the current number of Michael's trophies
def michael_current_trophies : ℕ := 30

-- Define the number of trophies Michael will have in three years
def michael_trophies_in_three_years : ℕ := michael_current_trophies + 100

-- Define the number of trophies Jack will have in three years
def jack_trophies_in_three_years : ℕ := 10 * michael_current_trophies

-- Define the total number of trophies Jack and Michael will have after three years
def total_trophies_in_three_years : ℕ := michael_trophies_in_three_years + jack_trophies_in_three_years

-- Prove that the total number of trophies after three years is 430
theorem total_trophies_correct : total_trophies_in_three_years = 430 :=
by
  sorry -- proof is omitted

end NUMINAMATH_GPT_total_trophies_correct_l1260_126063


namespace NUMINAMATH_GPT_range_of_k_for_distinct_real_roots_l1260_126081

theorem range_of_k_for_distinct_real_roots (k : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (k - 1) * x1^2 - 2 * x1 + 1 = 0 ∧ (k - 1) * x2^2 - 2 * x2 + 1 = 0) →
    k < 2 ∧ k ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_for_distinct_real_roots_l1260_126081


namespace NUMINAMATH_GPT_solve_custom_operation_l1260_126055

theorem solve_custom_operation (x : ℤ) (h : ((4 * 3 - (12 - x)) = 2)) : x = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_custom_operation_l1260_126055


namespace NUMINAMATH_GPT_jungkook_colored_paper_count_l1260_126082

theorem jungkook_colored_paper_count :
  (3 * 10) + 8 = 38 :=
by sorry

end NUMINAMATH_GPT_jungkook_colored_paper_count_l1260_126082


namespace NUMINAMATH_GPT_price_of_second_candy_l1260_126074

variables (X P : ℝ)

-- Conditions
def total_weight (X : ℝ) := X + 6.25 = 10
def total_value (X P : ℝ) := 3.50 * X + 6.25 * P = 40

-- Proof problem
theorem price_of_second_candy (h1 : total_weight X) (h2 : total_value X P) : P = 4.30 :=
by 
  sorry

end NUMINAMATH_GPT_price_of_second_candy_l1260_126074


namespace NUMINAMATH_GPT_q_at_2_l1260_126085

noncomputable def q (x : ℝ) : ℝ :=
  Real.sign (3 * x - 2) * |3 * x - 2|^(1/4) +
  2 * Real.sign (3 * x - 2) * |3 * x - 2|^(1/6) +
  |3 * x - 2|^(1/8)

theorem q_at_2 : q 2 = 4 := by
  -- Proof attempt needed
  sorry

end NUMINAMATH_GPT_q_at_2_l1260_126085


namespace NUMINAMATH_GPT_exists_integers_a_b_part_a_l1260_126096

theorem exists_integers_a_b_part_a : 
  ∃ a b : ℤ, (∀ x : ℝ, x^2 + a * x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + a * x + (b : ℝ) = 0) := 
sorry

end NUMINAMATH_GPT_exists_integers_a_b_part_a_l1260_126096


namespace NUMINAMATH_GPT_find_b_l1260_126051

noncomputable def func (x a b : ℝ) := (1 / 12) * x^2 + a * x + b

theorem  find_b (a b : ℝ) (x1 x2 : ℝ):
    (func x1 a b = 0) →
    (func x2 a b = 0) →
    (b = (x1 * x2) / 12) →
    ((3 - x1) = (x2 - 3)) →
    (b = -6) :=
by
    sorry

end NUMINAMATH_GPT_find_b_l1260_126051
