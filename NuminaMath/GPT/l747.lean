import Mathlib

namespace NUMINAMATH_GPT_A_minus_3B_A_minus_3B_independent_of_y_l747_74734

variables (x y : ℝ)
def A : ℝ := 3*x^2 - x + 2*y - 4*x*y
def B : ℝ := x^2 - 2*x - y + x*y - 5

theorem A_minus_3B (x y : ℝ) : A x y - 3 * B x y = 5*x + 5*y - 7*x*y + 15 :=
by
  sorry

theorem A_minus_3B_independent_of_y (x : ℝ) (hyp : ∀ y : ℝ, A x y - 3 * B x y = 5*x + 5*y - 7*x*y + 15) :
  5 - 7*x = 0 → x = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_A_minus_3B_A_minus_3B_independent_of_y_l747_74734


namespace NUMINAMATH_GPT_fourth_square_state_l747_74797

inductive Shape
| Circle
| Triangle
| LineSegment
| Square

inductive Position
| TopLeft
| TopRight
| BottomLeft
| BottomRight

structure SquareState where
  circle : Position
  triangle : Position
  line_segment_parallel_to : Bool -- True = Top & Bottom; False = Left & Right
  square : Position

def move_counterclockwise : Position → Position
| Position.TopLeft => Position.BottomLeft
| Position.BottomLeft => Position.BottomRight
| Position.BottomRight => Position.TopRight
| Position.TopRight => Position.TopLeft

def update_square_states (s1 s2 s3 : SquareState) : Prop :=
  move_counterclockwise s1.circle = s2.circle ∧
  move_counterclockwise s2.circle = s3.circle ∧
  move_counterclockwise s1.triangle = s2.triangle ∧
  move_counterclockwise s2.triangle = s3.triangle ∧
  s1.line_segment_parallel_to = !s2.line_segment_parallel_to ∧
  s2.line_segment_parallel_to = !s3.line_segment_parallel_to ∧
  move_counterclockwise s1.square = s2.square ∧
  move_counterclockwise s2.square = s3.square

theorem fourth_square_state (s1 s2 s3 s4 : SquareState) (h : update_square_states s1 s2 s3) :
  s4.circle = move_counterclockwise s3.circle ∧
  s4.triangle = move_counterclockwise s3.triangle ∧
  s4.line_segment_parallel_to = !s3.line_segment_parallel_to ∧
  s4.square = move_counterclockwise s3.square :=
sorry

end NUMINAMATH_GPT_fourth_square_state_l747_74797


namespace NUMINAMATH_GPT_proof_complement_U_A_l747_74748

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set A
def A : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := { x ∈ U | x ∉ A }

-- The theorem statement
theorem proof_complement_U_A :
  complement_U_A = {1, 5} :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_proof_complement_U_A_l747_74748


namespace NUMINAMATH_GPT_tangent_line_at_point_is_x_minus_y_plus_1_eq_0_l747_74754

noncomputable def tangent_line (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_at_point_is_x_minus_y_plus_1_eq_0:
  tangent_line 0 = 1 →
  ∀ x y, y = tangent_line x → x - y + 1 = 0 → y = x * Real.exp x + 1 →
  x = 0 ∧ y = 1 → x - y + 1 = 0 :=
by
  intro h_point x y h_tangent h_eq h_coord
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_is_x_minus_y_plus_1_eq_0_l747_74754


namespace NUMINAMATH_GPT_same_function_C_l747_74793

theorem same_function_C (x : ℝ) (hx : x ≠ 0) : (x^0 = 1) ∧ ((1 / x^0) = 1) :=
by
  -- Definition for domain exclusion
  have h1 : x ^ 0 = 1 := by 
    sorry -- proof skipped
  have h2 : 1 / x ^ 0 = 1 := by 
    sorry -- proof skipped
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_same_function_C_l747_74793


namespace NUMINAMATH_GPT_bank_record_withdrawal_l747_74705

def deposit (x : ℤ) := x
def withdraw (x : ℤ) := -x

theorem bank_record_withdrawal : withdraw 500 = -500 :=
by
  sorry

end NUMINAMATH_GPT_bank_record_withdrawal_l747_74705


namespace NUMINAMATH_GPT_sales_decrease_percentage_l747_74798

theorem sales_decrease_percentage 
  (P S : ℝ) 
  (P_new : ℝ := 1.30 * P) 
  (R : ℝ := P * S) 
  (R_new : ℝ := 1.04 * R) 
  (x : ℝ) 
  (S_new : ℝ := S * (1 - x/100)) 
  (h1 : 1.30 * P * S * (1 - x/100) = 1.04 * P * S) : 
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_sales_decrease_percentage_l747_74798


namespace NUMINAMATH_GPT_totalCerealInThreeBoxes_l747_74762

def firstBox := 14
def secondBox := firstBox / 2
def thirdBox := secondBox + 5
def totalCereal := firstBox + secondBox + thirdBox

theorem totalCerealInThreeBoxes : totalCereal = 33 := 
by {
  sorry
}

end NUMINAMATH_GPT_totalCerealInThreeBoxes_l747_74762


namespace NUMINAMATH_GPT_positive_difference_is_127_div_8_l747_74789

-- Defining the basic expressions
def eight_squared : ℕ := 8 ^ 2 -- 64

noncomputable def expr1 : ℝ := (eight_squared + eight_squared) / 8
noncomputable def expr2 : ℝ := (eight_squared / eight_squared) / 8

-- Problem statement
theorem positive_difference_is_127_div_8 :
  (expr1 - expr2) = 127 / 8 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_is_127_div_8_l747_74789


namespace NUMINAMATH_GPT_ratio_of_eggs_used_l747_74775

theorem ratio_of_eggs_used (total_eggs : ℕ) (eggs_left : ℕ) (eggs_broken : ℕ) (eggs_bought : ℕ) :
  total_eggs = 72 →
  eggs_left = 21 →
  eggs_broken = 15 →
  eggs_bought = total_eggs - (eggs_left + eggs_broken) →
  (eggs_bought / total_eggs) = 1 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_ratio_of_eggs_used_l747_74775


namespace NUMINAMATH_GPT_sum_and_product_roots_l747_74750

structure quadratic_data where
  m : ℝ
  n : ℝ

def roots_sum_eq (qd : quadratic_data) : Prop :=
  qd.m / 3 = 9

def roots_product_eq (qd : quadratic_data) : Prop :=
  qd.n / 3 = 20

theorem sum_and_product_roots (qd : quadratic_data) :
  roots_sum_eq qd → roots_product_eq qd → qd.m + qd.n = 87 := by
  sorry

end NUMINAMATH_GPT_sum_and_product_roots_l747_74750


namespace NUMINAMATH_GPT_min_value_m2n_mn_l747_74773

theorem min_value_m2n_mn (m n : ℝ) 
  (h1 : (x - m)^2 + (y - n)^2 = 9)
  (h2 : x + 2 * y + 2 = 0)
  (h3 : 0 < m)
  (h4 : 0 < n)
  (h5 : m + 2 * n + 2 = 5)
  (h6 : ∃ l : ℝ, l = 4 ): (m + 2 * n) / (m * n) = 8/3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_m2n_mn_l747_74773


namespace NUMINAMATH_GPT_find_length_of_first_train_l747_74772

noncomputable def length_of_first_train (speed_train1 speed_train2 : ℕ) (time_to_cross : ℕ) (length_train2 : ℚ) : ℚ :=
  let relative_speed := (speed_train1 + speed_train2) * 1000 / 3600
  let combined_length := relative_speed * time_to_cross
  combined_length - length_train2

theorem find_length_of_first_train :
  length_of_first_train 120 80 9 280.04 = 220 := sorry

end NUMINAMATH_GPT_find_length_of_first_train_l747_74772


namespace NUMINAMATH_GPT_cos_C_value_l747_74791

theorem cos_C_value (a b c : ℝ) (A B C : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) : 
  Real.cos C = 7 / 25 :=
  sorry

end NUMINAMATH_GPT_cos_C_value_l747_74791


namespace NUMINAMATH_GPT_solve_for_a_l747_74702

theorem solve_for_a (a : ℝ) (h : a⁻¹ = (-1 : ℝ)^0) : a = 1 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l747_74702


namespace NUMINAMATH_GPT_two_numbers_with_difference_less_than_half_l747_74733

theorem two_numbers_with_difference_less_than_half
  (x1 x2 x3 : ℝ)
  (h1 : 0 ≤ x1) (h2 : x1 < 1)
  (h3 : 0 ≤ x2) (h4 : x2 < 1)
  (h5 : 0 ≤ x3) (h6 : x3 < 1) :
  ∃ a b, 
    (a = x1 ∨ a = x2 ∨ a = x3) ∧
    (b = x1 ∨ b = x2 ∨ b = x3) ∧
    a ≠ b ∧ 
    |b - a| < 1 / 2 :=
sorry

end NUMINAMATH_GPT_two_numbers_with_difference_less_than_half_l747_74733


namespace NUMINAMATH_GPT_symmetric_point_y_axis_l747_74738

theorem symmetric_point_y_axis (x y : ℝ) (hx : x = -3) (hy : y = 2) :
  (-x, y) = (3, 2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_y_axis_l747_74738


namespace NUMINAMATH_GPT_carmela_gives_each_l747_74795

noncomputable def money_needed_to_give_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) : ℕ :=
  let total_cousins_money := cousins * cousins_count
  let total_money := carmela + total_cousins_money
  let people_count := 1 + cousins_count
  let equal_share := total_money / people_count
  let total_giveaway := carmela - equal_share
  total_giveaway / cousins_count

theorem carmela_gives_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) (h_carmela : carmela = 7) (h_cousins : cousins = 2) (h_cousins_count : cousins_count = 4) :
  money_needed_to_give_each carmela cousins cousins_count = 1 :=
by
  rw [h_carmela, h_cousins, h_cousins_count]
  sorry

end NUMINAMATH_GPT_carmela_gives_each_l747_74795


namespace NUMINAMATH_GPT_point_in_second_quadrant_l747_74736

theorem point_in_second_quadrant (a : ℝ) :
  ∃ q : ℕ, q = 2 ∧ (-3 : ℝ) < 0 ∧ (a^2 + 1) > 0 := 
by sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l747_74736


namespace NUMINAMATH_GPT_range_of_M_l747_74721

theorem range_of_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) :
    ( (1 / a - 1) * (1 / b - 1) * (1 / c - 1) )  ≥ 8 := 
  sorry

end NUMINAMATH_GPT_range_of_M_l747_74721


namespace NUMINAMATH_GPT_probability_of_spinner_stopping_in_region_G_l747_74768

theorem probability_of_spinner_stopping_in_region_G :
  let pE := (1:ℝ) / 2
  let pF := (1:ℝ) / 4
  let y  := (1:ℝ) / 6
  let z  := (1:ℝ) / 12
  pE + pF + y + z = 1 → y = 2 * z → y = (1:ℝ) / 6 := by
  intros htotal hdouble
  sorry

end NUMINAMATH_GPT_probability_of_spinner_stopping_in_region_G_l747_74768


namespace NUMINAMATH_GPT_prime_power_sum_l747_74724

theorem prime_power_sum (a b p : ℕ) (hp : p = a ^ b + b ^ a) (ha_prime : Nat.Prime a) (hb_prime : Nat.Prime b) (hp_prime : Nat.Prime p) : 
  p = 17 := 
sorry

end NUMINAMATH_GPT_prime_power_sum_l747_74724


namespace NUMINAMATH_GPT_matrix_pow_2020_l747_74780

-- Define the matrix type and basic multiplication rule
def M : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![3, 1]]

theorem matrix_pow_2020 :
  M ^ 2020 = ![![1, 0], ![6060, 1]] := by
  sorry

end NUMINAMATH_GPT_matrix_pow_2020_l747_74780


namespace NUMINAMATH_GPT_songs_can_be_stored_l747_74779

def totalStorageGB : ℕ := 16
def usedStorageGB : ℕ := 4
def songSizeMB : ℕ := 30
def gbToMb : ℕ := 1000

def remainingStorageGB := totalStorageGB - usedStorageGB
def remainingStorageMB := remainingStorageGB * gbToMb
def numberOfSongs := remainingStorageMB / songSizeMB

theorem songs_can_be_stored : numberOfSongs = 400 :=
by
  sorry

end NUMINAMATH_GPT_songs_can_be_stored_l747_74779


namespace NUMINAMATH_GPT_sum_of_mapped_elements_is_ten_l747_74786

theorem sum_of_mapped_elements_is_ten (a b : ℝ) (h1 : a = 1) (h2 : b = 9) : a + b = 10 := by
  sorry

end NUMINAMATH_GPT_sum_of_mapped_elements_is_ten_l747_74786


namespace NUMINAMATH_GPT_functional_equation_l747_74717

theorem functional_equation 
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
  (h2 : f 0 ≠ 0) :
  f 2009 = 1 :=
sorry

end NUMINAMATH_GPT_functional_equation_l747_74717


namespace NUMINAMATH_GPT_solve_for_y_l747_74742

theorem solve_for_y (x y : ℝ) (h1 : 3 * x^2 + 4 * x + 7 * y + 2 = 0) (h2 : 3 * x + 2 * y + 5 = 0) : 4 * y^2 + 33 * y + 11 = 0 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l747_74742


namespace NUMINAMATH_GPT_max_profit_achieved_l747_74787

theorem max_profit_achieved :
  ∃ x : ℤ, 
    (x = 21) ∧ 
    (21 + 14 = 35) ∧ 
    (30 - 21 = 9) ∧ 
    (21 - 5 = 16) ∧
    (-x + 1965 = 1944) :=
by
  sorry

end NUMINAMATH_GPT_max_profit_achieved_l747_74787


namespace NUMINAMATH_GPT_problem_21_sum_correct_l747_74758

theorem problem_21_sum_correct (A B C D E : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10)
    (h_eq : (10 * A + B) * (10 * C + D) = 111 * E) : 
  A + B + C + D + E = 21 :=
sorry

end NUMINAMATH_GPT_problem_21_sum_correct_l747_74758


namespace NUMINAMATH_GPT_number_of_factors_l747_74710

theorem number_of_factors (K : ℕ) (hK : K = 2^4 * 3^3 * 5^2 * 7^1) : 
  ∃ n : ℕ, (∀ d e f g : ℕ, (0 ≤ d ∧ d ≤ 4) → (0 ≤ e ∧ e ≤ 3) → (0 ≤ f ∧ f ≤ 2) → (0 ≤ g ∧ g ≤ 1) → n = 120) :=
sorry

end NUMINAMATH_GPT_number_of_factors_l747_74710


namespace NUMINAMATH_GPT_chocolate_mixture_l747_74751

theorem chocolate_mixture (x : ℝ) (h_initial : 110 / 220 = 0.5)
  (h_equation : (110 + x) / (220 + x) = 0.75) : x = 220 := by
  sorry

end NUMINAMATH_GPT_chocolate_mixture_l747_74751


namespace NUMINAMATH_GPT_square_area_l747_74704

theorem square_area (y1 y2 y3 y4 : ℤ) 
  (h1 : y1 = 0) (h2 : y2 = 3) (h3 : y3 = 0) (h4 : y4 = -3) : 
  ∃ (area : ℤ), area = 36 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l747_74704


namespace NUMINAMATH_GPT_second_job_hourly_wage_l747_74785

-- Definitions based on conditions
def total_wages : ℕ := 160
def first_job_wages : ℕ := 52
def second_job_hours : ℕ := 12

-- Proof statement
theorem second_job_hourly_wage : 
  (total_wages - first_job_wages) / second_job_hours = 9 :=
by
  sorry

end NUMINAMATH_GPT_second_job_hourly_wage_l747_74785


namespace NUMINAMATH_GPT_average_total_goals_l747_74774

theorem average_total_goals (carter_avg shelby_avg judah_avg total_avg : ℕ) 
    (h1: carter_avg = 4) 
    (h2: shelby_avg = carter_avg / 2)
    (h3: judah_avg = 2 * shelby_avg - 3) 
    (h4: total_avg = carter_avg + shelby_avg + judah_avg) :
  total_avg = 7 :=
by
  sorry

end NUMINAMATH_GPT_average_total_goals_l747_74774


namespace NUMINAMATH_GPT_inequality_proof_l747_74788

theorem inequality_proof 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0)
  (hxz : x * z = 1) 
  (h₁ : x * (1 + z) > 1) 
  (h₂ : y * (1 + x) > 1) 
  (h₃ : z * (1 + y) > 1) :
  2 * (x + y + z) ≥ -1/x + 1/y + 1/z + 3 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l747_74788


namespace NUMINAMATH_GPT_solve_equation_l747_74767

theorem solve_equation (x : ℝ) :
  (x^2 + 2*x + 1 = abs (3*x - 2)) ↔ 
  (x = (-7 + Real.sqrt 37) / 2) ∨ 
  (x = (-7 - Real.sqrt 37) / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l747_74767


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l747_74728

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 < x) → ((x^2 < x) ↔ (0 < x ∧ x < 1)) ∧ ((1/x > 2) ↔ (0 < x ∧ x < 1/2)) := 
by 
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l747_74728


namespace NUMINAMATH_GPT_expression_a_equals_half_expression_c_equals_half_l747_74760

theorem expression_a_equals_half :
  (A : ℝ) = (1 / 2) :=
by
  let A := (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))
  sorry

theorem expression_c_equals_half :
  (C : ℝ) = (1 / 2) :=
by
  let C := (Real.tan (22.5 * Real.pi / 180)) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)
  sorry

end NUMINAMATH_GPT_expression_a_equals_half_expression_c_equals_half_l747_74760


namespace NUMINAMATH_GPT_ratio_in_sequence_l747_74757

theorem ratio_in_sequence (a1 a2 b1 b2 b3 : ℝ)
  (h1 : ∃ d, a1 = 1 + d ∧ a2 = 1 + 2 * d ∧ 9 = 1 + 3 * d)
  (h2 : ∃ r, b1 = 1 * r ∧ b2 = 1 * r^2 ∧ b3 = 1 * r^3 ∧ 9 = 1 * r^4) :
  b2 / (a1 + a2) = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_ratio_in_sequence_l747_74757


namespace NUMINAMATH_GPT_only_valid_pairs_l747_74708

theorem only_valid_pairs (a b : ℕ) (h₁ : a ≥ 1) (h₂ : b ≥ 1) :
  a^b^2 = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3) :=
by
  sorry

end NUMINAMATH_GPT_only_valid_pairs_l747_74708


namespace NUMINAMATH_GPT_simplify_fractions_l747_74756

theorem simplify_fractions :
  (240 / 20) * (6 / 180) * (10 / 4) = 1 :=
by sorry

end NUMINAMATH_GPT_simplify_fractions_l747_74756


namespace NUMINAMATH_GPT_cos_squared_sin_pi_over_2_plus_alpha_l747_74737

variable (α : ℝ)

-- Given conditions
def cond1 : Prop := (Real.pi / 2) < α * Real.pi
def cond2 : Prop := Real.cos α = -3 / 5

-- Proof goal
theorem cos_squared_sin_pi_over_2_plus_alpha :
  cond1 α → cond2 α →
  (Real.cos (Real.sin (Real.pi / 2 + α)))^2 = 8 / 25 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_cos_squared_sin_pi_over_2_plus_alpha_l747_74737


namespace NUMINAMATH_GPT_algebra_expr_eval_l747_74730

theorem algebra_expr_eval {x y : ℝ} (h : x - 2 * y = 3) : 5 - 2 * x + 4 * y = -1 :=
by sorry

end NUMINAMATH_GPT_algebra_expr_eval_l747_74730


namespace NUMINAMATH_GPT_gcd_12345_6789_l747_74777

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_12345_6789_l747_74777


namespace NUMINAMATH_GPT_last_three_digits_of_expression_l747_74784

theorem last_three_digits_of_expression : 
  let prod := 301 * 402 * 503 * 604 * 646 * 547 * 448 * 349
  (prod ^ 3) % 1000 = 976 :=
by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_expression_l747_74784


namespace NUMINAMATH_GPT_percentage_difference_l747_74731

variables (P P' : ℝ)

theorem percentage_difference (h : P' = 1.25 * P) :
  ((P' - P) / P') * 100 = 20 :=
by sorry

end NUMINAMATH_GPT_percentage_difference_l747_74731


namespace NUMINAMATH_GPT_reverse_difference_198_l747_74749

theorem reverse_difference_198 (a : ℤ) : 
  let N := 100 * (a - 1) + 10 * a + (a + 1)
  let M := 100 * (a + 1) + 10 * a + (a - 1)
  M - N = 198 := 
by
  sorry

end NUMINAMATH_GPT_reverse_difference_198_l747_74749


namespace NUMINAMATH_GPT_bus_stops_per_hour_l747_74796

-- Define the speeds as constants
def speed_excluding_stoppages : ℝ := 60
def speed_including_stoppages : ℝ := 50

-- Formulate the main theorem
theorem bus_stops_per_hour :
  (1 - speed_including_stoppages / speed_excluding_stoppages) * 60 = 10 := 
by
  sorry

end NUMINAMATH_GPT_bus_stops_per_hour_l747_74796


namespace NUMINAMATH_GPT_find_x_l747_74769

theorem find_x (p : ℕ) (hprime : Nat.Prime p) (hgt5 : p > 5) (x : ℕ) (hx : x ≠ 0) :
    (∀ n : ℕ, 0 < n → (5 * p + x) ∣ (5 * p ^ n + x ^ n)) ↔ x = p := by
  sorry

end NUMINAMATH_GPT_find_x_l747_74769


namespace NUMINAMATH_GPT_Brad_has_9_green_balloons_l747_74763

theorem Brad_has_9_green_balloons
  (total_balloons : ℕ)
  (red_balloons : ℕ)
  (green_balloons : ℕ)
  (h1 : total_balloons = 17)
  (h2 : red_balloons = 8)
  (h3 : total_balloons = red_balloons + green_balloons) :
  green_balloons = 9 := 
sorry

end NUMINAMATH_GPT_Brad_has_9_green_balloons_l747_74763


namespace NUMINAMATH_GPT_distance_to_town_l747_74735

theorem distance_to_town (fuel_efficiency : ℝ) (fuel_used : ℝ) (distance : ℝ) : 
  fuel_efficiency = 70 / 10 → 
  fuel_used = 20 → 
  distance = fuel_efficiency * fuel_used → 
  distance = 140 :=
by
  intros
  sorry

end NUMINAMATH_GPT_distance_to_town_l747_74735


namespace NUMINAMATH_GPT_inequality_always_true_l747_74739

variable (a b c : ℝ)

theorem inequality_always_true (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end NUMINAMATH_GPT_inequality_always_true_l747_74739


namespace NUMINAMATH_GPT_manager_salary_4200_l747_74765

theorem manager_salary_4200
    (avg_salary_employees : ℕ → ℕ → ℕ) 
    (total_salary_employees : ℕ → ℕ → ℕ)
    (new_avg_salary : ℕ → ℕ → ℕ)
    (total_salary_with_manager : ℕ → ℕ → ℕ) 
    (n_employees : ℕ)
    (employee_salary : ℕ) 
    (n_total : ℕ)
    (total_salary_before : ℕ)
    (avg_increase : ℕ)
    (new_employee_salary : ℕ) 
    (total_salary_after : ℕ) 
    (manager_salary : ℕ) :
    n_employees = 15 →
    employee_salary = 1800 →
    avg_increase = 150 →
    avg_salary_employees n_employees employee_salary = 1800 →
    total_salary_employees n_employees employee_salary = 27000 →
    new_avg_salary employee_salary avg_increase = 1950 →
    new_employee_salary = 1950 →
    total_salary_with_manager (n_employees + 1) new_employee_salary = 31200 →
    total_salary_before = 27000 →
    total_salary_after = 31200 →
    manager_salary = total_salary_after - total_salary_before →
    manager_salary = 4200 := 
by 
  intros 
  sorry

end NUMINAMATH_GPT_manager_salary_4200_l747_74765


namespace NUMINAMATH_GPT_calculation_correct_l747_74722

def calculation : ℝ := 1.23 * 67 + 8.2 * 12.3 - 90 * 0.123

theorem calculation_correct : calculation = 172.20 := by
  sorry

end NUMINAMATH_GPT_calculation_correct_l747_74722


namespace NUMINAMATH_GPT_find_diameter_endpoint_l747_74753

def circle_center : ℝ × ℝ := (4, 1)
def diameter_endpoint_1 : ℝ × ℝ := (1, 5)

theorem find_diameter_endpoint :
  let (h, k) := circle_center
  let (x1, y1) := diameter_endpoint_1
  (2 * h - x1, 2 * k - y1) = (7, -3) :=
by
  let (h, k) := circle_center
  let (x1, y1) := diameter_endpoint_1
  sorry

end NUMINAMATH_GPT_find_diameter_endpoint_l747_74753


namespace NUMINAMATH_GPT_car_collision_frequency_l747_74701

theorem car_collision_frequency
  (x : ℝ)
  (h_collision : ∀ t : ℝ, t > 0 → ∃ n : ℕ, t = n * x)
  (h_big_crash : ∀ t : ℝ, t > 0 → ∃ n : ℕ, t = n * 20)
  (h_total_accidents : 240 / x + 240 / 20 = 36) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_car_collision_frequency_l747_74701


namespace NUMINAMATH_GPT_prime_condition_l747_74718

theorem prime_condition (p : ℕ) (hp : Nat.Prime p) (h2p : Nat.Prime (p + 2)) : p = 3 ∨ 6 ∣ (p + 1) := 
sorry

end NUMINAMATH_GPT_prime_condition_l747_74718


namespace NUMINAMATH_GPT_remainder_when_divided_by_8_l747_74759

theorem remainder_when_divided_by_8:
  ∀ (n : ℕ), (∃ (q : ℕ), n = 7 * q + 5) → n % 8 = 1 :=
by
  intro n h
  rcases h with ⟨q, hq⟩
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_8_l747_74759


namespace NUMINAMATH_GPT_min_value_of_quadratic_l747_74790

def quadratic_function (x : ℝ) : ℝ := x^2 + 6 * x + 13

theorem min_value_of_quadratic :
  (∃ x : ℝ, quadratic_function x = 4) ∧ (∀ y : ℝ, quadratic_function y ≥ 4) :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l747_74790


namespace NUMINAMATH_GPT_division_sum_l747_74792

theorem division_sum (quotient divisor remainder : ℕ) (hquot : quotient = 65) (hdiv : divisor = 24) (hrem : remainder = 5) : 
  (divisor * quotient + remainder) = 1565 := by 
  sorry

end NUMINAMATH_GPT_division_sum_l747_74792


namespace NUMINAMATH_GPT_sequence_an_correct_l747_74781

theorem sequence_an_correct (S_n : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, S_n n = n^2 + 1) :
  (a 1 = 2) ∧ (∀ n, n ≥ 2 → a n = 2 * n - 1) :=
by
  -- We assume S_n is defined such that S_n = n^2 + 1
  -- From this, we have to show that:
  -- for n = 1, a_1 = 2,
  -- and for n ≥ 2, a_n = 2n - 1
  sorry

end NUMINAMATH_GPT_sequence_an_correct_l747_74781


namespace NUMINAMATH_GPT_digit_150_of_17_div_70_is_2_l747_74700

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_digit_150_of_17_div_70_is_2_l747_74700


namespace NUMINAMATH_GPT_tan_a_div_tan_b_l747_74719

variable {a b : ℝ}

-- Conditions
axiom sin_a_plus_b : Real.sin (a + b) = 1/2
axiom sin_a_minus_b : Real.sin (a - b) = 1/4

-- Proof statement (without the explicit proof)
theorem tan_a_div_tan_b : (Real.tan a) / (Real.tan b) = 3 := by
  sorry

end NUMINAMATH_GPT_tan_a_div_tan_b_l747_74719


namespace NUMINAMATH_GPT_solve_for_a_l747_74755

theorem solve_for_a (x a : ℝ) (h : x = 5) (h_eq : a * x - 8 = 10 + 4 * a) : a = 18 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l747_74755


namespace NUMINAMATH_GPT_gas_cost_is_4_l747_74707

theorem gas_cost_is_4
    (mileage_rate : ℝ)
    (truck_efficiency : ℝ)
    (profit : ℝ)
    (trip_distance : ℝ)
    (trip_cost : ℝ)
    (gallons_used : ℝ)
    (cost_per_gallon : ℝ) :
  mileage_rate = 0.5 →
  truck_efficiency = 20 →
  profit = 180 →
  trip_distance = 600 →
  trip_cost = mileage_rate * trip_distance - profit →
  gallons_used = trip_distance / truck_efficiency →
  cost_per_gallon = trip_cost / gallons_used →
  cost_per_gallon = 4 :=
by
  sorry

end NUMINAMATH_GPT_gas_cost_is_4_l747_74707


namespace NUMINAMATH_GPT_fraction_of_work_completed_l747_74712

-- Definitions
def work_rate_x : ℚ := 1 / 14
def work_rate_y : ℚ := 1 / 20
def work_rate_z : ℚ := 1 / 25

-- Given the combined work rate and time
def combined_work_rate : ℚ := work_rate_x + work_rate_y + work_rate_z
def time_worked : ℚ := 5

-- The fraction of work completed
def fraction_work_completed : ℚ := combined_work_rate * time_worked

-- Statement to prove
theorem fraction_of_work_completed : fraction_work_completed = 113 / 140 := by
  sorry

end NUMINAMATH_GPT_fraction_of_work_completed_l747_74712


namespace NUMINAMATH_GPT_range_of_m_for_log_function_domain_l747_74778

theorem range_of_m_for_log_function_domain (m : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + m > 0) → m > 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_log_function_domain_l747_74778


namespace NUMINAMATH_GPT_sequence_inequality_l747_74761

theorem sequence_inequality (a : ℕ → ℝ) 
  (h₀ : a 0 = 5) 
  (h₁ : ∀ n, a (n + 1) * a n - a n ^ 2 = 1) : 
  35 < a 600 ∧ a 600 < 35.1 :=
sorry

end NUMINAMATH_GPT_sequence_inequality_l747_74761


namespace NUMINAMATH_GPT_symmetric_point_correct_l747_74725

-- Define the point and line
def point : ℝ × ℝ := (-1, 2)
def line (x : ℝ) : ℝ := x - 1

-- Define a function that provides the symmetric point with respect to the line
def symmetric_point (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ × ℝ :=
  -- Since this function is a critical part of the problem, we won't define it explicitly. Using a placeholder.
  sorry

-- The proof problem
theorem symmetric_point_correct : symmetric_point point line = (3, -2) :=
  sorry

end NUMINAMATH_GPT_symmetric_point_correct_l747_74725


namespace NUMINAMATH_GPT_range_of_a_l747_74766

noncomputable def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

noncomputable def q (a : ℝ) : Prop :=
  a < 1 ∧ a ≠ 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) :
  (1 ≤ a ∧ a < 2) ∨ a ≤ -2 ∨ a = 0 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l747_74766


namespace NUMINAMATH_GPT_cody_paid_amount_l747_74723

/-- Cody buys $40 worth of stuff,
    the tax rate is 5%,
    he receives an $8 discount after taxes,
    and he and his friend split the final price equally.
    Prove that Cody paid $17. -/
theorem cody_paid_amount
  (initial_cost : ℝ)
  (tax_rate : ℝ)
  (discount : ℝ)
  (final_split : ℝ)
  (H1 : initial_cost = 40)
  (H2 : tax_rate = 0.05)
  (H3 : discount = 8)
  (H4 : final_split = 2) :
  (initial_cost * (1 + tax_rate) - discount) / final_split = 17 :=
by
  sorry

end NUMINAMATH_GPT_cody_paid_amount_l747_74723


namespace NUMINAMATH_GPT_greatest_positive_integer_difference_l747_74747

-- Define the conditions
def condition_x (x : ℝ) : Prop := 4 < x ∧ x < 6
def condition_y (y : ℝ) : Prop := 6 < y ∧ y < 10

-- Define the problem statement
theorem greatest_positive_integer_difference (x y : ℕ) (hx : condition_x x) (hy : condition_y y) : y - x = 4 :=
sorry

end NUMINAMATH_GPT_greatest_positive_integer_difference_l747_74747


namespace NUMINAMATH_GPT_expr1_simplified_expr2_simplified_l747_74782

variable (a x : ℝ)

theorem expr1_simplified : (-a^3 + (-4 * a^2) * a) = -5 * a^3 := 
by
  sorry

theorem expr2_simplified : (-x^2 * (-x)^2 * (-x^2)^3 - 2 * x^10) = -x^10 := 
by
  sorry

end NUMINAMATH_GPT_expr1_simplified_expr2_simplified_l747_74782


namespace NUMINAMATH_GPT_usual_time_to_school_l747_74715

theorem usual_time_to_school (R : ℝ) (T : ℝ) (h : (17 / 13) * (T - 7) = T) : T = 29.75 :=
sorry

end NUMINAMATH_GPT_usual_time_to_school_l747_74715


namespace NUMINAMATH_GPT_factor_expression_l747_74726

theorem factor_expression (x : ℤ) : 63 * x + 28 = 7 * (9 * x + 4) :=
by sorry

end NUMINAMATH_GPT_factor_expression_l747_74726


namespace NUMINAMATH_GPT_moles_of_C6H5CH3_formed_l747_74746

-- Stoichiometry of the reaction
def balanced_reaction (C6H6 CH4 C6H5CH3 H2 : ℝ) : Prop :=
  C6H6 + CH4 = C6H5CH3 + H2

-- Given conditions
def reaction_conditions (initial_CH4 : ℝ) (initial_C6H6 final_C6H5CH3 final_H2 : ℝ) : Prop :=
  balanced_reaction initial_C6H6 initial_CH4 final_C6H5CH3 final_H2 ∧ initial_CH4 = 3 ∧ final_H2 = 3

-- Theorem to prove
theorem moles_of_C6H5CH3_formed (initial_CH4 final_C6H5CH3 : ℝ) : reaction_conditions initial_CH4 3 final_C6H5CH3 3 → final_C6H5CH3 = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_moles_of_C6H5CH3_formed_l747_74746


namespace NUMINAMATH_GPT_three_digit_numbers_l747_74732

theorem three_digit_numbers (a b c n : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
    (h5 : 0 ≤ c) (h6 : c ≤ 9) (h7 : n = 100 * a + 10 * b + c) (h8 : 10 * b + c = (100 * a + 10 * b + c) / 5) :
    n = 125 ∨ n = 250 ∨ n = 375 := 
by 
  sorry

end NUMINAMATH_GPT_three_digit_numbers_l747_74732


namespace NUMINAMATH_GPT_tom_needs_44000_pounds_salt_l747_74794

theorem tom_needs_44000_pounds_salt 
  (flour_needed : ℕ)
  (flour_bag_weight : ℕ)
  (flour_bag_cost : ℕ)
  (salt_cost_per_pound : ℝ)
  (promotion_cost : ℕ)
  (ticket_price : ℕ)
  (tickets_sold : ℕ)
  (total_revenue : ℕ) 
  (expected_salt_cost : ℝ) 
  (S : ℝ) : 
  flour_needed = 500 → 
  flour_bag_weight = 50 → 
  flour_bag_cost = 20 → 
  salt_cost_per_pound = 0.2 → 
  promotion_cost = 1000 → 
  ticket_price = 20 → 
  tickets_sold = 500 → 
  total_revenue = 8798 → 
  0.2 * S = (500 * 20) - (500 / 50) * 20 - 1000 →
  S = 44000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_tom_needs_44000_pounds_salt_l747_74794


namespace NUMINAMATH_GPT_painting_time_equation_l747_74727

theorem painting_time_equation
  (Hannah_rate : ℝ)
  (Sarah_rate : ℝ)
  (combined_rate : ℝ)
  (temperature_factor : ℝ)
  (break_time : ℝ)
  (t : ℝ)
  (condition1 : Hannah_rate = 1 / 6)
  (condition2 : Sarah_rate = 1 / 8)
  (condition3 : combined_rate = (Hannah_rate + Sarah_rate) * temperature_factor)
  (condition4 : temperature_factor = 0.9)
  (condition5 : break_time = 1.5) :
  (combined_rate * (t - break_time) = 1) ↔ (t = 1 + break_time + 1 / combined_rate) :=
by
  sorry

end NUMINAMATH_GPT_painting_time_equation_l747_74727


namespace NUMINAMATH_GPT_triangle_area_l747_74743

theorem triangle_area (a b c : ℝ)
    (h1 : Polynomial.eval a (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (h2 : Polynomial.eval b (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (h3 : Polynomial.eval c (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (sum_roots : a + b + c = 4)
    (sum_prod_roots : a * b + a * c + b * c = 5)
    (prod_roots : a * b * c = 1):
    Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) = 1 :=
  sorry

end NUMINAMATH_GPT_triangle_area_l747_74743


namespace NUMINAMATH_GPT_part1_part2_l747_74740

theorem part1 (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi) (h_trig : Real.sin α + Real.cos α = 1 / 5) :
  Real.sin α - Real.cos α = 7 / 5 := sorry

theorem part2 (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi) (h_trig : Real.sin α + Real.cos α = 1 / 5) :
  Real.sin (2 * α + Real.pi / 3) = -12 / 25 - 7 * Real.sqrt 3 / 50 := sorry

end NUMINAMATH_GPT_part1_part2_l747_74740


namespace NUMINAMATH_GPT_jillian_distance_l747_74752

theorem jillian_distance : 
  ∀ (x y z : ℝ),
  (1 / 63) * x + (1 / 77) * y + (1 / 99) * z = 11 / 3 →
  (1 / 63) * z + (1 / 77) * y + (1 / 99) * x = 13 / 3 →
  x + y + z = 308 :=
by
  sorry

end NUMINAMATH_GPT_jillian_distance_l747_74752


namespace NUMINAMATH_GPT_bus_speed_l747_74776

def distance : ℝ := 350.028
def time : ℝ := 10
def speed_kmph : ℝ := 126.01

theorem bus_speed :
  (distance / time) * 3.6 = speed_kmph := 
sorry

end NUMINAMATH_GPT_bus_speed_l747_74776


namespace NUMINAMATH_GPT_complement_A_in_U_l747_74783

/-- Problem conditions -/
def is_universal_set (x : ℕ) : Prop := (x - 6) * (x + 1) ≤ 0
def A : Set ℕ := {1, 2, 4}
def U : Set ℕ := { x | is_universal_set x }

/-- Proof statement -/
theorem complement_A_in_U : (U \ A) = {3, 5, 6} :=
by
  sorry  -- replacement for the proof

end NUMINAMATH_GPT_complement_A_in_U_l747_74783


namespace NUMINAMATH_GPT_simplify_expression_l747_74799

theorem simplify_expression (x y : ℝ) : (5 - 4 * y) - (6 + 5 * y - 2 * x) = -1 - 9 * y + 2 * x := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l747_74799


namespace NUMINAMATH_GPT_sum_of_n_values_l747_74729

theorem sum_of_n_values (n_values : List ℤ) 
  (h : ∀ n ∈ n_values, ∃ k : ℤ, 24 = k * (2 * n - 1)) : n_values.sum = 2 :=
by
  -- Proof to be provided.
  sorry

end NUMINAMATH_GPT_sum_of_n_values_l747_74729


namespace NUMINAMATH_GPT_ratio_of_45_and_9_l747_74745

theorem ratio_of_45_and_9 : (45 / 9) = 5 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_45_and_9_l747_74745


namespace NUMINAMATH_GPT_fraction_sum_is_integer_l747_74764

theorem fraction_sum_is_integer (n : ℤ) : 
  ∃ k : ℤ, (n / 3 + (n^2) / 2 + (n^3) / 6) = k := 
sorry

end NUMINAMATH_GPT_fraction_sum_is_integer_l747_74764


namespace NUMINAMATH_GPT_standard_deviation_of_applicants_ages_l747_74770

noncomputable def average_age : ℝ := 30
noncomputable def max_different_ages : ℝ := 15

theorem standard_deviation_of_applicants_ages 
  (σ : ℝ)
  (h : max_different_ages = 2 * σ) 
  : σ = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_standard_deviation_of_applicants_ages_l747_74770


namespace NUMINAMATH_GPT_natasha_average_speed_l747_74711

theorem natasha_average_speed :
  (4 * 2.625 * 2) / (4 + 2) = 3.5 := 
by
  sorry

end NUMINAMATH_GPT_natasha_average_speed_l747_74711


namespace NUMINAMATH_GPT_contrapositive_of_x_squared_gt_1_l747_74709

theorem contrapositive_of_x_squared_gt_1 (x : ℝ) (h : x ≤ 1) : x^2 ≤ 1 :=
sorry

end NUMINAMATH_GPT_contrapositive_of_x_squared_gt_1_l747_74709


namespace NUMINAMATH_GPT_total_amount_of_money_l747_74720

theorem total_amount_of_money (N50 N500 : ℕ) (h1 : N50 = 97) (h2 : N50 + N500 = 108) : 
  50 * N50 + 500 * N500 = 10350 := by
  sorry

end NUMINAMATH_GPT_total_amount_of_money_l747_74720


namespace NUMINAMATH_GPT_fraction_to_terminating_decimal_l747_74703

theorem fraction_to_terminating_decimal :
  (47 : ℚ) / (2^2 * 5^4) = 0.0188 :=
sorry

end NUMINAMATH_GPT_fraction_to_terminating_decimal_l747_74703


namespace NUMINAMATH_GPT_sum_of_tetrahedron_properties_eq_14_l747_74706

-- Define the regular tetrahedron properties
def regular_tetrahedron_edges : ℕ := 6
def regular_tetrahedron_vertices : ℕ := 4
def regular_tetrahedron_faces : ℕ := 4

-- State the theorem that needs to be proven
theorem sum_of_tetrahedron_properties_eq_14 :
  regular_tetrahedron_edges + regular_tetrahedron_vertices + regular_tetrahedron_faces = 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_tetrahedron_properties_eq_14_l747_74706


namespace NUMINAMATH_GPT_hannah_total_savings_l747_74771

theorem hannah_total_savings :
  let a1 := 4
  let a2 := 2 * a1
  let a3 := 2 * a2
  let a4 := 2 * a3
  let a5 := 20
  a1 + a2 + a3 + a4 + a5 = 80 :=
by
  sorry

end NUMINAMATH_GPT_hannah_total_savings_l747_74771


namespace NUMINAMATH_GPT_scoops_arrangement_count_l747_74716

theorem scoops_arrangement_count :
  (5 * 4 * 3 * 2 * 1 = 120) :=
by
  sorry

end NUMINAMATH_GPT_scoops_arrangement_count_l747_74716


namespace NUMINAMATH_GPT_eighth_term_of_arithmetic_sequence_l747_74713

theorem eighth_term_of_arithmetic_sequence
  (a l : ℕ) (n : ℕ) (h₁ : a = 4) (h₂ : l = 88) (h₃ : n = 30) :
  (a + 7 * (l - a) / (n - 1) = (676 : ℚ) / 29) :=
by
  sorry

end NUMINAMATH_GPT_eighth_term_of_arithmetic_sequence_l747_74713


namespace NUMINAMATH_GPT_fraction_sum_eq_l747_74714

variable {x : ℝ}

theorem fraction_sum_eq (h : x ≠ -1) : 
  (x / (x + 1) ^ 2) + (1 / (x + 1) ^ 2) = 1 / (x + 1) := 
by
  sorry

end NUMINAMATH_GPT_fraction_sum_eq_l747_74714


namespace NUMINAMATH_GPT_bisection_interval_l747_74741

def f(x : ℝ) := x^3 - 2 * x - 5

theorem bisection_interval :
  f 2 < 0 ∧ f 3 > 0 ∧ f 2.5 > 0 →
  ∃ a b : ℝ, a = 2 ∧ b = 2.5 ∧ f a * f b ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_bisection_interval_l747_74741


namespace NUMINAMATH_GPT_largest_divisor_of_462_and_231_l747_74744

def is_factor (a b : ℕ) : Prop := a ∣ b

def largest_common_divisor (a b c : ℕ) : Prop :=
  is_factor c a ∧ is_factor c b ∧ (∀ d, (is_factor d a ∧ is_factor d b) → d ≤ c)

theorem largest_divisor_of_462_and_231 :
  largest_common_divisor 462 231 231 :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_462_and_231_l747_74744
