import Mathlib

namespace NUMINAMATH_GPT_number_of_triangles_from_8_points_on_circle_l1182_118230

-- Definitions based on the problem conditions
def points_on_circle : ℕ := 8

-- Problem statement without the proof
theorem number_of_triangles_from_8_points_on_circle :
  ∃ n : ℕ, n = (points_on_circle.choose 3) ∧ n = 56 := 
by
  sorry

end NUMINAMATH_GPT_number_of_triangles_from_8_points_on_circle_l1182_118230


namespace NUMINAMATH_GPT_check_blank_value_l1182_118256

/-- Define required constants and terms. -/
def six_point_five : ℚ := 6 + 1/2
def two_thirds : ℚ := 2/3
def three_point_five : ℚ := 3 + 1/2
def one_and_eight_fifteenths : ℚ := 1 + 8/15
def blank : ℚ := 3 + 1/20
def seventy_one_point_ninety_five : ℚ := 71 + 95/100

/-- The translated assumption and statement to be proved: -/
theorem check_blank_value :
  (six_point_five - two_thirds) / three_point_five - one_and_eight_fifteenths * (blank + seventy_one_point_ninety_five) = 1 :=
sorry

end NUMINAMATH_GPT_check_blank_value_l1182_118256


namespace NUMINAMATH_GPT_circle_tangent_to_y_axis_l1182_118292

/-- The relationship between the circle with the focal radius |PF| of the parabola y^2 = 2px (where p > 0)
as its diameter and the y-axis -/
theorem circle_tangent_to_y_axis
  (p : ℝ) (hp : p > 0)
  (x1 y1 : ℝ)
  (focus : ℝ × ℝ := (p / 2, 0))
  (P : ℝ × ℝ := (x1, y1))
  (center : ℝ × ℝ := ((2 * x1 + p) / 4, y1 / 2))
  (radius : ℝ := (2 * x1 + p) / 4) :
  -- proof that the circle with PF as its diameter is tangent to the y-axis
  ∃ k : ℝ, k = radius ∧ (center.1 = k) :=
sorry

end NUMINAMATH_GPT_circle_tangent_to_y_axis_l1182_118292


namespace NUMINAMATH_GPT_sequence_is_increasing_l1182_118243

variable (a_n : ℕ → ℝ)

def sequence_positive_numbers (a_n : ℕ → ℝ) : Prop :=
∀ n, 0 < a_n n

def sequence_condition (a_n : ℕ → ℝ) : Prop :=
∀ n, a_n (n + 1) = 2 * a_n n

theorem sequence_is_increasing 
  (h1 : sequence_positive_numbers a_n) 
  (h2 : sequence_condition a_n) : 
  ∀ n, a_n (n + 1) > a_n n :=
by
  sorry

end NUMINAMATH_GPT_sequence_is_increasing_l1182_118243


namespace NUMINAMATH_GPT_find_A_coords_find_AC_equation_l1182_118212

theorem find_A_coords
  (B : ℝ × ℝ) (hB : B = (1, -2))
  (median_CM : ∀ x y, 2 * x - y + 1 = 0)
  (angle_bisector_BAC : ∀ x y, x + 7 * y - 12 = 0) :
  ∃ A : ℝ × ℝ, A = (-2, 2) :=
by
  sorry

theorem find_AC_equation
  (A B : ℝ × ℝ) (hA : A = (-2, 2)) (hB : B = (1, -2))
  (median_CM : ∀ x y, 2 * x - y + 1 = 0)
  (angle_bisector_BAC : ∀ x y, x + 7 * y - 12 = 0) :
  ∃ k b : ℝ, ∀ x y, y = k * x + b ↔ 3 * x - 4 * y + 14 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_A_coords_find_AC_equation_l1182_118212


namespace NUMINAMATH_GPT_Beth_peas_count_l1182_118217

-- Definitions based on conditions
def number_of_corn : ℕ := 10
def number_of_peas (number_of_corn : ℕ) : ℕ := 2 * number_of_corn + 15

-- Theorem that represents the proof problem
theorem Beth_peas_count : number_of_peas 10 = 35 :=
by
  sorry

end NUMINAMATH_GPT_Beth_peas_count_l1182_118217


namespace NUMINAMATH_GPT_factor_expression_l1182_118263

noncomputable def expression (x : ℝ) : ℝ := (15 * x^3 + 80 * x - 5) - (-4 * x^3 + 4 * x - 5)

theorem factor_expression (x : ℝ) : expression x = 19 * x * (x^2 + 4) := 
by 
  sorry

end NUMINAMATH_GPT_factor_expression_l1182_118263


namespace NUMINAMATH_GPT_min_sum_of_squares_l1182_118267

theorem min_sum_of_squares (a b c d : ℤ) (h1 : a^2 ≠ b^2 ∧ a^2 ≠ c^2 ∧ a^2 ≠ d^2 ∧ b^2 ≠ c^2 ∧ b^2 ≠ d^2 ∧ c^2 ≠ d^2)
                           (h2 : (a * b + c * d)^2 + (a * d - b * c)^2 = 2004) :
  a^2 + b^2 + c^2 + d^2 = 2 * Int.sqrt 2004 :=
sorry

end NUMINAMATH_GPT_min_sum_of_squares_l1182_118267


namespace NUMINAMATH_GPT_find_m_in_arith_seq_l1182_118226

noncomputable def arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem find_m_in_arith_seq (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0) 
  (h_seq : arith_seq a d) 
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32) 
  (h_am : ∃ m, a m = 8) : 
  ∃ m, m = 8 := 
sorry

end NUMINAMATH_GPT_find_m_in_arith_seq_l1182_118226


namespace NUMINAMATH_GPT_pentagon_largest_angle_l1182_118251

theorem pentagon_largest_angle (x : ℝ) (h : 2 * x + 3 * x + 4 * x + 5 * x + 6 * x = 540) : 6 * x = 162 :=
sorry

end NUMINAMATH_GPT_pentagon_largest_angle_l1182_118251


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l1182_118247

theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ n → 6 + (k - 1) * 2 = 202)) ∧ n = 99 :=
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l1182_118247


namespace NUMINAMATH_GPT_min_points_in_set_M_l1182_118236
-- Import the necessary library

-- Define the problem conditions and the result to prove
theorem min_points_in_set_M :
  ∃ (M : Finset ℝ) (C₁ C₂ C₃ C₄ C₅ C₆ C₇ : Finset ℝ),
  C₇.card = 7 ∧
  C₆.card = 6 ∧
  C₅.card = 5 ∧
  C₄.card = 4 ∧
  C₃.card = 3 ∧
  C₂.card = 2 ∧
  C₁.card = 1 ∧
  C₇ ⊆ M ∧
  C₆ ⊆ M ∧
  C₅ ⊆ M ∧
  C₄ ⊆ M ∧
  C₃ ⊆ M ∧
  C₂ ⊆ M ∧
  C₁ ⊆ M ∧
  M.card = 12 :=
sorry

end NUMINAMATH_GPT_min_points_in_set_M_l1182_118236


namespace NUMINAMATH_GPT_remaining_yards_is_720_l1182_118239

-- Definitions based on conditions:
def marathon_miles : Nat := 25
def marathon_yards : Nat := 500
def yards_in_mile : Nat := 1760
def num_of_marathons : Nat := 12

-- Total distance for one marathon in yards
def one_marathon_total_yards : Nat :=
  marathon_miles * yards_in_mile + marathon_yards

-- Total distance for twelve marathons in yards
def total_distance_yards : Nat :=
  num_of_marathons * one_marathon_total_yards

-- Remaining yards after converting the total distance into miles and yards
def y : Nat :=
  total_distance_yards % yards_in_mile

-- Condition ensuring y is the remaining yards and is within the bounds 0 ≤ y < 1760
theorem remaining_yards_is_720 : 
  y = 720 := sorry

end NUMINAMATH_GPT_remaining_yards_is_720_l1182_118239


namespace NUMINAMATH_GPT_stickers_per_student_l1182_118211

theorem stickers_per_student 
  (gold_stickers : ℕ) 
  (silver_stickers : ℕ) 
  (bronze_stickers : ℕ) 
  (students : ℕ)
  (h1 : gold_stickers = 50)
  (h2 : silver_stickers = 2 * gold_stickers)
  (h3 : bronze_stickers = silver_stickers - 20)
  (h4 : students = 5) : 
  (gold_stickers + silver_stickers + bronze_stickers) / students = 46 :=
by
  sorry

end NUMINAMATH_GPT_stickers_per_student_l1182_118211


namespace NUMINAMATH_GPT_hcf_of_three_numbers_l1182_118209

theorem hcf_of_three_numbers (a b c : ℕ) (h1 : a + b + c = 60)
  (h2 : Nat.lcm (Nat.lcm a b) c = 180)
  (h3 : (1:ℚ)/a + 1/b + 1/c = 11/120)
  (h4 : a * b * c = 900) :
  Nat.gcd (Nat.gcd a b) c = 5 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_three_numbers_l1182_118209


namespace NUMINAMATH_GPT_total_goals_scored_l1182_118206

-- Definitions based on the problem conditions
def kickers_first_period_goals : ℕ := 2
def kickers_second_period_goals : ℕ := 2 * kickers_first_period_goals
def spiders_first_period_goals : ℕ := kickers_first_period_goals / 2
def spiders_second_period_goals : ℕ := 2 * kickers_second_period_goals

-- The theorem we need to prove
theorem total_goals_scored : 
  kickers_first_period_goals + kickers_second_period_goals +
  spiders_first_period_goals + spiders_second_period_goals = 15 := 
by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_total_goals_scored_l1182_118206


namespace NUMINAMATH_GPT_log_base_change_l1182_118215

theorem log_base_change (a b : ℝ) (h₁ : Real.log 2 / Real.log 10 = a) (h₂ : Real.log 3 / Real.log 10 = b) :
    Real.log 18 / Real.log 5 = (a + 2 * b) / (1 - a) := by
  sorry

end NUMINAMATH_GPT_log_base_change_l1182_118215


namespace NUMINAMATH_GPT_hoseok_result_l1182_118228

theorem hoseok_result :
  ∃ X : ℤ, (X - 46 = 15) ∧ (X - 29 = 32) :=
by
  sorry

end NUMINAMATH_GPT_hoseok_result_l1182_118228


namespace NUMINAMATH_GPT_multiplication_to_squares_l1182_118214

theorem multiplication_to_squares :
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_to_squares_l1182_118214


namespace NUMINAMATH_GPT_ratio_buses_to_cars_l1182_118235

theorem ratio_buses_to_cars (B C : ℕ) (h1 : B = C - 60) (h2 : C = 65) : B / C = 1 / 13 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_buses_to_cars_l1182_118235


namespace NUMINAMATH_GPT_mrs_oaklyn_rugs_l1182_118204

theorem mrs_oaklyn_rugs (buying_price selling_price total_profit : ℕ) (h1 : buying_price = 40) (h2 : selling_price = 60) (h3 : total_profit = 400) : 
  ∃ (num_rugs : ℕ), num_rugs = 20 :=
by
  sorry

end NUMINAMATH_GPT_mrs_oaklyn_rugs_l1182_118204


namespace NUMINAMATH_GPT_order_of_magnitude_l1182_118229

theorem order_of_magnitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let m := a / Real.sqrt b + b / Real.sqrt a
  let n := Real.sqrt a + Real.sqrt b
  let p := Real.sqrt (a + b)
  m ≥ n ∧ n > p := 
sorry

end NUMINAMATH_GPT_order_of_magnitude_l1182_118229


namespace NUMINAMATH_GPT_chloes_test_scores_l1182_118286

theorem chloes_test_scores :
  ∃ (scores : List ℕ),
  scores = [93, 92, 86, 82, 79, 78] ∧
  (List.take 4 scores).sum = 339 ∧
  scores.sum / 6 = 85 ∧
  List.Nodup scores ∧
  ∀ score ∈ scores, score < 95 :=
by
  sorry

end NUMINAMATH_GPT_chloes_test_scores_l1182_118286


namespace NUMINAMATH_GPT_solution_set_inequality_system_l1182_118244

theorem solution_set_inequality_system (
  x : ℝ
) : (x + 1 ≥ 0 ∧ (x - 1) / 2 < 1) ↔ (-1 ≤ x ∧ x < 3) := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_system_l1182_118244


namespace NUMINAMATH_GPT_min_value_z_l1182_118276

theorem min_value_z (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  ∃ z_min, z_min = (x + 1 / x) * (y + 1 / y) ∧ z_min = 33 / 4 :=
sorry

end NUMINAMATH_GPT_min_value_z_l1182_118276


namespace NUMINAMATH_GPT_merchant_profit_l1182_118216

theorem merchant_profit (C S : ℝ) (h: 20 * C = 15 * S) : 
  (S - C) / C * 100 = 33.33 := by
sorry

end NUMINAMATH_GPT_merchant_profit_l1182_118216


namespace NUMINAMATH_GPT_area_of_field_l1182_118264

theorem area_of_field : ∀ (L W : ℕ), L = 20 → L + 2 * W = 88 → L * W = 680 :=
by
  intros L W hL hEq
  rw [hL] at hEq
  sorry

end NUMINAMATH_GPT_area_of_field_l1182_118264


namespace NUMINAMATH_GPT_ishaan_age_eq_6_l1182_118220

-- Variables for ages
variable (I : ℕ) -- Ishaan's current age

-- Constants for ages
def daniel_current_age := 69
def years := 15
def daniel_future_age := daniel_current_age + years

-- Lean theorem statement
theorem ishaan_age_eq_6 
    (h1 : daniel_current_age = 69)
    (h2 : daniel_future_age = 4 * (I + years)) : 
    I = 6 := by
  sorry

end NUMINAMATH_GPT_ishaan_age_eq_6_l1182_118220


namespace NUMINAMATH_GPT_largest_sum_of_watch_digits_l1182_118284

theorem largest_sum_of_watch_digits : ∃ s : ℕ, s = 23 ∧ 
  (∀ h m : ℕ, h < 24 → m < 60 → s ≤ (h / 10 + h % 10 + m / 10 + m % 10)) :=
by
  sorry

end NUMINAMATH_GPT_largest_sum_of_watch_digits_l1182_118284


namespace NUMINAMATH_GPT_N_prime_iff_k_eq_2_l1182_118237

/-- Define the number N for a given k -/
def N (k : ℕ) : ℕ := (10 ^ (2 * k) - 1) / 99

/-- Statement: Prove that N is prime if and only if k = 2 -/
theorem N_prime_iff_k_eq_2 (k : ℕ) : Prime (N k) ↔ k = 2 := by
  sorry

end NUMINAMATH_GPT_N_prime_iff_k_eq_2_l1182_118237


namespace NUMINAMATH_GPT_matrix_exp_1000_l1182_118224

-- Define the matrix as a constant
noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![2, 1]]

-- The property of matrix exponentiation
theorem matrix_exp_1000 :
  A^1000 = ![![1, 0], ![2000, 1]] :=
by
  sorry

end NUMINAMATH_GPT_matrix_exp_1000_l1182_118224


namespace NUMINAMATH_GPT_toms_animal_robots_l1182_118278

theorem toms_animal_robots (h : ∀ (m t : ℕ), t = 2 * m) (hmichael : 8 = m) : ∃ t, t = 16 := 
by
  sorry

end NUMINAMATH_GPT_toms_animal_robots_l1182_118278


namespace NUMINAMATH_GPT_equal_partition_of_weights_l1182_118202

theorem equal_partition_of_weights 
  (weights : Fin 2009 → ℕ) 
  (h1 : ∀ i : Fin 2008, (weights i + 1 = weights (i + 1)) ∨ (weights i = weights (i + 1) + 1))
  (h2 : ∀ i : Fin 2009, weights i ≤ 1000)
  (h3 : (Finset.univ.sum weights) % 2 = 0) :
  ∃ (A B : Finset (Fin 2009)), (A ∪ B = Finset.univ ∧ A ∩ B = ∅ ∧ A.sum weights = B.sum weights) :=
sorry

end NUMINAMATH_GPT_equal_partition_of_weights_l1182_118202


namespace NUMINAMATH_GPT_sum_of_y_neg_l1182_118232

-- Define the conditions from the problem
def condition1 (x y : ℝ) : Prop := x + y = 7
def condition2 (x z : ℝ) : Prop := x * z = -180
def condition3 (x y z : ℝ) : Prop := (x + y + z)^2 = 4

-- Define the main theorem to prove
theorem sum_of_y_neg (x y z : ℝ) (S : ℝ) :
  (condition1 x y) ∧ (condition2 x z) ∧ (condition3 x y z) →
  (S = (-29) + (-13)) →
  -S = 42 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_y_neg_l1182_118232


namespace NUMINAMATH_GPT_rabbits_in_cage_l1182_118268

theorem rabbits_in_cage (heads legs : ℝ) (total_heads : heads = 40) 
  (condition : legs = 8 + 10 * (2 * (heads - rabbits))) :
  ∃ rabbits : ℝ, rabbits = 33 :=
by
  sorry

end NUMINAMATH_GPT_rabbits_in_cage_l1182_118268


namespace NUMINAMATH_GPT_sum_of_roots_l1182_118208

theorem sum_of_roots (a b c : ℝ) (h : 6 * a^3 + 7 * a^2 - 12 * a = 0) : 
  - (7 / 6 : ℝ) = -1.17 := 
sorry

end NUMINAMATH_GPT_sum_of_roots_l1182_118208


namespace NUMINAMATH_GPT_calculate_total_houses_built_l1182_118257

theorem calculate_total_houses_built :
  let initial_houses := 1426
  let final_houses := 2000
  let rate_a := 25
  let time_a := 6
  let rate_b := 15
  let time_b := 9
  let rate_c := 30
  let time_c := 4
  let total_houses_built := (rate_a * time_a) + (rate_b * time_b) + (rate_c * time_c)
  total_houses_built = 405 :=
by
  sorry

end NUMINAMATH_GPT_calculate_total_houses_built_l1182_118257


namespace NUMINAMATH_GPT_smallest_positive_angle_l1182_118258

theorem smallest_positive_angle (theta : ℝ) (h_theta : theta = -2002) :
  ∃ α : ℝ, 0 < α ∧ α < 360 ∧ ∃ k : ℤ, theta = α + k * 360 ∧ α = 158 := 
by
  sorry

end NUMINAMATH_GPT_smallest_positive_angle_l1182_118258


namespace NUMINAMATH_GPT_probability_final_marble_red_l1182_118274

theorem probability_final_marble_red :
  let P_wA := 5/8
  let P_bA := 3/8
  let P_gB_p1 := 6/10
  let P_rB_p2 := 4/10
  let P_gC_p1 := 4/7
  let P_rC_p2 := 3/7
  let P_wr_b_g := P_wA * P_gB_p1 * P_rB_p2
  let P_blk_g_red := P_bA * P_gC_p1 * P_rC_p2
  (P_wr_b_g + P_blk_g_red) = 79/980 :=
by {
  let P_wA := 5/8
  let P_bA := 3/8
  let P_gB_p1 := 6/10
  let P_rB_p2 := 4/10
  let P_gC_p1 := 4/7
  let P_rC_p2 := 3/7
  let P_wr_b_g := P_wA * P_gB_p1 * P_rB_p2
  let P_blk_g_red := P_bA * P_gC_p1 * P_rC_p2
  show (P_wr_b_g + P_blk_g_red) = 79/980
  sorry
}

end NUMINAMATH_GPT_probability_final_marble_red_l1182_118274


namespace NUMINAMATH_GPT_roots_sum_condition_l1182_118262

theorem roots_sum_condition (a b : ℝ) 
  (h1 : ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 9) 
    ∧ (x * y + y * z + x * z = a) ∧ (x * y * z = b)) :
  a + b = 38 := 
sorry

end NUMINAMATH_GPT_roots_sum_condition_l1182_118262


namespace NUMINAMATH_GPT_tan_sum_eq_tan_product_l1182_118240

theorem tan_sum_eq_tan_product {α β γ : ℝ} 
  (h_sum : α + β + γ = π) : 
    Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_eq_tan_product_l1182_118240


namespace NUMINAMATH_GPT_fibonacci_problem_l1182_118265

theorem fibonacci_problem 
  (F : ℕ → ℕ)
  (h1 : F 1 = 1)
  (h2 : F 2 = 1)
  (h3 : ∀ n ≥ 3, F n = F (n - 1) + F (n - 2))
  (a b c : ℕ)
  (h4 : F c = 2 * F b - F a)
  (h5 : F c - F a = F a)
  (h6 : a + c = 1700) :
  a = 849 := 
sorry

end NUMINAMATH_GPT_fibonacci_problem_l1182_118265


namespace NUMINAMATH_GPT_quadratic_function_value_l1182_118290

theorem quadratic_function_value
  (p q r : ℝ)
  (h1 : p + q + r = 3)
  (h2 : 4 * p + 2 * q + r = 12) :
  p + q + 3 * r = -5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_value_l1182_118290


namespace NUMINAMATH_GPT_motherGaveMoney_l1182_118210

-- Define the given constants and fact
def initialMoney : Real := 0.85
def foundMoney : Real := 0.50
def toyCost : Real := 1.60
def remainingMoney : Real := 0.15

-- Define the unknown amount given by his mother
def motherMoney (M : Real) := initialMoney + M + foundMoney - toyCost = remainingMoney

-- Statement to prove
theorem motherGaveMoney : ∃ M : Real, motherMoney M ∧ M = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_motherGaveMoney_l1182_118210


namespace NUMINAMATH_GPT_find_CD_squared_l1182_118266

noncomputable def first_circle (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 25
noncomputable def second_circle (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 25

theorem find_CD_squared : ∃ C D : ℝ × ℝ, 
  (first_circle C.1 C.2 ∧ second_circle C.1 C.2) ∧ 
  (first_circle D.1 D.2 ∧ second_circle D.1 D.2) ∧ 
  (C ≠ D) ∧ 
  ((D.1 - C.1)^2 + (D.2 - C.2)^2 = 50) :=
by
  sorry

end NUMINAMATH_GPT_find_CD_squared_l1182_118266


namespace NUMINAMATH_GPT_find_value_of_fraction_l1182_118241

noncomputable def a : ℝ := 5 * (Real.sqrt 2) + 7

theorem find_value_of_fraction (h : (20 * a) / (a^2 + 1) = Real.sqrt 2) (h1 : 1 < a) : 
  (14 * a) / (a^2 - 1) = 1 := 
by 
  have h_sqrt : 20 * a = Real.sqrt 2 * a^2 + Real.sqrt 2 := by sorry
  have h_rearrange : Real.sqrt 2 * a^2 - 20 * a + Real.sqrt 2 = 0 := by sorry
  have h_solution : a = 5 * (Real.sqrt 2) + 7 := by sorry
  have h_asquare : a^2 = 99 + 70 * (Real.sqrt 2) := by sorry
  exact sorry

end NUMINAMATH_GPT_find_value_of_fraction_l1182_118241


namespace NUMINAMATH_GPT_abc_equality_l1182_118275

theorem abc_equality (a b c : ℕ) (h1 : b = a^2 - a) (h2 : c = b^2 - b) (h3 : a = c^2 - c) : 
  a = 2 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end NUMINAMATH_GPT_abc_equality_l1182_118275


namespace NUMINAMATH_GPT_find_a_from_complex_condition_l1182_118253

theorem find_a_from_complex_condition (a : ℝ) (x y : ℝ) 
  (h : x = -1 ∧ y = -2 * a)
  (h_line : x - y = 0) : a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_from_complex_condition_l1182_118253


namespace NUMINAMATH_GPT_sum_of_x_y_l1182_118271

theorem sum_of_x_y (x y : ℕ) (x_square_condition : ∃ x, ∃ n : ℕ, 450 * x = n^2)
                   (y_cube_condition : ∃ y, ∃ m : ℕ, 450 * y = m^3) :
                   x = 2 ∧ y = 4 → x + y = 6 := 
sorry

end NUMINAMATH_GPT_sum_of_x_y_l1182_118271


namespace NUMINAMATH_GPT_volume_of_solid_of_revolution_l1182_118289

theorem volume_of_solid_of_revolution (a : ℝ) : 
  let h := a / 2
  let r := (Real.sqrt 3 / 2) * a
  2 * (1 / 3) * π * r^2 * h = (π * a^3) / 4 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_solid_of_revolution_l1182_118289


namespace NUMINAMATH_GPT_average_salary_l1182_118223

theorem average_salary
  (num_technicians : ℕ) (avg_salary_technicians : ℝ)
  (num_other_workers : ℕ) (avg_salary_other_workers : ℝ)
  (total_num_workers : ℕ) (avg_salary_all_workers : ℝ) :
  num_technicians = 7 →
  avg_salary_technicians = 14000 →
  num_other_workers = total_num_workers - num_technicians →
  avg_salary_other_workers = 6000 →
  total_num_workers = 28 →
  avg_salary_all_workers = (num_technicians * avg_salary_technicians + num_other_workers * avg_salary_other_workers) / total_num_workers →
  avg_salary_all_workers = 8000 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_average_salary_l1182_118223


namespace NUMINAMATH_GPT_initial_position_of_M_l1182_118252

theorem initial_position_of_M :
  ∃ x : ℤ, (x + 7) - 4 = 0 ∧ x = -3 :=
by sorry

end NUMINAMATH_GPT_initial_position_of_M_l1182_118252


namespace NUMINAMATH_GPT_volume_of_pyramid_l1182_118299

noncomputable def pyramid_volume : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 0)
  let C : ℝ × ℝ := (12, 20)
  let D : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2) -- Midpoint of BC
  let E : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) -- Midpoint of AC
  let F : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) -- Midpoint of AB
  let height : ℝ := 8.42 -- Vertically above the orthocenter
  let base_area : ℝ := 110 -- Area of the midpoint triangle
  (1 / 3) * base_area * height

theorem volume_of_pyramid : pyramid_volume = 309.07 :=
  by
    sorry

end NUMINAMATH_GPT_volume_of_pyramid_l1182_118299


namespace NUMINAMATH_GPT_age_difference_l1182_118218

variable (y m e : ℕ)

theorem age_difference (h1 : m = y + 3) (h2 : e = 3 * y) (h3 : e = 15) : 
  ∃ x, e = y + m + x ∧ x = 2 := by
  sorry

end NUMINAMATH_GPT_age_difference_l1182_118218


namespace NUMINAMATH_GPT_combination_sum_l1182_118200

-- Definition of combination, also known as binomial coefficient
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem statement
theorem combination_sum :
  (combination 8 2) + (combination 8 3) = 84 :=
by
  sorry

end NUMINAMATH_GPT_combination_sum_l1182_118200


namespace NUMINAMATH_GPT_num_mystery_shelves_l1182_118281

def num_books_per_shelf : ℕ := 9
def num_picture_shelves : ℕ := 2
def total_books : ℕ := 72
def num_books_from_picture_shelves : ℕ := num_picture_shelves * num_books_per_shelf
def num_books_from_mystery_shelves : ℕ := total_books - num_books_from_picture_shelves

theorem num_mystery_shelves :
  num_books_from_mystery_shelves / num_books_per_shelf = 6 := by
sorry

end NUMINAMATH_GPT_num_mystery_shelves_l1182_118281


namespace NUMINAMATH_GPT_xyz_zero_unique_solution_l1182_118219

theorem xyz_zero_unique_solution {x y z : ℝ} (h1 : x^2 * y + y^2 * z + z^2 = 0)
                                 (h2 : z^3 + z^2 * y + z * y^3 + x^2 * y = 1 / 4 * (x^4 + y^4)) :
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end NUMINAMATH_GPT_xyz_zero_unique_solution_l1182_118219


namespace NUMINAMATH_GPT_polynomial_geometric_roots_k_value_l1182_118293

theorem polynomial_geometric_roots_k_value 
    (j k : ℝ)
    (h : ∃ (a r : ℝ), a ≠ 0 ∧ r ≠ 0 ∧ 
      (∀ u v : ℝ, (u = a ∨ u = a * r ∨ u = a * r^2 ∨ u = a * r^3) →
        (v = a ∨ v = a * r ∨ v = a * r^2 ∨ v = a * r^3) →
        u ≠ v) ∧ 
      (a + a * r + a * r^2 + a * r^3 = 0) ∧
      (a^4 * r^6 = 900)) :
  k = -900 :=
sorry

end NUMINAMATH_GPT_polynomial_geometric_roots_k_value_l1182_118293


namespace NUMINAMATH_GPT_value_of_fraction_zero_l1182_118234

theorem value_of_fraction_zero (x : ℝ) (h1 : x^2 - 1 = 0) (h2 : 1 - x ≠ 0) : x = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fraction_zero_l1182_118234


namespace NUMINAMATH_GPT_faster_train_length_l1182_118294

theorem faster_train_length
  (speed_faster : ℝ)
  (speed_slower : ℝ)
  (time_to_cross : ℝ)
  (relative_speed_limit: ℝ)
  (h1 : speed_faster = 108 * 1000 / 3600)
  (h2: speed_slower = 36 * 1000 / 3600)
  (h3: time_to_cross = 17)
  (h4: relative_speed_limit = 2) :
  (speed_faster - speed_slower) * time_to_cross = 340 := 
sorry

end NUMINAMATH_GPT_faster_train_length_l1182_118294


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1182_118296

theorem neither_sufficient_nor_necessary (a b : ℝ) (h : a^2 > b^2) : 
  ¬(a > b) ∨ ¬(b > a) := sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1182_118296


namespace NUMINAMATH_GPT_find_number_l1182_118277

theorem find_number (x : ℝ) (h : 3034 - x / 20.04 = 2984) : x = 1002 :=
sorry

end NUMINAMATH_GPT_find_number_l1182_118277


namespace NUMINAMATH_GPT_matrix_B_power_103_l1182_118297

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem matrix_B_power_103 :
  B ^ 103 = B :=
by
  sorry

end NUMINAMATH_GPT_matrix_B_power_103_l1182_118297


namespace NUMINAMATH_GPT_jake_hours_of_work_l1182_118231

def initialDebt : ℕ := 100
def amountPaid : ℕ := 40
def workRate : ℕ := 15
def remainingDebt : ℕ := initialDebt - amountPaid

theorem jake_hours_of_work : remainingDebt / workRate = 4 := by
  sorry

end NUMINAMATH_GPT_jake_hours_of_work_l1182_118231


namespace NUMINAMATH_GPT_even_function_properties_l1182_118283

theorem even_function_properties 
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_increasing : ∀ x y : ℝ, 5 ≤ x ∧ x ≤ y ∧ y ≤ 7 → f x ≤ f y)
  (h_min_value : ∀ x : ℝ, 5 ≤ x ∧ x ≤ 7 → 6 ≤ f x) :
  (∀ x y : ℝ, -7 ≤ x ∧ x ≤ y ∧ y ≤ -5 → f y ≤ f x) ∧ (∀ x : ℝ, -7 ≤ x ∧ x ≤ -5 → 6 ≤ f x) :=
by
  sorry

end NUMINAMATH_GPT_even_function_properties_l1182_118283


namespace NUMINAMATH_GPT_lindsey_squat_weight_l1182_118280

-- Define the conditions
def num_bands : ℕ := 2
def resistance_per_band : ℤ := 5
def dumbbell_weight : ℤ := 10

-- Define the weight Lindsay will squat
def total_weight : ℤ := num_bands * resistance_per_band + dumbbell_weight

-- State the theorem
theorem lindsey_squat_weight : total_weight = 20 :=
by
  sorry

end NUMINAMATH_GPT_lindsey_squat_weight_l1182_118280


namespace NUMINAMATH_GPT_total_cleaning_validation_l1182_118259

-- Define the cleaning frequencies and their vacations
def Michael_bath_week := 2
def Michael_shower_week := 1
def Michael_vacation_weeks := 3

def Angela_shower_day := 1
def Angela_vacation_weeks := 2

def Lucy_bath_week := 3
def Lucy_shower_week := 2
def Lucy_alter_weeks := 4
def Lucy_alter_shower_day := 1
def Lucy_alter_bath_week := 1

def weeks_year := 52
def days_week := 7

-- Calculate Michael's total cleaning times in a year
def Michael_total := (Michael_bath_week * weeks_year) + (Michael_shower_week * weeks_year)
def Michael_vacation_reduction := Michael_vacation_weeks * (Michael_bath_week + Michael_shower_week)
def Michael_cleaning_times := Michael_total - Michael_vacation_reduction

-- Calculate Angela's total cleaning times in a year
def Angela_total := (Angela_shower_day * days_week * weeks_year)
def Angela_vacation_reduction := Angela_vacation_weeks * (Angela_shower_day * days_week)
def Angela_cleaning_times := Angela_total - Angela_vacation_reduction

-- Calculate Lucy's total cleaning times in a year
def Lucy_baths_total := Lucy_bath_week * weeks_year
def Lucy_showers_total := Lucy_shower_week * weeks_year
def Lucy_alter_showers := Lucy_alter_shower_day * days_week * Lucy_alter_weeks
def Lucy_alter_baths_reduction := (Lucy_bath_week - Lucy_alter_bath_week) * Lucy_alter_weeks
def Lucy_cleaning_times := Lucy_baths_total + Lucy_showers_total + Lucy_alter_showers - Lucy_alter_baths_reduction

-- Calculate the total times they clean themselves in 52 weeks
def total_cleaning_times := Michael_cleaning_times + Angela_cleaning_times + Lucy_cleaning_times

-- The proof statement
theorem total_cleaning_validation : total_cleaning_times = 777 :=
by simp [Michael_cleaning_times, Angela_cleaning_times, Lucy_cleaning_times, total_cleaning_times]; sorry

end NUMINAMATH_GPT_total_cleaning_validation_l1182_118259


namespace NUMINAMATH_GPT_minimum_perimeter_l1182_118288

noncomputable def minimum_perimeter_triangle (l m n : ℕ) : ℕ :=
  l + m + n

theorem minimum_perimeter :
  ∀ (l m n : ℕ),
    (l > m) → (m > n) → 
    ((∃ k : ℕ, 10^4 ∣ 3^l - 3^m + k * 10^4) ∧ (∃ k : ℕ, 10^4 ∣ 3^m - 3^n + k * 10^4) ∧ (∃ k : ℕ, 10^4 ∣ 3^l - 3^n + k * 10^4)) →
    minimum_perimeter_triangle l m n = 3003 :=
by
  intros l m n hlm hmn hmod
  sorry

end NUMINAMATH_GPT_minimum_perimeter_l1182_118288


namespace NUMINAMATH_GPT_probability_of_AB_not_selected_l1182_118295

-- The definition for the probability of not selecting both A and B 
def probability_not_selected : ℚ :=
  let total_ways := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial (4 - 2))
  let favorable_ways := 1 -- Only the selection of C and D
  favorable_ways / total_ways

-- The theorem stating the desired probability
theorem probability_of_AB_not_selected : probability_not_selected = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_AB_not_selected_l1182_118295


namespace NUMINAMATH_GPT_circle_diameter_line_eq_l1182_118205

theorem circle_diameter_line_eq (x y : ℝ) :
  x^2 + y^2 - 2*x + 6*y + 8 = 0 → (2 * 1 + (-3) + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_line_eq_l1182_118205


namespace NUMINAMATH_GPT_set_intersection_eq_l1182_118242

def A : Set ℝ := {x | |x - 1| ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x > 0}

theorem set_intersection_eq :
  A ∩ (Set.univ \ B) = {x | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_set_intersection_eq_l1182_118242


namespace NUMINAMATH_GPT_bamboo_consumption_correct_l1182_118270

-- Define the daily bamboo consumption for adult and baby pandas
def adult_daily_bamboo : ℕ := 138
def baby_daily_bamboo : ℕ := 50

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total bamboo consumed by an adult panda in a week
def adult_weekly_bamboo := adult_daily_bamboo * days_in_week

-- Define the total bamboo consumed by a baby panda in a week
def baby_weekly_bamboo := baby_daily_bamboo * days_in_week

-- Define the total bamboo consumed by both pandas in a week
def total_bamboo_consumed := adult_weekly_bamboo + baby_weekly_bamboo

-- The theorem states that the total bamboo consumption in a week is 1316 pounds
theorem bamboo_consumption_correct : total_bamboo_consumed = 1316 := by
  sorry

end NUMINAMATH_GPT_bamboo_consumption_correct_l1182_118270


namespace NUMINAMATH_GPT_sugar_amount_l1182_118213

-- Definitions based on conditions
variables (S F B C : ℝ) -- S = amount of sugar, F = amount of flour, B = amount of baking soda, C = amount of chocolate chips

-- Conditions
def ratio_sugar_flour (S F : ℝ) : Prop := S / F = 5 / 4
def ratio_flour_baking_soda (F B : ℝ) : Prop := F / B = 10 / 1
def ratio_baking_soda_chocolate_chips (B C : ℝ) : Prop := B / C = 3 / 2
def new_ratio_flour_baking_soda_chocolate_chips (F B C : ℝ) : Prop :=
  F / (B + 120) = 16 / 3 ∧ F / (C + 50) = 16 / 2

-- Prove that the current amount of sugar is 1714 pounds
theorem sugar_amount (S F B C : ℝ) (h1 : ratio_sugar_flour S F)
  (h2 : ratio_flour_baking_soda F B) (h3 : ratio_baking_soda_chocolate_chips B C)
  (h4 : new_ratio_flour_baking_soda_chocolate_chips F B C) : 
  S = 1714 :=
sorry

end NUMINAMATH_GPT_sugar_amount_l1182_118213


namespace NUMINAMATH_GPT_exists_additive_function_close_to_f_l1182_118227

variable (f : ℝ → ℝ)

theorem exists_additive_function_close_to_f (h : ∀ x y : ℝ, |f (x + y) - f x - f y| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x : ℝ, |f x - g x| ≤ 1) ∧ (∀ x y : ℝ, g (x + y) = g x + g y) := by
  sorry

end NUMINAMATH_GPT_exists_additive_function_close_to_f_l1182_118227


namespace NUMINAMATH_GPT_incorrect_statement_g2_l1182_118246

def g (x : ℚ) : ℚ := (2 * x + 3) / (x - 2)

theorem incorrect_statement_g2 : g 2 ≠ 0 := by
  sorry

end NUMINAMATH_GPT_incorrect_statement_g2_l1182_118246


namespace NUMINAMATH_GPT_max_not_divisible_by_3_l1182_118298

theorem max_not_divisible_by_3 (a b c d e f : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) (h7 : 3 ∣ (a * b * c * d * e * f)) : 
  ∃ x y z u v, ((x = a ∧ y = b ∧ z = c ∧ u = d ∧ v = e) ∨ (x = a ∧ y = b ∧ z = c ∧ u = d ∧ v = f) ∨ (x = a ∧ y = b ∧ z = c ∧ u = e ∧ v = f) ∨ (x = a ∧ y = b ∧ z = d ∧ u = e ∧ v = f) ∨ (x = a ∧ y = c ∧ z = d ∧ u = e ∧ v = f) ∨ (x = b ∧ y = c ∧ z = d ∧ u = e ∧ v = f)) ∧ (¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z) ∧ ¬ (3 ∣ u) ∧ ¬ (3 ∣ v)) :=
sorry

end NUMINAMATH_GPT_max_not_divisible_by_3_l1182_118298


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l1182_118222

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n, a (n + 1) = a n + 2

-- Statement of the theorem with conditions and conclusion
theorem arithmetic_sequence_a5 :
  ∃ a : ℕ → ℕ, is_arithmetic_sequence a ∧ a 1 = 1 ∧ a 5 = 9 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l1182_118222


namespace NUMINAMATH_GPT_factorize_expression_l1182_118225

theorem factorize_expression (a : ℝ) : a^3 + 2*a^2 + a = a*(a+1)^2 :=
  sorry

end NUMINAMATH_GPT_factorize_expression_l1182_118225


namespace NUMINAMATH_GPT_no_rearrangement_to_positive_and_negative_roots_l1182_118250

theorem no_rearrangement_to_positive_and_negative_roots (a b c : ℝ) :
  (∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ a ≠ 0 ∧ b = -a * (x1 + x2) ∧ c = a * x1 * x2) →
  (∃ y1 y2 : ℝ, y1 > 0 ∧ y2 > 0 ∧ a ≠ 0 ∧ b != 0 ∧ c != 0 ∧ 
    (∃ b' c' : ℝ, b' ≠ b ∧ c' ≠ c ∧ 
      b' = -a * (y1 + y2) ∧ c' = a * y1 * y2)) →
  False := by
  sorry

end NUMINAMATH_GPT_no_rearrangement_to_positive_and_negative_roots_l1182_118250


namespace NUMINAMATH_GPT_freeze_time_l1182_118261

theorem freeze_time :
  ∀ (minutes_per_smoothie total_minutes num_smoothies freeze_time: ℕ),
    minutes_per_smoothie = 3 →
    total_minutes = 55 →
    num_smoothies = 5 →
    freeze_time = total_minutes - (num_smoothies * minutes_per_smoothie) →
    freeze_time = 40 :=
by
  intros minutes_per_smoothie total_minutes num_smoothies freeze_time
  intros H1 H2 H3 H4
  subst H1
  subst H2
  subst H3
  subst H4
  sorry

end NUMINAMATH_GPT_freeze_time_l1182_118261


namespace NUMINAMATH_GPT_length_of_CD_l1182_118273

theorem length_of_CD {L : ℝ} (h₁ : 16 * Real.pi * L + (256 / 3) * Real.pi = 432 * Real.pi) :
  L = (50 / 3) :=
by
  sorry

end NUMINAMATH_GPT_length_of_CD_l1182_118273


namespace NUMINAMATH_GPT_solution_l1182_118269

theorem solution (x y : ℝ) (h1 : x ≠ y) (h2 : x^2 - 2000 * x = y^2 - 2000 * y) : 
  x + y = 2000 := 
by 
  sorry

end NUMINAMATH_GPT_solution_l1182_118269


namespace NUMINAMATH_GPT__l1182_118254

@[simp] theorem upper_base_length (ABCD is_trapezoid: Boolean)
  (point_M: Boolean)
  (perpendicular_DM_AB: Boolean)
  (MC_eq_CD: Boolean)
  (AD_eq_d: ℝ)
  : BC = d / 2 := sorry

end NUMINAMATH_GPT__l1182_118254


namespace NUMINAMATH_GPT_sum_of_numbers_is_37_l1182_118233

theorem sum_of_numbers_is_37 :
  ∃ (A B : ℕ), 
    1 ≤ A ∧ A ≤ 50 ∧ 1 ≤ B ∧ B ≤ 50 ∧ A ≠ B ∧
    (50 * B + A = k^2) ∧ Prime B ∧ B > 10 ∧
    A + B = 37 
  := by
    sorry

end NUMINAMATH_GPT_sum_of_numbers_is_37_l1182_118233


namespace NUMINAMATH_GPT_wire_length_l1182_118207

theorem wire_length (S L : ℝ) (h1 : S = 10) (h2 : S = (2 / 5) * L) : S + L = 35 :=
by
  sorry

end NUMINAMATH_GPT_wire_length_l1182_118207


namespace NUMINAMATH_GPT_work_days_B_l1182_118255

theorem work_days_B (A_days B_days : ℕ) (hA : A_days = 12) (hTogether : (1/12 + 1/A_days) = (1/8)) : B_days = 24 := 
by
  revert hTogether -- reversing to tackle proof
  sorry

end NUMINAMATH_GPT_work_days_B_l1182_118255


namespace NUMINAMATH_GPT_pages_written_in_a_year_l1182_118201

-- Definitions based on conditions
def pages_per_letter : ℕ := 3
def letters_per_week : ℕ := 2
def friends : ℕ := 2
def weeks_per_year : ℕ := 52

-- Definition to calculate total pages written in a week
def weekly_pages (pages_per_letter : ℕ) (letters_per_week : ℕ) (friends : ℕ) : ℕ :=
  pages_per_letter * letters_per_week * friends

-- Definition to calculate total pages written in a year
def yearly_pages (weekly_pages : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weekly_pages * weeks_per_year

-- Theorem to prove the total pages written in a year
theorem pages_written_in_a_year : yearly_pages (weekly_pages pages_per_letter letters_per_week friends) weeks_per_year = 624 :=
by 
  sorry

end NUMINAMATH_GPT_pages_written_in_a_year_l1182_118201


namespace NUMINAMATH_GPT_range_of_m_l1182_118287

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (mx-1)*(x-2) > 0 ↔ (1/m < x ∧ x < 2)) → m < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1182_118287


namespace NUMINAMATH_GPT_time_for_B_and_C_l1182_118248

variables (a b c : ℝ)

-- Conditions
axiom cond1 : a = (1 / 2) * b
axiom cond2 : b = 2 * c
axiom cond3 : a + b + c = 1 / 26
axiom cond4 : a + b = 1 / 13
axiom cond5 : a + c = 1 / 39

-- Statement to prove
theorem time_for_B_and_C (a b c : ℝ) (cond1 : a = (1 / 2) * b)
                                      (cond2 : b = 2 * c)
                                      (cond3 : a + b + c = 1 / 26)
                                      (cond4 : a + b = 1 / 13)
                                      (cond5 : a + c = 1 / 39) :
  (1 / (b + c)) = 104 / 3 :=
sorry

end NUMINAMATH_GPT_time_for_B_and_C_l1182_118248


namespace NUMINAMATH_GPT_length_of_AB_l1182_118279

-- Define the problem variables
variables (AB CD : ℝ)
variables (h : ℝ)

-- Define the conditions
def ratio_condition (AB CD : ℝ) : Prop :=
  AB / CD = 7 / 3

def length_condition (AB CD : ℝ) : Prop :=
  AB + CD = 210

-- Lean statement combining the conditions and the final result
theorem length_of_AB (h : ℝ) (AB CD : ℝ) (h_ratio : ratio_condition AB CD) (h_length : length_condition AB CD) : 
  AB = 147 :=
by
  -- Definitions and proof would go here
  sorry

end NUMINAMATH_GPT_length_of_AB_l1182_118279


namespace NUMINAMATH_GPT_largest_s_value_l1182_118221

theorem largest_s_value (r s : ℕ) (h_r : r ≥ 3) (h_s : s ≥ 3) 
  (h_angle : (r - 2) * 180 / r = (5 * (s - 2) * 180) / (4 * s)) : s ≤ 130 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_s_value_l1182_118221


namespace NUMINAMATH_GPT_tan_4530_l1182_118285

noncomputable def tan_of_angle (deg : ℝ) : ℝ := Real.tan (deg * Real.pi / 180)

theorem tan_4530 : tan_of_angle 4530 = -1 / Real.sqrt 3 := sorry

end NUMINAMATH_GPT_tan_4530_l1182_118285


namespace NUMINAMATH_GPT_find_T_l1182_118203

variable (a b c T : ℕ)

theorem find_T (h1 : a + b + c = 84) (h2 : a - 5 = T) (h3 : b + 9 = T) (h4 : 5 * c = T) : T = 40 :=
sorry

end NUMINAMATH_GPT_find_T_l1182_118203


namespace NUMINAMATH_GPT_gcf_of_24_and_16_l1182_118282

theorem gcf_of_24_and_16 :
  let n := 24
  let lcm := 48
  gcd n 16 = 8 :=
by
  sorry

end NUMINAMATH_GPT_gcf_of_24_and_16_l1182_118282


namespace NUMINAMATH_GPT_habitable_fraction_of_earth_l1182_118245

theorem habitable_fraction_of_earth :
  (1 / 2) * (1 / 4) = 1 / 8 := by
  sorry

end NUMINAMATH_GPT_habitable_fraction_of_earth_l1182_118245


namespace NUMINAMATH_GPT_mary_mortgage_payment_l1182_118249

theorem mary_mortgage_payment :
  let a1 := 400
  let r := 2
  let n := 11
  let sum_geom_series (a1 r : ℕ) (n : ℕ) : ℕ := (a1 * (1 - r^n)) / (1 - r)
  sum_geom_series a1 r n = 819400 :=
by
  let a1 := 400
  let r := 2
  let n := 11
  let sum_geom_series (a1 r : ℕ) (n : ℕ) : ℕ := (a1 * (1 - r^n)) / (1 - r)
  have h : sum_geom_series a1 r n = 819400 := sorry
  exact h

end NUMINAMATH_GPT_mary_mortgage_payment_l1182_118249


namespace NUMINAMATH_GPT_value_of_t_l1182_118238

theorem value_of_t (k m r s t : ℕ) 
  (hk : 1 ≤ k) (hm : 2 ≤ m) (hr : r = 13) (hs : s = 14)
  (h : k < m) (h' : m < r) (h'' : r < s) (h''' : s < t)
  (average_condition : (k + m + r + s + t) / 5 = 10) :
  t = 20 := 
sorry

end NUMINAMATH_GPT_value_of_t_l1182_118238


namespace NUMINAMATH_GPT_probability_at_least_5_heads_l1182_118272

def fair_coin_probability_at_least_5_heads : ℚ :=
  (Nat.choose 7 5 + Nat.choose 7 6 + Nat.choose 7 7) / 2^7

theorem probability_at_least_5_heads :
  fair_coin_probability_at_least_5_heads = 29 / 128 := 
  by
    sorry

end NUMINAMATH_GPT_probability_at_least_5_heads_l1182_118272


namespace NUMINAMATH_GPT_div_add_fraction_l1182_118291

theorem div_add_fraction : (3 / 7) / 4 + 2 = 59 / 28 :=
by
  sorry

end NUMINAMATH_GPT_div_add_fraction_l1182_118291


namespace NUMINAMATH_GPT_total_points_l1182_118260

noncomputable def Darius_points : ℕ := 10
noncomputable def Marius_points : ℕ := Darius_points + 3
noncomputable def Matt_points : ℕ := Darius_points + 5
noncomputable def Sofia_points : ℕ := 2 * Matt_points

theorem total_points : Darius_points + Marius_points + Matt_points + Sofia_points = 68 :=
by
  -- Definitions are directly from the problem statement, proof skipped 
  sorry

end NUMINAMATH_GPT_total_points_l1182_118260
