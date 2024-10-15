import Mathlib

namespace NUMINAMATH_GPT_painting_cost_conversion_l702_70238

def paintingCostInCNY (paintingCostNAD : ℕ) (usd_to_nad : ℕ) (usd_to_cny : ℕ) : ℕ :=
  paintingCostNAD * (1 / usd_to_nad) * usd_to_cny

theorem painting_cost_conversion :
  (paintingCostInCNY 105 7 6 = 90) :=
by
  sorry

end NUMINAMATH_GPT_painting_cost_conversion_l702_70238


namespace NUMINAMATH_GPT_three_digit_integer_one_more_than_LCM_l702_70282

theorem three_digit_integer_one_more_than_LCM:
  ∃ (n : ℕ), (n > 99 ∧ n < 1000) ∧ (∃ (k : ℕ), n = k + 1 ∧ (∃ m, k = 3 * 4 * 5 * 7 * 2^m)) :=
  sorry

end NUMINAMATH_GPT_three_digit_integer_one_more_than_LCM_l702_70282


namespace NUMINAMATH_GPT_find_rainy_days_l702_70232

theorem find_rainy_days 
  (n d T H P R : ℤ) 
  (h1 : R + (d - R) = d)
  (h2 : 3 * (d - R) = T)
  (h3 : n * R = H)
  (h4 : T = H + P)
  (hd : 1 ≤ d ∧ d ≤ 31)
  (hR_range : 0 ≤ R ∧ R ≤ d) :
  R = (3 * d - P) / (n + 3) :=
sorry

end NUMINAMATH_GPT_find_rainy_days_l702_70232


namespace NUMINAMATH_GPT_polygon_triangle_existence_l702_70289

theorem polygon_triangle_existence (n : ℕ) (h₁ : n > 1)
  (h₂ : ∀ (k₁ k₂ : ℕ), k₁ ≠ k₂ → (4 ≤ k₁) → (4 ≤ k₂) → k₁ ≠ k₂) :
  ∃ k, k = 3 :=
by
  sorry

end NUMINAMATH_GPT_polygon_triangle_existence_l702_70289


namespace NUMINAMATH_GPT_a_squared_divisible_by_b_l702_70249

theorem a_squared_divisible_by_b (a b : ℕ) (h1 : a < 1000) (h2 : b > 0) 
    (h3 : ∃ k, a ^ 21 = b ^ 10 * k) : ∃ m, a ^ 2 = b * m := 
by
  sorry

end NUMINAMATH_GPT_a_squared_divisible_by_b_l702_70249


namespace NUMINAMATH_GPT_total_cost_paper_plates_and_cups_l702_70259

theorem total_cost_paper_plates_and_cups :
  ∀ (P C : ℝ), (20 * P + 40 * C = 1.20) → (100 * P + 200 * C = 6.00) := by
  intros P C h
  sorry

end NUMINAMATH_GPT_total_cost_paper_plates_and_cups_l702_70259


namespace NUMINAMATH_GPT_program_output_l702_70296

-- Define the initial conditions
def initial_a := 1
def initial_b := 3

-- Define the program transformations
def a_step1 (a b : ℕ) := a + b
def b_step2 (a b : ℕ) := a - b

-- Define the final values after program execution
def final_a := a_step1 initial_a initial_b
def final_b := b_step2 final_a initial_b

-- Statement to prove
theorem program_output :
  final_a = 4 ∧ final_b = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_program_output_l702_70296


namespace NUMINAMATH_GPT_miranda_saved_per_month_l702_70208

-- Definition of the conditions and calculation in the problem
def total_cost : ℕ := 260
def sister_contribution : ℕ := 50
def months : ℕ := 3
def miranda_savings : ℕ := total_cost - sister_contribution
def saved_per_month : ℕ := miranda_savings / months

-- Theorem statement with the expected answer
theorem miranda_saved_per_month : saved_per_month = 70 :=
by
  sorry

end NUMINAMATH_GPT_miranda_saved_per_month_l702_70208


namespace NUMINAMATH_GPT_helen_hand_washing_time_l702_70212

theorem helen_hand_washing_time :
  (52 / 4) * 30 / 60 = 6.5 := by
  sorry

end NUMINAMATH_GPT_helen_hand_washing_time_l702_70212


namespace NUMINAMATH_GPT_cost_of_one_dozen_pens_l702_70270

-- Define the initial conditions
def cost_pen : ℕ := 65
def cost_pencil := cost_pen / 5
def total_cost (pencils : ℕ) := 3 * cost_pen + pencils * cost_pencil

-- State the theorem
theorem cost_of_one_dozen_pens (pencils : ℕ) (h : total_cost pencils = 260) :
  12 * cost_pen = 780 :=
by
  -- Preamble to show/conclude that the proofs are given
  sorry

end NUMINAMATH_GPT_cost_of_one_dozen_pens_l702_70270


namespace NUMINAMATH_GPT_tan_sum_pi_over_4_sin_cos_fraction_l702_70229

open Real

variable (α : ℝ)

axiom tan_α_eq_2 : tan α = 2

theorem tan_sum_pi_over_4 (α : ℝ) (h : tan α = 2) : tan (α + π / 4) = -3 :=
sorry

theorem sin_cos_fraction (α : ℝ) (h : tan α = 2) : (sin α + cos α) / (sin α - cos α) = 3 :=
sorry

end NUMINAMATH_GPT_tan_sum_pi_over_4_sin_cos_fraction_l702_70229


namespace NUMINAMATH_GPT_least_number_to_subtract_l702_70298

theorem least_number_to_subtract (x : ℕ) (h : x = 1234567890) : ∃ n, x - n = 5 := 
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l702_70298


namespace NUMINAMATH_GPT_k_range_correct_l702_70243

noncomputable def k_range (k : ℝ) : Prop :=
  (∀ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∧
  (∀ x : ℝ, k * x ^ 2 + k * x + 1 > 0) ∧
  ((∃ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∨
   (∃ x : ℝ, k * x ^ 2 + k * x + 1 > 0)) ∧
  ¬ ((∃ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∧
    (∃ x : ℝ, k * x ^ 2 + k * x + 1 > 0))

theorem k_range_correct (k : ℝ) : k_range k ↔ (-3 < k ∧ k < 0) ∨ (3 ≤ k ∧ k < 4) :=
sorry

end NUMINAMATH_GPT_k_range_correct_l702_70243


namespace NUMINAMATH_GPT_find_MN_sum_l702_70209

noncomputable def M : ℝ := sorry -- Placeholder for the actual non-zero solution M
noncomputable def N : ℝ := M ^ 2

theorem find_MN_sum :
  (M^2 = N) ∧ (Real.log N / Real.log M = Real.log M / Real.log N) ∧ (M ≠ N) ∧ (M ≠ 1) ∧ (N ≠ 1) → (M + N = 6) :=
by
  intros h
  exact sorry -- Will be replaced by the actual proof


end NUMINAMATH_GPT_find_MN_sum_l702_70209


namespace NUMINAMATH_GPT_square_vectors_l702_70283

theorem square_vectors (AB CD AD : ℝ × ℝ)
  (side_length: ℝ)
  (M N : ℝ × ℝ)
  (x y: ℝ)
  (MN : ℝ × ℝ):
  side_length = 2 →
  M = ((AB.1 + CD.1) / 2, (AB.2 + CD.2) / 2) →
  N = ((CD.1 + AD.1) / 2, (CD.2 + AD.2) / 2) →
  MN = (x * AB.1 + y * AD.1, x * AB.2 + y * AD.2) →
  (x = -1/2) ∧ (y = 1/2) →
  (x * y = -1/4) ∧ ((N.1 - M.1) * AD.1 + (N.2 - M.2) * AD.2 - (N.1 - M.1) * AB.1 - (N.2 - M.2) * AB.2 = -1) :=
by
  intros side_length_cond M_cond N_cond MN_cond xy_cond
  sorry

end NUMINAMATH_GPT_square_vectors_l702_70283


namespace NUMINAMATH_GPT_final_net_earnings_l702_70299

-- Declare constants representing the problem conditions
def connor_hourly_rate : ℝ := 7.20
def connor_hours_worked : ℝ := 8.0
def emily_hourly_rate : ℝ := 2 * connor_hourly_rate
def sarah_hourly_rate : ℝ := 5 * connor_hourly_rate
def emily_hours_worked : ℝ := 10.0
def connor_deduction_rate : ℝ := 0.05
def emily_deduction_rate : ℝ := 0.08
def sarah_deduction_rate : ℝ := 0.10

-- Combined final net earnings for the day
def combined_final_net_earnings (connor_hourly_rate emily_hourly_rate sarah_hourly_rate
                                  connor_hours_worked emily_hours_worked
                                  connor_deduction_rate emily_deduction_rate sarah_deduction_rate : ℝ) : ℝ :=
  let connor_gross := connor_hourly_rate * connor_hours_worked
  let emily_gross := emily_hourly_rate * emily_hours_worked
  let sarah_gross := sarah_hourly_rate * connor_hours_worked
  let connor_net := connor_gross * (1 - connor_deduction_rate)
  let emily_net := emily_gross * (1 - emily_deduction_rate)
  let sarah_net := sarah_gross * (1 - sarah_deduction_rate)
  connor_net + emily_net + sarah_net

-- The theorem statement proving their combined final net earnings
theorem final_net_earnings : 
  combined_final_net_earnings 7.20 14.40 36.00 8.0 10.0 0.05 0.08 0.10 = 498.24 :=
by sorry

end NUMINAMATH_GPT_final_net_earnings_l702_70299


namespace NUMINAMATH_GPT_spadesuit_evaluation_l702_70201

-- Define the operation
def spadesuit (x y : ℚ) : ℚ := x - (1 / y)

-- Prove the main statement
theorem spadesuit_evaluation : spadesuit 3 (spadesuit 3 (3 / 2)) = 18 / 7 :=
by
  sorry

end NUMINAMATH_GPT_spadesuit_evaluation_l702_70201


namespace NUMINAMATH_GPT_algebraic_expression_value_l702_70245

-- Define the problem conditions and the final proof statement.
theorem algebraic_expression_value : 
  (∀ m n : ℚ, (2 * m - 1 = 0) → (1 / 2 * n - 2 * m = 0) → m ^ 2023 * n ^ 2022 = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l702_70245


namespace NUMINAMATH_GPT_perp_case_parallel_distance_l702_70292

open Real

-- Define the line equations
def l1 (x y : ℝ) := 2 * x + y + 4 = 0
def l2 (a x y : ℝ) := a * x + 4 * y + 1 = 0

-- Perpendicular condition between l1 and l2
def perpendicular (a : ℝ) := (∃ x y : ℝ, l1 x y ∧ l2 a x y ∧ (2 * -a) / 4 = -1)

-- Parallel condition between l1 and l2
def parallel (a : ℝ) := (∃ x y : ℝ, l1 x y ∧ l2 a x y ∧ a = 8)

noncomputable def intersection_point : (ℝ × ℝ) := (-3/2, -1)

noncomputable def distance_between_lines : ℝ := (3 * sqrt 5) / 4

-- Statement for the intersection point when perpendicular
theorem perp_case (a : ℝ) : perpendicular a → ∃ x y, l1 x y ∧ l2 (-2) x y := 
by
  sorry

-- Statement for the distance when parallel
theorem parallel_distance {a : ℝ} : parallel a → distance_between_lines = (3 * sqrt 5) / 4 :=
by
  sorry

end NUMINAMATH_GPT_perp_case_parallel_distance_l702_70292


namespace NUMINAMATH_GPT_sin_120_eq_sqrt3_div_2_l702_70291

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_120_eq_sqrt3_div_2_l702_70291


namespace NUMINAMATH_GPT_flag_design_l702_70265

/-- Given three colors and a flag with three horizontal stripes where no adjacent stripes can be the 
same color, there are exactly 12 different possible flags. -/
theorem flag_design {colors : Finset ℕ} (h_colors : colors.card = 3) : 
  ∃ n : ℕ, n = 12 ∧ (∃ f : ℕ → ℕ, (∀ i, f i ∈ colors) ∧ (∀ i < 2, f i ≠ f (i + 1))) :=
sorry

end NUMINAMATH_GPT_flag_design_l702_70265


namespace NUMINAMATH_GPT_initial_range_calculation_l702_70272

variable (initial_range telescope_range : ℝ)
variable (increased_by : ℝ)
variable (h_telescope : telescope_range = increased_by * initial_range)

theorem initial_range_calculation 
  (h_telescope_range : telescope_range = 150)
  (h_increased_by : increased_by = 3)
  (h_telescope : telescope_range = increased_by * initial_range) :
  initial_range = 50 :=
  sorry

end NUMINAMATH_GPT_initial_range_calculation_l702_70272


namespace NUMINAMATH_GPT_solve_for_angle_B_solutions_l702_70261

noncomputable def number_of_solutions_for_angle_B (BC AC : ℝ) (angle_A : ℝ) : ℕ :=
  if (BC = 6 ∧ AC = 8 ∧ angle_A = 40) then 2 else 0

theorem solve_for_angle_B_solutions : number_of_solutions_for_angle_B 6 8 40 = 2 :=
  by sorry

end NUMINAMATH_GPT_solve_for_angle_B_solutions_l702_70261


namespace NUMINAMATH_GPT_total_students_l702_70279

-- Definition of variables and conditions
def M := 50
def E := 4 * M - 3

-- Statement of the theorem to prove
theorem total_students : E + M = 247 := by
  sorry

end NUMINAMATH_GPT_total_students_l702_70279


namespace NUMINAMATH_GPT_find_m_l702_70262

theorem find_m (m : ℕ) : (11 - m + 1 = 5) → m = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l702_70262


namespace NUMINAMATH_GPT_Thabo_harcdover_nonfiction_books_l702_70252

theorem Thabo_harcdover_nonfiction_books 
  (H P F : ℕ)
  (h1 : P = H + 20)
  (h2 : F = 2 * P)
  (h3 : H + P + F = 180) : 
  H = 30 :=
by
  sorry

end NUMINAMATH_GPT_Thabo_harcdover_nonfiction_books_l702_70252


namespace NUMINAMATH_GPT_length_of_AD_in_parallelogram_l702_70236

theorem length_of_AD_in_parallelogram
  (x : ℝ)
  (AB BC CD : ℝ)
  (AB_eq : AB = x + 3)
  (BC_eq : BC = x - 4)
  (CD_eq : CD = 16)
  (parallelogram_ABCD : AB = CD ∧ AD = BC) :
  AD = 9 := by
sorry

end NUMINAMATH_GPT_length_of_AD_in_parallelogram_l702_70236


namespace NUMINAMATH_GPT_discount_percentage_is_30_l702_70213

theorem discount_percentage_is_30 
  (price_per_pant : ℝ) (num_of_pants : ℕ)
  (price_per_sock : ℝ) (num_of_socks : ℕ)
  (total_spend_after_discount : ℝ)
  (original_pants_price := num_of_pants * price_per_pant)
  (original_socks_price := num_of_socks * price_per_sock)
  (original_total_price := original_pants_price + original_socks_price)
  (discount_amount := original_total_price - total_spend_after_discount)
  (discount_percentage := (discount_amount / original_total_price) * 100) :
  (price_per_pant = 110) ∧ 
  (num_of_pants = 4) ∧ 
  (price_per_sock = 60) ∧ 
  (num_of_socks = 2) ∧ 
  (total_spend_after_discount = 392) →
  discount_percentage = 30 := by
  sorry

end NUMINAMATH_GPT_discount_percentage_is_30_l702_70213


namespace NUMINAMATH_GPT_lucas_raspberry_candies_l702_70219

-- Define the problem conditions and the question
theorem lucas_raspberry_candies :
  ∃ (r l : ℕ), (r = 3 * l) ∧ ((r - 5) = 4 * (l - 5)) ∧ (r = 45) :=
by
  sorry

end NUMINAMATH_GPT_lucas_raspberry_candies_l702_70219


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l702_70228

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4 * x^2 = (3 - 2 * x) * (3 + 2 * x) :=
sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l702_70228


namespace NUMINAMATH_GPT_solution_set_for_inequality_l702_70253

theorem solution_set_for_inequality : 
  { x : ℝ | x * (x - 1) < 2 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l702_70253


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l702_70274

theorem isosceles_triangle_perimeter (a b : ℝ) (h_iso : a = 4 ∨ b = 4) (h_iso2 : a = 8 ∨ b = 8) : 
  (a = 4 ∧ b = 8 ∧ 4 + a + b = 16 ∨ 
  a = 4 ∧ b = 8 ∧ b + a + a = 20 ∨ 
  a = 8 ∧ b = 4 ∧ a + a + b = 20) :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l702_70274


namespace NUMINAMATH_GPT_find_q_l702_70273

open Real

noncomputable def q := (9 + 3 * Real.sqrt 5) / 2

theorem find_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l702_70273


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_l702_70221

theorem inequality_holds_for_all_x (m : ℝ) : (∀ x : ℝ, (m^2 + 4*m - 5)*x^2 - 4*(m - 1)*x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_holds_for_all_x_l702_70221


namespace NUMINAMATH_GPT_correct_value_l702_70246

theorem correct_value (x : ℝ) (h : x / 3.6 = 2.5) : (x * 3.6) / 2 = 16.2 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_correct_value_l702_70246


namespace NUMINAMATH_GPT_extreme_values_of_f_max_min_values_on_interval_l702_70239

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (Real.exp x)

theorem extreme_values_of_f : 
  (∃ x_max : ℝ, f x_max = 2 / Real.exp 1 ∧ ∀ x : ℝ, f x ≤ 2 / Real.exp 1) :=
sorry

theorem max_min_values_on_interval : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, 
    (f 1 = 2 / Real.exp 1 ∧ ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → f x ≤ 2 / Real.exp 1)
     ∧ (f 2 = 4 / (Real.exp 2) ∧ ∀ x ∈ Set.Icc (1/2 : ℝ) 2, 4 / (Real.exp 2) ≤ f x)) :=
sorry

end NUMINAMATH_GPT_extreme_values_of_f_max_min_values_on_interval_l702_70239


namespace NUMINAMATH_GPT_enclosed_area_of_curve_l702_70286

noncomputable def radius_of_arcs := 1

noncomputable def arc_length := (1 / 2) * Real.pi

noncomputable def side_length_of_octagon := 3

noncomputable def area_of_octagon (s : ℝ) := 
  2 * (1 + Real.sqrt 2) * s ^ 2

noncomputable def area_of_sectors (n : ℕ) (arc_radius : ℝ) (arc_theta : ℝ) := 
  n * (1 / 4) * Real.pi

theorem enclosed_area_of_curve : 
  area_of_octagon side_length_of_octagon + area_of_sectors 12 radius_of_arcs arc_length 
  = 54 + 54 * Real.sqrt 2 + 3 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_enclosed_area_of_curve_l702_70286


namespace NUMINAMATH_GPT_max_value_t_min_value_y_l702_70293

-- 1. Prove that the maximum value of t given |2x+5| + |2x-1| - t ≥ 0 is s = 6.
theorem max_value_t (t : ℝ) (x : ℝ) :
  (abs (2*x + 5) + abs (2*x - 1) - t ≥ 0) → (t ≤ 6) :=
by sorry

-- 2. Given s = 6 and 4a + 5b = s, prove that the minimum value of y = 1/(a+2b) + 4/(3a+3b) is y = 3/2.
theorem min_value_y (a b : ℝ) (s : ℝ) :
  s = 6 → (4*a + 5*b = s) → (a > 0) → (b > 0) → 
  (1/(a + 2*b) + 4/(3*a + 3*b) ≥ 3/2) :=
by sorry

end NUMINAMATH_GPT_max_value_t_min_value_y_l702_70293


namespace NUMINAMATH_GPT_P_neither_l702_70257

-- Definition of probabilities according to given conditions
def P_A : ℝ := 0.63      -- Probability of answering the first question correctly
def P_B : ℝ := 0.50      -- Probability of answering the second question correctly
def P_A_and_B : ℝ := 0.33  -- Probability of answering both questions correctly

-- Theorem to prove the probability of answering neither of the questions correctly
theorem P_neither : (1 - (P_A + P_B - P_A_and_B)) = 0.20 := by
  sorry

end NUMINAMATH_GPT_P_neither_l702_70257


namespace NUMINAMATH_GPT_circle_radius_five_iff_l702_70200

noncomputable def circle_eq_radius (x y : ℝ) (k : ℝ) : Prop :=
  x^2 + 8*x + y^2 + 4*y - k = 0

def is_circle_with_radius (r : ℝ) (x y : ℝ) (k : ℝ) : Prop :=
  circle_eq_radius x y k ↔ r = 5 ∧ k = 5

theorem circle_radius_five_iff (k : ℝ) :
  (∃ x y : ℝ, circle_eq_radius x y k) ↔ k = 5 :=
sorry

end NUMINAMATH_GPT_circle_radius_five_iff_l702_70200


namespace NUMINAMATH_GPT_max_water_bottles_one_athlete_l702_70204

-- Define variables and key conditions
variable (total_bottles : Nat := 40)
variable (total_athletes : Nat := 25)
variable (at_least_one : ∀ i, i < total_athletes → Nat.succ i ≥ 1)

-- Define the problem as a theorem
theorem max_water_bottles_one_athlete (h_distribution : total_bottles = 40) :
  ∃ max_bottles, max_bottles = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_water_bottles_one_athlete_l702_70204


namespace NUMINAMATH_GPT_simplify_expression_l702_70294

theorem simplify_expression :
  2 + 3 / (4 + 5 / (6 + 7 / 8)) = 137 / 52 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l702_70294


namespace NUMINAMATH_GPT_rhombus_area_l702_70256

-- Define the lengths of the diagonals
def d1 : ℝ := 25
def d2 : ℝ := 30

-- Statement to prove that the area of the rhombus is 375 square centimeters
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 25) (h2 : d2 = 30) : 
  (d1 * d2) / 2 = 375 := by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_rhombus_area_l702_70256


namespace NUMINAMATH_GPT_num_ordered_triples_pos_int_l702_70281

theorem num_ordered_triples_pos_int
  (lcm_ab: lcm a b = 180)
  (lcm_ac: lcm a c = 450)
  (lcm_bc: lcm b c = 1200)
  (gcd_abc: gcd (gcd a b) c = 3) :
  ∃ n: ℕ, n = 4 :=
sorry

end NUMINAMATH_GPT_num_ordered_triples_pos_int_l702_70281


namespace NUMINAMATH_GPT_length_of_each_piece_l702_70231

theorem length_of_each_piece (rod_length : ℝ) (num_pieces : ℕ) (h₁ : rod_length = 42.5) (h₂ : num_pieces = 50) : (rod_length / num_pieces * 100) = 85 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_each_piece_l702_70231


namespace NUMINAMATH_GPT_least_positive_integer_special_property_l702_70267

theorem least_positive_integer_special_property : ∃ (N : ℕ) (a b c : ℕ), 
  N = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 10 * b + c = N / 29 ∧ N = 725 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_special_property_l702_70267


namespace NUMINAMATH_GPT_barbed_wire_cost_l702_70288

theorem barbed_wire_cost
  (A : ℕ)          -- Area of the square field (sq m)
  (cost_per_meter : ℕ)  -- Cost per meter for the barbed wire (Rs)
  (gate_width : ℕ)      -- Width of each gate (m)
  (num_gates : ℕ)       -- Number of gates
  (side_length : ℕ)     -- Side length of the square field (m)
  (perimeter : ℕ)       -- Perimeter of the square field (m)
  (total_length : ℕ)    -- Total length of the barbed wire needed (m)
  (total_cost : ℕ)      -- Total cost of drawing the barbed wire (Rs)
  (h1 : A = 3136)       -- Given: Area = 3136 sq m
  (h2 : cost_per_meter = 1)  -- Given: Cost per meter = 1 Rs/m
  (h3 : gate_width = 1)      -- Given: Width of each gate = 1 m
  (h4 : num_gates = 2)       -- Given: Number of gates = 2
  (h5 : side_length * side_length = A)  -- Side length calculated from the area
  (h6 : perimeter = 4 * side_length)    -- Perimeter of the square field
  (h7 : total_length = perimeter - (num_gates * gate_width))  -- Actual barbed wire length after gates
  (h8 : total_cost = total_length * cost_per_meter)           -- Total cost calculation
  : total_cost = 222 :=      -- The result we need to prove
sorry

end NUMINAMATH_GPT_barbed_wire_cost_l702_70288


namespace NUMINAMATH_GPT_dave_total_time_l702_70227

variable (W J : ℕ)

-- Given conditions
def time_walked := W = 9
def ratio := J / W = 4 / 3

-- Statement to prove
theorem dave_total_time (time_walked : time_walked W) (ratio : ratio J W) : W + J = 21 := 
by
  sorry

end NUMINAMATH_GPT_dave_total_time_l702_70227


namespace NUMINAMATH_GPT_find_x_minus_y_l702_70218

theorem find_x_minus_y {x y z : ℤ} (h1 : x - (y + z) = 5) (h2 : x - y + z = -1) : x - y = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_minus_y_l702_70218


namespace NUMINAMATH_GPT_figure_can_be_rearranged_to_square_l702_70248

def can_form_square (n : ℕ) : Prop :=
  let s := Nat.sqrt n
  s * s = n

theorem figure_can_be_rearranged_to_square (n : ℕ) :
  (∃ a b c : ℕ, a + b + c = n) → (can_form_square n) → (n % 1 = 0) :=
by
  intros _ _
  sorry

end NUMINAMATH_GPT_figure_can_be_rearranged_to_square_l702_70248


namespace NUMINAMATH_GPT_bales_stored_in_barn_l702_70207

-- Defining the conditions
def bales_initial : Nat := 28
def bales_stacked : Nat := 28
def bales_already_there : Nat := 54

-- Formulate the proof statement
theorem bales_stored_in_barn : bales_already_there + bales_stacked = 82 := by
  sorry

end NUMINAMATH_GPT_bales_stored_in_barn_l702_70207


namespace NUMINAMATH_GPT_area_not_covered_correct_l702_70203

-- Define the dimensions of the rectangle
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 8

-- Define the side length of the square
def square_side_length : ℕ := 5

-- The area of the rectangle
def rectangle_area : ℕ := rectangle_length * rectangle_width

-- The area of the square
def square_area : ℕ := square_side_length * square_side_length

-- The area of the region not covered by the square
def area_not_covered : ℕ := rectangle_area - square_area

-- The theorem statement asserting the required area
theorem area_not_covered_correct : area_not_covered = 55 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_area_not_covered_correct_l702_70203


namespace NUMINAMATH_GPT_consecutive_product_even_product_divisible_by_6_l702_70215

theorem consecutive_product_even (n : ℕ) : ∃ k, n * (n + 1) = 2 * k := 
sorry

theorem product_divisible_by_6 (n : ℕ) : 6 ∣ (n * (n + 1) * (2 * n + 1)) :=
sorry

end NUMINAMATH_GPT_consecutive_product_even_product_divisible_by_6_l702_70215


namespace NUMINAMATH_GPT_tickets_distribution_correct_l702_70223

def tickets_distribution (tickets programs : nat) (A_tickets_min : nat) : nat :=
sorry

theorem tickets_distribution_correct :
  tickets_distribution 6 4 3 = 17 :=
by
  sorry

end NUMINAMATH_GPT_tickets_distribution_correct_l702_70223


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l702_70255

theorem sufficient_not_necessary_condition (x : ℝ) : (x^2 - 2 * x < 0) → (|x - 1| < 2) ∧ ¬( (|x - 1| < 2) → (x^2 - 2 * x < 0)) :=
by sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l702_70255


namespace NUMINAMATH_GPT_find_AD_find_a_rhombus_l702_70250

variable (a : ℝ) (AB AD : ℝ)

-- Problem 1: Given AB = 2, find AD
theorem find_AD (h1 : AB = 2)
    (h_quad : ∀ x, x^2 - (a-4)*x + (a-1) = 0 → x = AB ∨ x = AD) : AD = 5 := sorry

-- Problem 2: Find the value of a such that ABCD is a rhombus
theorem find_a_rhombus (h_quad : ∀ x, x^2 - (a-4)*x + (a-1) = 0 → x = 2 → AB = AD → x = a ∨ AB = AD → x = 10) :
    a = 10 := sorry

end NUMINAMATH_GPT_find_AD_find_a_rhombus_l702_70250


namespace NUMINAMATH_GPT_find_principal_6400_l702_70240

theorem find_principal_6400 (CI SI P : ℝ) (R T : ℝ) 
  (hR : R = 5) (hT : T = 2) 
  (hSI : SI = P * R * T / 100) 
  (hCI : CI = P * (1 + R / 100) ^ T - P) 
  (hDiff : CI - SI = 16) : 
  P = 6400 := 
by 
  sorry

end NUMINAMATH_GPT_find_principal_6400_l702_70240


namespace NUMINAMATH_GPT_solution_set_of_x_squared_gt_x_l702_70242

theorem solution_set_of_x_squared_gt_x :
  { x : ℝ | x^2 > x } = { x : ℝ | x < 0 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_x_squared_gt_x_l702_70242


namespace NUMINAMATH_GPT_primes_pos_int_solutions_l702_70226

theorem primes_pos_int_solutions 
  (p : ℕ) [hp : Fact (Nat.Prime p)] (a b : ℕ) (h1 : ∃ k : ℤ, (4 * a + p : ℤ) + k * (4 * b + p : ℤ) = b * k * a)
  (h2 : ∃ m : ℤ, (a^2 : ℤ) + m * (b^2 : ℤ) = b * m * a) : a = b ∨ a = b * p :=
  sorry

end NUMINAMATH_GPT_primes_pos_int_solutions_l702_70226


namespace NUMINAMATH_GPT_total_points_scored_l702_70244

theorem total_points_scored (m2 m3 m1 o2 o3 o1 : ℕ) 
  (H1 : m2 = 25) 
  (H2 : m3 = 8) 
  (H3 : m1 = 10) 
  (H4 : o2 = 2 * m2) 
  (H5 : o3 = m3 / 2) 
  (H6 : o1 = m1 / 2) : 
  (2 * m2 + 3 * m3 + m1) + (2 * o2 + 3 * o3 + o1) = 201 := 
by
  sorry

end NUMINAMATH_GPT_total_points_scored_l702_70244


namespace NUMINAMATH_GPT_solve_for_x_l702_70264

theorem solve_for_x (x y : ℝ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 6) : x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l702_70264


namespace NUMINAMATH_GPT_complex_fraction_sum_l702_70254

theorem complex_fraction_sum :
  let a := (1 : ℂ)
  let b := (0 : ℂ)
  (a + b) = 1 :=
by
  sorry

end NUMINAMATH_GPT_complex_fraction_sum_l702_70254


namespace NUMINAMATH_GPT_sum_of_cubes_roots_poly_l702_70214

theorem sum_of_cubes_roots_poly :
  (∀ (a b c : ℂ), (a^3 - 2*a^2 + 2*a - 3 = 0) ∧ (b^3 - 2*b^2 + 2*b - 3 = 0) ∧ (c^3 - 2*c^2 + 2*c - 3 = 0) → 
  a^3 + b^3 + c^3 = 5) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_roots_poly_l702_70214


namespace NUMINAMATH_GPT_mayoral_election_l702_70234

theorem mayoral_election :
  ∀ (X Y Z : ℕ), (X = Y + (Y / 2)) → (Y = Z - (2 * Z / 5)) → (Z = 25000) → X = 22500 :=
by
  intros X Y Z h1 h2 h3
  -- Proof here, not necessary for the task
  sorry

end NUMINAMATH_GPT_mayoral_election_l702_70234


namespace NUMINAMATH_GPT_total_tape_area_l702_70297

theorem total_tape_area 
  (long_side_1 short_side_1 : ℕ) (boxes_1 : ℕ)
  (long_side_2 short_side_2 : ℕ) (boxes_2 : ℕ)
  (long_side_3 short_side_3 : ℕ) (boxes_3 : ℕ)
  (overlap : ℕ) (tape_width : ℕ) :
  long_side_1 = 30 → short_side_1 = 15 → boxes_1 = 5 →
  long_side_2 = 40 → short_side_2 = 40 → boxes_2 = 2 →
  long_side_3 = 50 → short_side_3 = 20 → boxes_3 = 3 →
  overlap = 2 → tape_width = 2 →
  let total_length_1 := boxes_1 * (long_side_1 + overlap + 2 * (short_side_1 + overlap))
  let total_length_2 := boxes_2 * 3 * (long_side_2 + overlap)
  let total_length_3 := boxes_3 * (long_side_3 + overlap + 2 * (short_side_3 + overlap))
  let total_length := total_length_1 + total_length_2 + total_length_3
  let total_area := total_length * tape_width
  total_area = 1740 :=
  by
  -- Add the proof steps here
  -- sorry can be used to skip the proof
  sorry

end NUMINAMATH_GPT_total_tape_area_l702_70297


namespace NUMINAMATH_GPT_parabola_vertex_l702_70285

theorem parabola_vertex (x y : ℝ) : y^2 + 6*y + 2*x + 5 = 0 → (x, y) = (2, -3) :=
sorry

end NUMINAMATH_GPT_parabola_vertex_l702_70285


namespace NUMINAMATH_GPT_min_val_xy_l702_70205

theorem min_val_xy (x y : ℝ) 
  (h : 2 * (Real.cos (x + y - 1))^2 = ((x + 1)^2 + (y - 1)^2 - 2 * x * y) / (x - y + 1)) : 
  xy ≥ (1 / 4) :=
sorry

end NUMINAMATH_GPT_min_val_xy_l702_70205


namespace NUMINAMATH_GPT_expression_value_l702_70266

theorem expression_value (x y z : ℤ) (hx : x = -2) (hy : y = 1) (hz : z = 1) : 
  x^2 * y * z - x * y * z^2 = 6 :=
by
  rw [hx, hy, hz]
  rfl

end NUMINAMATH_GPT_expression_value_l702_70266


namespace NUMINAMATH_GPT_mildred_oranges_l702_70235

theorem mildred_oranges (original after given : ℕ) (h1 : original = 77) (h2 : after = 79) (h3 : given = after - original) : given = 2 :=
by
  sorry

end NUMINAMATH_GPT_mildred_oranges_l702_70235


namespace NUMINAMATH_GPT_man_fraction_ownership_l702_70263

theorem man_fraction_ownership :
  ∀ (F : ℚ), (3 / 5 * F = 15000) → (75000 = 75000) → (F / 75000 = 1 / 3) :=
by
  intros F h1 h2
  sorry

end NUMINAMATH_GPT_man_fraction_ownership_l702_70263


namespace NUMINAMATH_GPT_distance_symmetric_parabola_l702_70230

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def parabola (x : ℝ) : ℝ := 3 - x^2

theorem distance_symmetric_parabola (A B : ℝ × ℝ) 
  (hA : A.2 = parabola A.1) 
  (hB : B.2 = parabola B.1)
  (hSym : A.1 + A.2 = 0 ∧ B.1 + B.2 = 0) 
  (hDistinct : A ≠ B) :
  distance A B = 3 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_symmetric_parabola_l702_70230


namespace NUMINAMATH_GPT_result_of_4_times_3_l702_70237

def operation (a b : ℕ) : ℕ :=
  a^2 + a * Nat.factorial b - b^2

theorem result_of_4_times_3 : operation 4 3 = 31 := by
  sorry

end NUMINAMATH_GPT_result_of_4_times_3_l702_70237


namespace NUMINAMATH_GPT_woman_stop_time_l702_70258

-- Conditions
def man_speed := 5 -- in miles per hour
def woman_speed := 15 -- in miles per hour
def wait_time := 4 -- in minutes
def man_speed_mpm : ℚ := man_speed * (1 / 60) -- convert to miles per minute
def distance_covered := man_speed_mpm * wait_time

-- Definition of the relative speed between the woman and the man
def relative_speed := woman_speed - man_speed
def relative_speed_mpm : ℚ := relative_speed * (1 / 60) -- convert to miles per minute

-- The Proof statement
theorem woman_stop_time :
  (distance_covered / relative_speed_mpm) = 2 :=
by
  sorry

end NUMINAMATH_GPT_woman_stop_time_l702_70258


namespace NUMINAMATH_GPT_speed_of_stream_l702_70260

-- Definitions
variable (b s : ℝ)
def downstream_distance : ℝ := 120
def downstream_time : ℝ := 4
def upstream_distance : ℝ := 90
def upstream_time : ℝ := 6

-- Equations
def downstream_eq : Prop := downstream_distance = (b + s) * downstream_time
def upstream_eq : Prop := upstream_distance = (b - s) * upstream_time

-- Main statement
theorem speed_of_stream (h₁ : downstream_eq b s) (h₂ : upstream_eq b s) : s = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l702_70260


namespace NUMINAMATH_GPT_parabola_standard_equation_l702_70295

/-- Given that the directrix of a parabola coincides with the line on which the circles 
    x^2 + y^2 - 4 = 0 and x^2 + y^2 + y - 3 = 0 lie, the standard equation of the parabola 
    is x^2 = 4y.
-/
theorem parabola_standard_equation :
  (∀ x y : ℝ, x^2 + y^2 - 4 = 0 → x^2 + y^2 + y - 3 = 0 → y = -1) →
  ∀ p : ℝ, 4 * (p / 2) = 4 → x^2 = 4 * p * y :=
by
  sorry

end NUMINAMATH_GPT_parabola_standard_equation_l702_70295


namespace NUMINAMATH_GPT_car_quotient_div_15_l702_70275

/-- On a straight, one-way, single-lane highway, cars all travel at the same speed
    and obey a modified safety rule: the distance from the back of the car ahead
    to the front of the car behind is exactly two car lengths for each 20 kilometers
    per hour of speed. A sensor by the road counts the number of cars that pass in
    one hour. Each car is 5 meters long. 
    Let N be the maximum whole number of cars that can pass the sensor in one hour.
    Prove that when N is divided by 15, the quotient is 266. -/
theorem car_quotient_div_15 
  (speed : ℕ) 
  (d : ℕ) 
  (sensor_time : ℕ) 
  (car_length : ℕ)
  (N : ℕ)
  (h1 : ∀ m, speed = 20 * m)
  (h2 : d = 2 * car_length)
  (h3 : car_length = 5)
  (h4 : sensor_time = 1)
  (h5 : N = 4000) : 
  N / 15 = 266 := 
sorry

end NUMINAMATH_GPT_car_quotient_div_15_l702_70275


namespace NUMINAMATH_GPT_ratio_speed_car_speed_bike_l702_70225

def speed_of_tractor := 575 / 23
def speed_of_bike := 2 * speed_of_tractor
def speed_of_car := 540 / 6
def ratio := speed_of_car / speed_of_bike

theorem ratio_speed_car_speed_bike : ratio = 9 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_speed_car_speed_bike_l702_70225


namespace NUMINAMATH_GPT_range_of_x_l702_70241

theorem range_of_x (x : ℝ) (h : 2 * x - 4 ≥ 0) : x ≥ 2 :=
sorry

end NUMINAMATH_GPT_range_of_x_l702_70241


namespace NUMINAMATH_GPT_intersection_S_T_eq_T_l702_70211

noncomputable def S : Set ℝ := { y | ∃ x : ℝ, y = 3^x }
noncomputable def T : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end NUMINAMATH_GPT_intersection_S_T_eq_T_l702_70211


namespace NUMINAMATH_GPT_new_boarders_joined_l702_70202

theorem new_boarders_joined (initial_boarders new_boarders initial_day_students total_boarders total_day_students: ℕ)
  (h1: initial_boarders = 60)
  (h2: initial_day_students = 150)
  (h3: total_boarders = initial_boarders + new_boarders)
  (h4: total_day_students = initial_day_students)
  (h5: 2 * initial_day_students = 5 * initial_boarders)
  (h6: 2 * total_boarders = total_day_students) :
  new_boarders = 15 :=
by
  sorry

end NUMINAMATH_GPT_new_boarders_joined_l702_70202


namespace NUMINAMATH_GPT_maximum_value_of_z_l702_70206

theorem maximum_value_of_z :
  ∃ x y : ℝ, (x - y ≥ 0) ∧ (x + y ≤ 2) ∧ (y ≥ 0) ∧ (∀ u v : ℝ, (u - v ≥ 0) ∧ (u + v ≤ 2) ∧ (v ≥ 0) → 3 * u - v ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_z_l702_70206


namespace NUMINAMATH_GPT_solve_arctan_equation_l702_70210

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan (1 / x) + Real.arctan (1 / (x^3))

theorem solve_arctan_equation (x : ℝ) (hx : x = (1 + Real.sqrt 5) / 2) :
  f x = Real.pi / 4 :=
by
  rw [hx]
  sorry

end NUMINAMATH_GPT_solve_arctan_equation_l702_70210


namespace NUMINAMATH_GPT_max_weak_quartets_120_l702_70217

noncomputable def max_weak_quartets (n : ℕ) : ℕ :=
  -- Placeholder definition to represent the maximum weak quartets
  sorry  -- To be replaced with the actual mathematical definition

theorem max_weak_quartets_120 : max_weak_quartets 120 = 4769280 := by
  sorry

end NUMINAMATH_GPT_max_weak_quartets_120_l702_70217


namespace NUMINAMATH_GPT_problem1_l702_70269

theorem problem1 (x y : ℤ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 :=
sorry

end NUMINAMATH_GPT_problem1_l702_70269


namespace NUMINAMATH_GPT_collinear_iff_real_simple_ratio_l702_70222

theorem collinear_iff_real_simple_ratio (a b c : ℂ) : (∃ k : ℝ, a = k * b + (1 - k) * c) ↔ ∃ r : ℝ, (a - b) / (a - c) = r :=
sorry

end NUMINAMATH_GPT_collinear_iff_real_simple_ratio_l702_70222


namespace NUMINAMATH_GPT_alpha_beta_identity_l702_70287

open Real

theorem alpha_beta_identity 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h : cos β = tan α * (1 + sin β)) : 
  2 * α + β = π / 2 :=
by
  sorry

end NUMINAMATH_GPT_alpha_beta_identity_l702_70287


namespace NUMINAMATH_GPT_elisa_math_books_l702_70290

theorem elisa_math_books (N M L : ℕ) (h₀ : 24 + M + L + 1 = N + 1) (h₁ : (N + 1) % 9 = 0) (h₂ : (N + 1) % 4 = 0) (h₃ : N < 100) : M = 7 :=
by
  sorry

end NUMINAMATH_GPT_elisa_math_books_l702_70290


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_extreme_values_of_f_on_interval_l702_70280

noncomputable def f (x : ℝ) : ℝ :=
  let a : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 * Real.cos x)
  let b : ℝ × ℝ := (Real.cos x, 2 * Real.sin x)
  a.1 * b.1 + a.2 * b.2

theorem smallest_positive_period_of_f :
  ∃ p > 0, ∀ x : ℝ, f (x + p) = f x ∧ p = Real.pi := sorry

theorem extreme_values_of_f_on_interval :
  ∃ max_val min_val, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max_val) ∧
                     (∀ x ∈ Set.Icc 0 (Real.pi / 2), min_val ≤ f x) ∧
                     max_val = 3 ∧ min_val = 0 := sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_extreme_values_of_f_on_interval_l702_70280


namespace NUMINAMATH_GPT_complement_union_l702_70284

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 4, 5}
def complementU (A B : Set ℕ) : Set ℕ := U \ (A ∪ B)

theorem complement_union :
  complementU A B = {2, 6} := by
  sorry

end NUMINAMATH_GPT_complement_union_l702_70284


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l702_70271

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m)
  (h_a2 : a 2 = 3)
  (h_a7 : a 7 = 13) : 
  ∃ d, ∀ n, a n = a 1 + (n - 1) * d ∧ d = 2 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l702_70271


namespace NUMINAMATH_GPT_find_ab_value_l702_70220

-- Definitions from the conditions
def ellipse_eq (a b : ℝ) : Prop := b^2 - a^2 = 25
def hyperbola_eq (a b : ℝ) : Prop := a^2 + b^2 = 49

-- Main theorem statement
theorem find_ab_value {a b : ℝ} (h_ellipse : ellipse_eq a b) (h_hyperbola : hyperbola_eq a b) : 
  |a * b| = 2 * Real.sqrt 111 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_ab_value_l702_70220


namespace NUMINAMATH_GPT_triangle_is_right_l702_70233

variable {a b c : ℝ}

theorem triangle_is_right
  (h : a^3 + (Real.sqrt 2 / 4) * b^3 + (Real.sqrt 3 / 9) * c^3 - (Real.sqrt 6 / 2) * a * b * c = 0) :
  (a * a + b * b = c * c) :=
sorry

end NUMINAMATH_GPT_triangle_is_right_l702_70233


namespace NUMINAMATH_GPT_students_catching_up_on_homework_l702_70224

theorem students_catching_up_on_homework
  (total_students : ℕ)
  (half_doing_silent_reading : ℕ)
  (third_playing_board_games : ℕ)
  (remain_catching_up_homework : ℕ) :
  total_students = 24 →
  half_doing_silent_reading = total_students / 2 →
  third_playing_board_games = total_students / 3 →
  remain_catching_up_homework = total_students - (half_doing_silent_reading + third_playing_board_games) →
  remain_catching_up_homework = 4 :=
by
  intros h_total h_half h_third h_remain
  sorry

end NUMINAMATH_GPT_students_catching_up_on_homework_l702_70224


namespace NUMINAMATH_GPT_term_217_is_61st_l702_70247

variables {a_n : ℕ → ℝ}

def arithmetic_sequence (a_n : ℕ → ℝ) (a_15 a_45 : ℝ) : Prop :=
  ∃ (a₁ d : ℝ), (∀ n, a_n n = a₁ + (n - 1) * d) ∧ a_n 15 = a_15 ∧ a_n 45 = a_45

theorem term_217_is_61st (h : arithmetic_sequence a_n 33 153) : a_n 61 = 217 := sorry

end NUMINAMATH_GPT_term_217_is_61st_l702_70247


namespace NUMINAMATH_GPT_gcf_3465_10780_l702_70277

theorem gcf_3465_10780 : Nat.gcd 3465 10780 = 385 := by
  sorry

end NUMINAMATH_GPT_gcf_3465_10780_l702_70277


namespace NUMINAMATH_GPT_complex_number_in_first_quadrant_l702_70251

def is_in_first_quadrant (z : ℂ) : Prop :=
  0 < z.re ∧ 0 < z.im

theorem complex_number_in_first_quadrant (z : ℂ) (h : 0 < z.re ∧ 0 < z.im) : is_in_first_quadrant z :=
by sorry

end NUMINAMATH_GPT_complex_number_in_first_quadrant_l702_70251


namespace NUMINAMATH_GPT_arithmetic_mean_of_normal_distribution_l702_70276

theorem arithmetic_mean_of_normal_distribution
  (σ : ℝ) (hσ : σ = 1.5)
  (value : ℝ) (hvalue : value = 11.5)
  (hsd : value = μ - 2 * σ) :
  μ = 14.5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_normal_distribution_l702_70276


namespace NUMINAMATH_GPT_platform_length_is_correct_l702_70216

noncomputable def length_of_platform (train1_speed_kmph : ℕ) (train2_speed_kmph : ℕ) (cross_time_s : ℕ) (platform_time_s : ℕ) : ℕ :=
  let train1_speed_mps := train1_speed_kmph * 5 / 18
  let train2_speed_mps := train2_speed_kmph * 5 / 18
  let relative_speed := train1_speed_mps + train2_speed_mps
  let total_distance := relative_speed * cross_time_s
  let train1_length := 2 * total_distance / 3
  let platform_length := train1_speed_mps * platform_time_s
  platform_length

theorem platform_length_is_correct : length_of_platform 48 42 12 45 = 600 :=
by
  sorry

end NUMINAMATH_GPT_platform_length_is_correct_l702_70216


namespace NUMINAMATH_GPT_meal_cost_is_seven_l702_70268

-- Defining the given conditions
def total_cost : ℕ := 21
def number_of_meals : ℕ := 3

-- The amount each meal costs
def meal_cost : ℕ := total_cost / number_of_meals

-- Prove that each meal costs 7 dollars given the conditions
theorem meal_cost_is_seven : meal_cost = 7 :=
by
  -- The result follows directly from the definition of meal_cost
  unfold meal_cost
  have h : 21 / 3 = 7 := by norm_num
  exact h


end NUMINAMATH_GPT_meal_cost_is_seven_l702_70268


namespace NUMINAMATH_GPT_angle_C_in_triangle_l702_70278

theorem angle_C_in_triangle (A B C : ℝ)
  (hA : A = 60)
  (hAC : C = 2 * B)
  (hSum : A + B + C = 180) : C = 80 :=
sorry

end NUMINAMATH_GPT_angle_C_in_triangle_l702_70278
